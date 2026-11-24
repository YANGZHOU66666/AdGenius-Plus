"""ServerDistillationTrainer that projects to the vocabulary only at the positions the loss uses.

TRL 1.18's server-distillation `compute_loss` runs the student with full logits over prompt + completion
and then keeps `logits[:, prompt_length - 1 : -1]`. With a 248k vocabulary and ~1,200-token prompts this
materializes (and accelerate upcasts to fp32) >1 GB per sequence that is immediately discarded.

This subclass computes the same prompt boundary first and calls the model with
`logits_to_keep = seq_len - prompt_length + 1`, i.e. the last prompt position plus every completion position,
then takes `logits[:, :-1]`. The selected positions, and therefore the loss, are identical; the backbone still
processes the whole sequence. Everything else is copied unchanged from TRL 1.18.0.
"""

import torch
import torch.nn.functional as F
from trl.experimental.server_distillation import ServerDistillationTrainer


class CompletionLogitsServerDistillationTrainer(ServerDistillationTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_labels = torch.where(completion_mask.bool(), completion_ids, torch.full_like(completion_ids, -100))
        labels = torch.cat([torch.full_like(prompt_ids, -100), completion_labels], dim=1)
        inputs = {
            **inputs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_attention_mask": prompt_mask,
        }

        # Changed: use the padded prompt-block width as the boundary. On-policy batches are left-padded prompts
        # followed by right-padded completions, so every completion starts at prompt_ids.shape[1]. TRL's
        # _compute_prompt_length returns the shortest real prompt length instead, which in a batch of mixed
        # prompt lengths keeps hundreds of extra (fully masked) prompt positions. Then project only those positions.
        prompt_length = prompt_ids.shape[1]
        keep = inputs["input_ids"].shape[1] - prompt_length + 1
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            logits_to_keep=keep,
        )
        labels = inputs["labels"][:, prompt_length:]
        completion_tokens = inputs["input_ids"][:, prompt_length:]

        teacher_result = self._get_teacher_token_logprobs_from_server(inputs, prompt_length)

        # Changed: the kept logits start at position prompt_length - 1, so drop only the final position.
        student_logits = student_outputs.logits[:, :-1, :]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        comp_len = teacher_result["actual_logprobs"].shape[1]
        completion_tokens = completion_tokens[:, :comp_len]
        trimmed_labels = labels[:, :comp_len]

        if self.beta > 0:
            loss = self._compute_server_sparse_top_1_divergence_loss(
                teacher_result=teacher_result,
                student_log_probs=student_log_probs[:, :comp_len, :],
                completion_tokens=completion_tokens,
                labels=trimmed_labels,
            )
        else:
            loss = self._compute_server_forward_kl_loss(
                teacher_result=teacher_result,
                student_log_probs=student_log_probs[:, :comp_len, :],
                labels=trimmed_labels,
            )

        if self.model.training:
            loss = loss / self.current_gradient_accumulation_steps

        return (loss, student_outputs) if return_outputs else loss
