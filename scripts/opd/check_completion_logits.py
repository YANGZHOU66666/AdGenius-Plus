"""Check that CompletionLogitsServerDistillationTrainer gives the same loss and gradients as TRL's original.

Real Qwen3.5-0.8B (text-only) on CPU; a 3-sample batch with very different prompt lengths (left padding) and
completion lengths (right padding); TRL's real teacher-alignment code with a deterministic fake teacher client.
Checked for beta=1 and beta=0.5.
"""

import torch
from transformers import AutoModelForCausalLM
from trl.experimental.server_distillation import ServerDistillationTrainer

from completion_logits_trainer import CompletionLogitsServerDistillationTrainer

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
model.train()

B, P, C = 3, 20, 7
prompt_ids = torch.randint(100, 5000, (B, P))
prompt_mask = torch.ones(B, P, dtype=torch.long)
prompt_mask[1, :12] = 0                    # sample 1: prompt of 8 real tokens
prompt_mask[2, :5] = 0                     # sample 2: prompt of 15 real tokens
completion_ids = torch.randint(100, 5000, (B, C))
completion_mask = torch.ones(B, C, dtype=torch.long)
completion_mask[0, 5:] = 0                 # sample 0: 5 completion tokens
completion_mask[2, 3:] = 0                 # sample 2: 3 completion tokens
inputs = {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
          "completion_ids": completion_ids, "completion_mask": completion_mask}


class FakeTeacherClient:
    """Deterministic per-token teacher logprobs that depend only on the token id and its position."""

    def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
        out = {"logprobs": [], "logprob_token_ids": [], "actual_logprobs": [], "actual_token_ids": []}
        for seq, start in zip(sequences, prompt_lengths):
            tail = seq[start:]
            out["actual_logprobs"].append([[-((t * 7 + i) % 97) / 30 - 0.05] for i, t in enumerate(tail)])
            out["actual_token_ids"].append([[t] for t in tail])
            out["logprobs"].append([[-((t * 3 + i) % 89) / 40 - 0.01] for i, t in enumerate(tail)])
            out["logprob_token_ids"].append([[(t * 13 + i) % 5000 + 100] for i, t in enumerate(tail)])
        return out


def run(cls, beta):
    trainer = object.__new__(cls)
    trainer.beta, trainer.temperature, trainer.loss_top_k, trainer.loss_add_tail = beta, 1.0, 1, True
    trainer.reverse_kl_top_1_mode = "sampled"
    trainer.model, trainer.current_gradient_accumulation_steps = model, 4
    trainer.teacher_client = FakeTeacherClient()
    compute = getattr(cls.compute_loss, "__wrapped__", cls.compute_loss)
    model.zero_grad()
    loss = compute(trainer, model, dict(inputs))
    loss.backward()
    return (loss.detach(), model.model.layers[0].mlp.down_proj.weight.grad.clone(),
            model.lm_head.weight.grad.clone())


for beta in (1.0, 0.5):
    loss_a, grad_a, head_a = run(ServerDistillationTrainer, beta)
    loss_b, grad_b, head_b = run(CompletionLogitsServerDistillationTrainer, beta)
    print(f"beta={beta}: original {loss_a.item():.8f} | completion-only {loss_b.item():.8f} | "
          f"loss diff {abs(loss_a - loss_b).item():.2e} | layer-0 grad diff {(grad_a - grad_b).abs().max().item():.2e} | "
          f"lm_head grad diff {(head_a - head_b).abs().max().item():.2e}")
    assert torch.allclose(loss_a, loss_b, atol=1e-6), "loss mismatch"
    assert torch.allclose(grad_a, grad_b, atol=1e-6) and torch.allclose(head_a, head_b, atol=1e-6), "grad mismatch"
print("EQUIVALENT")
