"""Local CPU checks before any GPU time: data order vs OPD, completion-only loss mask, sampler order.

Needs a local environment with trl 1.18 and transformers 5.23 (CPU is enough):
    <trl-python> scripts/sft/check_sft_local.py --data artifacts/train/sft_data/v1/sft_train_v1.jsonl
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OPD_RUN = REPO / "results/training/opd_lr_sweep_v1/runs/beta0.5_lr2e-06/manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=REPO / "artifacts/train/sft_data/v1/sft_train_v1.jsonl")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    import sft_train

    # 1. ordering function is the OPD one, and the first 1k products match the OPD sweep
    opd_src = (REPO / "scripts/opd/opd_train.py").read_text()
    opd_fn = re.search(r"def ordered_prompts\(.*?\n(?=\n\nclass|\n\ndef)", opd_src, re.S).group(0)
    sft_fn = inspect.getsource(sft_train.ordered_rows)
    body = lambda s: [l for l in s.splitlines()[1:] if not l.strip().startswith('"""')]
    assert body(opd_fn) == body(sft_fn), "ordering code differs from OPD"
    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    ordered = sft_train.ordered_rows(rows, 20261123)[:1000]
    opd_first = json.loads(OPD_RUN.read_text())["data"]["first_product_ids"]
    assert [r["product_id"] for r in ordered[:5]] == opd_first, ([r["product_id"] for r in ordered[:5]], opd_first)
    assert {r["rank"] for r in ordered} == set(range(1000))
    print("order: identical function to OPD; first 1k = ranks 0-999; first ids match the OPD run")

    # 2. tokenization and completion-only mask, with TRL's own preprocessing
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
    sample = ordered[:8]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"],
                             "chat_template_kwargs": {"enable_thinking": False}} for r in sample])
    cfg = SFTConfig(output_dir=str(REPO / "artifacts/train/_sft_check_out"), max_steps=1, per_device_train_batch_size=4,
                    completion_only_loss=True, max_length=2048, report_to="none", shuffle_dataset=False, train_sampling_strategy="sequential",
                    use_cpu=True, bf16=False)
    trainer = SFTTrainer(model=model, args=cfg, processing_class=tok, train_dataset=ds)
    ex = trainer.train_dataset[0]
    print("dataset fields:", sorted(ex.keys()))
    ids, mask = ex["input_ids"], [l != -100 for l in ex["labels"]]
    completion = tok.decode([t for t, m in zip(ids, mask) if m])
    prompt_part = tok.decode([t for t, m in zip(ids, mask) if not m])
    expected_prompt = tok.apply_chat_template(sample[0]["prompt"], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    print("prompt part ends with:", repr(prompt_part[-60:]))
    print("loss part:", repr(completion))
    assert prompt_part == expected_prompt, "prompt part differs from the inference prompt"
    assert completion == sample[0]["completion"][0]["content"] + "<|im_end|>\n" or \
        completion == sample[0]["completion"][0]["content"] + "<|im_end|>", "loss covers more than the ad"
    for i, r in enumerate(sample):
        e = trainer.train_dataset[i]
        got = tok.decode([t for t, l in zip(e["input_ids"], e["labels"]) if l != -100])
        assert all(l in (-100, t) for t, l in zip(e["input_ids"], e["labels"])), "labels are not input ids"
        assert got.startswith(r["completion"][0]["content"]), i
    print("mask: loss covers exactly the teacher ad plus the end-of-turn token; prompt equals the inference prompt")

    # 3. the train sampler keeps dataset order
    sampler = trainer._get_train_sampler(trainer.train_dataset)
    print("sampler:", type(sampler).__name__, list(iter(sampler))[:8])
    assert list(iter(sampler))[:8] == list(range(8)), "sampler shuffles the data"
    print("ALL_CHECKS_OK")


if __name__ == "__main__":
    main()
