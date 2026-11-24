# SFT（监督微调）

学生 Qwen3.5-0.8B 模仿冻结的 397B 老师广告（`data/train/v1/teacher_qwen397b_train.jsonl`），只对广告部分计算交叉熵。方案见 [docs/sft_opd_experiment_plan.md](../../docs/sft_opd_experiment_plan.md)。

```text
scripts/sft/
├── lr_sweep/    选学习率（待做）
└── formal/      正式训练（待做）
```

与 OPD 共用 `scripts/build_training_prompts.py`（同一份 prompt）和 `scripts/dev_generate.py`（同样的 dev 生成设置）。
