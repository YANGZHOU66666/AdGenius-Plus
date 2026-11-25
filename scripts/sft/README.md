# SFT（监督微调）

学生 Qwen3.5-0.8B 模仿冻结的 397B 老师广告（`data/train/v1/teacher_qwen397b_train.jsonl`），只对广告部分计算交叉熵。方案见 [docs/sft_opd_experiment_plan.md](../../docs/sft_opd_experiment_plan.md)。

```text
scripts/
├── build_training_prompts.py   OPD 和 SFT 共用：冻结训练和 dev prompt
├── dev_generate.py             OPD 和 SFT 共用：在 VM 上对 dev 商品贪心生成
├── formal/                     OPD 和 SFT 共用：正式训练编排（已完成：sft_formal_v1）
└── sft/
    ├── build_sft_data.py       本地：把冻结 prompt 和老师广告拼成 SFT 样本（sft_train_v1.jsonl）
    ├── check_sft_local.py      本地 CPU 检查：数据顺序与 OPD 相同、loss 只覆盖广告和结束符、sampler 按顺序取
    ├── sft_train.py            VM：单个 SFT run，选学习率和正式训练共用
    └── lr_sweep/               选学习率（已完成：sft_lr_sweep_v1，选定 5e-6）
```

**数据顺序要特别注意**：TRL 的 `shuffle_dataset=False` 只是不在训练前打乱数据集，transformers 的 Trainer 默认仍用 `RandomSampler` 取样。`sft_train.py` 设了 `train_sampling_strategy="sequential"`，并在训练前检查 sampler，不是按顺序取就直接报错。否则 1k/2k/4k/8k checkpoint 就不再对应“只用前 N 个商品”。
