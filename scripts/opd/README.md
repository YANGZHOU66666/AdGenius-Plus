# OPD（在线策略蒸馏）

学生 Qwen3.5-0.8B 自己采样，老师 Qwen3.5-397B（Tinker）逐 token 打分，用 TRL `ServerDistillationTrainer` 训练。方案见 [docs/sft_opd_experiment_plan.md](../../docs/sft_opd_experiment_plan.md)。

```text
scripts/
├── build_training_prompts.py        OPD 和 SFT 共用：冻结训练和 dev prompt
├── dev_generate.py                  OPD 和 SFT 共用：在 VM 上对 dev 商品贪心生成
└── opd/
    ├── opd_train.py                 单个 OPD run：按块内打乱的顺序读取 prompt，写 checkpoint 和 run_report.json
    ├── completion_logits_trainer.py 只对回答部分计算学生 logits，loss 和梯度与 TRL 原实现完全一致
    ├── check_completion_logits.py   验证上面这一点（输出 EQUIVALENT）
    ├── teacher_proxy.py             兼容 vLLM 接口的老师端点，转发到 Tinker；actual 模式给 beta=1，top1 模式给 beta<1
    ├── lr_sweep/                    选学习率（已完成：opd_lr_sweep_v1）
    └── formal/                      正式训练（待做）
```

本目录下的训练代码由选学习率和正式训练共用，运行在 Colab GPU VM 上，依赖见 `pyproject.toml` 的可选组 `training`。
