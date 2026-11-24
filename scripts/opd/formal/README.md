# OPD 正式训练（待做）

用选学习率得到的设置（`beta=0.5`、学习率 2e-6，见[选择报告](../../../results/training/opd_lr_sweep_v1/report.md)），1 个种子，从头训练到 8k 个 prompt（每步 40 个 prompt × 4 次采样，共 200 步）。训练代码复用上一级的 `opd_train.py`，这里将放正式训练的编排脚本。

计划：

- 在 1k、2k、4k、8k（第 25、50、100、200 步）保存权重，作为数据规模曲线的四个点，全部在 test 上评测；
- 定期把完整训练状态（约 9 GB）存到 Google Drive，会话被回收后从最近的 checkpoint 续训；
- 每个规模的最终权重（bf16，约 1.6 GB）另外保存；
- 预计 A100 约 3–3.5 小时、约 24 CU，Tinker 约 2600 万 token。
