# SFT 学习率选择（dev，2026-11-24）

实验 `sft_lr_sweep_v1`，对应[实验方案](../../../docs/sft_opd_experiment_plan.md) §5 的 SFT 学习率选择。选择规则在看到评审结果之前就已写定，与 [OPD 的选择](../opd_lr_sweep_v1/report.md)相同，原文记录在 [manifest.json](manifest.json) 的 `evaluation.rules`。

## 结论

**按规则选定学习率 5e-6。** 但三个学习率在 dev 上**分不出明显高下**：

| 学习率 | 平均胜率（两场比较的平均） |
|---|---:|
| 5e-6 | 0.506 |
| **1e-5** | **0.520**（第一） |
| 2e-5 | 0.474 |

- 第一名 1e-5 对第二名 5e-6 的胜率是 0.515（即 5e-6 对 1e-5 是 0.485），95% 置信区间 [0.46, 0.57] 含 0.5，按规则视为打平，选两者中较小的 **5e-6**。
- 三场比较的置信区间**全部**含 0.5，任何两个学习率之间都没有可靠差别。
- 5e-6 恰好是网格的下边界。它被选中是因为打平时取较小值，不是因为它明显最好；按事先定的规则不往外扩。

## 设置

| 项目 | 值 |
|---|---|
| 学生 | Qwen3.5-0.8B（纯文本 checkpoint），全参训练，fp32 主权重 + bf16 |
| 数据 | `data/train/v1` 前 1k 个商品：prompt 与 gold、老师数据、OPD 逐字节相同；目标是冻结的 397B 老师广告 |
| 顺序 | 与 OPD 完全相同：按 rank，4 个块内用种子 20261123 打乱；前 5 个商品与 OPD 一致 |
| loss | 只对老师广告和结束符计算交叉熵（已用 TRL 的预处理在本地核对）；关闭思考 |
| 每步 | 40 条（micro batch 4 × 累积 10），共 25 步；3 步 warmup 后学习率恒定 |
| 优化 | AdamW（fused），gradient checkpointing，`max_length=2048`，按顺序取样（`SequentialSampler`） |
| dev 生成 | 200 个 dev 商品，vLLM 贪心解码，最多 200 token，与 OPD 相同 |
| 评审 | 仓库 `scripts/evaluate_pairwise.py`：DeepSeek（deepseek-flash），pairwise-v0.2，只给广告文本，AB/BA 各一次 |
| 硬件和软件 | Colab NVIDIA L4（23 GB），驱动 580.82.07；torch 2.13.0、vllm 0.30.0、trl 1.18.0、transformers 5.23.0 |

## 评审结果

| 比较（A 对 B） | A 胜率 | 95% CI | A 稳定胜 | B 稳定胜 | 两次结论不一致 | 只看稳定的 A 胜率 |
|---|---:|---|---:|---:|---:|---:|
| 5e-6 对 1e-5 | 0.485 | [0.43, 0.54] | 60 | 66 | 74 | 0.476 |
| 5e-6 对 2e-5 | 0.5275 | [0.4675, 0.585] | 77 | 66 | 57 | 0.538 |
| 1e-5 对 2e-5 | 0.525 | [0.4675, 0.5825] | 75 | 65 | 60 | 0.536 |

- 1200 次 DeepSeek 调用，没有一次失败；没有需要按规则直接定分的商品（没有文本完全相同或为空的情况）。
- 两次结论不一致的商品占 29–37%，远高于 OPD 调参时的 18–25%，说明三个模型的输出非常接近，评委在很多商品上主要是在看位置。
- 换成“只看两次结论一致的商品”的口径，排序不变，也同样分不出高下。

## 训练和护栏指标

| 学习率 | 训练耗时 | 每步 | 显存峰值 | loss：开始 → 最后 5 步 | dev 平均词数 | 超过 35 词 | 截断 |
|---|---:|---:|---:|---|---:|---:|---:|
| 未训练学生 | – | – | – | – | 51.0 | 154 | 3 |
| 5e-6 | 329 秒 | 13 秒 | 14.8 GB | 1.74 → 1.19 | 26.1 | 5 | 0 |
| 1e-5 | 331 秒 | 13 秒 | 14.8 GB | 1.74 → 1.23 | 25.8 | 3 | 0 |
| 2e-5 | 331 秒 | 13 秒 | 14.8 GB | 1.74 → 1.33 | 26.2 | 3 | 0 |
| 老师 397B（参考） | – | – | – | – | 26.7 | 0 | – |

- 三个学习率都把长度从 51 词拉到了老师的水平（约 26 词），都没有截断，都没有发散。
- 学习率越大，最后的训练 loss 越高（1.19 → 1.23 → 1.33）。只训练 25 步，学习率大时更新幅度大，可能已经开始过冲；评审结果上 2e-5 也略差，但差距在误差范围内。
- 示例（`train_product_708999`，磁性日历白板）：
  - **5e-6：** Organize your fridge with this 17x12 magnetic dry-erase calendar set. Includes monthly, weekly, daily planners, six fine-tip markers, and an eraser for clear, stain-free writing.
  - **1e-5：** Organize your fridge with this 17x12 magnetic whiteboard set featuring 17 fine-tip markers, erasers, and a monthly weekly daily calendar. …
  - **2e-5：** Organize your family fridge with this 17x12" magnetic calendar set featuring six fine markers and an eraser. …

  1e-5 这条把马克笔数量写成了 17 支。主评委不检查事实，这类问题要靠后面的 rubric 评测。

## 与 OPD 调参的对照

- 同一份 prompt、同样的数据顺序、同样的 dev 生成设置和评审规则。
- OPD 的三个学习率差别明显（组内最强的一场胜率 0.77），SFT 三个学习率差别很小。这说明 SFT 在这个学习率范围内对学习率不敏感，选哪个对正式训练影响不大。

## 费用和用时

| 项目 | 用量 |
|---|---|
| Colab L4 | 02:19–03:06，含安装、小测试和 3 个 run；45.48 → 44.04 CU，共 **1.44 CU** |
| DeepSeek | 1200 次调用；输入 55 万 token（其中 23 万命中缓存），输出 7.5 万 token；约 **$0.19**（按高峰价） |
| Tinker | 不需要 |
| 用时 | 拿到 L4 约 1 分钟；小测试 8 分钟；3 个 run 共 27 分钟（每个约 9 分钟：训练 6 分钟，dev 生成 3 分钟）；评审约 10 分钟 |

## 运行中的问题

1. **SFTTrainer 默认打乱数据（已修复）**：本地检查发现，即使设了 `shuffle_dataset=False`，transformers 默认仍用 `RandomSampler` 打乱数据。`shuffle_dataset` 只控制训练前是否先打乱数据集。改为 `train_sampling_strategy="sequential"`，并在训练开始前检查 sampler，不是按顺序取就直接报错。VM 上三个 run 都确认是 `SequentialSampler`。OPD 的 `ServerDistillationTrainer` 用的是 TRL 自己的 `RepeatSampler`，`shuffle_dataset=False` 会生效，**OPD 调参的顺序没有受影响**。这个问题对 1k 的调参影响不大（每个样本反正只用一次），但正式训练必须修，否则 1k/2k/4k/8k checkpoint 就不再对应“只用前 N 个商品”。
2. **防回收**：沿用 OPD 的做法，单元一直执行到训练结束，watchdog 每 10 分钟同步一次；连不上 Colab 时跳过检查，不当成回收。这次会话从头到尾没有被回收，全部完成后 watchdog 自动停掉了 L4。
3. **第一次申请 L4 返回 Service Unavailable**，1 分钟后第二次申请成功。

## 与方案的差异

- **每步 40 条**，方案原定 32 条。这样每 1k 条正好是 25 步，1k/2k/4k/8k 都落在整步上，与 OPD 一致。
- **得分用两场比较胜率的平均**，方案原文是“赢得最多”；3 个候选时两者排序相同。
- **各 run 的 checkpoint 没有保留**，正式训练用选定的学习率从头训练。

## 文件

| 内容 | 位置 |
|---|---|
| 3 个 run 的 dev 输出（GenerationResult v1.1） | [generations/](generations/)；未训练学生的输出与 OPD 共用 [opd_lr_sweep_v1/generations/base.jsonl](../opd_lr_sweep_v1/generations/base.jsonl) |
| 送评的 CandidatePair 和按规则直接定分的商品 | [pairs/](pairs/) |
| 每个 run 的训练记录（超参数、逐步 loss、耗时、sampler、哈希） | [runs/](runs/) |
| 选择结果 | [selection.json](selection.json) |
| 实验 manifest（输入、规则、评审 run、环境、费用、搬迁说明） | [manifest.json](manifest.json) |
| DeepSeek 评审结果 | [results/pairwise/sft_sweep_dev_*](../../pairwise/) |
| 脚本（仓库版本，与运行版本的差异见其 README） | [scripts/sft/](../../../scripts/sft/)、[scripts/sft/lr_sweep/](../../../scripts/sft/lr_sweep/) |
| 本地（不进 Git）：训练日志、VM 存档、SFT 样本、实际运行的代码 | `artifacts/train/sft_lr_sweep_v1/` |
| 本地（不进 Git）：评审调用日志 | `artifacts/pairwise/sft_sweep_dev_*/calls.jsonl` |

vLLM 的 dev 生成日志在 VM 停掉前没有同步下来，没有保留；dev 输出本身完整。

## 待办

1. **正式训练**：学习率 5e-6，3 个种子，各从头训练到 8k（200 步），在 1k、2k、4k、8k 保存权重。保留 `train_sampling_strategy="sequential"` 和训练前的 sampler 检查。
