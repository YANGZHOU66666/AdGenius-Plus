# OPD 学习率与 beta 选择（dev，2026-11-23）

实验 `opd_lr_sweep_v1`，对应[实验方案](../../../docs/sft_opd_experiment_plan.md) §5 的 OPD 学习率选择。选择规则在看到任何评审结果之前就已写定，原文记录在 [manifest.json](manifest.json) 的 `evaluation.rules`。

## 结论

**选定 `beta=0.5`（广义 JSD），`lr=2e-6`。**

| 步骤 | 结果 |
|---|---|
| beta=1 组内 | lr=2e-6 胜出（对 5e-7 胜率 0.695，对 1e-6 胜率 0.768，CI 都不含 0.5） |
| beta=0.5 组内 | lr=2e-6 胜出（对 5e-7 胜率 0.765，对 1e-6 胜率 0.593，CI 都不含 0.5） |
| 最终对决 | beta=0.5 lr=2e-6 对 beta=1 lr=2e-6 胜率 **0.62**，95% CI [0.5625, 0.6775]，不含 0.5，不触发“平局取 beta=1”的规则 |

**需要注意：两个 beta 都是网格里最大的学习率 2e-6 胜出，最优值可能在 2e-6 以上。** 按规则仍选 2e-6；是否在正式训练前加试 4e-6，由你决定（见文末）。

## 设置

| 项目 | 值 |
|---|---|
| 学生 | Qwen3.5-0.8B（纯文本 checkpoint `Qwen3_5ForCausalLM`） |
| 老师 | Tinker `Qwen/Qwen3.5-397B-A17B`。beta=1 只取实际 token 的 logprob；beta=0.5 另取老师 top-1 |
| 训练数据 | `data/train/v1`（提交 `f98a45f`）按顺序的前 1k 个 prompt，每个采样 4 条，块内按种子打乱 |
| 每步 | 40 个 prompt × 4 = 160 条序列，共 25 步。方案原定每步 32 个 prompt，改成 40 是为了让 1k/2k/4k/8k 的边界都落在整步上 |
| 优化 | AdamW（fused），`constant_with_warmup`（3 步 warmup），bf16，gradient checkpointing，`loss_top_k=1`，temperature 1.0，`max_completion_length=128`，种子 20261123 |
| dev 生成 | 200 个 dev 商品，vLLM 贪心解码（temperature 0），最多 200 token，`enable_thinking=False` |
| 评审 | 仓库的 `scripts/evaluate_pairwise.py`：DeepSeek（deepseek-flash），pairwise-v0.2，只给广告文本，AB/BA 各评一次 |
| 硬件和软件 | Colab NVIDIA A100-SXM4-40GB，驱动 580.82.07；torch 2.13.0、vllm 0.30.0、trl 1.18.0、transformers 5.23.0、accelerate 1.15.0、tinker 0.40.0 |

## 计分规则（评审前写定）

- **分母固定为全部 200 个 dev 商品。** 每个商品，A 的得分是成功判断中 A 胜的比例（0、0.5 或 1）。
- **两条文本完全相同：** 记 0.5，不调用评委。**一边为空：** 非空一方胜。**两次判断都失败：** 记 0.5（本次没有发生）。
- **截断的输出保留原文照常评审。** 截断率单独报告。
- **不重新生成任何模型输出**（贪心解码，每个商品只生成一次），只有评委调用可以重试。
- **组内：** 每个学习率的分数是它两场比较胜率的平均。第一名胜出，除非它对第二名的 bootstrap 95% CI 含 0.5，这时取两者中较小的学习率。**组间：** 两组胜者对决，CI 含 0.5 时取 beta=1。
- **Bootstrap：** 按商品重抽样 10000 次，种子 20261123。

## 评审结果

| 比较（A 对 B） | A 胜率 | 95% CI | AB/BA 不一致的商品 | 按规则直接定分 |
|---|---:|---|---:|---:|
| beta=1：5e-7 对 1e-6 | 0.555 | [0.495, 0.6175] | 40 | 2（文本相同） |
| beta=1：5e-7 对 2e-6 | 0.305 | [0.25, 0.36] | 44 | 0 |
| beta=1：1e-6 对 2e-6 | 0.2325 | [0.185, 0.2825] | 41 | 0 |
| beta=0.5：5e-7 对 1e-6 | 0.295 | [0.24, 0.35] | 43 | 1 |
| beta=0.5：5e-7 对 2e-6 | 0.235 | [0.185, 0.2875] | 36 | 0 |
| beta=0.5：1e-6 对 2e-6 | 0.4075 | [0.35, 0.4675] | 46 | 1 |
| **最终：beta=1 2e-6 对 beta=0.5 2e-6** | **0.38** | **[0.3225, 0.4375]** | 50 | 0 |

组内平均分：

| 学习率 | beta=1 | beta=0.5 |
|---|---:|---:|
| 5e-7 | 0.430 | 0.265 |
| 1e-6 | 0.339 | 0.556 |
| 2e-6 | **0.731** | **0.679** |

- **评委稳定性：** 7 场比较共 2792 次 DeepSeek 调用，没有失败。AB/BA 结论不一致的商品占 18–25%，按两次各半计入。
- **beta=1 不单调：** 1e-6 反而不如 5e-7（胜率 0.555，CI 下限 0.495，刚好含 0.5），只有 2e-6 明显更好。1e-6 是用修好的代理重跑的那个 run，训练时的随机采样也不同。这一点不影响选择，但说明单种子的组内差异有噪声。

## 护栏指标（200 条 dev 输出，只描述，不参与选择）

| 系统 | 平均词数 | 超过 35 词 | 被截断 | 空输出 |
|---|---:|---:|---:|---:|
| 老师 397B（参考） | 26.7 | 0 | – | 0 |
| 未训练学生 | 51.0 | 154 | 3 | 0 |
| beta=1 lr=5e-7 | 19.5 | 5 | 5 | 0 |
| beta=1 lr=1e-6 | 16.7 | 4 | 5 | 0 |
| beta=1 lr=2e-6 | 17.9 | 3 | 0 | 0 |
| beta=0.5 lr=5e-7 | 18.0 | 4 | 2 | 0 |
| beta=0.5 lr=1e-6 | 16.8 | 2 | 0 | 0 |
| **beta=0.5 lr=2e-6（选定）** | **25.2** | **2** | **0** | **0** |

选定的设置长度最接近老师，没有截断。示例（`train_product_708999`）：

- **老师：** Organize family life with this 4-piece magnetic dry erase calendar set. The 17x12" monthly and weekly boards include markers, an eraser, and magnets for easy fridge mounting.
- **未训练学生：** The 4-Board Magnetic Dry Erase Calendar Bundle for your fridge is a family planner that keeps your schedule organized. Featuring 17x12" whiteboards with colorful designs, this set includes 8 fine-tip markers…（写到上限被截断）
- **beta=1 lr=2e-6：** Keep your family's daily routines organized with this 17x12" magnetic whiteboard. Use it to track family chores, schedule events, and plan meals.
- **beta=0.5 lr=2e-6：** Organize your fridge with this 17x12" magnetic dry-erase calendar bundle. Featuring monthly, weekly, and daily planners, it helps you keep track of family life and chores.

## 训练记录

| Run | 训练耗时 | 每步约 | Tinker 发送 token |
|---|---:|---:|---:|
| beta=1 lr=5e-7 | 25.5 分 | 60 秒 | 3.28M |
| beta=1 lr=1e-6（第 14 步失败） | – | – | 1.92M（作废） |
| beta=1 lr=1e-6（重跑） | 19.9 分 | 47 秒 | 3.26M |
| beta=1 lr=2e-6 | 19.7 分 | 46 秒 | 3.25M |
| beta=0.5 lr=5e-7 | 19.6 分 | 46 秒 | 3.27M |
| beta=0.5 lr=1e-6 | 19.6 分 | 46 秒 | 3.26M |
| beta=0.5 lr=2e-6 | 19.5 分 | 46 秒 | 3.25M |

显存峰值 24.7–27.5 GB。每个 run 的 checkpoint 没有保留（正式训练会用选定的参数从头训练）。

## 费用

| 项目 | 用量 | 费用 |
|---|---|---|
| Colab A100 | 本次会话 19:43–22:49，66.24 → 45.48 CU | 20.8 CU |
| Colab（之前两次被回收） | 71.46 → 66.24 CU | 5.2 CU（作废） |
| Tinker | 21.49M token（其中 1.92M 作废） | 按实测 $0.72/M 约 $15.5，以 Tinker 账单为准 |
| DeepSeek | 2792 次调用；输入 1.23M（其中 0.55M 命中缓存），输出 0.17M | 约 $0.41（按高峰价） |

## 运行中的问题

1. **Colab 回收：** 之前两次会话约 22 分钟后被回收。原因是训练放在一个脱离 notebook 单元的后台进程里，内核看起来是空闲的。改成让单元一直执行到全部训练结束，并加了每 10 分钟检查一次的 watchdog：拉回已完成的 run，会话丢了就只重跑缺的 run。之后没有再被回收。
2. **beta=1 lr=1e-6 第 14 步失败：** 老师在 462 个位置中缺了 44 个。代理原来用“最后一次出现的 assistant 开头”来定位回答起点，学生输出里如果再冒出对话开头就会取错位置。修复：改成第一次出现的位置；回答中只要有缺口，就整条序列重新打分并记录。用修好的代理重跑成功，所有 run 都没有触发兜底。没有发生泄漏时，新旧代理的输出完全一样，所以前面用旧代理跑完的 run 不受影响。
3. **每组 token 上限：** 从 12.5M 调到 15M，避免重跑时碰到上限。

## 与方案的差异

- **每步 40 个 prompt × 4 次采样**，方案原定 32 个 prompt。这样每 1k 个 prompt 正好是 25 步，1k/2k/4k/8k 都落在整步上。
- **组内得分用两场比较胜率的平均**，方案原文是“赢得最多”；3 个候选时两者排序相同。
- **没有保留选定 run 的 1k checkpoint**，所以无法按方案 §5 第 5 步从它续训到 8k。正式训练用相同设置从头训练。
- 两个 beta 都扫了，组间对决在分不出高下时取 beta=1。

## 文件

| 内容 | 位置 |
|---|---|
| 7 个模型的 dev 输出（GenerationResult v1.1） | [generations/](generations/) |
| 送评的 CandidatePair 和按规则直接定分的商品 | [pairs/](pairs/) |
| 每个 run 的训练记录（超参数、逐步 loss、耗时、Tinker 用量、代理版本） | [runs/](runs/) |
| 选择结果 | [selection.json](selection.json) |
| 实验 manifest（输入、规则、评审 run、环境、费用、搬迁说明） | [manifest.json](manifest.json) |
| DeepSeek 评审结果 | [results/pairwise/opd_sweep_dev_*](../../pairwise/) |
| 本地（不进 Git）：训练和生成日志、VM 存档、冻结 prompt、代码 | `artifacts/train/opd_lr_sweep_v1/` |
| 本地（不进 Git）：评审调用日志 | `artifacts/pairwise/opd_sweep_dev_*/calls.jsonl` |

**代理版本**：beta=1 的 lr=5e-7 和 lr=2e-6 由修复前的代理服务，其余 4 个 run 由修复后的代理服务。没有发生对话开头泄漏时两者输出完全相同；修复前的代理一旦遇到泄漏会让训练报错，而这两个 run 都正常跑完，说明它们没有遇到泄漏。`run_report.json` 里的代码哈希是在训练结束写报告时取的，不能用来判断代理版本，以各 run manifest 的 `teacher.proxy_version` 为准。

## 待办

1. **是否加试 lr=4e-6（beta=0.5）：** 2e-6 在网格边上。一个 run 约 32 分钟、4 CU、Tinker 约 $2.4，评审约 $0.1。
2. **正式训练：** 定期把完整状态（约 9 GB）存到 Drive 以便续训，并保存最终权重（bf16，约 1.6 GB）。
3. **训练代码进仓库：** 目前只在 `artifacts/` 里，manifest 记录了每个文件的哈希。正式训练时整理进 `src/adgenius/training/` 并配测试。
