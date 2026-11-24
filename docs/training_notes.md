# 训练实验杂记

这份杂记记录[SFT 与 OPD 蒸馏实验方案](sft_opd_experiment_plan.md)背后的依据、试点结果、冒烟测试经验和工程备忘。正式结论以 `results/` 为准。

文中的试点和冒烟测试放在仓库外的本地目录 `../analysis/`，没有纳入 Git。它们只用于指导设计，不能当作实验结论引用。

## 1. 教师选择

**为什么只能用 397B**：OPD 要求教师给学生采样出的每个 token 打出 logprob，所以教师必须和学生共用同一个分词器，还要能对序列打分。

- GLM、LongCat 的分词器和学生不同，Qwen4-Max 只能通过 API 调用，三者都无法担任 OPD 教师。
- 397B 是 Qwen3.5 家族中最大的模型。397B、0.8B、2B 三者的 `tokenizer.json` 完全一致：248,044 个词加 26 个特殊 token，`vocab_size` 为 248,320。

**教师试点**（2026-11-16）：DeepSeek 作评委，gold 100 个商品，AB/BA 取平均，括号内为 95% bootstrap 区间。

| 候选 vs 参照 | 候选胜率 |
|---|---|
| 397B（关闭思考）vs GLM-5.3-Flash | 37.5%（30.5–45.0） |
| 397B（开启思考，预算 2048）vs GLM | 24.0%（18.0–30.0） |
| Qwen4-Max（关闭思考）vs GLM | 63.0%（55.5–70.5） |
| Qwen4-Max（开启思考）vs GLM | 55.0%（48.0–62.0） |
| 397B（关闭思考）vs Qwen3.5-2B | 96.0%（92.5–99.0） |

从中得出：
- 教师和学生之间差距很大。
- 关闭思考的效果更好。
- 397B 不是最强的写手，所以实验结论只能说"哪种方法更好地传递了这个教师的能力"。
- 如果用 397B 当评测参照，学生的胜率起点只有约 4%。

## 2. 评委校准

完整数据见[四评委对比](../results/pairwise/comparison/report.md)，摘要如下：

| 评委 | 与人工一致率 | 两个顺序一致率 |
|---|---:|---:|
| DeepSeek（主评委） | 83.5% | 87% |
| GPT-6 Luna | 80.0% | 92% |
| Qwen Flash | 79.0% | 72% |
| Claude Haiku | 73.5% | 87% |

## 3. 冒烟测试经验

测试时间为 2026-11-16 至 11-18，在 Mac（MPS）上进行，用了 32 个训练池 prompt 和 4 个开发集商品。OPD 的教师先用本地 2B 模拟，之后换成 Tinker 397B。

### 3.1 训练方法

- **第一步必须 warmup。** Adam 的第一步更新几乎只取梯度的符号，每个参数都会移动约一个学习率的幅度。在学习率 1e-5 下只跑 1 步 SFT，4 条 dev 输出中有 3 条变成了空字符串；OPD 也出现了同样的截断。
- **OPD 的学习率要比 SFT 小一个量级。** 用 1e-5 时，即使加了 warmup，4 步之内采样长度就从 54 降到 15 个 token，反向 KL 反而上升。改用 1e-6（也是 TRL 蒸馏训练器的默认值）后可以正常训练。
- **"每 token 平均 logprob"会误导。** OPD 用 1e-6 训练 16 步后，397B 给 4 条 dev 输出的整句总 logprob 全部提高，比如从 −66.6 升到 −18.7；但每 token 的平均值反而下降，从 −1.22 降到 −1.55。原因是句子变短后，最差的 3 个 token 占了扣分的 41–89%。
- **小规模 OPD 的典型毛病**：
  - 照抄商品标题；
  - 套用 397B 不认可的开头，比如 "Get the perfect…"，其中 "Get" 的 logprob 为 −7.4；
  - 写得太短，丢掉了信息。

  原因是在 `beta=1, loss_top_k=1` 下，loss 只能压低学生写错的词，看不到教师首选的 token，也就没法把它往上拉。
- **SFT 也有自己的毛病**：编造信息（列出一串颜色，"four" 的 logprob 为 −10.8）、"This/The X features…" 式开头、偶尔超过 35 词。

### 3.2 工程

- **TRL 1.18**：
  - SFT 关闭思考，需要在每条数据里写 `chat_template_kwargs`；
  - TRL 会自动把 completion mask 转成 `labels`；
  - loss 使用 `fused_lm_head`，不会生成完整的 logits。
- **`ServerDistillationTrainer`**：
  - `beta>0` 时要求 `loss_top_k=1`；
  - `VLLMClient` 要求本机能导入 vLLM，在 Mac 上需要绕过这个检查；
  - 它不发送 API key，所以远程 vLLM 要通过 SSH 隧道或本地代理访问。
- **内存（0.8B，batch 1）**：
  - fp32 加 AdamW 的峰值约 13.7 GB；
  - bf16 混合精度几乎不降低峰值；
  - Adafactor 可以降到 7.6 GB；
  - MPS 的缓存会持续增长，需要设置 `torch_empty_cache_steps=1`。
- **Qwen3.5 小模型使用 Gated DeltaNet。** Mac 上没有 `flash-linear-attention` 和 `causal-conv1d` 的加速内核，速度很慢。
- **Tinker**：
  - `compute_logprobs` 只返回实际 token 的 logprob；
  - `sample()` 支持 `topk_prompt_logprobs` 和 `target_prompt_logprobs`；
  - 只用它给序列打分，学生不在 Tinker 上训练（Tinker 的训练只支持 LoRA）。
- **百炼和 Tinker 是否同一个模型**：Tinker 上的 397B 给百炼 397B 生成的广告打出的平均 logprob 是 −0.15 到 −0.31，本地 2B 给出的是 −1.14。这间接说明两边是同一个模型。Tinker 对同一条序列重复打分，结果约有 0.03 nats/token 的波动。

## 4. 资源与成本

| 资源 | 实测或标价 |
|---|---|
| Tinker | prefill 标价 $1.65/百万 token；冒烟测试实际为 $0.72/百万（11 万 token 扣费 $0.08），可能得益于前缀缓存 |
| 百炼 | 输入 ¥1.2/百万，输出 ¥7.2/百万 |
| Colab（`colab` CLI） | L4（22.5 GB）1.94 单元/小时；A100 40GB 6.73 单元/小时；可以同时开 2 个会话 |
| DeepSeek 评委 | 每次调用约 460 个输入 token、62 个输出 token |

**本方案的粗略估算**（尚未实测）：

| 项目 | 估算 |
|---|---|
| OPD 1 个种子（扫学习率加续训到 8k） | 约 4 万次打分，Tinker 约 $25–35 |
| SFT 教师数据（8k 条） | 约 ¥10 |
| GPU 时间（L4） | 约 7–11 小时 |
| DeepSeek 评委（三层评测） | 约 5 万次调用 |

最大的不确定性是 Tinker 在每步 128 条序列并发打分时的吞吐。目前只测过 4 条同时打分，用了 2.9 秒。

## 5. 仓库安排（计划）

```text
prompts/generation/ad_copy_v1.txt        从 gold manifest 抽出的冻结生成 prompt
configs/generation/train_teacher.toml    教师采样设置
configs/training/{sft,opd}.toml          训练配置
src/adgenius/training/                   data / sft / opd / teacher_proxy / callbacks / generate
src/adgenius/eval/guardrails.py          护栏指标
scripts/                                 build_train_split / generate_teacher_data / train_sft / train_opd /
                                         generate_student / evaluate_training
data/train/v1/                           products、dev_products、teacher_397b、manifest（共享）
results/training/<实验名>/                报告与指标（共享）
artifacts/train/、artifacts/checkpoints/  运行日志与权重（被忽略）
```

- 每次运行都记录 git 提交、配置哈希、数据哈希、种子、硬件、耗时和 API 花费。
- 训练依赖放在 `pyproject.toml` 的可选组 `[training]`。
- 按里程碑分批合并：数据 → SFT → 评测 → OPD。

## 6. 工程待办

- [ ] 在 L4 上测速：SFT 跑 20 步，OPD 跑 2 步（每步 128 次采样），同时测出 Tinker 的吞吐。
- [ ] 逐 token 比较百炼和 Tinker 的 397B，确认两边一致。
- [x] 如果选用 `beta=0.5`，把代理升级为同时返回教师首选的 token（通过 Tinker `sample(topk_prompt_logprobs=1)`）。已实现为代理的 top1 模式，见 §7。

## 7. OPD 学习率选择的工程经验（2026-11-23）

结果见[选择报告](../results/training/opd_lr_sweep_v1/report.md)；本节只记工程经验。

- **Colab 空闲回收**：Colab 只按 notebook 单元是否在执行来判断会话是否在用，不看 GPU。把训练放进脱离单元的后台进程后，内核显得空闲，约 22 分钟后会话被回收。改为让单元一直执行到训练结束，并在 Mac 上用每 10 分钟检查一次的 watchdog：拉回已完成的 run，会话丢失时只重跑缺的 run，内核意外空闲时立即重启单元。之后没有再被回收。
- **VM 磁盘不保留**：会话一结束，checkpoint 和日志都会丢失。每个 run 结束后立刻把 dev 输出和日志拉回本地；正式训练需要定期把完整状态（约 9 GB）存到 Drive。
- **代理定位回答起点的 bug**：代理只把回答部分发给 Tinker 打分。原来用最后一次出现的 assistant 开头定位起点，学生采样里如果再出现对话开头，前面的位置就不会打分，TRL 报 “missing actual-token logprobs”。改为第一次出现的位置，并在发现缺口时整条序列重新打分、记录日志。修复后所有 run 都没有触发兜底。
- **代码哈希要在训练开始时取**：这次的 `run_report.json` 在训练结束时才计算代码哈希，中途替换文件后会记录错版本。正式训练应在启动时记录。
- **A100 实测**（每步 160 条序列，micro batch 16）：
  - beta=1 和 beta=0.5（top1 模式）每步都约 46 秒，25 步约 20 分钟；加上加载、保存和 dev 生成，每个 run 约 32 分钟；
  - 显存峰值约 27.5 GB；
  - Tinker 每个 run 约 326 万 token，按实测单价约 $2.4。
- **整轮选择的用量**：Colab 20.8 单元（另有 5.2 单元耗在两次被回收的会话上）；Tinker 约 2150 万 token；DeepSeek 2792 次调用，约 $0.41。

