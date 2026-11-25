# SFT 与 OPD 正式训练报告（2026-11-24）

本报告只包含训练过程和输出的描述性统计，**不含评委评测**，test 评测的设计待定。方案见 [docs/sft_opd_experiment_plan.md](../../docs/sft_opd_experiment_plan.md)，工程经验见 [docs/training_notes.md §9](../../docs/training_notes.md)。

## 结论

两种方法的正式训练都已完成，所有输出齐全：

| | SFT（[sft_formal_v1](sft_formal_v1/manifest.json)） | OPD（[opd_formal_v1](opd_formal_v1/manifest.json)） |
|---|---|---|
| 设置 | 学习率 5e-6 | `beta=0.5`，学习率 2e-6 |
| 训练 | 200 步（8k 商品，每步 40 条），一次跑完，没有中断或续训 | 200 步（8k 个 prompt × 4 次采样，每步 160 条），一次跑完，没有中断或续训 |
| Drive 上的完整 checkpoint | 8/8（第 25–200 步，每 25 步一份） | 8/8（第 25–200 步，每 25 步一份） |
| 输出 | 4 个规模 × (test 1000 + dev 200) | 4 个规模 × (test 1000 + dev 200) |

参照系统（[test_references_v1](test_references_v1/manifest.json)）：未训练学生和 397B 老师在 test 上各 1000 条输出。

## 设置（事先写定）

| 项目 | 值 |
|---|---|
| 学生 | Qwen3.5-0.8B（纯文本 checkpoint），全参训练，fp32 主权重 + bf16 混合精度 |
| 数据 | `data/train/v1` 的 8,000 个商品，顺序与选学习率时相同（按 rank，四个块内用种子 20261123 打乱） |
| 种子 | 两种方法都只用 20261123 |
| 调度 | 3 步 warmup 后学习率恒定，AdamW（fused） |
| checkpoint | 每 25 步（每 1k 个商品）保存完整续训状态；1k/2k/4k/8k 对应第 25/50/100/200 步 |
| 生成 | 每个规模的 checkpoint 转为 bf16，vLLM 贪心解码，最多 200 token，关闭思考；test 1000 个商品、dev 200 个商品 |
| 硬件 | OPD：Colab A100-SXM4-40GB；SFT：Colab L4（23 GB）。torch 2.13.0、vllm 0.30.0、trl 1.18.0、transformers 5.23.0 |
| SFT 取样 | 强制按顺序取样（`train_sampling_strategy="sequential"`），训练前检查 sampler |

**与方案的差异**：训练和生成的长度上限从 2048 提高到 4096。训练集 rank 1543 的 prompt 有 2307 个 token，test 里最长的有 1968 个 token，原上限会让 OPD 中途崩溃、让 SFT 截断或丢弃样本。能放进 2048 的 prompt，输出不受影响。

## 输出的描述性统计

| 系统 | test 平均词数 | test 超过 35 词 | test 截断 | dev 平均词数 | dev 超过 35 词 |
|---|---:|---:|---:|---:|---:|
| 未训练学生 | 50.2 | 716 | 21 | – | – |
| SFT 1k | 25.4 | 13 | 0 | 26.0 | 4 |
| SFT 2k | 24.8 | 3 | 0 | 25.0 | 0 |
| SFT 4k | 27.6 | 29 | 0 | 28.0 | 6 |
| SFT 8k | 26.1 | 7 | 0 | 26.5 | 0 |
| OPD 1k | 25.0 | 11 | 1 | 25.5 | 2 |
| OPD 2k | 25.1 | 13 | 1 | 25.5 | 2 |
| OPD 4k | 26.5 | 15 | 0 | 26.4 | 4 |
| OPD 8k | 26.5 | 16 | 0 | 26.9 | 2 |
| 397B 老师 | 26.6 | 7 | 0 | 26.7 | 1 |

所有输出都没有空文本。截断的输出保留原文，记为 `truncated`。两种方法在 1k 时就把长度从约 50 词压到老师的水平（约 26 词）。这些数字只描述长度和格式，不代表广告质量。dev 上未训练学生的输出见 [opd_lr_sweep_v1/generations/base.jsonl](opd_lr_sweep_v1/generations/base.jsonl)。

## 训练曲线（每 25 步，即每 1k 个商品的平均）

| 规模区间 | SFT loss | OPD loss | OPD 学生采样长度（token） |
|---|---:|---:|---:|
| 0–1k | 1.313 | 0.0765 | 36.9 |
| 1k–2k | 1.125 | 0.0694 | 37.2 |
| 2k–3k | 1.065 | 0.0682 | 37.6 |
| 3k–4k | 1.036 | 0.0669 | 37.5 |
| 4k–5k | 1.002 | 0.0662 | 37.9 |
| 5k–6k | 0.998 | 0.0656 | 37.7 |
| 6k–7k | 0.973 | 0.0645 | 38.2 |
| 7k–8k | 0.979 | 0.0645 | 37.8 |

- SFT 的 loss 是模仿老师广告的交叉熵；OPD 的 loss 是学生与老师分布的差距（beta=0.5 的广义 JSD），两者不能直接比较。
- 两条曲线都是前 2k 下降最快，之后趋缓，没有发散。
- SFT：训练 48 分钟，每步 13.4 秒，显存峰值 15.5 GB。OPD：训练 2 小时 40 分钟，每步 46.9 秒，显存峰值 28.6 GB。
- 每步的记录在各 run 的 `training_log.jsonl` 中。

## 文件位置

| 内容 | 位置 |
|---|---|
| 输出（GenerationResult v1.1） | `{sft,opd}_formal_v1/generations/{test,dev}/scale_{1k,2k,4k,8k}.jsonl` |
| 参照系统的 test 输出 | `test_references_v1/generations/test/{teacher_qwen397b,student_untrained}.jsonl` |
| 设置、哈希、时间线、问题记录 | 各目录的 `manifest.json` |
| 每步训练记录 | `{sft,opd}_formal_v1/training_log.jsonl` |
| Drive 上每个文件的大小和 md5 | `{sft,opd}_formal_v1/drive_checkpoints.json` |
| 原始日志、VM 状态、实际运行的代码（不进 Git） | `artifacts/train/{sft,opd}_formal_v1/`；冒烟测试在 `artifacts/train/formal_smoke_v1/`；397B 调用日志在 `artifacts/generation/teacher_qwen397b_test_v1/` |

**Google Drive（`gdrive:adgenius_checkpoints/`）**

| 内容 | SFT（`sft_formal_v1/`，78.5 GB） | OPD（`opd_formal_v1/`，74.0 GB） |
|---|---|---|
| 完整 checkpoint `full/step_0025 … step_0200` | 8/8 | 8/8 |
| bf16 推理权重 `weights_bf16/scale_*` | 1k、2k、4k、8k 都齐全 | **只有 8k 齐全**；1k、2k、4k 因限流没传完（缺 `model.safetensors`） |
| 输出与日志 `outputs/`、`logs/` | 齐全 | 齐全 |

OPD 缺的 3 份 bf16 副本不影响结果：生成用的是 VM 上的副本；需要时可以从 Drive 上的 fp32 checkpoint 用同样的转换重新导出（转换是确定性的，但没有和丢失的文件逐位核对过）。

## 费用

| 项目 | 用量 |
|---|---|
| Colab | 本轮共 90.5 CU（244.0 → 153.6），包括冒烟测试、等 Drive 时的空转和两次正式训练 |
| Tinker | OPD 正式训练 2,000 次请求、32,000 条序列、2612 万 token；加上冒烟测试约 2680 万 token，按实测单价约 $16 |
| 百炼 | 397B 老师 test 生成：输入 74 万 token、输出 3.9 万 token，约 ¥1.2 |
| DeepSeek | 本轮未使用 |

## 遇到的问题和处理

| 问题 | 影响 | 处理 |
|---|---|---|
| rclone 共用的 client ID 撞上 Google 配额限制（`RATE_LIMIT_EXCEEDED`） | OPD 冒烟测试晚了约 2.5 小时，正式训练因此晚了约 3 小时；收尾时 OPD 的 1k、2k、4k 和 SFT 的 1k、4k 各卡在 bf16 上传上，最长 1 小时 | 训练不再等 Drive；加大重试；bf16 上传改为后台进行（仓库里的新版本；两次运行用的还是旧版本） |
| **续训会悄悄失效** | checkpoint 中的权重名称（`model.language_model.*`）与训练中的模型不一致，Trainer 续训时实际没有加载权重 | 续训时改用 `from_pretrained` 加载，中断再续训与不中断的 loss 只差 0.001 以内，已在冒烟测试中验证。正式训练没有发生中断，没有用到续训 |
| 长度上限 | rank 1543 的 prompt 超过 2048 token，会让 OPD 中途崩溃 | 上限提高到 4096 |
| GPU 备选逻辑 | 原代码要等约 2 小时才会换 H100 | 改为按 30 分钟计时；最后等到了 A100，没用上 H100 |
| `colab exec` 通过标准输入提交时挂起 | 两条命令没有执行，若延迟执行会干扰训练 | 发现后终止；改为用 `-f` 执行脚本文件 |
| 重启内核被权限检查拦下 | OPD 收尾无法换上新代码 | 等旧程序逐个超时；多花约 3 小时 A100 |

## 待办

1. **test 评测方案**：DeepSeek 两两比较（两层），参照系统包括未训练学生、397B 老师，以及是否还要 GLM。
2. **（可选）补齐 OPD 的 bf16 副本**：从 Drive 上的 fp32 checkpoint 重新转换并上传 1k、2k、4k。
3. **清理 Drive**：冒烟测试留下的 `adgenius_checkpoints/_smoke/`（42.3 GB）。
