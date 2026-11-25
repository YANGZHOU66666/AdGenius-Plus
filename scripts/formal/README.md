# 正式训练（SFT 和 OPD 共用）

两种方法各用选学习率得到的设置，1 个种子（20261123），从头训练到 8k（每步 40 个商品，共 200 步），每 25 步存一份完整 checkpoint 到 Google Drive，在 1k、2k、4k、8k（第 25、50、100、200 步）导出 bf16 权重并在 dev 和 test 上贪心生成。结果见 [results/training/sft_formal_v1](../../results/training/sft_formal_v1/manifest.json)、[opd_formal_v1](../../results/training/opd_formal_v1/manifest.json) 和[报告](../../results/training/formal_v1_report.md)。

| | SFT | OPD |
|---|---|---|
| 设置 | 学习率 5e-6 | `beta=0.5`，学习率 2e-6，Tinker 老师（top1） |
| GPU | L4，备选 A100 | A100，备选 H100 |
| 训练代码 | [`../sft/sft_train.py`](../sft/sft_train.py) | [`../opd/opd_train.py`](../opd/opd_train.py)、`completion_logits_trainer.py`、`teacher_proxy.py` |

```text
scripts/formal/
├── watchdog_formal.py   Mac 上运行：申请 GPU、安装依赖、上传代码/数据/密钥、启动阻塞 cell；每 10 分钟同步状态和输出，
│                        内核空闲或 cell 退出时重启（从最新 checkpoint 续训），完成后等 Drive 上传、核对输出、停机
├── formal_cell.py       在 Colab 内核里阻塞运行 run_formal.py，内核一直忙，不会因空闲被回收
├── run_formal.py        VM 上的编排：建纯文本学生 →（OPD）启动老师代理 → 训练（失败重试，从最新 checkpoint 续训）
│                        → 每个规模导出 bf16 → 生成 dev/test →（SFT）未训练学生的 test → 上传 → DONE
├── drive_sync.py        训练回调：每次保存后后台 rclone 上传完整 checkpoint，写 UPLOAD_OK；写 progress.json；
│                        找最新的完整 checkpoint（本地没有就从 Drive 下载）
├── gen_outputs.py       vLLM 贪心生成（与 ../dev_generate.py 相同设置，只把 max_model_len 提高到 4096）
├── wait_uploads_cell.py DONE 之后让内核保持忙碌，直到所有 checkpoint 和 bf16 权重都上传到 Drive（有时间上限）
└── smoke.py             冒烟测试：4 步、模拟崩溃、删本地 checkpoint 强制从 Drive 续训、2 次导出、各 10 条生成
```

## 运行

需要：`colab` CLI 已登录；rclone 远端 `gdrive`（`~/.config/rclone/rclone.conf`）；OPD 还需要 `.env` 里的 `TINKER_API_KEY`；冻结的 prompt 和 SFT 数据在 `artifacts/train/{prompts,sft_data}/v1/`（由 `../build_training_prompts.py`、`../sft/build_sft_data.py` 生成）。

```bash
python3 scripts/formal/smoke.py --method sft
```

```bash
nohup caffeinate -i python3 scripts/formal/watchdog_formal.py --method sft > /dev/null 2>&1 &
```

```bash
nohup caffeinate -i python3 scripts/formal/watchdog_formal.py --method opd > /dev/null 2>&1 &
```

VM 的状态、日志和输出同步到 `artifacts/train/<run>/`（不进 Git），watchdog 日志是其中的 `watchdog_<method>.log`。Drive 上的位置是 `gdrive:adgenius_checkpoints/<run>/{full/step_NNNN, weights_bf16/scale_*, outputs, logs}`。

密钥只以 0600 临时文件上传：Tinker key 在 cell 读取后立即删除；rclone 配置留在 `/content/run/.rclone.conf`，随 VM 释放。日志里不写会话 URL 和密钥。

## 与实际运行版本的差异

`sft_formal_v1` 和 `opd_formal_v1` 运行时的训练代码与这里逐字节相同（哈希记录在各 run 的 `manifest.json`）。编排代码有两处不同，都不影响训练和输出：

- `run_formal.py`：两次运行时 bf16 权重的 Drive 上传是同步的（等上传结束才生成）；这里改为后台上传，收尾时由 `wait_uploads_cell.py` 补传。
- 路径：原来所有文件在一个目录里，这里改为从仓库各处读取（`watchdog_formal.py` 的 `source()`），本地副本写到 `--work-dir`。

实际运行的版本保存在 `artifacts/train/<run>/code/`（不进 Git）。
