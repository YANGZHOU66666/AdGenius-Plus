# OPD 学习率选择

[opd_lr_sweep_v1](../../../results/training/opd_lr_sweep_v1/report.md) 实验的编排脚本：beta ∈ {1, 0.5} × 学习率 ∈ {5e-7, 1e-6, 2e-6}，每个 run 训练前 1k 个 prompt，在 200 个 dev 商品上用 DeepSeek 两两比较后按事先定好的规则选择。训练代码本身在上一级 [scripts/opd/](../README.md)。

## 脚本

| 脚本 | 在哪里运行 | 作用 |
|---|---|---|
| `watchdog.py` | 本地，脱离会话运行 | 每 10 分钟检查一次 Colab 会话：拉回已完成的 run；会话被回收时重新申请 GPU，只重跑缺的 run；内核意外空闲时重启单元；全部完成后停掉 VM。连不上 Colab 时跳过这一轮，不当作回收 |
| `sweep_cell.py` | VM，notebook 单元 | 一直执行到所有分组跑完，保证内核不会因空闲被回收；读取一次性 key 文件后立即删除 |
| `vm_runner.py` | VM | 一个 beta 分组：启动 Tinker 代理，依次训练每个学习率并生成 dev 输出 |
| `prepare_eval.py` | 本地 | 给每条 dev 输出分配稳定的 `candidate_id`，生成 CandidatePair，并按规则直接处理文本相同或为空的商品 |
| `run_judge.sh` | 本地 | 对一场比较调用仓库的 `scripts/evaluate_pairwise.py`（DeepSeek，pairwise-v0.2） |
| `select_from_pairwise.py` | 本地 | 按事先定好的规则计分并选择 |

watchdog 会把 `scripts/dev_generate.py`、`scripts/opd/` 下的训练代码和本目录的 VM 脚本，一起平铺上传到 VM 的 `/content/sweep/`。

## 复现

```sh
# 1. 冻结 prompt（与实验所用文件的哈希一致：train 13144069…，dev 2e35dec3…）
PYTHONPATH=src .venv/bin/python scripts/build_training_prompts.py

# 2. 启动 watchdog（需要 colab CLI 已登录，.env 中有 TINKER_API_KEY）
nohup caffeinate -i python3 scripts/opd/lr_sweep/watchdog.py --work-dir artifacts/train/opd_lr_sweep_rerun \
  > /dev/null 2>&1 &

# 3. 评审和选择（每场比较一个 run_judge.sh，可以并行）
python3 scripts/opd/lr_sweep/prepare_eval.py --work-dir artifacts/train/opd_lr_sweep_rerun beta1_lr5e-07__vs__beta1_lr1e-06 ...
scripts/opd/lr_sweep/run_judge.sh artifacts/train/opd_lr_sweep_rerun beta1_lr5e-07__vs__beta1_lr1e-06 <judge_run_id>
python3 scripts/opd/lr_sweep/select_from_pairwise.py --work-dir artifacts/train/opd_lr_sweep_rerun
```

训练时的采样有随机性，重跑得到的模型输出不会和原实验逐字相同。原实验的结果以 `results/` 中的文件为准。

## 与实验当时所用版本的差异

实验实际运行的版本保存在本地被忽略的 `artifacts/train/opd_lr_sweep_v1/code/`（当时放在同一个目录里），文件哈希记录在[实验 manifest](../../../results/training/opd_lr_sweep_v1/manifest.json) 的 `code.sha256`。仓库里的版本只做了以下改动，训练、生成、计分逻辑都没有变：

| 脚本 | 改动 |
|---|---|
| `completion_logits_trainer.py`、`check_completion_logits.py`、`dev_generate.py`、`vm_runner.py`、`sweep_cell.py` | 无改动 |
| `opd_train.py` | 在训练开始前计算代码和数据哈希。原版在训练结束写报告时才计算，中途替换文件会记成错误的版本 |
| `teacher_proxy.py` | `.env` 路径改为相对仓库；`tinker` 和 `transformers` 改为在用到时才导入，方便测试。这里是修复后的版本（按第一次出现的 assistant 开头定位回答，有缺口时整条重新打分） |
| `watchdog.py` | 工作目录、prompt 目录、会话名、GPU、beta 和学习率改为命令行参数；从三个目录收集要上传的文件；连不上 Colab 时跳过检查，不再误判为回收 |
| `prepare_eval.py`、`select_from_pairwise.py`、`run_judge.sh` | 工作目录改为命令行参数。用这里的版本重新处理实验数据，得到的比较对和 `selection.json` 与 `results/` 中的完全相同 |
| `build_training_prompts.py` | 新增：实验当时的 prompt 是临时生成的，这个脚本可以按原哈希重建 |

实验期间用过、现已不再需要的启动脚本（逐个会话的启动器、旧监控脚本、旧评审脚本）没有放进来。

**正式训练之后的改动**：`opd_train.py` 已更新为正式训练实际运行的版本（见 [../../formal/](../../formal/README.md)）：每 25 步存完整 checkpoint（原来只在最后一步存）、vLLM 长度上限 4096（原来 2048）、续训时用 `from_pretrained` 加载权重、通过 `drive_sync.py` 上传到 Drive。`watchdog.py` 因此也会上传 `../../formal/drive_sync.py`。在 dev 的 1k 训练里这些改动不改变 loss 和输出（dev prompt 都短于 2048 token），但每 25 步会多存一份约 9 GB 的 checkpoint；要逐字节复现选学习率时的代码，用提交 `7b9ccde` 中的版本（与实际运行版本的差异见上表）或 `artifacts/train/opd_lr_sweep_v1/code/`。
