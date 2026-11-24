# SFT 学习率选择

[sft_lr_sweep_v1](../../../results/training/sft_lr_sweep_v1/report.md) 实验的编排脚本：学习率 ∈ {5e-6, 1e-5, 2e-5}，每个 run 训练前 1k 个商品，在 200 个 dev 商品上用 DeepSeek 两两比较后按事先定好的规则选择（与 OPD 相同）。训练代码本身在上一级 [scripts/sft/](../README.md)。

## 脚本

| 脚本 | 在哪里运行 | 作用 |
|---|---|---|
| `watchdog_sft.py` | 本地，脱离会话运行 | 每 10 分钟检查一次 Colab L4 会话：拉回已完成的 run；会话被回收时重新申请 GPU，只重跑缺的 run；内核意外空闲时重启单元；全部完成后停掉 VM。连不上 Colab 时跳过这一轮，不当作回收。不需要 API key |
| `sweep_cell_sft.py` | VM，notebook 单元 | 一直执行到所有 run 跑完，保证内核不会因空闲被回收；记录 GPU 和库版本 |
| `vm_runner_sft.py` | VM | 依次训练每个学习率并生成 dev 输出；loss 不是有限值的 run 标记为发散，不生成 dev 输出 |
| `prepare_eval_sft.py` | 本地 | 给每条 dev 输出分配稳定的 `candidate_id`，生成 CandidatePair，并按规则直接处理文本相同或为空的商品。与 OPD 的版本只差生成 `run_id` 的前缀 |
| `run_judge.sh` | 本地 | 对一场比较调用仓库的 `scripts/evaluate_pairwise.py`（DeepSeek，pairwise-v0.2） |
| `select_sft.py` | 本地 | 按事先定好的规则计分并选择 |

watchdog 会把 `scripts/dev_generate.py`、`scripts/sft/sft_train.py`、本目录的 VM 脚本和数据文件，一起平铺上传到 VM 的 `/content/sweep/`。

## 复现

```sh
# 1. 冻结 prompt，再拼出 SFT 样本（与实验所用文件的哈希一致：595f1c37…）
PYTHONPATH=src .venv/bin/python scripts/build_training_prompts.py
python3 scripts/sft/build_sft_data.py

# 2. 本地检查顺序、loss 范围和 sampler（需要装有 trl 1.18 和 transformers 5.23 的环境，CPU 即可）
<trl-python> scripts/sft/check_sft_local.py

# 3. 启动 watchdog（需要 colab CLI 已登录）
nohup caffeinate -i python3 scripts/sft/lr_sweep/watchdog_sft.py --work-dir artifacts/train/sft_lr_sweep_rerun \
  > /dev/null 2>&1 &

# 4. 评审和选择（每场比较一个 run_judge.sh，可以并行）
python3 scripts/sft/lr_sweep/prepare_eval_sft.py --work-dir artifacts/train/sft_lr_sweep_rerun sft_lr5e-06__vs__sft_lr1e-05 ...
scripts/sft/lr_sweep/run_judge.sh artifacts/train/sft_lr_sweep_rerun sft_lr5e-06__vs__sft_lr1e-05 sft_sweep_dev_lr5e6_vs_lr1e5
python3 scripts/sft/lr_sweep/select_sft.py --work-dir artifacts/train/sft_lr_sweep_rerun
```

`select_sft.py` 默认从 `artifacts/train/opd_lr_sweep_v1/runs/base/dev.jsonl` 读取未训练学生的 dev 输出，用来算护栏指标；没有这个文件时用 `--base-dev` 指定。

## 与实验当时所用版本的差异

实验实际运行的版本保存在本地被忽略的 `artifacts/train/sft_lr_sweep_v1/code/`（当时放在同一个目录里），文件哈希记录在[实验 manifest](../../../results/training/sft_lr_sweep_v1/manifest.json) 的 `code.sha256`。仓库里的版本只改了路径和命令行参数，训练、生成、计分逻辑都没有变：

| 脚本 | 改动 |
|---|---|
| `sft_train.py`、`vm_runner_sft.py`、`sweep_cell_sft.py`、`prepare_eval_sft.py` | 无改动 |
| `build_sft_data.py` | prompt 目录和输出目录改为命令行参数，默认在 `artifacts/train/` 下 |
| `check_sft_local.py` | 数据路径改为参数；OPD 的前几个商品改从仓库的 OPD run manifest 读取 |
| `watchdog_sft.py` | 工作目录、数据目录、会话名、GPU 和学习率改为命令行参数；从三个目录收集要上传的文件 |
| `select_sft.py` | 未训练学生的 dev 输出路径改为参数。用这里的版本重新处理实验数据，得到的比较对和 `selection.json` 与 `results/` 中的完全相同 |
| `run_judge.sh` | 仓库根目录改为相对本目录定位 |
