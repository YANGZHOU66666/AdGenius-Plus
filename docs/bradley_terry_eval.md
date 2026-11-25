# Bradley–Terry 评测链路

这条链路把多个生成系统之间的 pairwise 判断换算成每个系统一个分数，同时给出每组对比的胜、平、负。它对应 SFT 与 OPD 实验方案（训练分支的 `docs/sft_opd_experiment_plan.md`）第 6.2 节的两层评测：第 1 层看同规模下 OPD 对 SFT 的胜负，第 2 层用 Bradley–Terry（BT）拟合所有系统的总排名。评委是现有的 pairwise 评委，见 [pairwise 评委使用说明](pairwise_llm_judge.md)。

## 1. 原理

给每个系统一个分数 s。系统 A 对 B 的预期胜率只由分差决定：

P(A 赢 B) = 1 / (1 + e^−(s_A − s_B))

分差 0 对应胜率 50%，分差 1 约 73%，分差 2 约 88%。拟合是在所有比较上求最大似然（Hunter 2004 的 MM 算法）。两个系统即使没有直接比过，也能通过中间系统放到同一把尺子上。

- **每一对怎么记分：** 同一对的 AB、BA 两个顺序取平均。两次都赢记 1，两次结论相反记 0.5（算作平），两次都输记 0。评委有任何一个顺序失败的配对不计分，单独计数（可以续跑补齐）。
- **生成失败判负（暂定）：** 每组对比的商品集合固定取自商品文件，不因生成结果而缩小。某个商品上一方没有可用广告（生成 `failed`，或 `include_truncated = false` 时的截断广告），这一对不送评委，直接判这一方输；双方都没有可用广告时判平。这类配对记为 forfeit，计入胜/平/负和拟合。规则与训练分支 GenerationResult v1.1 的约定一致：给模型排名时分母固定为全部商品，failed 不剔除。
- **缺记录报错：** 被比较的系统在某个商品上完全没有生成记录时，生成配对会直接报错。这是数据不全，应先补齐生成。
- **基准：** 分数只有差值有意义，设计里的 `anchor` 系统固定为 0 分。
- **伪计数：** 每组对比额外加 0.5 个虚拟的平局，保证即使一方全胜，分数也是有限值。每组 1,000 对时这个影响可以忽略。
- **区间：** 按商品 bootstrap，默认重抽 2,000 次。同一商品在各组对比中的判断一起重抽，每次都重新拟合。
- **吻合检查：** 用拟合出的分数算出每组的预测胜率，看它是否落在这组观察胜率的区间内。只有比较图里有环（`cycle_rank > 0`）时，这项检查才有意义；如果比较只构成一棵树，拟合会完全复现每组的观察值。观察区间退化成一个值（比如一方全胜）的对比无法检查，会标为 n/t。

## 2. 设计文件

设计写在 `configs/eval/bt/<name>.toml`：

```toml
[design]
name = "test_sft_opd_v1"
products = "data/eval/test/product_items.jsonl"
anchor = "base"                       # 0 分的系统
pairs = "data/eval/test/bt_pairs_test_sft_opd_v1.jsonl"   # build_bt_pairs.py 的输出
include_truncated = true              # 截断的广告照常送评委（默认）；false 时按生成失败判负
seed = 20261124                       # 商品顺序和 max_products 抽样（默认）

[systems]
base = "data/eval/test/results_base.jsonl"    # 每个系统一份 GenerationResult
sft_1k = "data/eval/test/results_sft_1k.jsonl"

[[comparisons]]
systems = ["sft_1k", "base"]          # 先写预期更强的一方，胜率按它统计
# max_products = 400                  # 可选：只用打乱后的前 N 个商品
```

读取时会检查：比较中的系统都已列出，每个系统至少参加一组比较，没有重复的组合，所有系统通过比较连成一张图。

现有两份设计：

| 设计 | 内容 |
|---|---|
| [test_sft_opd_v1.toml](../configs/eval/bt/test_sft_opd_v1.toml) | 正式 test 评测：11 个系统、18 组对比，cycle rank 8。每组 1,000 对，共 18,000 对、36,000 次评委调用 |
| [gold_generators.toml](../configs/eval/bt/gold_generators.toml) | gold 上 4 个生成模型的两两对比，用于冒烟测试 |

`test_sft_opd_v1` 的 18 组对比：

| 类别 | 对比 | 组数 |
|---|---|---:|
| 主结论 | OPD 对 SFT，1k / 2k / 4k / 8k | 4 |
| 规模链 | SFT 1k–2k、2k–4k、4k–8k；OPD 同样三组 | 6 |
| 起点 | 训练前学生对 SFT 1k、OPD 1k、SFT 8k、OPD 8k | 4 |
| 参照 | GLM 对 SFT 8k、GLM 对 OPD 8k、GLM 对 397B、397B 对 SFT 8k | 4 |

## 3. 运行

在仓库根目录执行，前缀 `PYTHONPATH=src .venv/bin/python`。

**① 导入学生输出。** 训练机上生成的原始输出（每行 `product_id`、`text`、`finish_reason`）转成 GenerationResult v1.1。`finish_reason` 为 `stop` 记为 success，为 `length` 记为 truncated（保留截断后的原文），没有文字记为 failed。请显式给出 `--run-id`；默认值取自输入文件的修改时间，同步到本地后往往不是生成时间。

```sh
scripts/import_generation_outputs.py --system sft_1k --run-id gen_sft_1k_20261124 \
  --input <outputs/scale_1k/test.jsonl> --products data/eval/test/product_items.jsonl \
  --output data/eval/test/results_sft_1k.jsonl
```

参照模型（GLM、397B 教师）在 test 上的广告用 `generate_candidates.py` 或教师脚本生成，本身就是 GenerationResult。

**② 生成配对。** 先用 `--check` 看计划：每组的商品数、送评委的配对数、因生成失败直接判出的 forfeit 数、总调用次数、cycle rank。确认后去掉 `--check`，写出配对文件和 manifest。脚本不会覆盖已有文件。

```sh
scripts/build_bt_pairs.py --design configs/eval/bt/test_sft_opd_v1.toml --check
```

配对按商品交错排列：同一商品的各组对比挨在一起，商品顺序打乱。所以先用 `--limit` 小规模试跑，也能覆盖到每一组对比。

**③ 跑评委。** 用现有的 pairwise 评委，不带人工标签：

```sh
scripts/evaluate_pairwise.py --pairs data/eval/test/bt_pairs_test_sft_opd_v1.jsonl \
  --no-human-annotations --run-id deepseek_test_sft_opd_v1 --workers 32
```

**④ 统计。** 不调用 API，几秒内完成：

```sh
scripts/summarize_bt.py --design configs/eval/bt/test_sft_opd_v1.toml \
  --results results/pairwise/deepseek_test_sft_opd_v1/results.jsonl
```

`--results` 可以传多个文件，比如分批跑的几个 run，同一配对不能出现两次。`--pairs` 可以替换设计里的配对文件，比如对 gold 冻结的 `candidate_pairs.jsonl` 做冒烟测试。

## 4. 输出

默认写入 `results/bt/<design name>/`：

| 文件 | 内容 |
|---|---|
| `metrics.json` | 格式见 [BTMetrics v1](../metadata/schemas/bt_metrics.schema.json)，写出前校验。每个系统的分数和 95% 区间；每组对比的配对数、胜/平/负（含 forfeit）、forfeit 数、胜率和区间、BT 预测胜率、是否落在区间外；配对计数（评委计分、forfeit、不完整、未判） |
| `report.md` | 上面两张表的可读版本 |
| `manifest.json` | 设计、配对、评委结果、各系统文件和代码的哈希 |

## 5. 解读时注意

- 分数是相对于基准的对数几率，不是绝对质量。比较两个系统，看分差和两者的区间；同规模 OPD 对 SFT 的主结论，直接看那一组的胜率和区间。
- 每对只有一次评委判断（两个顺序），同一配对重跑时，约有 5% 的结论会变（见仓库外的并发测试记录）。按商品 bootstrap 的区间已经包含这部分波动。
- 截断的广告默认参加比较。训练前学生在 test 上有 21 条输出被截断；如果把它们排除，等于去掉它最差的一部分输出，分数会偏高。
- 生成失败目前暂定判负。截至 2026-11-24，gold、教师数据和已有的学生 test 输出中都没有 failed；可能出现的情形是训坏的 checkpoint 输出空文本，或 API 参照模型补跑后仍失败。报告中按组列出 forfeit 数，失败多时应单独说明。
- 吻合检查发现预测落在区间外时，说明"每个系统一个分数"不足以描述这些比较，比如某个系统对某一类对手特别强。这时应以各组对比的直接胜率为准。

## 6. 代码与测试

- 实现：[eval/bradley_terry.py](../src/adgenius/eval/bradley_terry.py)。
- 入口：[import_generation_outputs.py](../scripts/import_generation_outputs.py)、[build_bt_pairs.py](../scripts/build_bt_pairs.py)、[summarize_bt.py](../scripts/summarize_bt.py)。
- 测试：[tests/test_bradley_terry.py](../tests/test_bradley_terry.py)，覆盖设计校验、配对生成（截断、生成失败判负、双方失败判平、缺记录报错、交错、`max_products`）、已知分数的恢复、全胜时的有限分数、记分方向与平局、导入映射。
