# Rubric judge 人机一致率：事实有依据（`factual_support`）

本文记录用四个 LLM 评委判断单条广告是否与商品资料冲突，并与人工标签比较的实验和结论。状态截至 **2026-11-24**。逐条结果在 `results/rubric/`，统计在 [metrics.json](../results/rubric/comparison/metrics.json)。

## 1. 结论

**选择 DeepSeek（`deepseek-flash`，即 DeepSeek-V4.1-Flash）作为 rubric 的默认评委，配置为 [configs/eval/rubric_deepseek.toml](../configs/eval/rubric_deepseek.toml)。**

- 它的一致率和 kappa 在四个模型中最高：0.938，kappa 0.716。
- 误报最少：只把 2 条人工判为 True 的广告判成冲突。用评委比较生成模型时，误报会冤枉好广告。
- 130 次调用全部一次成功，输出 9,283 token，约为 Claude 的三分之一。没有出现"判了冲突却引用不出资料原文"的输出。
- 它的弱点是漏判偏多：18 条人工冲突中只抓到 12 条，所以用它统计的冲突率会偏低。

DeepSeek、Claude、GPT 三者的 95% 区间大幅重叠，这 130 条不足以说明它们之间有显著差异。Qwen Flash 的 kappa 接近 0，不适合作为 rubric 评委。

## 2. 数据与人工标签

- **样本：** [rubric_sample.jsonl](../data/eval/gold/rubric_sample.jsonl) 共 130 条，包括 qwen_local 全部 100 条，以及 Gemma、GLM、LongCat 各 10 条（来自 30 个不同商品）。抽样方法见[数据说明](data_pipeline.md)第 9 节。
- **人工标签：** [rubric_annotations_zhou.jsonl](../data/eval/gold/rubric_annotations_zhou.jsonl)，按 [rubric v0.2](individual_rubric_v0.2.md) 判定。共 18 条 False，全部来自 qwen_local；其他三个模型的 30 条都是 True。

## 3. 评委设置

| 运行 ID | 模型 | 推理设置 | 配置 |
|---|---|---|---|
| `deepseek41_rubric_v1` | `deepseek-flash` | temperature 0，关闭 thinking，最多 1,024 输出 token | `rubric_deepseek.toml` |
| `claude55_rubric_v1` | `claude-haiku-5-5` | 不设 temperature，adaptive thinking，effort low，最多 2,048 输出 token，JSON schema 输出 | `rubric_claude.toml` |
| `gpt6_luna_rubric_v1` | `gpt-6-luna` | temperature 0，reasoning effort none，最多 1,024 输出 token | `rubric_gpt6_luna.toml` |
| `qwen37_rubric_v1` | `qwen3.7-flash` | temperature 0，关闭 thinking，最多 1,024 输出 token | `rubric_qwen.toml` |

推理设置与 pairwise 评委相同。

- **输入：** 评委看到的商品资料由 `render_product_information` 渲染，和生成广告时发给模型的、以及人工标注时看到的完全一样；另加一条广告。评委看不到生成模型的名称，也看不到人工标签。
- **Prompt：** [system_v1.txt](../prompts/eval/rubric/system_v1.txt) 包含 v0.2 的核心规则和订书机示例，不含规则文档中的已裁定边界情况表；[user_v1.txt](../prompts/eval/rubric/user_v1.txt) 是商品资料加广告。
- **输出：** `{"value", "conflicts": [{"ad", "information"}], "reason"}`。写入结果时，冲突原文并入 RubricResult 的 `reason` 字段。
- **调用：** 每条广告调用一次。网络错误或 JSON 解析失败最多重试，每条最多 4 次。

## 4. 指标

正类为"冲突"（`factual_support = false`）。

| 指标 | 含义 |
|---|---|
| 一致率 | 评委与人工判断相同的比例 |
| Cohen's kappa | 扣除偶然一致后的一致程度。人工 86.2% 为 True，评委全判 True 时一致率就有 0.862，kappa 为 0，所以必须一起看 |
| 抓到 / 漏判 | 人工判为冲突的 18 条中，评委判对 / 判错的条数 |
| 误报 | 人工判为 True、评委判为冲突的条数 |

95% 区间使用逐条 bootstrap，10,000 次，seed 20261124。

## 5. 结果

| 评委 | 一致率 [95% 区间] | kappa [95% 区间] | 抓到 / 18 | 误报 | 尝试次数（失败） | 输入 / 输出 token |
|---|---|---|---:|---:|---|---|
| DeepSeek | 0.938 [0.892, 0.977] | 0.716 [0.504, 0.884] | 12 | 2 | 130 (0) | 134,765 / 9,283 |
| GPT-6 Luna | 0.915 [0.862, 0.962] | 0.669 [0.460, 0.836] | 14 | 7 | 130 (0) | 130,146 / 8,123 |
| Claude Haiku | 0.915 [0.862, 0.962] | 0.653 [0.435, 0.834] | 13 | 6 | 130 (0) | 242,653 / 26,708 |
| Qwen Flash | 0.746 [0.669, 0.823] | 0.047 [-0.123, 0.241] | 4 | 19 | 154 (24) | 167,892 / 56,662 |

- 抓到冲突的 95% 区间都很宽（DeepSeek 为 0.44–0.88），原因是人工冲突只有 18 条。
- 在其他三个生成模型的 30 条广告上（人工全部判 True），DeepSeek 和 GPT 没有误报，Claude 误报 1 条，Qwen 误报 5 条。
- Claude 有 2 条判为冲突，却没有引用任何资料原文（`false_without_quoted_conflict`）。按规则"引用不出资料原文就不是冲突"把它们改回 True 后，Claude 的一致率为 0.931、kappa 0.703，仍低于 DeepSeek。
- Qwen 有 24 次失败尝试，都是 JSON 没写完，其中 23 次输出达到了 1,024 token 上限。它抓到的冲突少，误报又多，判断接近随机。

## 6. 主要分歧

三个较强评委（DeepSeek、Claude、GPT）中，有两个以上同时漏判的：

| item | 情况 | 漏判的评委 |
|---|---|---|
| 087 | 资料中高度范围一处写 44.9、其余写 42.9，文案写 42.9；评委按"多数一致"判为无冲突 | 三个都漏 |
| 102 | 文案写 "each set includes 100 screws"，资料是一盒 100 套 | 三个都漏 |
| 028 | 文案写 "brand-new"，标题写 "Remanufactured" | DeepSeek、Claude |
| 016 | 文案说内部是不锈钢，资料说内部是防刮衬里 | DeepSeek、Claude |

两个以上评委同时误报的只有两条：003，评委把 details 里的 Item Weight 2.6 磅当作与正文 2 磅冲突；052，评委把资料没说的 "leather pages" 当作冲突。v0.2 的第 3 条规则（正文数值优先于 details 数值字段）就是针对前一类补充的。其余误报都只出现在单个评委上。

## 7. 限制

- **单人标签：** 人工标签只有一位标注者，没有两人独立标注的一致率可作参照，不能视为共识标签。
- **规则与 prompt 来自同一批数据：** v0.2 的规则和边界裁定是在这 130 条上定的。评委选择也在同一批数据上完成，结果偏乐观。若要报告可推广的一致率，应另取一批未参与定规则的广告人工标注后再验证。
- **事后修改：** 016 的人工标签和 v0.2 第 3 条规则是在看过评委结果后才定的。016 改为 False 后，只有 GPT 在这条上与人工一致，GPT 的一致率因此比改前高一条。改前（17 条冲突）的统计：DeepSeek 0.946 / kappa 0.744、Claude 0.923 / 0.678、GPT 0.908 / 0.631、Qwen 0.754 / 0.058。DeepSeek 仍然第一，Claude 和 GPT 的先后因此对调。
- **Prompt 早于规则补充：** `system_v1.txt` 写于第 3 条规则补充之前，没有包含这一条，也没有重跑。
- **运行来源：** 四次运行是 2026-11-24 用一个探索版脚本跑的。prompt、模型设置、请求构造、解析和重试次数都与 [scripts/evaluate_rubric.py](../scripts/evaluate_rubric.py) 相同，之后用仓库代码的记录格式导入。导入时重新分配了 trace ID，未记录调用时间和延迟；各运行 manifest 的 `provenance` 字段有说明。

## 8. 复现与新运行

重新计算统计（不调用 API）：

```sh
PYTHONPATH=src .venv/bin/python scripts/compare_rubric.py
```

运行一个新的评委（默认写入 `results/rubric/<run_id>/`，调用日志写入 `artifacts/rubric/<run_id>/calls.jsonl`）：

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_rubric.py \
  --eval-config configs/eval/rubric_deepseek.toml --run-id deepseek41_rubric_v2
```

修改 prompt、配置、数据或代码后，请使用新的 run ID。已有运行目录中的 manifest 记录了输入和代码哈希，配置不一致时会拒绝续跑。重新调用模型不保证得到相同的判断。
