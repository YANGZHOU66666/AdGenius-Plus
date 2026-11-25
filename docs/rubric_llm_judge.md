# Rubric LLM Judge

rubric 评委逐条检查一条广告是否与它的商品资料冲突，目前只判一个标准：事实有依据（`factual_support`）。判定规则见 [rubric v0.2](individual_rubric_v0.2.md)，评委选择的依据见[人机一致率报告](rubric_judge_agreement.md)。

**默认评委是 DeepSeek**（`deepseek-flash`），配置为 [configs/eval/rubric_deepseek.toml](../configs/eval/rubric_deepseek.toml)。不传 `--eval-config` 时就用它，需要 `DEEPSEEK_API_KEY`。

## 输入与协议

评委每次只看一条广告，和该广告对应的商品资料：

- **商品资料：** 由 `render_product_information` 渲染，与生成广告时发给模型的、以及人工标注时看到的完全相同。
- **Prompt：** [system_v1.txt](../prompts/eval/rubric/system_v1.txt) 写明 v0.2 的规则；[user_v1.txt](../prompts/eval/rubric/user_v1.txt) 是商品资料加广告。
- **评委看不到：** 生成模型名称、候选 ID、人工标签，以及其他广告。
- **协议：** `rubric-v0.2`，`input_mode = "product_and_ad"`，`criteria = ["factual_support"]`。

评委返回严格 JSON：

```json
{"value": false, "conflicts": [{"ad": "…", "information": "…"}], "reason": "…"}
```

`value` 为 `false` 表示冲突。写入结果时，`conflicts` 中引用的原文并入 [RubricResult v1](../metadata/schemas/rubric_result.schema.json) 的 `reason` 字段，格式为 `<reason> | conflicts: ad "…" vs information "…"`。

脚本有两种输入：

| 模式 | 参数 | 判哪些广告 | 用途 |
|---|---|---|---|
| rubric 样本（默认） | `--gold-dir`，默认 `data/eval/gold` | [rubric_sample.jsonl](../data/eval/gold/rubric_sample.jsonl) 的 130 条 | 与人工标签比较，校准评委 |
| 生成结果 | `--candidates <results_*.jsonl …>`，`--products <ProductItem>` | 给定文件中每条有广告的结果（success 和 truncated） | 评测生成模型 |

## 配置

| 配置 | 密钥 | 推理设置 |
|---|---|---|
| `configs/eval/rubric_deepseek.toml`（默认） | `DEEPSEEK_API_KEY` | 关闭 thinking，temperature 0，最多 1,024 输出 token |
| `configs/eval/rubric_gpt6_luna.toml` | `OPENAI_API_KEY` | reasoning effort none，temperature 0，最多 1,024 输出 token |
| `configs/eval/rubric_claude.toml` | `ANTHROPIC_API_KEY` | adaptive thinking，effort low，不设 temperature，最多 2,048 输出 token |
| `configs/eval/rubric_qwen.toml` | `DASHSCOPE_API_KEY` | 关闭 thinking，temperature 0，最多 1,024 输出 token；kappa 接近 0，不建议使用 |

各家的推理参数不同，请直接选对应的配置文件，不要只改 `model_alias`。Claude 通过原生 `output_config.format` 接收 JSON schema；其他三家使用 JSON mode，再在本地校验。

## 运行

在仓库根目录执行，密钥放在 `.env` 中。`--dry-run` 不需要密钥，不调用 API，也不写文件，只核对输入条数和输出位置。

**在 rubric 样本上跑一个评委：**

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_rubric.py \
  --eval-config configs/eval/rubric_deepseek.toml --run-id deepseek41_rubric_v2 --dry-run
```

**评测生成模型：** 把生成结果（GenerationResult 格式，例如 `generate_candidates.py` 写出的 `results_<model>.jsonl`）传给 `--candidates`，可以一次传多个文件：

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_rubric.py \
  --candidates path/to/results_new_model.jsonl data/eval/gold/results_qwen_local.jsonl \
  --products data/eval/gold/product_items.jsonl \
  --run-id deepseek41_rubric_v1_new_model --dry-run
```

- `--products` 是这些广告对应的 ProductItem 文件，默认是 gold 的 100 件商品。换成 test 商品时传 test 的 ProductItem 文件。广告引用了文件里没有的商品时，脚本会报错。
- 生成模型名取自生成运行 ID `gen_<model>_<时间>`；不符合这个格式的，用整个运行 ID 作为模型名。也可以用 `--generator-names` 按文件顺序给每个 `--candidates` 文件指定一个短名，比如 `sft_1k opd_1k`。
- 因长度上限被截断（`truncated`）的广告照常判，因为这就是生成模型实际给出的内容。生成失败（`failed`）的条目没有广告可判，会跳过，并在 `skipped_failed_generations` 中按模型计数。不同文件中出现重复的 `candidate_id` 时会报错。
- 比较新模型时，把旧模型的结果放进同一次运行，这样评委、prompt 和商品都相同。

去掉 `--dry-run` 才会真正调用 API。先用 `--limit N` 取前 N 条、换一个单独的 run ID 试跑。`--workers` 是并发请求数，默认为 6；大批量时可以调到 32。

## 重试与续跑

- 每条广告最多尝试 `max_attempts` 次（默认 4 次），两次之间等待 2、4、8 秒。网络错误和 JSON 解析失败都会重试。
- 用完尝试次数仍失败的，结果记为 `status = "failed"`、`error = "judge_failed"`，运行状态为 `completed_with_failures`。
- 每次尝试都在调用日志末尾追加一行并立即落盘，日志是续跑的依据。`results.jsonl` 在运行中最多每 15 秒重写一次，结束或中断时按输入顺序完整写出。按 Ctrl+C 时，未开始的条目会取消，已发出的请求会等它返回并记入日志。
- 中断或有失败时，用同一个 run ID 重新运行即可：已成功的条目会保留，只重做缺失或失败的条目。如果某次成功已经写进日志、还没来得及写进 `results.jsonl`，续跑时会从日志重建这条结果，不再调用 API。日志最后一行因崩溃只写了一半时，这一行会被丢弃，对应条目重新判。
- 结果已全部成功时再运行，不需要密钥，也不调用 API，只重算 `metrics.json`。
- manifest 记录了配置、prompt、输入文件、代码与 schema 的哈希，以及条目顺序。其中任何一项变化，续跑都会被拒绝，这时请使用新的 run ID。不要为了续跑去改旧的 manifest。
- 同一个运行目录不要同时开两个进程，目前没有加锁。

## 输出

| 位置 | 文件 | Git |
|---|---|---|
| `results/rubric/<run_id>/` | `results.jsonl`、`metrics.json`、`manifest.json` | 共享 |
| `artifacts/rubric/<run_id>/` | `calls.jsonl` | 忽略 |

- `results.jsonl`：每条广告一条 RubricResult model 记录，顺序与输入一致。`result_id` 由 run ID 和 `candidate_id` 确定，`trace_id` 指向调用日志中的那次成功尝试。
- `calls.jsonl`：每次尝试一行，包括失败的尝试，记录原始返回文本、token 用量和延迟。读结果不需要它，排查问题时才用得上。
- `manifest.json`：输入模式、配置、模型设置、各项哈希、条目列表和运行状态。
- `--output-dir` 和 `--calls-dir` 可以改这两个目录。

当前四次正式运行（130 条样本上的四个评委）是由探索脚本跑出后导入的，没有 `metrics.json`，见各自 manifest 的 `provenance` 字段。

## 指标

**每次运行的 `metrics.json`**：按生成模型统计评委的判断。格式见 [RubricGeneratorMetrics v1](../metadata/schemas/rubric_generator_metrics.schema.json)，写出前校验，带 `schema_version` 字段。

| 字段 | 含义 |
|---|---|
| `by_generator.<模型>.false`、`false_rate` | 判为冲突的条数，以及占成功判断条数的比例 |
| `false_rate_ci95` | 按商品 bootstrap（10,000 次，seed 20261124）得到的 95% 区间 |
| `judge_failed` | 评委失败、未计入比例的条数 |
| `false_without_quoted_conflict` | 判了冲突却没有引用原文的条数；按规则，这些不算冲突，应人工复查 |
| `paired_differences` | 同一批商品上，两两模型冲突率之差（前者减后者），及按商品 bootstrap 得到的 95% 区间 |
| `calls` | 尝试次数、失败次数和 token 总数 |

解读时要注意：

- 这是评委判出的冲突率，不是人工标签。默认评委在 130 条样本中只抓到 18 条人工冲突中的 12 条，所以绝对冲突率会偏低。它适合在同一次运行、同一评委下比较生成模型。
- 评委只在 qwen_local 那类冲突上验证过；对其他模型可能出现的其他类型错误，判得准不准还不知道。
- 一个模型的冲突为 0 条时，区间会退化成 [0, 0]，但这并不说明结果没有不确定性。
- 比较两个模型时，看 `paired_differences` 的区间是否跨过 0，比分别看两个模型各自的区间更准确。

**与人工标签比较：** [scripts/compare_rubric.py](../scripts/compare_rubric.py) 读取样本上的评委运行和 [rubric_annotations_zhou.jsonl](../data/eval/gold/rubric_annotations_zhou.jsonl)，计算一致率、Cohen's kappa、抓到 / 漏判 / 误报条数和 bootstrap 区间，写入 `results/rubric/comparison/`。这一步不调用 API：

```sh
PYTHONPATH=src .venv/bin/python scripts/compare_rubric.py
```

`--runs` 可以指定参与比较的运行，默认是当前四个评委。

## 代码与测试

- 入口：[scripts/evaluate_rubric.py](../scripts/evaluate_rubric.py)；实现在 [eval/rubric.py](../src/adgenius/eval/rubric.py)。
- 统计：[eval/rubric_metrics.py](../src/adgenius/eval/rubric_metrics.py)。
- 离线测试：[tests/test_rubric_judge.py](../tests/test_rubric_judge.py)，覆盖配置、请求构造、解析、重试、续跑、生成结果输入和统计。

新增其他 rubric 标准时，需要同时修改配置校验（目前只接受 `factual_support`）、输出 schema 和 prompt。
