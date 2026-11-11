# Gold 候选广告生成 v1

## 目的

为 100 个 gold 商品生成候选广告，供后续 **pairwise gold 人工标注**使用，并复用为 rubric 人工标注与人机一致率评估的输入。生成阶段只产出候选文案；人工完成判断后，才形成 gold 标签。

## 生成思路

- 输入是 `data/eval/gold/v1/product_items.jsonl` 中固定的 100 个 Office Products 商品，不使用最终 test 集或训练池。
- 每个商品分别调用四个模型，各生成一条英文广告，因此每个商品有四条候选，共 400 条。
- 选择不同模型系列及预期能力档位，以覆盖不同文案质量和写法。强、中、弱只是选型预期，不代表每条广告的质量；最终由人工标注判断。

| 模型别名 | 模型 | 预期档位 | 调用方式 |
| --- | --- | --- | --- |
| `qwen_local` | Qwen3.5-2B | 弱 | 本地 Ollama |
| `gemma_31b` | Gemma 4 31B | 中档对照 | Google Gemini API |
| `glm53_flash` | GLM-5.3-Flash | 中档对照 | 智谱 BigModel API |
| `longcat` | LongCat-2.5 | 强 | LongCat API |

四个模型使用相同的商品信息和生成任务：写一到两句简洁、面向消费者的英文推广文案，说明商品是什么，并传达一到两个具体、有用的商品事实。所有声明应有商品资料支持，避免编造规格、效果、保证或排名，只返回广告正文。

商品信息以文本列举方式呈现，使用 `title`、`categories`、`features`、`description` 和筛选后的 `details`；不传价格、评分、评论数或图片等信息。本次统一设置 `temperature=0.2`、`max_output_tokens=2048`。模型专属选项保持各自配置，例如 Qwen 关闭思考、GLM 开启思考。

## 结果与补跑

原始生成脚本为 `scripts/generate_candidates.py`，产物保存在 `artifacts/eval/gold_generation/v1/`。每个模型分别保存 `results_<alias>.jsonl` 和 `calls_<alias>.jsonl`，前者遵循 `GenerationResult` schema，后者记录请求、响应、usage 和错误等调用信息。

首次生成中，其余三个模型各成功 100 条；Gemma 成功 79 条，失败 21 条（11 条超时、10 条 HTTP 500）。随后通过 `scripts/retry_failed_candidates.py` 只补跑失败项，保留原来的模型、prompt、生成参数和候选 ID，将请求超时提高到 180 秒，并对临时错误做有限次数的退避重试。

**补跑后四个模型均为 100/100 成功，共 400 条。后续标注使用 `artifacts/eval/gold_generation/v1/retried/` 中的四份完整 results 文件。** 同目录的 calls 文件保留原调用历史并追加补跑记录；原始目录中的文件保留不变。

## 后续人工标注

1. **Pairwise gold：** 每个商品从四条候选中随机抽取两个不同模型的结果，组成一对，共 100 对。抽样时固定并记录随机种子；两位标注者独立标注同一组 100 对。标注材料展示商品信息和两条广告，隐藏模型来源，随机安排 A/B 位置；允许选择 A、B 或平局，并保存理由。人工结果用于校准 pairwise judge、评估人机一致率。抽样和标注是后续步骤，生成阶段没有产生这些标签。
2. **Rubric gold：** 保留全部 400 条候选，作为 rubric 人工标注与人机一致率评估的输入，不因 pairwise 的四选二抽样丢弃其他候选。具体标注规则由 rubric 协议约定。

这些商品及其候选属于 gold 校准集，不用于训练；最终模型比较使用独立冻结的 test 集。
