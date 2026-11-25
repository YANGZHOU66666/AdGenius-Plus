# AdGenius+

Office Products 广告生成、离线评测与小模型微调实验。仓库结构见 [docs/architecture.md](docs/architecture.md)。

所有现有数据的含义、来源、生产步骤、依赖关系和重建能力统一见 [数据说明与生产链路](docs/data_pipeline.md)。

小模型训练实验（SFT 与 OPD 对比）的训练与评测设计见 [SFT 与 OPD 蒸馏实验方案](docs/sft_opd_experiment_plan.md)（方案阶段）；设计依据、试点结果和工程备忘见 [训练实验杂记](docs/training_notes.md)。

## 安装与 Key

Python 3.11+：

```sh
python -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

在 `.env` 填需要调用的平台 Key。共享包装器支持 OpenAI 兼容协议、Claude Messages、Google generateContent 和本地 Ollama；入口脚本显式加载 `.env`。配置见 `configs/models.toml`，Key 和虚拟环境不提交。

| 模型别名 | 平台 / 环境变量 |
|---|---|
| `qwen` | 阿里云中国区百炼，`DASHSCOPE_API_KEY` |
| `deepseek` | DeepSeek，`DEEPSEEK_API_KEY` |
| `claude` | Anthropic，`ANTHROPIC_API_KEY` |
| `gpt6_luna` | OpenAI，`OPENAI_API_KEY` |
| `gemma_31b` | Google，`GEMINI_API_KEY` |
| `glm53_flash` | 智谱，`ZAI_API_KEY` |
| `longcat` | LongCat，`LONGCAT_API_KEY` |
| `mistral` | Mistral，`MISTRAL_API_KEY` |
| `qwen_local` | 本地 Ollama，无需 Key |

阿里云默认北京兼容端点；可用 `DASHSCOPE_BASE_URL` 设置同地区工作空间的兼容 API 根地址。完整参数和调用记录格式见 [metadata/README.md](metadata/README.md) 与共享客户端代码。

```sh
PYTHONPATH=src .venv/bin/python scripts/llm_smoke.py --model deepseek
```

本地 Qwen 需要先启动 Ollama 并准备对应模型：

```sh
ollama pull qwen3.5:2b
PYTHONPATH=src .venv/bin/python scripts/llm_smoke.py --model qwen_local
```

## 当前正式数据

**队友 clone 后可直接读取完整 400 条广告、100 对 pair 和100 条当前人工标签，无需重新调用模型。**正式数据统一在 `data/eval/gold/`，完整索引见 [data/README.md](data/README.md)。

```text
data/eval/gold/
├── product_items.jsonl
├── results_qwen_local.jsonl
├── results_gemma_31b.jsonl
├── results_glm53_flash.jsonl
├── results_longcat.jsonl
├── candidate_pairs.jsonl
├── candidate_pairs.manifest.json
├── pairwise_annotations_zhou.jsonl
└── manifest.json
```

商品、候选、pair 的内容和 ID 已冻结。人工标签仅提交最终 100 条，15 条修改已由用户确认；原始标注与复核记录仅在本地 `deprecated/` 保存，不提交 Git。`data/splits/v1/` 保留 gold 100、test 1,000、训练候选池 709,403 条的固定划分；gold/test 都不能进入训练。大规模原始数据和训练池索引保留获取/重建信息，不上传普通 Git。

```sh
# 离线校验400条候选、100对、100个人标和四模型当前统计，不用 Key。
PYTHONPATH=src .venv/bin/python scripts/validate_gold_data.py
```

## 标注与评测

启动英文人工标注台，默认加载正式 gold 及已有标签：

```sh
PYTHONPATH=src .venv/bin/python tools/pairwise_annotation.py
```

访问 http://127.0.0.1:8765。随机左右、保存结论、目录跳转和自动下一题的设计见 [标注台文档](docs/pairwise_annotation_workbench.md)。

[四模型最新统计](results/pairwise/comparison/report.md)已随仓库共享。Qwen、DeepSeek、Claude 和 GPT-6 Luna 使用相同的广告文本进行独立 AB/BA 判断；说明与参数见 [Judge 文档](docs/pairwise_llm_judge.md)。新运行使用新的 run ID，标准结果写入 results/pairwise/<run_id>/，调用日志写入 artifacts/pairwise/<run_id>/：

默认评委已选定为 **DeepSeek**，综合依据见四模型对比报告。省略 `--eval-config` 时加载 `configs/eval/pairwise.toml`，使用 `DEEPSEEK_API_KEY`、temperature 0、关闭 thinking、最多 1,024 个输出 tokens。Qwen 对照使用 `configs/eval/pairwise_qwen.toml`。

```sh
# 只检查计划，不联网、不写产物。
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py --run-id deepseek_gold_new --dry-run

# 真实全量评测：100对，200次初始调用。
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py --run-id deepseek_gold_new
```

## 候选生成与补跑

现有400条直接使用 `data/eval/gold/results_*.jsonl`。生成其他批次时指定新的运行目录；产物仍是四份 results 和四份 calls，已有文件不会覆盖：

```sh
PYTHONPATH=src .venv/bin/python scripts/generate_candidates.py   --input data/eval/gold/product_items.jsonl   --output-dir artifacts/generation/new_run

PYTHONPATH=src .venv/bin/python scripts/retry_failed_candidates.py   --source-dir artifacts/generation/new_run --dry-run
```

生成、补跑的默认商品入口已统一到正式 gold；默认运行输出目录为 `artifacts/generation/gold/`，里面是原始运行快照和调用历史。修改运行输出不会自动替换冻结的数据集。详见 [生成与抽样方法](docs/gold_candidate_generation.md)。

## 在代码中调用

```python
from adgenius.shared.llm import LLMClient, ModelConfig

config = ModelConfig.from_toml("configs/models.toml", "deepseek")
with LLMClient(config) as client:
    response = client.generate(
        [{"role": "user", "content": "Write a short product ad."}],
        max_output_tokens=256,
        temperature=0.2,
    )
print(response.text)
```

可传 `json_mode` 和原生 `provider_options`，后者直接进入请求 JSON body，不能覆盖 model/messages/token 上限等核心字段。模型配置的 request_options 用于 thinking 等平台参数。支持有限重试并保留结构化错误；不会自动修复模型 JSON 或隐藏失败。

## Git 与归档

共享：data/ 中的正式数据、results/ 中的逐条结果与统计，以及配置、代码和文档。artifacts/ 整体作为本地运行目录忽略。

本地保存：`.env`；artifacts/ 下的完整 calls 日志、训练池缓存、checkpoint 和运行中间产物；以及外部大原始数据。旧试跑、废弃数据、原始人工标注及复核记录放在 [deprecated/](deprecated/README.md)；迁移索引记录移动前后的文件哈希。

## 离线检查

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
node --test tests/test_pairwise_annotation_frontend.cjs
PYTHONPATH=src .venv/bin/python scripts/validate_gold_data.py
```

## 输入输出位置

| 工作 | 输入 | 输出 |
|---|---|---|
| 生成 / 补跑 | data/eval/gold/product_items.jsonl | artifacts/generation/<run>/，默认 gold |
| Pair 抽样 | data/eval/gold/results_*.jsonl | 由 --output 明确指定，正式数据放 data/ |
| 人工标注 | data/eval/gold/ | 同目录的人工标签 |
| Pairwise Judge | data/eval/gold/ | results/pairwise/<run_id>/；calls 单独写 artifacts/pairwise/<run_id>/ |
| 数据划分 | 外部原始 gzip | 小划分和商品放 data/；大训练池放 artifacts/cache/office_products/ |

详见 [正式结果索引](results/README.md)。生成、评测脚本提供 --output-dir；Judge 另有 --calls-dir。已有实验的原始 manifest 保留历史路径，新的输入输出采用当前结构。
