# AdGenius+

Office Products 广告文案生成、离线评测与小模型微调实验。仓库结构见 [docs/architecture.md](docs/architecture.md)。

## LLM API 包装

Python 3.11+，使用 `httpx` 直接适配 OpenAI 兼容协议、Claude Messages、Google generateContent 和本地 Ollama，无需安装厂商 SDK。当前支持同步、非流式、纯文本消息。

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/llm_smoke.py
```

默认使用 mock，不联网、不需要 key。真实调用先在 [configs/models.toml](configs/models.toml) 确认模型 ID 和地域，再在 shell 设置环境变量，或填到仓库根目录 `.env`：

```sh
export DASHSCOPE_API_KEY='your-key'
python scripts/llm_smoke.py --model qwen
```

也支持 `--model qwen_local`（本地 Ollama，无需 Key）、`--model mistral`（`MISTRAL_API_KEY`）、`--model gemma_31b`（Google 官方 Gemini API，`GEMINI_API_KEY`）、`--model glm53_flash`（智谱官方，`ZAI_API_KEY`）、`--model longcat`（`LONGCAT_API_KEY`）、`--model deepseek`（`DEEPSEEK_API_KEY`）或 `--model claude`（`ANTHROPIC_API_KEY`）。smoke 脚本显式加载根目录 `.env`；后续 eval/train 入口也应在程序启动时显式调用 `dotenv.load_dotenv()`。`.env` 已加入 `.gitignore`。

当前候选生成选择四个模型来源，档位是实验预期，实际文案质量由标注检验：

| Alias | 模型 | 预期档位 | 调用渠道 |
| --- | --- | --- | --- |
| `qwen_local` | Qwen3.5-2B | 弱 | 本地 Ollama |
| `gemma_31b` | Gemma 4 31B | 中档对照 | Google Gemini API 官方托管 |
| `glm53_flash` | GLM-5.3-Flash，320B 总参数 / 18B 激活 | 中档对照 | 智谱 BigModel 官方 |
| `longcat` | LongCat-2.5 | 强 | LongCat 官方 |

Gemma 使用 Google 官方 Gemini API 的原生 `generateContent` 接口，模型 ID 为 `gemma-4-31b-it`；[官方部署页](https://gemma.ai/deployment/gemini-api/)明确支持 Gemma 4 31B 和 26B-A4B。调用时使用 `GEMINI_API_KEY`，在 [Google AI Studio](https://aistudio.google.com/apikey) 创建。当前官方价格表将 Gemma 4 的输入和输出标为免费，但 RPM、TPM、RPD 按项目和账号状态动态分配，在 [AI Studio 配额页](https://aistudio.google.com/rate-limit)查看。账户余额不代表一定获得更高 Gemma 配额。GLM 的百炼 pilot 返回产品未激活，且当前账号控制台未找到该模型；因此 `glm53_flash` 已改走智谱官方，API ID 为 `glm-5.3-flash`，需要新的 `ZAI_API_KEY`，不复用百炼 Key。[智谱模型文档](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)列明该模型仅支持启用思考；试跑可先给 2048 输出预算，再检查是否截断或只有思考没有正文，这不是已验证足够的预算。

本地配置使用 Ollama 原生 `/api/chat` 接口，并通过 `think=false` 关闭 Qwen 的思考输出；如果使用 vLLM 或其他 OpenAI-compatible 服务，把 provider 改为 `openai_compatible_local`、base URL 改为该服务的 `/v1` 地址即可。只允许本地 loopback HTTP，远程 API 仍要求 HTTPS。LongCat 的官方 SDK 示例把 `/v1` 隐含在 SDK 中；本仓库的 wrapper 只会在 `base_url` 后追加 `/chat/completions`，所以配置文件中使用完整的 `https://api.longcat.chat/openai/v1`。

连通性检查（每条命令都会产生一次真实 API 调用并计费）：

```sh
export MISTRAL_API_KEY='your-key'
PYTHONPATH=src python scripts/llm_smoke.py --model mistral

# GLM-5.3-Flash through BigModel directly
export ZAI_API_KEY='your-key'
PYTHONPATH=src python scripts/llm_smoke.py --model glm53_flash

export GEMINI_API_KEY='your-key'
PYTHONPATH=src python scripts/llm_smoke.py --model gemma_31b

export LONGCAT_API_KEY='your-key'
PYTHONPATH=src python scripts/llm_smoke.py --model longcat
```

本地 Qwen 需要先启动 Ollama 服务：

```sh
ollama pull qwen3.5:2b
ollama serve
PYTHONPATH=src python scripts/llm_smoke.py --model qwen_local
```

本地调用不需要 `QWEN_LOCAL_API_KEY`，也不会产生 API 费用。

生成 gold 候选时，`scripts/generate_candidates.py` 读取 100 个 `ProductItem`，四个模型分别生成一条广告，共 400 条候选记录。每个模型独立保存一份 `GenerationResult` JSONL 和一份调用日志，通过 `--output-dir` 指定保存目录：

```sh
PYTHONPATH=src python scripts/generate_candidates.py \
  --input data/eval/gold/v1/product_items.jsonl \
  --output-dir artifacts/eval/gold_generation/v1 \
  --models qwen_local gemma_31b glm53_flash longcat \
  --max-output-tokens 2048
```

```text
artifacts/eval/gold_generation/v1/
├── results_qwen_local.jsonl
├── results_gemma_31b.jsonl
├── results_glm53_flash.jsonl
├── results_longcat.jsonl
├── calls_qwen_local.jsonl
├── calls_gemma_31b.jsonl
├── calls_glm53_flash.jsonl
└── calls_longcat.jsonl
```

完整跑完时，每个 results 文件 100 条记录；每个 calls 文件记录所有尝试，发生重试时会超过 100 行。每次尝试都有独立 `trace_id`，`attempt` 是本轮尝试序号，`retry_of_trace_id` 关联上一次失败调用；结果中的 `trace_id` 指向最终一次尝试。失败也保留一条结果，需区分记录总数与成功候选数。`GenerationResult` schema 保持不变。

调用前会检查全部配置、所需 Key 和八个输出路径，避免调用完成后才发现文件已存在。每个商品调用完成后立即写入对应 results/calls 文件并刷新，可在运行中查看进度；完整生成入口已有文件不会覆盖。候选 ID 根据 `run_id`、`product_id` 和 `sample_index` 生成稳定 hash，每次调用日志使用独立 `trace_id`；候选 ID 不包含模型名。截断正文记为 `output_too_long`。

生成入口默认对超时、网络错误、408/429、5xx 和空正文最多额外重试 3 次，即每条最多 4 次调用；等待采用指数退避和随机抖动，上限 60 秒。通过 `--max-retries` 和 `--retry-delay` 调整，`--max-retries 0` 禁用自动重试。鉴权、余额、参数错误、拒答和输出截断直接记录为失败。脚本关闭 wrapper 内层重试，由这一层统一限制调用次数并逐次记录；超时重试可能重复执行服务端已经完成的工作，日志保留 `may_have_completed`。Gemma 默认请求超时已提高到 180 秒，也可以用 `--timeout-seconds` 覆盖。

已生成结果中的失败项使用单独入口补跑：

```sh
# 仅检查，既不联网也不创建文件。
PYTHONPATH=src python scripts/retry_failed_candidates.py --dry-run

# 默认检查 gold_generation/v1，仅补跑失败候选。
PYTHONPATH=src python scripts/retry_failed_candidates.py
```

默认从 `data/eval/gold/v1/product_items.jsonl` 和 `artifacts/eval/gold_generation/v1/` 读取，直接更新 `v1/` 中的四份 results 和四份 calls，不创建 `retried/` 子目录。成功记录原样保留，失败项以补跑结果替换，调用日志保留历史并追加新尝试。后续 rubric/pairwise 直接使用 `v1/` 中的完整 results；calls 中出现失败尝试不表示最终结果仍失败。首次更新前的八份 JSONL 备份在同级 `artifacts/eval/gold_generation/v1_original/`。

补跑从历史调用日志恢复模型、system prompt、输出预算、temperature 和 provider options，并核对商品输入，避免误用默认参数；仅把请求超时设为 180 秒、重试设为最多额外 3 次。`candidate_id`、`product_id`、`run_id`、`sample_index` 保留，重试设置及实际请求写入新调用日志。只需要有失败项的模型的 Key；其他模型无需启动本地服务或提供 Key。

每补完一条立即原子更新结果；再次运行同一命令会跳过已经成功的记录，并继续补跑仍失败的部分。如果中断前成功调用已落盘但结果尚未更新，也会从调用日志恢复。`retry_manifest.json` 记录输入和原始备份文件摘要，备份或输入变化时拒绝混用；同一输出目录同时只允许一个补跑进程。可用 `--models gemma_31b` 限制要补跑的模型，用 `--input`、`--source-dir` 指定另一批数据，或显式指定 `--output-dir` 另存结果。默认原地补跑，备份目录为源目录同级的 `<目录名>_original/`。全部成功退出码为 0，仍有失败时为 1。

本步保留全部 400 条候选供 rubric 人机一致率评估使用，不抽样、不调用 judge、不生成标签。pairwise 的每商品四选二抽样是下一步独立工作，复用同一批候选。

先用 mock 验证流程，不会联网：

```sh
PYTHONPATH=src python scripts/generate_candidates.py \
  --input data/eval/gold/v1/product_items.jsonl \
  --output-dir /tmp/adgenius-gold-mock-v1 \
  --models mock --limit 2
```

Qwen 示例配置使用北京 endpoint。API key、endpoint、可用模型必须属于相同地域：

| 地域 | OpenAI 兼容 base_url |
|---|---|
| 北京 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 新加坡 | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

北京/新加坡的上表地址是官方兼容 endpoint。新工作空间建议使用控制台显示的 workspace 专属域名；不要根据所在国家猜 endpoint。Qwen 配置支持用 `DASHSCOPE_BASE_URL` 覆盖默认地址。

### 使用阿里云中国区（当前 Qwen 默认配置）

1. 在[阿里云中国站注册](https://account.aliyun.com/register/register.htm)，按页面要求完成[实名认证](https://myaccount.console.aliyun.com/cert-info)。
2. 用主账号打开[百炼北京区控制台](https://bailian.console.aliyun.com/cn-beijing/model/market)，同意协议开通服务。开通本身免费。
3. 在[北京区 API Key 页面](https://bailian.console.aliyun.com/cn-beijing/model/settings/api-key)创建 API Key，选择默认业务空间即可。这里需要的是百炼 API Key，不是阿里云 AccessKey ID/Secret。
4. 复制弹窗中的 API Host。推荐业务空间专属地址形如 `https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。设置 `DASHSCOPE_BASE_URL` 为控制台给出的完整 Base URL（不带 `/chat/completions`）；不设置时使用官方北京兼容域名 `https://dashscope.aliyuncs.com/compatible-mode/v1`。API key 和地址都必须选北京地域。
5. 在[阿里云账户充值](https://billing-cost.console.aliyun.com/fortune/fund-management/recharge)页面充值。普通模型 API 按量计费，从阿里云账户余额扣款，无需购买服务器或包月套餐。可先查看[北京区免费额度](https://bailian.console.aliyun.com/cn-beijing/costing-balance/free-quota)；是否有赠送额度及有效期以账号页面为准，额度用完后可能转为按量计费。

从仓库根目录运行（Python 环境已安装依赖）：

```sh
export DASHSCOPE_API_KEY='你的北京区百炼 API Key'
export DASHSCOPE_BASE_URL='控制台复制的北京区 OpenAI 兼容 Base URL'
PYTHONPATH=src .venv/bin/python scripts/llm_smoke.py --model qwen
```

如果继续使用默认的北京 DashScope 地址，可省略 `DASHSCOPE_BASE_URL`。

实际请求会发送到国内北京接口。不要把 key 提交到 Git 或贴到运行日志中。

### 在 eval / train 代码里调用

```python
from adgenius.shared.llm import LLMClient, ModelConfig
from dotenv import load_dotenv

load_dotenv()  # Load the project .env at application startup.

config = ModelConfig.from_toml("configs/models.toml", "qwen")
with LLMClient(config) as client:
    response = client.generate(
        [
            {"role": "system", "content": "Return a JSON object with a supported boolean."},
            {"role": "user", "content": "Product: blue stapler. Copy: A blue stapler."},
        ],
        max_output_tokens=256,
        json_mode=True,  # OpenAI 兼容服务的 JSON object 请求，能力由具体模型决定
    )
    print(response.text)
    print(response.usage)
    record = response.to_dict()
```

`temperature=None` 表示使用服务端默认值。需要控制采样时显式设置；各模型对采样参数的支持/生效范围仍以官方文档为准。

响应包含 `text`、provider、请求/实际模型名、usage、finish_reason、request_id、endpoint、安全的 config 快照、耗时、attempts、实际请求体和原始响应。`to_dict()` 可直接保存为 JSONL；不包含鉴权请求头或 key 值。usage 不存在的字段为 `None`，Claude 输入 tokens 统一为普通输入 + cache read + cache write，原始 usage 仍完整保留。拒答或其他缺少文本的响应标记 `content_missing=True`；截断仍保留 finish_reason，由评测层决定有效性。

Google `generateContent` 将 `system` 转为 `systemInstruction`，`assistant` 转为 `model`，输出参数放入 `generationConfig`。带 `thought=true` 的部分不进入返回正文；原始响应仍保留。`output_tokens` 包含 `candidatesTokenCount` 和 `thoughtsTokenCount`，后者另记为 `reasoning_tokens`；`MAX_TOKENS` 保持 Google 原值并由候选生成脚本记为失败。Google 原生扩展可通过 `provider_options={"generationConfig": {...}}` 传入，例如 `thinkingConfig`；不得覆盖统一接口的 token 上限和 temperature。

Qwen / DeepSeek 的 thinking 参数放在配置的 `request_options`；示例已关闭 thinking。单次调用可通过 `provider_options` 覆盖它们。这里的 options 直接进入请求 JSON body，**不是 SDK 的 `extra_body`**。不可覆盖 model/messages/token 上限等核心字段。

Claude 的 `json_mode=True` 会明确报错，因为它与 OpenAI 协议的 JSON object 参数不同；需要时通过 `provider_options` 传 Claude 原生 `output_config`，并确认该模型支持。严格 JSON schema 也用各服务原生参数，当前不做跨 provider 的能力假设。

### 错误与重试

- 仅重试连接失败、连接超时、408、429 和暂时性 5xx，默认最多重试两次；退避最多等待 60 秒。
- 401/403、参数错误和无效响应结构不会自动重试。`LLMError` 提供 attempts、status_code、request_id 和脱敏错误响应 raw_response。
- 生成文本即使不是合法 JSON 也原样返回；解析、校验和记录无效率由评测层负责。
- 读写超时/中断无法确认服务端是否完成调用，因此直接报错并标记 `may_have_completed=True`，不自动重复提交。HTTP 服务端错误的重试仍可能重复计费；需要完全禁用重试时设置 `max_retries=0`。
- 不做自动跨模型切换，避免实验混入不同 judge。密钥只从环境变量读取，不能写进 TOML 或运行记录。

可显式保存一次响应（不会覆盖已有文件）：

```sh
python scripts/llm_smoke.py --model mock --output artifacts/eval/mock_response.json
```

### 本地检查

```sh
python -m unittest discover -s tests -v
```

测试使用 `httpx.MockTransport`，不访问真实 API。

### 官方文档

- [阿里 Model Studio API key](https://www.alibabacloud.com/help/en/model-studio/get-api-key)
- [百炼中国区首次 API 调用](https://help.aliyun.com/zh/model-studio/first-api-call-to-qwen)
- [百炼中国区 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)
- [百炼中国区地域与业务空间域名](https://help.aliyun.com/zh/model-studio/regions)
- [阿里 OpenAI 兼容接口](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)
- [阿里地域与 API 域名](https://www.alibabacloud.com/help/en/model-studio/regions)
- [DeepSeek API](https://api-docs.deepseek.com/)
- [DeepSeek thinking 配置](https://api-docs.deepseek.com/guides/thinking_mode)
- [Claude Messages API](https://platform.claude.com/docs/en/api/messages)
