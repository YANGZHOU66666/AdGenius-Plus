# AdGenius+

Office Products 广告文案生成、离线评测与小模型微调实验。仓库结构见 [docs/architecture.md](docs/architecture.md)。

## LLM API 包装

Python 3.11+，使用 `httpx` 直接适配 OpenAI 兼容协议和 Claude Messages API，无需安装两套 SDK。当前支持同步、非流式、纯文本消息。

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

也支持 `--model deepseek`（`DEEPSEEK_API_KEY`）或 `--model claude`（`ANTHROPIC_API_KEY`）。smoke 脚本显式加载根目录 `.env`；后续 eval/train 入口也应在程序启动时显式调用 `dotenv.load_dotenv()`。`.env` 已加入 `.gitignore`。

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
