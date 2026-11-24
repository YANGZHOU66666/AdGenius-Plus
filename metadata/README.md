# 公共数据约定

`metadata/` 保存评测与训练共用的数据结构定义。

| Schema | 用途 |
| --- | --- |
| [ProductItem](schemas/product_item.schema.json) | 完整商品资料，用于广告生成、人工界面展示及后续 rubric 事实核查。 |
| [GenerationResult](schemas/generation_result.schema.json) | 单条候选广告的生成结果。 |
| [CandidatePair](schemas/candidate_pair.schema.json) | 两条成功候选的冻结比较对。 |
| [PairwiseResult](schemas/pairwise_result.schema.json) | 模型 AB/BA 两次判断，或一位人工的一次独立判断。 |
| [RubricResult](schemas/rubric_result.schema.json) | 模型或一位人工对单条候选的逐项评分。 |

所有 schema 使用 JSON Schema Draft 2020-12。评估结果中的 `protocol_version` 指向评判规则版本，与数据结构版本分开管理。

本文中的 JSON 均为独立结构示例，不是冻结数据或真实标注结果。

## ProductItem v1

JSON Schema：[schemas/product_item.schema.json](schemas/product_item.schema.json)，使用 JSON Schema Draft 2020-12。

数据采用 JSONL 格式，一行一个商品对象，恰好包含以下三个一级字段：

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `product_id` | string | 是 | 项目内稳定且唯一的商品 ID，供生成结果与评估结果关联商品。 |
| `source_id` | string | 是 | 原始数据集的商品 ID，当前对应 Amazon 的 `parent_asin`。 |
| `product_info` | object | 是 | 除源 ID 外的全部原始商品属性。 |

两个 ID 都不能是空字符串或仅包含空白的字符串。不规定 `product_id` 的生成算法；正式构建数据集时必须固定 ID 映射，不能因为重排、抽样、重新划分或重跑而重新编号。同一商品在 gold、test 或训练数据等不同用途下沿用同一项目 ID。

### Amazon Office Products 映射

- `source_id` 等于原始记录的 `parent_asin`，不转换值。
- `product_info` 包含原始记录除 `parent_asin` 外的所有字段；`parent_asin` 不在内部重复保存。
- 保留原字段名、JSON 类型和值，不清洗、改写、筛选或补齐商品属性。缺失字段继续缺失，空值与空数组原样保留。
- `price` 可能为数字、字符串或 `null`，不强制转为数字。`details` 等嵌套内容以及 `subtitle`、`author` 等额外字段也原样保留。
- 通用 schema 只要求 `product_info` 是对象，不预先限制其内部字段或类型。上述源字段映射及完整性由数据集构建时校验。

这里的属性保留是 JSON 值层面的保留，不要求序列化后的空白、字段顺序或文件字节与原始 JSONL 相同。原始数据文件继续保留。

以下是虚构示例，仅展示结构：

```json
{
  "product_id": "prod_000001",
  "source_id": "DEMO-ASIN-001",
  "product_info": {
    "main_category": "Office Products",
    "title": "Metal Stapler",
    "average_rating": 4.6,
    "rating_number": 120,
    "features": [
      "Metal body",
      "Staples up to 20 sheets"
    ],
    "description": [],
    "price": 12.99,
    "images": [],
    "videos": [],
    "store": "Example Store",
    "categories": ["Office Products", "Staplers"],
    "details": {
      "Material": "Metal"
    },
    "bought_together": null
  }
}
```

### 数据集级约束

逐行 schema 校验之外，构建数据集时还需检查：

- `product_id` 在数据集中唯一；多个数据集引用同一商品时 ID 映射一致。
- 当前单一 Amazon 来源中的 `source_id` 唯一，并与原始商品准确对应。未来多来源场景的 `source_id` 仅在其来源内唯一，需由来源映射区分。
- `product_info` 与原始记录移除源 ID 后的对象一致，没有遗漏字段或修改值。

数据集的 `manifest.json` 记录 schema 版本与 hash、数据来源及源文件 hash、源 ID 字段名和 ID 映射规则，商品记录本身不重复添加这些字段。

### 与模型输入的关系

完整保存商品资料与选择模型输入是两个步骤，各环节的输入约定如下：

- **广告生成**：按生成规则选择商品字段并渲染为文本，实际请求保存在调用日志中；不会直接发送完整 ProductItem JSON。
- **人工标注界面**：展示商品资料供标注者参考，展示字段与生成时使用相同的渲染函数。
- **Pairwise 模型评委**：当前 `pairwise-v0.2` 协议只接收两条广告文本，不接收原始商品资料、生成模型身份或人工标签；判断信息流场景中的广告传播效果，事实核查由 rubric 负责。详见 [Pairwise Judge 协议与运行说明](../docs/pairwise_llm_judge.md)。
- **Rubric 评估**：事实核查应以候选生成所依据的商品资料为参照；具体输入字段和渲染方式由后续 rubric 协议规定。当前仓库仅定义 rubric schema，尚未实现运行链路。

本次仅定义数据格式，不转换或覆盖现有 `data/processed/v1/`、`data/splits/v1/` 或原始数据文件。

## GenerationResult v1.1

JSON Schema：[schemas/generation_result.schema.json](schemas/generation_result.schema.json)。数据采用 JSONL 格式，一行代表一个商品候选文案的一次标准化生成结果。

```json
{
  "candidate_id": "cand_000001",
  "product_id": "B000123",
  "run_id": "gen_qwen_v1_20261105",
  "sample_index": 0,
  "generation_result": {
    "status": "success",
    "text": "A reliable metal stapler for everyday office work.",
    "error": null
  },
  "trace_id": "call_000001"
}
```

字段约定：

| 字段 | 含义 |
| --- | --- |
| `candidate_id` | 单条候选文案的稳定唯一 ID，供候选对和 rubric 结果引用。 |
| `product_id` | 关联 `ProductItem` 的项目内商品 ID。 |
| `run_id` | 本次生成运行的 ID；模型、prompt、参数、数据版本和代码版本等放在对应运行 manifest 中。 |
| `sample_index` | 同一商品、同一运行中的采样序号，从 `0` 开始。 |
| `generation_result.status` | `success`、`truncated` 或 `failed`，含义见下方状态表。 |
| `generation_result.text` | `success` 时为完整文案；`truncated` 时为截断后的原文；`failed` 时为 `null`。 |
| `generation_result.error` | `success` 时为 `null`；`truncated` 时固定为 `output_too_long`；`failed` 时为固定错误代码字符串。 |
| `trace_id` | 关联调用日志中一次具体 API 调用的 ID；没有调用日志时为 `null`。 |

失败示例：

```json
{
  "candidate_id": "cand_000002",
  "product_id": "B000123",
  "run_id": "gen_qwen_v1_20261105",
  "sample_index": 1,
  "generation_result": {
    "status": "failed",
    "text": null,
    "error": "timeout"
  },
  "trace_id": "call_000002"
}
```

允许的错误代码为：`timeout`、`network_error`、`rate_limit`、`auth_error`、`provider_error`、`empty_output`、`invalid_output` 和 `output_too_long`。所有 API 重试和原始响应记录在运行产物的调用日志中；`GenerationResult` 只保留最终标准化结果。

### 生成状态

| `status` | `text` | `error` | 何时使用 |
| --- | --- | --- | --- |
| `success` | 非空完整文案 | `null` | 模型正常结束。 |
| `truncated` | 非空的截断原文 | `output_too_long` | 模型因输出长度上限停止，但已写出文案。 |
| `failed` | `null` | 错误代码 | 没有得到可用文案：请求失败、内容被拒、空输出，或长度上限耗尽时正文为空（例如思考用完了预算）。 |

截断示例：

```json
{
  "candidate_id": "cand_000003",
  "product_id": "B000123",
  "run_id": "gen_student_v1_20261123",
  "sample_index": 0,
  "generation_result": {
    "status": "truncated",
    "text": "A reliable metal stapler for everyday office work, with a sturdy base, a smooth",
    "error": "output_too_long"
  },
  "trace_id": null
}
```

`truncated` 只是如实记录模型在既定长度上限下的输出，不代表文案可用或不可用。各用途按自己的规则处理：

- **给模型排名的评测**（如 test 集上比较 base、SFT、OPD）：分母固定为全部商品，`success` 和 `truncated` 都照常评审，截断率单独报告；`failed` 也不剔除，按事先写定的评测规则计分（例如判负）。每个系统对每个商品只生成一次，不按输出内容重新生成；只有基础设施故障导致整批输出缺失时才整批重跑。评委调用的失败可以重试。
- **评委校准用的 gold 比较对**和**用作训练目标的老师数据**：只使用 `success`。`truncated` 不进入这些数据集。
- `scripts/retry_failed_candidates.py` 只修复 `failed`，不重试 `truncated`。

## CandidatePair v1

JSON Schema：[schemas/candidate_pair.schema.json](schemas/candidate_pair.schema.json)。它是 pairwise 专用的候选文案对快照，一行代表一个待比较的文案对。

```json
{
  "pair_id": "pair_000001",
  "product_id": "B000123",
  "candidates": [
    {
      "candidate_id": "cand_000001",
      "text": "A reliable metal stapler for everyday office work."
    },
    {
      "candidate_id": "cand_000002",
      "text": "A durable stapler designed for busy offices."
    }
  ]
}
```

约定：

- `candidates` 必须恰好包含两条候选，且两条 `candidate_id` 不同。
- 两条候选必须对应同一个 `product_id`，并且来源 `GenerationResult` 的状态都必须是 `success`。
- `text` 是从 `GenerationResult` 复制的冻结快照；构建 pair 时校验其与来源记录一致。来源生成记录仍是候选内容的事实来源。
- `CandidatePair` 不重复保存 `run_id`、`status`、`error` 或 `trace_id`。模型来源和运行信息通过候选 ID 及运行 manifest 追溯，并且不放入评委可见内容。
- 数组顺序只用于稳定存储，不代表 A/B 展示位置，也不代表文案优劣。实际评测中的 A/B 映射放在 `PairwiseResult` 中，以支持随机交换展示顺序。
- `pair_id` 必须唯一；同一组候选的正序和逆序视为同一个 pair，数据集级构建时去重。

逐行 schema 不能检查两个候选的 `product_id` 是否一致、`candidate_id` 是否互异、候选是否确实来自成功的 `GenerationResult`，这些属于构建 CandidatePair 时的数据集级校验。生成失败的候选不进入有效 pair；排除数量和原因记录在构建 manifest 或运行报告中。

## PairwiseResult v1

JSON Schema：[schemas/pairwise_result.schema.json](schemas/pairwise_result.schema.json)。同一个 schema 根据最外层 `judge_type` 区分 `model` 和 `human`；两个分支各有必填字段，不混用模型运行字段和人工标注字段。

### 模型版本

一条记录代表同一个模型评委对一个 pair 的 AB、BA 两次独立判断：

```json
{
  "result_id": "pw_000001",
  "judge_type": "model",
  "pair_id": "pair_000001",
  "judge_run_id": "pairwise_llm_run_001",
  "protocol_version": "pairwise-v0.1",
  "candidate_mapping": {
    "A": "cand_000001",
    "B": "cand_000002"
  },
  "pairwise_result": [
    {
      "prompt_order": "AB",
      "status": "success",
      "choice": "A",
      "winner_candidate_id": "cand_000001",
      "reason": "A communicates the 20-sheet capacity more clearly.",
      "error": null,
      "trace_id": "call_pw_001_ab"
    },
    {
      "prompt_order": "BA",
      "status": "success",
      "choice": "A",
      "winner_candidate_id": "cand_000001",
      "reason": "A provides more specific product information.",
      "error": null,
      "trace_id": "call_pw_001_ba"
    }
  ]
}
```

- `candidate_mapping` 固定 A/B 对应的候选身份；`prompt_order` 只改变提示词中的展示先后。BA 时选择 A 仍然指 `cand_000001`。
- `pairwise_result` 必须恰好两项，一个 `AB`、一个 `BA`，schema 不强制数组本身的排列顺序。两次调用相互独立，不把第一次判断作为第二次输入。
- 每次调用都有自己的 `trace_id`；没有日志时允许为 `null`。某次失败仍保留对应 item，不能删掉或当成平局。
- `judge_run_id` 关联模型、实际 prompt、参数及运行 provenance。结果不重复添加 `judge_id`，一个评测运行固定使用一份评委配置。

### 人工版本

一条记录代表一位人工标注者对同一个 pair 的一次独立判断：

```json
{
  "annotation_id": "human_pw_000001",
  "judge_type": "human",
  "annotation_set_id": "gold_annotations_v1",
  "pair_id": "pair_000001",
  "annotator_id": "annotator_01",
  "protocol_version": "pairwise-v0.1",
  "presentation": {
    "A": "cand_000001",
    "B": "cand_000002"
  },
  "pairwise_result": {
    "status": "success",
    "choice": "A",
    "winner_candidate_id": "cand_000001",
    "reason": "A communicates the product capacity more directly.",
    "error": null
  }
}
```

- `presentation` 记录这位标注者实际看到的 A/B 映射，人工 `pairwise_result` 是一个对象，不要求 AB/BA 双次判断。
- `annotator_id` 标识独立标注者，`annotation_set_id` 标识所属标注批次；人工结果没有 API `trace_id`。
- 两位标注者分别保存记录。独立标签不被讨论后的共识覆盖。

### 判定与状态约定

模型的每个判断 item 和人工的单个判断对象使用同一套状态规则：

| 状态 | `choice` | `winner_candidate_id` | `error` |
| --- | --- | --- | --- |
| `success`，选出赢家 | `A` 或 `B` | 相应候选 ID | `null` |
| `success`，平局 | `tie` | `null` | `null` |
| `failed` | `null` | `null` | 非空错误代码字符串 |

`reason` 必须提供，值可以是字符串或 `null`。失败代码例如 `timeout`、`network_error`、`rate_limit`、`auth_error`、`provider_error`、`invalid_response`、`unjudgeable`。错误保持字符串，不嵌套对象。schema 允许其他非空错误代码，具体运行应保持代码含义一致。

`winner_candidate_id` 由程序根据 `choice` 和候选映射计算。schema 校验结构、状态和 AB/BA 数量；构建结果时另行校验 A/B 是不同候选、两者恰好属于引用的 `CandidatePair`，且赢家与选择一致。人工和模型的 A/B 映射可能不同，计算一致率时按候选 ID 对齐。

两次模型判断是否一致、如何合并，属于后续派生统计；不能把两次冲突直接记为平局，也不能把两次调用当成两个独立商品。

## RubricResult v1

JSON Schema：[schemas/rubric_result.schema.json](schemas/rubric_result.schema.json)。同一个 schema 根据 `judge_type` 区分模型与人工；两者共用 `rubric_result` 的逐项评分结构。

以下指标名称和版本仅作结构示例，尤其 `specific_product_info` 尚需与 rubric 负责人确认；schema 不预先锁定第二个指标的名称。

### 模型版本

```json
{
  "result_id": "rubric_000001",
  "judge_type": "model",
  "candidate_id": "cand_000001",
  "product_id": "B000123",
  "judge_run_id": "rubric_llm_run_001",
  "protocol_version": "rubric-v0.1",
  "rubric_result": {
    "status": "success",
    "criteria": [
      {
        "criterion_id": "factual_support",
        "value": true,
        "reason": "The metal body and 20-sheet capacity are supported by the input."
      },
      {
        "criterion_id": "specific_product_info",
        "value": true,
        "reason": "The copy states the material and stapling capacity."
      }
    ],
    "error": null
  },
  "trace_id": "call_rubric_001"
}
```

### 人工版本

```json
{
  "annotation_id": "human_rubric_000001",
  "judge_type": "human",
  "annotation_set_id": "gold_annotations_v1",
  "candidate_id": "cand_000001",
  "product_id": "B000123",
  "annotator_id": "annotator_01",
  "protocol_version": "rubric-v0.1",
  "rubric_result": {
    "status": "success",
    "criteria": [
      {
        "criterion_id": "factual_support",
        "value": true,
        "reason": "Both factual claims are supported by the product information."
      },
      {
        "criterion_id": "specific_product_info",
        "value": true,
        "reason": "Mentions the metal body and 20-sheet capacity."
      }
    ],
    "error": null
  }
}
```

### 判定与状态约定

- 成功：`status` 为 `success`，`criteria` 至少一项，`error` 为 `null`。每项必须有非空 `criterion_id`、布尔 `value`，以及字符串或 `null` 类型的 `reason`。
- 某项 `value` 为 `false` 仍是有效评分，不代表调用失败。
- 失败：`status` 为 `failed`，`criteria` 必须为空数组，`error` 是非空错误代码字符串。人工无法判断时使用 `unjudgeable`，不强行填写布尔评分。
- 模型结果包含必填但可为 `null` 的 `trace_id`；人工结果不包含 API 日志字段。
- 同一结果内的 `criterion_id` 不得重复，且必须完整对应 `protocol_version` 规定的指标集合。这项校验需读取协议配置，不能仅靠通用逐行 schema 完成。
- `candidate_id` 必须引用成功候选；`product_id` 必须与该候选的来源商品一致。人工和模型比较一致率时使用相同候选、相同商品输入和相同协议版本。

## 数据用途与结果来源

`ProductItem` 的三个一级字段保持不变。商品属于哪个 split、用于 gold 校准或最终 test 评测，记录在数据集 manifest 中；`judge_type` 表示评判来源，不表示商品用途。gold 商品本身不等于人工标签，标签通过 `pair_id` 或 `candidate_id` 引用固定评测对象。

本次仅增加格式定义和示例，不生成候选、人工标签或模型评估结果，也不变更已有冻结数据。
