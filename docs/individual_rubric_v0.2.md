# 单条广告文案人工标注 Rubric v0.2

本版只保留 **事实有依据（`factual_support`）** 一项。v0.1 的 **含具体商品信息（`concrete_product_info`）** 不再标注：生成的广告基本都会写到具体商品信息，这一项区分不出好坏。[v0.1](individual_rubric_v0.1.md) 保留原样，作为历史版本。

本版用于 `gold_rubric_v1` 的 130 条单条文案标注，记录中的 `protocol_version` 为 `rubric-v0.2`。本版于 **2026-11-24** 定稿。

## 与 v0.1 的区别

| | v0.1 | v0.2 |
|---|---|---|
| 判 False 的条件 | 与资料冲突，**或**资料没有足够依据 | 只有与资料**冲突**才判 False |
| 资料未提及、但也不冲突的说法 | False（“资料未提及”） | True |
| 资料内部自相矛盾 | 未规定 | 按下文“资料自相矛盾时”处理 |

改为“只看冲突”的原因：此前对 119 条候选的筛查中，确认的错误都是冲突类（数字不符、把一个部件或产品线的属性安到本产品上、范围说大），没有凭空捏造的具体规格；而“资料未提及”这一类判断分歧最大：同一批 99 条可疑候选，两个模型评委按 v0.1 分别判出 99 条和 14 条 False，难以稳定标注。

已知代价：资料没提、但也不冲突的具体承诺会判 True。例如资料没提保修，文案写“5 年保修”，本版判 True。

## 事实有依据（`factual_support`）

**问题：** 文案对商品说的任何内容，是否与商品资料冲突？

**商品资料** 指标注页面左侧显示的全部内容：商品名（标题）、类别、features、description 和 product details。各部分同等有效，标题也是资料。渲染规则与生成时发给模型的资料相同。

**冲突** 指两者不可能同时对这件商品成立。常见冲突：

- 同一事物给出不同的数字、单位、数量、尺寸、容量或产出页数；
- 把一个东西的属性安到另一个东西上：某个部件、型号变体、套装中的某一件、配件、包装或整个产品线的规格，被说成本商品（或另一部件）的规格；
- 资料排除了的材质、新旧状态、颜色或类型；
- 适用范围大于资料给出的限制（如资料给了尺寸范围，文案说“适合所有笔记本”）。

**不算冲突：**

- 漏写信息；
- 改写、同义词、缩写或概括；
- 资料既没说也没排除的说法，包括更具体但与资料一致的措辞（资料写 plastic，文案写 PVC）；
- 主观夸赞和适用场景（如 “great for home offices”）。

按普通购物者的理解读文案。简称或概括不读成“全部”，除非文案明确这样说；单复数、单位换算的舍入、语法不顺不构成冲突。

**资料自相矛盾时：**

1. 由标题决定：文案与标题一致就不算冲突；与标题冲突就算。标题明显是另一件商品（数据贴错）时，忽略标题。
2. 标题不能决定时，与资料任何一处冲突都算；但 product details 中明显不属于本商品的零散字段，以及明显的笔误，不算。

**判定：**

- `false`：至少有一处冲突。在理由中写出冲突的文案原文和资料原文。
- `true`：没有冲突。如果找不到能引用的、与之冲突的资料原文，就不是冲突。

### 示例

商品资料示例：订书机，金属机身，最多装订 20 页。

| 文案 | 事实有依据 | 理由 |
|---|---|---|
| “A metal stapler that handles up to 20 sheets.” | True | 与资料一致。 |
| “A metal stapler that handles up to 30 sheets.” | False | 30 页与 20 页冲突。 |
| “A metal stapler backed by a 5-year warranty.” | True | 资料没提保修，也没有冲突。 |
| “A stylish stapler for your desk.” | True | 只有主观修饰。 |

### 已裁定的边界情况

以下裁定来自本批 130 条及此前的筛查，后续标注和 judge prompt 按同样的思路处理。括号内是 `rubric_sample.jsonl` 中的 `item_id`。

| 情况 | 判定 |
|---|---|
| 标题写黑/青/品红/黄四色套装，details 有一行 “Color: orange”；文案写 “orange toner cartridges”（072） | False：与标题冲突 |
| 标题写 “Remanufactured”，一条 feature 写 “Brand New Product”；文案写 “brand-new”（028） | False：与标题冲突 |
| 同一商品，文案照标题写 “Remanufactured”（107） | True：与标题一致 |
| 标题写 “Textured”，产品线介绍提到 Pearl 和 Glossy 标签；文案写 “Pearl or glossy finish” | False：与标题冲突 |
| 标题写 “12 x 16”，details 的 Size 写 12x18；文案写 12x18（079） | False：与标题冲突 |
| 标题是一张音乐专辑，其余资料全部描述一支钢笔；文案描述这支钢笔（120） | True：标题明显贴错，忽略 |
| 描述、技术参数和尺寸写 29.9–42.9 英寸，一条 feature 写 29.9–44.9；文案写 29.9–42.9（087） | False：与正常的 feature 冲突 |
| feature 写 5oz，details 写 Item Weight 2 pounds；文案写 2 pounds（100） | False：与 feature 冲突 |
| 标题、feature、描述都写 “leather-look”，只有 details 写 “Cover Material: Leather”；文案写 leather-look（076） | True：details 零散字段不算 |
| 资料写 “leather-look cover”，文案写 “leather cover”（125、037 中的 simulated leather） | False：资料排除了真皮 |
| 标题、数量字段和“6 色各 100 个”都是 600 个，描述中一句写 1200；文案写 600（047、116） | True：明显笔误不算 |
| 重量或尺寸照正文写，只与 details 的 Item Weight / Product Dimensions 不同（003、008、018） | True：details 零散字段不算 |
| 资料写外壳是不锈钢、内部是防刮衬里；文案写 “stainless steel interior”（016） | False：把外壳材质安到了内部 |
| 资料列出若干 CX 型号；文案写 “for CX series printers”（111） | True：简称不读成“全部” |
| 资料写适合前端高度 0.39–0.71 英寸的 Tenkeyless 键盘；文案写 “Perfect for all Tenkeyless keyboards”（041） | False：大于资料给出的限制 |
| 资料把对齐功能归于 Active Feed，文案归于 Perfect Page；商品确实能对齐 | True：两者可以同时成立 |
| 一盒 100 套螺丝和螺柱；文案写 “each set includes 100 screws with 100 posts”（102） | False：数量错 |
| 资料写 paper and plastic，文案写 PVC（009） | True：更具体但一致 |
| 黑色 650 页、彩色 350 页；文案写 “650 pages per cartridge”（082） | False：把黑色的页数说成每个墨盒 |
| 资料写 3/4 inch 和 0.75 inch / 18 mm；文案写 3/4 inch（063） | True：换算舍入不算 |
| 资料写 1 块电池；文案写 “built-in batteries”（053） | True：单复数不算 |
| 资料写碳化钨笔珠、黄铜笔尖；文案写 “Tungsten carbide tips”（048） | True：常见说法 |
| 资料写 1 支笔加 8 个笔芯；文案写 “a set of reusable pens”（064） | True：笼统说法，不算数量声明 |

## 记录格式

每条标注是 [RubricResult v1](../metadata/schemas/rubric_result.schema.json) 的 human 记录：`judge_type="human"`、`annotation_set_id="gold_rubric_v1"`、`protocol_version="rubric-v0.2"`，`criteria` 只含 `factual_support`。False 的理由写成“文案原文 vs 资料原文”，True 可以不写理由。标注工具见[标注台文档](rubric_annotation_workbench.md)。

## 当前限制

- 当前只有一位标注者（`zhou`），没有第二人独立标注，不能写作两人一致的共识标签。
- 上面的边界裁定和“资料自相矛盾时”的规则是在这 130 条上定下来的。用这批标签评估 judge prompt 时，若 prompt 直接写入这些例子，一致率会偏高；应同时报告只用核心定义的 prompt 的结果，或另取一批未参与定规则的文案验证。
- 016 是在看过 judge 结果后改为 False 的（2026-11-24），属于事后修改，报告 judge 一致率时需注明。
