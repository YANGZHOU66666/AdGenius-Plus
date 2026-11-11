# AdGenius+ 仓库设计

本设计依据 [`proposal/main.tex`](../../proposal/main.tex)，围绕两条研究线组织：先校准离线评测协议，再研究质量筛选的合成数据能否改善小模型微调，并比较文案质量与计算成本。

## 1. 目录

```text
AdGenius-Plus/
├── configs/
│   ├── common.yaml               # 数据版本、任务约定、共享默认值
│   ├── eval/                     # individual、pairwise、统计配置
│   ├── train/                    # 合成、筛选、SFT / LoRA 配置
│   └── experiments/              # baseline 和消融的完整运行配置
├── prompts/
│   ├── common/                   # 任务和商品输入模板
│   ├── eval/                     # individual / pairwise judge prompt
│   └── train/                    # teacher / student generation prompt
├── src/adgenius/
│   ├── shared/                   # schemas、data、llm、inference、runs
│   ├── eval/                     # annotation、individual、pairwise、metrics
│   └── train/                    # synthesize、select、dataset、sft、checkpoint
├── data/
│   ├── raw/                      # 忽略：原始数据或外部数据挂载点
│   ├── processed/                # 忽略：清洗、去重、标准化后的商品表
│   ├── splits/                   # Git：gold/test/val/train_pool 冻结 manifest
│   ├── eval/                     # Git：人工标签、共识标签、协议版本
│   └── train/                    # 忽略：teacher 候选、评分、物化训练集
├── artifacts/
│   ├── eval/<run_id>/            # 忽略：候选、judge 响应、评测结果
│   ├── train/<run_id>/           # 忽略：合成运行、训练日志、比较结果
│   ├── checkpoints/              # 忽略：base model、LoRA adapter、tokenizer
│   └── reports/                  # 报告图表和可提交素材
├── notebooks/                    # 临时探索；正式流程进入 src
├── tools/                        # 本地人工标注页面等辅助工具
├── docs/
└── tests/
```

`eval/` 和 `train/` 是工作流边界；`shared/` 放两边都要用的能力。比如文案生成既会用于 eval baseline，也会用于 train 的 teacher 合成，所以放在 `shared/inference.py` 或 `shared/llm/`，而不是复制两套实现。

## 2. 数据和产物怎么区分

最重要的规则是：**`data/` 是可作为后续输入复用的数据集，`artifacts/` 是某一次运行的输出。** 运行输出只有经过确认并写入 provenance，才晋升为新的 dataset。

```text
data/
├── splits/
│   ├── v1_gold.jsonl
│   ├── v1_test.jsonl
│   ├── v1_val.jsonl
│   └── v1_train_pool.jsonl
├── eval/
│   ├── annotations_zhou.jsonl
│   ├── annotations_chengkai.jsonl
│   ├── consensus_v1.jsonl
│   └── protocol_v1.yaml
└── train/
    └── <dataset_id>/
        ├── manifest.jsonl
        ├── examples.jsonl
        └── provenance.json

artifacts/
├── eval/<run_id>/
│   ├── config.yaml
│   ├── candidates.jsonl
│   ├── requests.jsonl
│   ├── raw_responses.jsonl
│   ├── results.jsonl
│   └── metrics.json
└── train/<run_id>/
    ├── config.yaml
    ├── teacher_candidates.jsonl
    ├── selection.jsonl
    ├── train.log
    ├── metrics.json
    └── provenance.json
```

具体含义：

- `data/splits/` 只保存商品 ID、`group_id`、split、seed、源数据版本和内容 hash。100 个 `gold`、1,000 个 `test`、`val`、`train_pool` 先按商品及相关变体隔离，再生成文案。
- `data/eval/` 保存两人的独立标签、共识标签、标注说明和冻结的 rubric/pairwise 协议。这些文件小且需要共同审阅，可以进 Git。
- `data/train/<dataset_id>/` 保存确认后可复用的 SFT 数据。每个样本要带 `product_id`、`candidate_id`、筛选规则版本和来源 run。
- `artifacts/eval/<run_id>/` 保存一次评测的完整输入、原始 API 响应、解析结果、失败状态、usage 和统计指标。
- `artifacts/train/<run_id>/` 保存 teacher 候选、质量分数、保留/拒绝原因、训练日志和比较结果。确认后才把选中的数据物化到 `data/train/`。
- `artifacts/checkpoints/` 只放模型权重和 tokenizer；checkpoint 的 base model、dataset、代码版本和训练配置写在对应的 `provenance.json`。

原始商品、清洗后的完整表、teacher 候选、训练样本、API 原始响应和 checkpoint 都应被 `.gitignore` 忽略。用 `DATA_ROOT` 和 `ARTIFACT_ROOT` 指定实际存储位置；现有 Amazon 原始 gzip 文件继续放在仓库外的 `Graduate Project/data/raw/...`。

## 3. 两条流程

评测校准：

```text
清洗 / 去重 / 商品家族分组
  → gold 商品 + 固定候选文案
  → 两人独立标注同一批 100 条
  → human-human agreement 与共识标签
  → 比较 rubric / judge 与人工标签
  → 冻结评测协议
```

训练与最终比较：

```text
train_pool → teacher 候选 → 评分 → filtered / random 数据集
                                      → SFT / LoRA → checkpoint
val        → checkpoint 生成 → 选 checkpoint
test       → template / base student / teacher / tuned student
           → 同一个冻结评测入口 → 最终统计
```

`train` 不读取 `gold` 和 `test`，也不改变冻结后的 rubric。`test` 不用于选阈值、prompt、训练超参或 checkpoint。

## 4. 产物命名和复现

每次运行使用唯一的 `run_id`，例如 `eval_qwen_individual_20261027_001` 或 `train_filtered_lora_20261027_001`。每个 run 至少保存：

```text
config.yaml       # 解析后的完整配置
provenance.json   # 代码版本、环境、seed、数据/prompt/rubric hash
results.jsonl     # 逐对象标准化结果
metrics.json      # 汇总指标、有效数、失败率、置信区间
```

评测还保存 `raw_responses.jsonl` 和 `usage`；训练还保存 base model/tokenizer、实际 tokens、steps、训练时间、峰值显存和 checkpoint 路径。模型返回无效 JSON 要保留为失败状态，不能在包装层自动修复后抹掉失败率。

## 5. 最小实现范围

先实现 `shared/schemas.py`、`shared/llm/`、数据读取和 manifest、`eval/individual.py`、`eval/pairwise.py`，用 mock 跑通完整 JSONL 产物。之后接真实 judge，冻结 gold/test。训练阶段再实现 `train/synthesize.py`、`train/select.py`、`train/dataset.py` 和一条 SFT/LoRA 路径；DPO 和 on-policy distillation 作为扩展，不提前搭完整框架。
