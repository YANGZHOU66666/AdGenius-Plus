# AdGenius+ Repository Layout

The project studies product-grounded advertising generation, human-calibrated evaluation, and later synthetic-data selection for small-model training.

```text
configs/       Model, evaluation and future training configuration
metadata/      Shared JSON Schema definitions
prompts/       Current evaluator prompts
src/adgenius/  Shared API/product rendering, evaluation and future training code
scripts/       CLI generation, data preparation, validation and evaluation
tools/         Local human-annotation workbench
data/          Current reusable datasets and frozen split information
results/       Formal per-sample outcomes, metrics and reports shared in Git
artifacts/     Local runtime outputs, call journals, caches and weights
docs/          Current methods and usage
tests/         Offline tests
deprecated/    Retired datasets, original labels/reviews, pilots and old docs
```

## Current data boundary

`data/` holds inputs used by subsequent experiments. Once a runtime output is confirmed as a frozen dataset, publish the small standard records here with provenance. The current gold dataset is `data/eval/gold/`: 100 products, four generators with 100 candidates each, 100 sampled pairs, and 100 current human preferences. Those core files are committed so collaborators can run pairwise or rubric work without generating candidates again.

`results/` holds formal experiment outcomes: four completed judge runs and the current comparison. New pairwise evaluation writes standard outcomes to results/pairwise/<run_id>/ and its full HTTP journal to artifacts/pairwise/<run_id>/. `artifacts/` is ignored as a whole and holds generation runtime outputs, journals, caches, checkpoints and intermediate files. Publishing a generation as a reusable dataset creates a frozen snapshot under data/, without changing the original runtime records.

`deprecated/` contains retired v1 candidates/pairs, prompt and judge pilots, pre-retry backups, original human labels, review records and old docs. Original human labels and review history are local only and ignored by Git; only the final 100 human labels are shared under data/. Its migration index records original and destination paths/hashes. Archived records retain their captured provenance. Current call logs remain under artifacts. See [deprecated/README.md](../deprecated/README.md) for the Git boundary.

The large Office Products source gzip currently stays outside the repo at `../data/raw/...`. Frozen source splits remain under `data/splits/v1/` and original gold/test product records under `data/processed/v1/`. The 709,403-record train-pool index is cached under artifacts/cache/office_products/train_pool.jsonl, while the shared split manifest stays in data/splits/v1/. It is a candidate pool rather than a required training size. Gold and final test are excluded from training; validation is selected from the training pool.

## Workflows

Calibration: frozen gold products → four candidate ads per product → random fixed pairs / individual ads → human labels → judge agreement → evaluator selection and freezing.

Training: train pool → teacher candidates → quality judgments → matched random/filtered training data → SFT/LoRA → validation checkpoint selection → fixed test comparison. This training flow is planned; the current repo has no trained model or materialized SFT dataset.

Shared API and product rendering live in src/adgenius/shared. Pairwise evaluation lives in src/adgenius/eval. CLI scripts orchestrate those components rather than duplicating provider implementations. Schema definitions stay in metadata/schemas. Dataset manifests live beside the relevant data.

Use independent identifiers for independent concepts: source split, dataset, annotation revision, judge protocol, and run. The active gold directory has no v1/v2 layer; generator run IDs and the original gold_pairwise_v2 annotation-set ID remain unchanged for identity/provenance.

## Documentation

- [数据说明与生产链路](data_pipeline.md): unified dataset inventory, production steps, dependencies, ID joins and rebuild limitations.
- [Data index](../data/README.md): formal dataset locations, contents, hashes and sharing.
- [Source split method](data_split.md): raw source and frozen partition.
- [Gold candidate generation](gold_candidate_generation.md): final collection method and pair sampling.
- [Annotation workbench](pairwise_annotation_workbench.md): UI and persistence behavior.
- [Pairwise judge](pairwise_llm_judge.md): protocol, CLI, retries and metric definitions.
- [Current comparison](../results/pairwise/comparison/report.md): four-judge calibration snapshot.

## Input/output contract

- Candidate generation reads data/eval/gold/product_items.jsonl and writes runtime results/calls to artifacts/generation/<run>/ (default batch gold). Repair uses that same runtime directory.
- Pair construction reads the four formal GenerationResult files in data/eval/gold; its explicit --output selects a new dataset file. It refuses to overwrite frozen pairs.
- Annotation reads and updates the current human file in data/eval/gold.
- Pairwise evaluation reads the formal pairs/labels, writes standard results to results/pairwise/<run_id>, and writes calls to artifacts/pairwise/<run_id>.
- Data splitting writes small manifests/products to data/ and the full train pool to artifacts/cache/office_products/.
- Smoke responses, when requested with --output, belong under artifacts/smoke/.
- The offline data validator reads data/ and results/; runtime logs and large source downloads are optional.

## Git boundary

Shared in Git: the formal datasets in data/, per-item results and metrics in results/, configs, code and docs. `artifacts/` is ignored as a whole and holds full call journals, the train-pool cache, checkpoints and run intermediates. `.env` and large raw source downloads also stay local. Old pilots, retired data and the original human labels with their review history live in [deprecated/](../deprecated/README.md); its migration index records file hashes before and after each move.

## Calling the shared client from code

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

The client speaks the OpenAI-compatible protocol, Claude Messages, Google generateContent and local Ollama. `json_mode` and native `provider_options` can be passed; the latter go straight into the request JSON body but cannot override the model, messages or token limit. A model's `request_options` in models.toml carry platform settings such as thinking. Retries are bounded and errors stay structured; the client never repairs model JSON or hides a failure. The Alibaba Cloud endpoint defaults to Beijing; set `DASHSCOPE_BASE_URL` to use another workspace's compatible API root. Record formats are in [metadata/README.md](../metadata/README.md).
