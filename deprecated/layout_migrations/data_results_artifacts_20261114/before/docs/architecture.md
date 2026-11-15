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
artifacts/     Experiment outputs and call journals
docs/          Current methods and usage
tests/         Offline tests
deprecated/    Retired datasets, original labels/reviews, pilots and old docs
```

## Current data boundary

`data/` holds inputs used by subsequent experiments. Once a runtime output is confirmed as a frozen dataset, publish the small standard records here with provenance. The current gold dataset is `data/eval/gold/`: 100 products, four generators with 100 candidates each, 100 sampled pairs, and 100 current human preferences. Those core files are committed so collaborators can run pairwise or rubric work without generating candidates again.

`artifacts/` holds run outputs. Four complete judge runs' standard results, decisions and manifests, plus the latest reviewed-label comparison, are explicitly shared. Full HTTP call journals, unselected new runs, checkpoints and bulk training outputs remain local. The original generation runtime snapshots/logs remain alongside one another under artifacts; they are not a second formal gold dataset.

`deprecated/` contains retired v1 candidates/pairs, prompt and judge pilots, pre-retry backups, original human labels, review records and old docs. Its migration index records original and destination paths/hashes. Archived records retain their captured provenance. Current call logs remain under artifacts. See [deprecated/README.md](../deprecated/README.md) for the Git boundary.

The large Office Products source gzip currently stays outside the repo at `../data/raw/...`. Frozen source splits remain under `data/splits/v1/` and original gold/test product records under `data/processed/v1/`. Train-pool data is a candidate pool rather than a required training size. Gold and final test are excluded from training; validation is selected from the training pool.

## Workflows

Calibration: frozen gold products → four candidate ads per product → random fixed pairs / individual ads → human labels → judge agreement → evaluator selection and freezing.

Training: train pool → teacher candidates → quality judgments → matched random/filtered training data → SFT/LoRA → validation checkpoint selection → fixed test comparison. This training flow is planned; the current repo has no trained model or materialized SFT dataset.

Shared API and product rendering live in src/adgenius/shared. Pairwise evaluation lives in src/adgenius/eval. CLI scripts orchestrate those components rather than duplicating provider implementations. Schema definitions stay in metadata/schemas. Dataset manifests live beside the relevant data.

Use independent identifiers for independent concepts: source split, dataset, annotation revision, judge protocol, and run. The active gold directory has no v1/v2 layer; generator run IDs and the original gold_pairwise_v2 annotation-set ID remain unchanged for identity/provenance.

## Documentation

- [Data index](../data/README.md): formal dataset locations, contents, hashes and sharing.
- [Source split method](data_split.md): raw source and frozen partition.
- [Gold candidate generation](gold_candidate_generation.md): final collection method and pair sampling.
- [Annotation workbench](pairwise_annotation_workbench.md): UI and persistence behavior.
- [Pairwise judge](pairwise_llm_judge.md): protocol, CLI, retries and metric definitions.
- [Current comparison](../artifacts/eval/pairwise/comparison/report.md): four-judge calibration snapshot.
