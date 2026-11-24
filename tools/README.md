# Human Annotation Tools

The pairwise workbench uses `data/eval/gold/` as its default input and label directory. It loads 100 frozen product/pair records and the 100 current labels from disk. Model identity and judge results are hidden. A/B presentation is shuffled on entry while the saved winner follows its candidate ID.

Run from the repository root:

```sh
PYTHONPATH=src .venv/bin/python tools/pairwise_annotation.py --port 8765
```

Open http://127.0.0.1:8765/?annotator_id=zhou. The interface has forced A/B choice, a colored directory, previous/next navigation, auto-advance to an unlabeled pair, optional reasons, conflict protection and disk synchronization. Revisited pairs show the saved conclusion in the newly shuffled presentation.

Defaults use annotation set `gold_pairwise_v2` and protocol `pairwise-v0.1`, preserving existing record identities despite the simplified folder path. To label independently as another team member, set a different annotator ID; it writes a separate pairwise_annotations_<id>.jsonl rather than modifying Zhou's file.

```sh
PYTHONPATH=src .venv/bin/python tools/pairwise_annotation.py   --annotator chengkai --port 8766
```

Current labels are `data/eval/gold/pairwise_annotations_zhou.jsonl`. Archived original labels and review records are in deprecated. A label edit changes the dataset hash; update the published manifest and statistics after confirming a new snapshot. The workbench does not call a model or automatically recompute evaluator statistics.

The full design is [docs/pairwise_annotation_workbench.md](../docs/pairwise_annotation_workbench.md). Dataset contents and sharing are described in [data/README.md](../data/README.md).

## Rubric workbench

The rubric workbench labels single ads from the frozen sample `data/eval/gold/rubric_sample.jsonl` (130 ads). It shows the product information and one ad, and hides the generator. Labels are RubricResult v1 human records in `data/eval/gold/rubric_annotations_<annotator>.jsonl`, written on every click.

```sh
PYTHONPATH=src .venv/bin/python tools/rubric_annotation.py
```

Open http://127.0.0.1:8766/?annotator_id=zhou. By default it labels only `factual_support`, with protocol `rubric-v0.2`. Add `--criteria factual_support concrete_product_info` to label both. Keys: `1` = no conflict, `2` = conflict, `←`/`→` move, `N` = next unlabeled. A True label advances automatically. A False label focuses the reason box; write the ad sentence and the conflicting information text there. Selecting words in the ad highlights them in the product information.
