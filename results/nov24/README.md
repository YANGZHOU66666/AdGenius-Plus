# November 24 rubric comparisons

The [answer comparison](rubric_answers_comparison.md) summarizes labels on 100 ads. Each of the four generators contributed 25 ads. Of the 100 ads, 34 also have a reference from Zhou and all four judges. The agreement figures use those 34 shared ads. [Experiment notes and reproduction details](对照与复现.md) describe the separate sample of 130 ads used to compare the judges.

- [Reference records for the 100 ads](annotations/rubric_annotations_chengkai.jsonl) and [the frozen sample](blind100/rubric_sample.jsonl) match by candidate ID.
- [AI precheck](ai_precheck/) contains a separate set of labels tagged as AI. The protocol for the agreement comparison uses the True/False factual-support results from the reference records and the judges.
- [The agreement and generator comparison](gold_calibration.pdf) uses the original calibration sample. Local results for the comparison of the two readers are in [the reference comparison report](rubric_answers_comparison.md).

To reproduce the reference comparisons from GitHub, clone the repository with Git, select this branch, install the shared repo dependencies, and run the following from the repo root:

```sh
python -m pip install -e .
python scripts/score_blind_nov24.py
```

To reproduce the sign and size of the differences between the 130-ad judge scores, the split audit, and the Bradley–Terry reference analysis, check out the isolated source trees used for the analysis:

```sh
git fetch origin
git worktree add --detach ../adgenius-train-evidence f98a45ffd05300f9206ddff8f777fc85a461f7fa
git worktree add --detach ../adgenius-bt-evidence 2734af754d3e484285571f997d1d237ed11c6af4
python scripts/analyze_gold_nov24.py --training-root ../adgenius-train-evidence
```

In the separate BT worktree, run:

```sh
PYTHONPATH=src python scripts/summarize_bt.py --design configs/eval/bt/gold_generators.toml --pairs data/eval/gold/candidate_pairs.jsonl --results results/pairwise/deepseek41_gold_v2/results.jsonl --output-dir results/bt/gold_generators
```
