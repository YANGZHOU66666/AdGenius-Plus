"""Pick the SFT learning rate from the repo-format DeepSeek pairwise results (pairwise-v0.2, AB and BA).

Rules, fixed before any judgment was read (docs/sft_opd_experiment_plan.md §5; same as the OPD sweep):
- Runs whose training loss is not finite (diverged) are excluded before judging.
- Denominator is always the 200 dev products. A product's score for run A in a matchup is the mean over its
  successful orders of 1 (A wins) or 0 (B wins). Products settled without a judge call: identical texts -> 0.5;
  one side empty -> the non-empty side wins (both empty -> 0.5). If both orders failed after retries -> 0.5.
- Truncated outputs keep their text and are judged as written; truncation is reported, not penalised separately.
- A learning rate's score is the mean of its win rates in its matchups. The best one wins unless the
  product-bootstrap 95% CI of its win rate against the runner-up contains 0.5; then the smaller of the two
  learning rates is chosen. A winner at the edge of the grid is reported, not extended.
- No per-item regeneration of model outputs; only judge calls are retried (repo evaluator behaviour).

Usage: python select_sft.py --work-dir <dir> [--base-dev <untrained dev.jsonl>]
(reads <dir>/eval and <dir>/results, writes <dir>/eval/selection.json)
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

LRS = ("5e-06", "1e-05", "2e-05")
REPO = Path(__file__).resolve().parents[3]
BASE_DEV = REPO / "artifacts/train/opd_lr_sweep_v1/runs/base/dev.jsonl"  # shared untrained-student outputs
SHORT = {"5e-06": "5e6", "1e-05": "1e5", "2e-05": "2e5"}
WORK = EVAL = None


def configure(work_dir):
    global WORK, EVAL
    WORK = Path(work_dir)
    EVAL = WORK / "eval"


def run(lr):
    return f"sft_lr{lr}"


def judge_run_id(a, b):
    return f"sft_sweep_dev_lr{SHORT[a]}_vs_lr{SHORT[b]}"


def diverged(lr):
    report = WORK / "results" / run(lr) / "run_report.json"
    return json.loads(report.read_text()).get("diverged", False)


def scores(matchup, rid):
    meta = json.loads((EVAL / "pairs" / f"{matchup}.excluded.json").read_text())
    cand_a = meta["candidate_a"]
    out = {e["product_id"]: e["score_a"] for e in meta["excluded"]}
    failed_both = inconsistent = 0
    with (EVAL / "pairwise" / rid / "results.jsonl").open(encoding="utf-8") as handle:
        results = [json.loads(line) for line in handle if line.strip()]
    for row in results:
        pid = next(p for p, c in cand_a.items() if c in row["candidate_mapping"].values())
        wins = [1.0 if call["winner_candidate_id"] == cand_a[pid] else 0.0
                for call in row["pairwise_result"] if call["status"] == "success"]
        if not wins:
            failed_both += 1
        elif len(wins) == 2 and wins[0] != wins[1]:
            inconsistent += 1
        out[pid] = statistics.mean(wins) if wins else 0.5
    assert len(out) == 200, f"{matchup}: {len(out)} products scored"
    return out, {"order_inconsistent": inconsistent, "both_orders_failed": failed_both,
                 "settled_by_rule": len(meta["excluded"])}


def summarize(a, b):
    rid = judge_run_id(a, b)
    s, notes = scores(f"{run(a)}__vs__{run(b)}", rid)
    values = list(s.values())
    rng = random.Random(20261123)
    boots = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"a": run(a), "b": run(b), "judge_run_id": rid, "products": len(values),
            "a_win_rate": round(statistics.mean(values), 4), "ci95": [round(boots[249], 4), round(boots[9749], 4)],
            **notes}


def guardrails(path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    words = [len(re.findall(r"\S+", r["text"])) for r in rows]
    return {"mean_words": round(statistics.mean(words), 1), "over_35_words": sum(w > 35 for w in words),
            "truncated": sum(r["finish_reason"] == "length" for r in rows), "empty": sum(not r["text"].strip() for r in rows)}


def main():
    summary = {"rules": __doc__.split("Rules")[1].split("Usage")[0].strip()}
    live = [lr for lr in LRS if not diverged(lr)]
    summary["excluded_diverged"] = [run(lr) for lr in LRS if lr not in live]
    stats = [summarize(x, y) for i, x in enumerate(live) for y in live[i + 1:]]
    score = {run(lr): [] for lr in live}
    for s in stats:
        score[s["a"]].append(s["a_win_rate"])
        score[s["b"]].append(1 - s["a_win_rate"])
    mean_score = {k: round(statistics.mean(v), 4) for k, v in score.items()}
    ranked = sorted(mean_score, key=mean_score.get, reverse=True)
    best = ranked[0]
    chosen, tie, second = best, False, None
    if len(ranked) > 1:
        second = ranked[1]
        head = next(s for s in stats if {s["a"], s["b"]} == {best, second})
        tie = head["ci95"][0] <= 0.5 <= head["ci95"][1]
        lr_of = {run(lr): float(lr) for lr in live}
        chosen = min((best, second), key=lr_of.get) if tie else best
    summary.update(matchups=stats, mean_score=mean_score, best=best, runner_up=second,
                   best_vs_runner_up_ci_contains_half=tie, chosen=chosen,
                   at_grid_edge=chosen in (run(LRS[0]), run(LRS[-1])))
    base = BASE_DEV
    summary["guardrails"] = {"base": guardrails(base),
                             **{run(lr): guardrails(WORK / "results" / run(lr) / "dev.jsonl") for lr in live}}
    (EVAL / "selection.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("mean_score", "best", "runner_up", "best_vs_runner_up_ci_contains_half",
                                              "chosen", "at_grid_edge", "excluded_diverged")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--base-dev", type=Path, default=BASE_DEV)
    args = parser.parse_args()
    BASE_DEV = args.base_dev
    configure(args.work_dir)
    main()
