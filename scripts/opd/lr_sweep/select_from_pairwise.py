"""Pick the OPD setting from the repo-format DeepSeek pairwise results (pairwise-v0.2, AB and BA).

Rules, fixed before any judgment was read (docs/sft_opd_experiment_plan.md §5 and the sweep notes):
- Denominator is always the 200 dev products. A product's score for run A in a matchup is the mean over its
  successful orders of 1 (A wins) or 0 (B wins). Products settled without a judge call: identical texts -> 0.5;
  one side empty -> the non-empty side wins (both empty -> 0.5). If both orders failed after retries -> 0.5.
- Truncated outputs keep their text and are judged as written; truncation is reported, not penalised separately.
- Within each beta, a learning rate's score is the mean of its win rates in its two matchups. The best one wins
  unless the product-bootstrap 95% CI of its win rate against the runner-up contains 0.5; then the smaller of
  the two learning rates is chosen.
- The two beta winners are compared head to head. If that CI contains 0.5, beta=1 is chosen (simpler and
  cheaper: no teacher top-1 retrieval).
- No per-item regeneration of model outputs; only judge calls are retried (repo evaluator behaviour).

Reads <work-dir>/eval/pairwise/<run_id>/results.jsonl and <work-dir>/eval/pairs/<matchup>.excluded.json;
writes <work-dir>/eval/selection.json. Usage: python select_from_pairwise.py --work-dir <dir>
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

# Set by configure() / the command line.
WORK = EVAL = None


def configure(work_dir):
    global WORK, EVAL
    WORK = Path(work_dir)
    EVAL = WORK / "eval"
LRS = ("5e-07", "1e-06", "2e-06")
BETAS = ("1", "0.5")
SHORT = {"5e-07": "5e7", "1e-06": "1e6", "2e-06": "2e6"}


def run(beta, lr):
    return f"beta{beta}_lr{lr}"


def judge_run_id(beta, lr_a, lr_b):
    return f"opd_sweep_dev_b{beta.replace('.', '')}_lr{SHORT[lr_a]}_vs_lr{SHORT[lr_b]}"


def scores(matchup, rid):
    """Per-product score of the first run in the matchup, over all 200 products."""
    meta = json.loads((EVAL / "pairs" / f"{matchup}.excluded.json").read_text())
    cand_a = meta["candidate_a"]
    out = {e["product_id"]: e["score_a"] for e in meta["excluded"]}
    failed_both, inconsistent = 0, 0
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


def summarize(a, b, rid):
    s, notes = scores(f"{a}__vs__{b}", rid)
    values = list(s.values())
    rng = random.Random(20261123)
    boots = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"a": a, "b": b, "judge_run_id": rid, "products": len(values), "a_win_rate": round(statistics.mean(values), 4),
            "ci95": [round(boots[249], 4), round(boots[9749], 4)], **notes}


def guardrails(name):
    rows = [json.loads(l) for l in (WORK / "results" / (f"{name}/dev.jsonl" if name != "base" else "base_dev.jsonl")).open()]
    words = [len(re.findall(r"\S+", r["text"])) for r in rows]
    return {"mean_words": round(statistics.mean(words), 1), "over_35_words": sum(w > 35 for w in words),
            "truncated": sum(r["finish_reason"] == "length" for r in rows), "empty": sum(not r["text"].strip() for r in rows)}


def ties(stats):
    return stats["ci95"][0] <= 0.5 <= stats["ci95"][1]


def main():
    summary = {"rules": __doc__.split("Rules")[1].split("Reads")[0].strip(), "groups": {}}
    winners = {}
    for beta in BETAS:
        stats = [summarize(run(beta, x), run(beta, y), judge_run_id(beta, x, y))
                 for i, x in enumerate(LRS) for y in LRS[i + 1:]]
        score = {run(beta, lr): [] for lr in LRS}
        for s in stats:
            score[s["a"]].append(s["a_win_rate"])
            score[s["b"]].append(1 - s["a_win_rate"])
        mean_score = {k: round(statistics.mean(v), 4) for k, v in score.items()}
        best, second = sorted(mean_score, key=mean_score.get, reverse=True)[:2]
        head = next(s for s in stats if {s["a"], s["b"]} == {best, second})
        tie = ties(head)
        lr_of = {run(beta, lr): float(lr) for lr in LRS}
        winners[beta] = min((best, second), key=lr_of.get) if tie else best
        summary["groups"][beta] = {"matchups": stats, "mean_score": mean_score, "best": best, "runner_up": second,
                                   "best_vs_runner_up_ci_contains_half": tie, "chosen": winners[beta]}
    lr1 = winners["1"].split("_lr")[1]
    lr05 = winners["0.5"].split("_lr")[1]
    final_rid = f"opd_sweep_dev_final_b1_lr{SHORT[lr1]}_vs_b05_lr{SHORT[lr05]}"
    if (EVAL / "pairwise" / final_rid / "results.jsonl").exists():
        final = summarize(winners["1"], winners["0.5"], final_rid)
        tie = ties(final)
        summary["final"] = {"matchup": final, "ci_contains_half": tie,
                            "chosen": winners["1"] if tie or final["a_win_rate"] > 0.5 else winners["0.5"]}
    else:
        summary["final"] = {"pending": f"{winners['1']}__vs__{winners['0.5']}", "judge_run_id": final_rid}
    summary["guardrails"] = {name: guardrails(name) for name in
                             ["base"] + [run(b, lr) for b in BETAS for lr in LRS]}
    (EVAL / "selection.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({"group_winners": winners, "final": summary["final"]}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the frozen selection rules to the dev judgments.")
    parser.add_argument("--work-dir", required=True)
    configure(parser.parse_args().work_dir)
    main()
