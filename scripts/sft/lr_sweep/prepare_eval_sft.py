"""Build candidate and CandidatePair files for the SFT dev selection judged by the repo's evaluate_pairwise.py.
(Copy of scripts/opd/lr_sweep/prepare_eval.py with the SFT generation run_id prefix.)

For each finished run (results/<run>/dev.jsonl) this writes eval/candidates/<run>.jsonl with a stable candidate_id
derived exactly like scripts/generate_candidates.py: "cand_" + sha256(json([run_id, product_id, 0]))[:24].
Truncated outputs keep their text and are flagged (finish_reason == "length"); they are judged like any other
output. The final GenerationResult conversion waits for the planned "truncated" status in the shared schema.

For each matchup it writes eval/pairs/<matchup>.jsonl (CandidatePair v1, one pair per dev product) and
eval/pairs/<matchup>.excluded.json for products settled without a judge call, by rules fixed before judging:
- identical texts -> tie (0.5 each);
- one side empty -> the non-empty side wins; both empty -> tie.
The denominator is always all 200 dev products.

Usage: python prepare_eval.py --work-dir <dir> <matchup> [<matchup> ...], where a matchup is "runA__vs__runB".
Runs are read from <dir>/results and outputs go to <dir>/eval.
"""

import argparse
import hashlib
import json
from pathlib import Path

# Set by configure() / the command line.
RESULTS = EVAL = None


def configure(work_dir):
    global RESULTS, EVAL
    RESULTS, EVAL = Path(work_dir) / "results", Path(work_dir) / "eval"


def run_id(run):
    return f"gen_sft_lr_sweep_dev_{run.replace('.', 'p').replace('-', 'm')}"


def candidate_id(rid, product_id):
    key = json.dumps([rid, product_id, 0], ensure_ascii=False).encode("utf-8")
    return "cand_" + hashlib.sha256(key).hexdigest()[:24]


def candidates(run):
    source = RESULTS / "base_dev.jsonl" if run == "base" else RESULTS / run / "dev.jsonl"
    rows = [json.loads(line) for line in source.open() if line.strip()]
    assert len(rows) == 200 and len({r["product_id"] for r in rows}) == 200, f"{run}: expected 200 unique products"
    rid = run_id(run)
    out = [{"candidate_id": candidate_id(rid, r["product_id"]), "product_id": r["product_id"], "run_id": rid,
            "sample_index": 0, "text": r["text"], "finish_reason": r["finish_reason"],
            "truncated": r["finish_reason"] == "length", "completion_tokens": r["completion_tokens"]} for r in rows]
    (EVAL / "candidates").mkdir(parents=True, exist_ok=True)
    with (EVAL / "candidates" / f"{run}.jsonl").open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {row["product_id"]: row for row in out}


def build(matchup):
    a, b = matchup.split("__vs__")
    ca, cb = candidates(a), candidates(b)
    assert ca.keys() == cb.keys()
    pairs, excluded = [], []
    for pid in sorted(ca):
        ta, tb = ca[pid]["text"].strip(), cb[pid]["text"].strip()
        if ta == tb:
            excluded.append({"product_id": pid, "rule": "identical_text", "score_a": 0.5})
        elif not ta or not tb:
            excluded.append({"product_id": pid, "rule": "empty_output",
                             "score_a": 0.5 if not ta and not tb else (1.0 if ta else 0.0)})
        else:
            key = json.dumps([matchup, pid], ensure_ascii=False).encode("utf-8")
            pairs.append({"pair_id": "pair_" + hashlib.sha256(key).hexdigest()[:24], "product_id": pid,
                          "candidates": [{"candidate_id": ca[pid]["candidate_id"], "text": ca[pid]["text"]},
                                         {"candidate_id": cb[pid]["candidate_id"], "text": cb[pid]["text"]}]})
    (EVAL / "pairs").mkdir(parents=True, exist_ok=True)
    with (EVAL / "pairs" / f"{matchup}.jsonl").open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    (EVAL / "pairs" / f"{matchup}.excluded.json").write_text(json.dumps(
        {"a": a, "b": b, "candidate_a": {p: ca[p]["candidate_id"] for p in ca},
         "candidate_b": {p: cb[p]["candidate_id"] for p in cb}, "excluded": excluded}, indent=1))
    print(f"{matchup}: {len(pairs)} pairs to judge, {len(excluded)} settled by rule, "
          f"truncated a={sum(c['truncated'] for c in ca.values())} b={sum(c['truncated'] for c in cb.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build candidates and CandidatePairs for dev matchups.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("matchups", nargs="+")
    args = parser.parse_args()
    configure(args.work_dir)
    for m in args.matchups:
        build(m)
