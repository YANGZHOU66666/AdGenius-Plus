"""Evaluate saved pilot ads using the project's frozen DeepSeek pairwise and rubric protocols."""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/sft_pilot_nov24"


def rows(name):
    with name.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    env={**os.environ,"PYTHONPATH":str(ROOT/"src")}
    for kind,parameters in [
        ("pairwise",["--pairs",str(OUT/"candidate_pairs.jsonl"),"--no-human-annotations"]),
        ("rubric",["--products",str(OUT/"dev_products.jsonl"),"--candidates",str(OUT/"results_base.jsonl"),str(OUT/"results_sft.jsonl")])]:
        cmd=[sys.executable,f"scripts/evaluate_{kind}.py","--eval-config",f"configs/eval/{kind}_deepseek.toml",
             "--output-dir",str(OUT/kind),"--run-id",f"deepseek_{kind}_pilot_128x32","--workers","4",*parameters]
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if args.dry_run:
        return
    systems={r["candidate_id"]:s for s in ("base","sft") for r in rows(OUT/f"results_{s}.jsonl")}
    pref=Counter()
    for r in rows(OUT/"pairwise/decisions.jsonl"):
        ab,ba=r.get("ab_winner_candidate_id"),r.get("ba_winner_candidate_id")
        if ab not in systems or ba not in systems:
            pref["unscored"]+=1
        else:
            pref[systems[ab] if ab==ba else "order_disagreement"]+=1
    factual={s:Counter() for s in ("base","sft")}
    for r in rows(OUT/"rubric/results.jsonl"):
        result=r["rubric_result"]
        bucket="unscored"
        if result["status"]=="success":
            criterion=next(c for c in result["criteria"] if c["criterion_id"]=="factual_support")
            bucket="no_conflict" if criterion["value"] else "conflict"
        factual[systems[r["candidate_id"]]][bucket]+=1
    n=pref["base"]+pref["sft"]+pref["order_disagreement"]
    report={"matched_products":len(rows(OUT/"candidate_pairs.jsonl")),"sft_win_share_on_scored_pairs":
            (pref["sft"]+pref["order_disagreement"]/2)/n if n else None,
            "pairwise":pref,"rubric":factual}
    (OUT/"judge_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
