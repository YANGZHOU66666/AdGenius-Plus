"""Summarize the paired teacher likelihood and length outcomes, by dev product."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ci(delta):
    rng = random.Random(20261124)
    draws = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(10000))
    return [draws[249], draws[9749]]


def summarize(out):
    manifest=json.loads((out/"manifest.json").read_text())
    data=rows(out/"stats.jsonl")
    arms={}
    summaries={}
    for name in ["base","sft"]:
        arm={r["product_id"]:r for r in data if r["system"]==name}
        arms[name]=arm
        assert len(arm)==manifest["dev"]
        nll=statistics.mean(r["nll"] for r in arm.values())
        weighted=sum(r["target_tokens"]*r["nll"] for r in arm.values())/sum(r["target_tokens"] for r in arm.values())
        words=[r["words"] for r in arm.values()]
        summaries[name]=dict(n=len(arm),mean_product_nll=nll,token_weighted_nll=weighted,
                             teacher_perplexity=math.exp(weighted),mean_words=statistics.mean(words),
                             within35=sum(0<w<=35 for w in words),empty_count=words.count(0),
                             below25=sum(0<w<25 for w in words),token_limit_hits=sum(r["at_limit"] for r in arm.values()))
    a,b=arms["base"],arms["sft"]
    assert a.keys()==b.keys() and len(data)==len(a)*2
    diffs=[b[i]["nll"]-a[i]["nll"] for i in a]
    length_diffs=[int(0<b[i]["words"]<=35)-int(0<a[i]["words"]<=35) for i in a]
    result=dict(arms=summaries,better_reference_nll=sum(d<0 for d in diffs),
                paired_product_nll_difference=statistics.mean(diffs),nll_difference_ci95=ci(diffs),
                compliance_difference=statistics.mean(length_diffs),compliance_difference_ci95=ci(length_diffs))
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("results/sft_pilot_nov24"))
    args=ap.parse_args()
    print(json.dumps(summarize(args.out),indent=2))
