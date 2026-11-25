"""Compare real second-reader labels on shared candidates as they become available."""
import argparse
import json
from pathlib import Path
from analyze_gold_nov24 import GOLD, OUT, ROOT, JUDGES, index, read


def compare(human, reference, ids):
    values = [(human[c], reference[c]) for c in ids]
    n = len(values)
    if not n:
        return {"overlap": 0}
    agree = sum(a == b for a, b in values) / n
    a = sum(x for x, _ in values) / n
    b = sum(y for _, y in values) / n
    chance = a*b + (1-a)*(1-b)
    return {"overlap": n, "agreement": agree,
            "kappa": (agree-chance)/(1-chance) if chance < 1 else None}


def labels(path, criterion):
    records = index(read(path), "candidate_id")
    out = {}
    for cid, record in records.items():
        if record["rubric_result"]["status"] != "success":
            raise ValueError(f"Non-success result for {cid}")
        value = next(c["value"] for c in record["rubric_result"]["criteria"] if c["criterion_id"] == criterion)
        if type(value) is not bool:
            raise ValueError(f"Invalid value for {cid}")
        out[cid] = value
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=OUT/"annotations/rubric_annotations_chengkai.jsonl")
    args = parser.parse_args()
    sample = index(read(OUT/"blind100/rubric_sample.jsonl"), "candidate_id")
    reader = labels(args.labels, "factual_support") if args.labels.exists() else {}
    if not reader.keys() <= sample.keys():
        raise ValueError("Labels outside the balanced sample")
    refs = {"Zhou": GOLD/"rubric_annotations_zhou.jsonl"}
    refs.update({j: ROOT/f"results/rubric/{p}_rubric_v1/results.jsonl" for j,p in JUDGES.items()})
    result = {"labeled": len(reader), "planned": len(sample), "comparisons_on_overlap": {}}
    for name,path in refs.items():
        ref = labels(path, "factual_support")
        if name in JUDGES:
            run = {"DeepSeek":"deepseek", "GPT":"gpt6_luna", "Claude":"claude", "Qwen":"qwen"}[name]
            followup = ROOT/f"results/rubric/rubric_followup_{run}_20261124/results.jsonl"
            if followup.exists():
                extra = labels(followup, "factual_support")
                if extra.keys() & ref.keys() or not extra.keys() <= sample.keys():
                    raise ValueError("Unexpected extra reference records")
                ref.update(extra)
        result["comparisons_on_overlap"][name] = compare(reader, ref, reader.keys() & ref.keys())
    print(json.dumps(result, indent=2))
