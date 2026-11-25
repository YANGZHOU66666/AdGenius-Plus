"""Local offline analysis of the November 24 frozen gold runs (no API calls)."""
import argparse
import csv
import hashlib
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/eval/gold"
OUT = ROOT / "results/nov24"
SEED = 20261124
JUDGES = {"DeepSeek": "deepseek41", "GPT": "gpt6_luna", "Claude": "claude55", "Qwen": "qwen37"}


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), f"duplicate {key}"
    return result


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(row):
    criteria = row["rubric_result"]["criteria"]
    assert row["rubric_result"]["status"] == "success" and len(criteria) == 1
    assert criteria[0]["criterion_id"] == "factual_support" and type(criteria[0]["value"]) is bool
    return criteria[0]["value"]


def paired_interval(diffs):
    """Product-cluster bootstrap of an ad-weighted paired accuracy difference."""
    rng = random.Random(SEED)
    groups = [(sum(xs), len(xs)) for _, xs in sorted(diffs.items())]
    draws = []
    for _ in range(10000):
        s = n = 0
        for ds, dn in rng.choices(groups, k=len(groups)):
            s += ds
            n += dn
        draws.append(s / n)
    draws.sort()
    return draws[249], draws[9749]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    products = index(read(GOLD / "product_items.jsonl"), "product_id")
    gens = {f.stem.removeprefix("results_"): index(read(f), "product_id") for f in sorted(GOLD.glob("results_*.jsonl"))}
    assert len(products) == 100 and len(gens) == 4
    systems = {}
    lengths = []
    for generator, rows in gens.items():
        assert rows.keys() == products.keys()
        counts = []
        for row in rows.values():
            assert row["generation_result"]["status"] == "success"
            systems[row["candidate_id"]] = generator
            counts.append(len(row["generation_result"]["text"].split()))
        lengths.append(dict(generator=generator, n=len(counts), mean_words=statistics.mean(counts),
                            median_words=statistics.median(counts), over_35=sum(n > 35 for n in counts)))
    assert len(systems) == 400
    write_csv("ad_lengths.csv", lengths)
    pairs = index(read(GOLD / "candidate_pairs.jsonl"), "pair_id")
    hp = index(read(GOLD / "pairwise_annotations_zhou.jsonl"), "pair_id")
    hrs = index(read(GOLD / "rubric_annotations_zhou.jsonl"), "candidate_id")
    correct = {}
    results = []
    by_pair = defaultdict(list)
    for judge, prefix in JUDGES.items():
        dec = index(read(ROOT / f"results/pairwise/{prefix}_gold_v2/decisions.jsonl"), "pair_id")
        rr = index(read(ROOT / f"results/rubric/{prefix}_rubric_v1/results.jsonl"), "candidate_id")
        assert dec.keys() == hp.keys() == pairs.keys() and rr.keys() == hrs.keys()
        score = []
        for pid, row in dec.items():
            actual = hp[pid]["pairwise_result"]["winner_candidate_id"]
            a, b = row["ab_winner_candidate_id"], row["ba_winner_candidate_id"]
            score.append(((actual == a) + (actual == b)) / 2)
            if judge == "DeepSeek":
                candidates = pairs[pid]["candidates"]
                ids = sorted([c["candidate_id"] for c in candidates], key=systems.get)
                first, second = ids
                assert a in ids and b in ids and (actual in ids or actual is None)
                by_pair[systems[first], systems[second]].append(
                    (0.5 if actual is None else int(first == actual), ((first == a) + (first == b)) / 2))
        correct[judge] = {c: int(label(row) == label(hrs[c])) for c, row in rr.items()}
        tp = sum(not label(hrs[c]) and not label(row) for c, row in rr.items())
        fp = sum(label(hrs[c]) and not label(row) for c, row in rr.items())
        fn = sum(not label(hrs[c]) for c in rr) - tp
        results.append(dict(judge=judge, rubric_correct=sum(correct[judge].values()), rubric_n=len(rr),
                            conflict_tp=tp, conflict_fp=fp, conflict_fn=fn,
                            conflict_f1=2*tp/(2*tp+fp+fn), pairwise_order_mean=statistics.mean(score)))
    write_csv("judge_summary.csv", results)
    write_csv("human_vs_deepseek_generator_pairs.csv", [
        dict(first=k[0], second=k[1], n=len(vals), human_first_share=statistics.mean(h for h,_ in vals),
             deepseek_first_share=statistics.mean(m for _,m in vals)) for k,vals in sorted(by_pair.items())])
    differences = []
    for other in ("GPT", "Claude", "Qwen"):
        groups = defaultdict(list)
        for c, row in hrs.items():
            groups[row["product_id"]].append(correct["DeepSeek"][c]-correct[other][c])
        lo, hi = paired_interval(groups)
        differences.append(dict(comparison=f"DeepSeek minus {other}", items=len(hrs), products=len(groups),
                                difference=(sum(correct["DeepSeek"].values())-sum(correct[other].values()))/len(hrs),
                                lower_95=lo, upper_95=hi))
    write_csv("paired_cluster_intervals.csv", differences)
    rng = random.Random(SEED)
    ids = sorted(products)
    rng.shuffle(ids)
    choices = sorted(gens)*25
    rng.shuffle(choices)
    sample = [dict(item_id=f"blind_{i:03}", candidate_id=gens[g][pid]["candidate_id"],
                   product_id=pid, generator=g) for i,(pid,g) in enumerate(zip(ids,choices),1)]
    folder = OUT / "blind100"
    folder.mkdir(exist_ok=True)
    for f in [GOLD/"product_items.jsonl",*sorted(GOLD.glob("results_*.jsonl"))]:
        dst = folder/f.name
        if not dst.exists():
            dst.symlink_to(os.path.relpath(f,folder))
    samplefile = folder / "rubric_sample.jsonl"
    contents = "".join(json.dumps(s)+"\n" for s in sample)
    if samplefile.exists() and samplefile.read_text() != contents:
        raise ValueError("Existing sample differs; refusing to replace it")
    samplefile.write_text(contents)
    overlap = sum(s["candidate_id"] in hrs for s in sample)
    train = args.training_root / "data/train/v1"
    manifest = json.loads((train / "teacher_qwen397b.manifest.json").read_text())
    source_ids, group_ids = set(), set()
    audit = {}
    for split in ("gold","test","dev","train"):
        if split in ("gold","test"):
            rows = read(ROOT/f"data/splits/v1/{split}.jsonl")
            src = {r["product_id"] for r in rows}
        else:
            rows = [r for r in read(train/"split_index.jsonl") if r["split"] == split]
            src = {r["source_id"] for r in rows}
            info = manifest["splits"][split]
            for part in ("input","output"):
                f = args.training_root/info[part]["path"]
                assert hashlib.sha256(f.read_bytes()).hexdigest() == info[part]["sha256"]
            product_rows = read(args.training_root/info["input"]["path"])
            teacher = index(read(args.training_root/info["output"]["path"]), "product_id")
            assert set(teacher) == {r["product_id"] for r in rows} == {r["product_id"] for r in product_rows}
            assert all(r["generation_result"]["status"] == "success" and r["generation_result"]["text"].strip() for r in teacher.values())
        group = {r["group_id"] for r in rows}
        assert not source_ids.intersection(src) and not group_ids.intersection(group) and len(src) == len(rows)
        source_ids.update(src)
        group_ids.update(group)
        audit[split] = {"records":len(rows), "distinct_groups":len(group)}
    print(json.dumps({"training_split_audit":audit,"balanced_sample_overlap":overlap},indent=2))


if __name__ == "__main__":
    main()
