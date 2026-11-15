"""Validate the shared frozen gold data and judge comparison, entirely offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema

from adgenius.eval.pairwise_metrics import summarize
from build_candidate_pairs import build_pairs


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(root: Path) -> dict:
    gold = root / "data/eval/gold"
    manifest = json.loads((gold / "manifest.json").read_text())
    require(manifest["status"] == "frozen", "Expected the formal frozen dataset")

    def checked_rows(entry: dict, schema_name: str) -> list[dict]:
        path = root / entry["path"]
        require(digest(path) == entry["sha256"], f"Content hash mismatch: {path}")
        rows = read_rows(path)
        require(len(rows) == entry["records"], f"Record count mismatch: {path}")
        schema = json.loads((root / "metadata/schemas" / schema_name).read_text())
        validator = jsonschema.Draft202012Validator(schema)
        for row in rows:
            validator.validate(row)
        return rows

    files = manifest["files"]
    products = checked_rows(files["product_items.jsonl"], "product_item.schema.json")
    pairs = checked_rows(files["candidate_pairs.jsonl"], "candidate_pair.schema.json")
    labels = checked_rows(files["pairwise_annotations_zhou.jsonl"], "pairwise_result.schema.json")
    product_ids = {p["product_id"] for p in products}
    require(len(product_ids) == len(products) == 100, "Expected 100 distinct gold products")
    generations = []
    for alias, entry in manifest["generators"].items():
        rows = checked_rows(entry, "generation_result.schema.json")
        require(len(rows) == 100 and {r["product_id"] for r in rows} == product_ids,
                f"Incomplete product coverage: {alias}")
        require(all(r["generation_result"]["status"] == "success" for r in rows),
                f"Unsuccessful candidate in frozen dataset: {alias}")
        generations.extend(rows)
    require(len(generations) == len({r["candidate_id"] for r in generations}) == 400,
            "Expected 400 distinct candidates")
    pair_manifest_path = root / files["candidate_pairs.manifest.json"]["path"]
    require(digest(pair_manifest_path) == files["candidate_pairs.manifest.json"]["sha256"],
            "Pair construction manifest hash mismatch")
    provenance = json.loads(pair_manifest_path.read_text())
    require(digest(root / provenance["products"]) == provenance["products_sha256"],
            "Pair provenance refers to different products")
    for source in provenance["inputs"]:
        require(digest(root / source["path"]) == source["sha256"],
                f"Pair source hash mismatch: {source['path']}")
    require(digest(gold / "candidate_pairs.jsonl") == provenance["output_sha256"],
            "Frozen pair output hash mismatch")
    expected, combinations = build_pairs(generations, list(manifest["generators"]),
                                         provenance["seed"], product_ids)
    require(expected == pairs and combinations == provenance["source_combination_counts"],
            "Pair sampling differs from the frozen seed and candidates")
    pair_map = {p["pair_id"]: p for p in pairs}
    humans = {r["pair_id"]: r for r in labels}
    require(len(pairs) == len(pair_map) == len(labels) == len(humans) == 100,
            "Expected 100 distinct pairs and human labels")
    require(set(humans) == set(pair_map), "Human pair coverage mismatch")
    for pid, row in humans.items():
        candidate_ids = {c["candidate_id"] for c in pair_map[pid]["candidates"]}
        result = row["pairwise_result"]
        require(row["judge_type"] == "human" and set(row["presentation"].values()) == candidate_ids,
                f"Invalid human presentation: {pid}")
        require(result["status"] == "success" and result["choice"] in {"A", "B"}
                and result["winner_candidate_id"] == row["presentation"][result["choice"]],
                f"Human winner mapping mismatch: {pid}")

    comparison = root / "results/pairwise/comparison"
    analysis = json.loads((comparison / "manifest.json").read_text())
    require(analysis["dataset_manifest"]["sha256"] == digest(gold / "manifest.json"),
            "Comparison uses a different dataset manifest")
    require(analysis["human_annotations"]["sha256"] == digest(gold / "pairwise_annotations_zhou.jsonl"),
            "Comparison uses different human labels")
    require(analysis["metrics_sha256"] == digest(comparison / "metrics.json"),
            "Comparison metrics hash mismatch")
    comparison_metrics = json.loads((comparison / "metrics.json").read_text())
    row_map = {r["run_id"]: r for r in comparison_metrics["models"]}
    require(len(analysis["results"]) == len(row_map) == 4, "Expected four complete judge runs")
    result_validator = jsonschema.Draft202012Validator(json.loads(
        (root / "metadata/schemas/pairwise_result.schema.json").read_text()))
    for source in analysis["results"]:
        path = root / source["path"]
        require(digest(path) == source["sha256"], f"Judge result hash mismatch: {path}")
        results = read_rows(path)
        require(len(results) == 100 and {r["pair_id"] for r in results} == set(pair_map),
                f"Judge pair coverage mismatch: {path}")
        for result in results:
            result_validator.validate(result)
            mapping = dict(zip("AB", (c["candidate_id"] for c in
                                       pair_map[result["pair_id"]]["candidates"])))
            require(result["candidate_mapping"] == mapping, "Judge candidate mapping mismatch")
            require(result["judge_run_id"] == path.parent.name, "Judge run ID mismatch")
            require(all(e["status"] == "success" and e["winner_candidate_id"] == mapping[e["choice"]]
                        for e in result["pairwise_result"]), "Invalid final judge winner")
        # Recompute agreement from published results; no call journal or API key needed.
        actual = summarize(results, humans, [], 100)
        row = row_map[path.parent.name]
        saved = comparison_metrics["metrics"][row["model"]]
        for key, value in actual["human_agreement"].items():
            require(saved["human_agreement"][key] == value,
                    f"Saved agreement differs from published labels: {path}, {key}")
        for key in ("order_consistency_rate", "first_presented_choice_rate", "stable_winner_count"):
            require(saved[key] == actual[key], f"Saved order statistics differ: {path}, {key}")
    return {"products": 100, "successful_candidates": 400, "pairs": 100,
            "human_labels": 100, "validated_judge_runs": 4,
            "judge_orders": 800, "api_calls": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = validate(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
