"""Bradley–Terry scores from pairwise judgments between generation systems.

A design config names the systems (each a GenerationResult file), the anchor system scored 0 and the
system pairs to compare. ``build_pairs`` turns it into CandidatePair records for the pairwise judge;
``summarize`` turns the judge's PairwiseResult records into per-comparison win rates, a Bradley–Terry
fit and product-level bootstrap intervals.

One pair is scored for its first system as the mean over its two presentation orders: 1 (won both),
0.5 (orders disagree) or 0 (lost both). Pairs without two successful orders are excluded and counted.

Every comparison covers a fixed product set taken from the ProductItem file. When one side has no usable
ad for a product (generation failed, or truncated with include_truncated = false), that side loses the
pair without a judge call (a forfeit); when both have none it is a tie. A product with no record at all
for a compared system is an error: regenerate it first.
Scores are natural-log odds relative to the anchor: a difference of 1 means a 73% expected win rate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "metadata" / "schemas"
USABLE_STATUSES = ("success", "truncated")
METRICS_SCHEMA_VERSION = "bt_metrics_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)


# ---------------------------------------------------------------- design


def load_design(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    design, systems, comparisons = raw.get("design"), raw.get("systems"), raw.get("comparisons")
    if not isinstance(design, dict) or not isinstance(systems, dict) or not isinstance(comparisons, list):
        raise ValueError("Design needs [design], [systems] and [[comparisons]]")
    required = {"name", "products", "anchor", "pairs"}
    optional = {"include_truncated", "seed"}
    if not required <= set(design) or set(design) - required - optional:
        raise ValueError(f"[design] requires {sorted(required)}; optional {sorted(optional)}")
    if not systems or not all(isinstance(v, str) for v in systems.values()):
        raise ValueError("[systems] maps each system name to a GenerationResult file")
    if design["anchor"] not in systems:
        raise ValueError(f"Anchor {design['anchor']!r} is not a listed system")
    edges, seen = [], set()
    for item in comparisons:
        names = item.get("systems") if isinstance(item, dict) else None
        if (not isinstance(names, list) or len(names) != 2 or names[0] == names[1]
                or set(item) - {"systems", "max_products"}):
            raise ValueError(f"Each comparison needs systems = [a, b] (distinct); optional max_products: {item}")
        unknown = [name for name in names if name not in systems]
        if unknown:
            raise ValueError(f"Comparison uses unknown systems: {unknown}")
        key = frozenset(names)
        if key in seen:
            raise ValueError(f"Duplicate comparison: {names}")
        seen.add(key)
        limit = item.get("max_products")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"max_products must be a positive integer: {names}")
        edges.append({"systems": list(names), "max_products": limit})
    unused = sorted(set(systems) - {name for edge in edges for name in edge["systems"]})
    if unused:
        raise ValueError(f"Systems not in any comparison: {unused}")
    components = _components(list(systems), [edge["systems"] for edge in edges])
    if len(components) > 1:
        raise ValueError(f"Comparisons do not connect all systems; separate groups: {components}")
    return {"path": path, "name": design["name"], "products": _resolve(design["products"]),
            "anchor": design["anchor"], "pairs": _resolve(design["pairs"]),
            "include_truncated": design.get("include_truncated", True), "seed": design.get("seed", 20261124),
            "systems": {name: _resolve(file) for name, file in systems.items()}, "comparisons": edges,
            "cycle_rank": len(edges) - len(systems) + 1}


def _components(nodes: list[str], edges: list[list[str]]) -> list[list[str]]:
    neighbours = {node: set() for node in nodes}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen, groups = set(), []
    for node in nodes:
        if node in seen:
            continue
        stack, group = [node], []
        seen.add(node)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in neighbours[current] - seen:
                seen.add(other)
                stack.append(other)
        groups.append(sorted(group))
    return groups


def load_generations(design: dict[str, Any]) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, int]]]:
    """system -> product_id -> sample_index 0 GenerationResult, plus status counts per system."""
    validator = _validator("generation_result.schema.json")
    products = {row["product_id"] for row in read_jsonl(design["products"])}
    by_system, counts, seen = {}, {}, set()
    for system, path in design["systems"].items():
        rows, status = {}, defaultdict(int)
        for row in read_jsonl(path):
            validator.validate(row)
            if row["candidate_id"] in seen:
                raise ValueError(f"candidate_id appears in more than one system file: {row['candidate_id']}")
            seen.add(row["candidate_id"])
            if row["product_id"] not in products:
                raise ValueError(f"{system}: product {row['product_id']} is not in {design['products']}")
            if row["sample_index"] != 0:
                continue
            if row["product_id"] in rows:
                raise ValueError(f"{system}: two sample_index 0 results for {row['product_id']}")
            rows[row["product_id"]] = row
            status[row["generation_result"]["status"]] += 1
        by_system[system], counts[system] = rows, dict(status)
    return by_system, counts


def plan(design: dict[str, Any]) -> dict[str, Any]:
    """Products per comparison, split into judged pairs and forfeits; deterministic from the design."""
    generations, status_counts = load_generations(design)
    usable = USABLE_STATUSES if design["include_truncated"] else ("success",)
    order = sorted(row["product_id"] for row in read_jsonl(design["products"]))
    random.Random(design["seed"]).shuffle(order)
    position = {product: index for index, product in enumerate(order)}
    pairs, forfeits, edges = [], [], []
    for edge_index, edge in enumerate(design["comparisons"]):
        a, b = edge["systems"]
        chosen = order[:edge["max_products"]] if edge["max_products"] else order
        missing = [(name, product) for product in chosen for name in (a, b) if product not in generations[name]]
        if missing:
            raise ValueError(f"No generation record for {len(missing)} compared (system, product) pairs, "
                             f"e.g. {missing[:3]}; regenerate them before building")
        judged = forfeited = 0
        for product in chosen:
            rows = {name: generations[name][product] for name in (a, b)}
            ok = {name: row["generation_result"]["status"] in usable for name, row in rows.items()}
            if ok[a] and ok[b]:
                candidates = sorted(({"candidate_id": row["candidate_id"], "text": row["generation_result"]["text"]}
                                     for row in rows.values()), key=lambda item: item["candidate_id"])
                pair_id = "pair_" + hashlib.sha256(_canonical(
                    [design["name"], product, [c["candidate_id"] for c in candidates]])).hexdigest()[:24]
                pairs.append((position[product], edge_index,
                              {"pair_id": pair_id, "product_id": product, "candidates": candidates}))
                judged += 1
            else:
                forfeits.append((product, edge_index, 0.5 if ok[a] == ok[b] else 1.0 if ok[a] else 0.0))
                forfeited += 1
        edges.append({"systems": [a, b], "products": len(chosen), "pairs": judged, "forfeits": forfeited})
    pairs.sort(key=lambda item: item[:2])
    return {"pairs": [row for _, _, row in pairs], "forfeits": forfeits, "comparisons": edges,
            "products": len(order), "status_counts": status_counts}


def build_pairs(design: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """CandidatePair records for every judged pair, interleaved by product so a --limit pilot covers all."""
    planned = plan(design)
    rows = planned["pairs"]
    validator = _validator("candidate_pair.schema.json")
    for row in rows:
        validator.validate(row)
    report = {"design": design["name"], "anchor": design["anchor"], "seed": design["seed"],
              "include_truncated": design["include_truncated"], "products": planned["products"],
              "systems": {name: {"path": relative(path), "sha256": sha256(path),
                                 "status_counts": planned["status_counts"][name]}
                          for name, path in design["systems"].items()},
              "comparisons": planned["comparisons"], "pairs": len(rows), "judge_calls_ab_ba": 2 * len(rows),
              "forfeits": len(planned["forfeits"]), "cycle_rank": design["cycle_rank"]}
    return rows, report


# ---------------------------------------------------------------- outcomes and fit


def outcomes(design: dict[str, Any], pairs: list[dict[str, Any]],
             results: list[dict[str, Any]]) -> tuple[list[tuple[str, int, float]], dict[str, Any]]:
    """(product_id, comparison index, score for its first system) per pair with two successful orders."""
    system_of = {}
    for system, path in design["systems"].items():
        for row in read_jsonl(path):
            system_of[row["candidate_id"]] = system
    edge_index = {}
    for index, edge in enumerate(design["comparisons"]):
        edge_index[tuple(edge["systems"])] = (index, False)
        edge_index[tuple(reversed(edge["systems"]))] = (index, True)
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    rows, seen, counts = [], set(), defaultdict(int)
    for result in results:
        pair = pair_map.get(result["pair_id"])
        if pair is None or result["pair_id"] in seen:
            raise ValueError(f"Unknown or duplicate pair in results: {result['pair_id']}")
        seen.add(result["pair_id"])
        mapping = result["candidate_mapping"]
        try:
            index, flipped = edge_index[(system_of[mapping["A"]], system_of[mapping["B"]])]
        except KeyError:
            counts["not_in_design"] += 1
            continue
        orders = result["pairwise_result"]
        if len(orders) != 2 or any(order["status"] != "success" for order in orders):
            counts["incomplete"] += 1
            continue
        score_a = sum(order["winner_candidate_id"] == mapping["A"] for order in orders) / 2
        rows.append((pair["product_id"], index, 1 - score_a if flipped else score_a))
    counts.update(pairs_in_file=len(pairs), results=len(results), not_judged=len(pairs) - len(seen),
                  scored=len(rows))
    return rows, dict(counts)


def _aggregate(rows: list[tuple[str, int, float]]) -> dict[str, dict[int, list[float]]]:
    """product -> comparison index -> [pairs, score sum]."""
    by_product: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for product, index, score in rows:
        cell = by_product[product].setdefault(index, [0, 0.0])
        cell[0] += 1
        cell[1] += score
    return by_product


def fit(totals: dict[int, list[float]], design: dict[str, Any], pseudo: float = 0.5,
        start: dict[str, float] | None = None, tol: float = 1e-10, max_iter: int = 100000) -> dict[str, float]:
    """Bradley–Terry maximum likelihood by minorization–maximization (Hunter, 2004).

    totals: comparison index -> [pairs, score sum for its first system]. ``pseudo`` adds that many
    virtual tied pairs to every compared edge so the estimate exists even when one side wins every pair.
    Returns natural-log strengths with the anchor at 0.
    """
    systems = list(design["systems"])
    wins = dict.fromkeys(systems, 0.0)
    counts: dict[str, dict[str, float]] = {name: defaultdict(float) for name in systems}
    for index, (n, score) in totals.items():
        if n <= 0:
            continue
        a, b = design["comparisons"][index]["systems"]
        wins[a] += score + pseudo / 2
        wins[b] += n - score + pseudo / 2
        counts[a][b] += n + pseudo
        counts[b][a] += n + pseudo
    present = [name for name in systems if counts[name]]
    groups = _components(present, [[a, b] for a in present for b in counts[a]])
    if len(groups) != 1 or len(present) != len(systems):
        raise ValueError(f"Judged comparisons do not connect all systems: {groups}")
    strength = {name: math.exp(start[name]) if start else 1.0 for name in systems}
    for _ in range(max_iter):
        updated = {name: wins[name] / sum(n / (strength[name] + strength[other]) for other, n in counts[name].items())
                   for name in systems}
        scale = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        updated = {name: value / scale for name, value in updated.items()}
        change = max(abs(math.log(updated[name] / strength[name])) for name in systems)
        strength = updated
        if change < tol:
            break
    anchor = math.log(strength[design["anchor"]])
    return {name: math.log(value) - anchor for name, value in strength.items()}


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    return [round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values)) - 1], 4)]


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def summarize(design: dict[str, Any], rows: list[tuple[str, int, float]], iterations: int = 2000,
              seed: int = 20261124, pseudo: float = 0.5,
              forfeits: list[tuple[str, int, float]] = ()) -> dict[str, Any]:
    """rows: judged outcomes; forfeits: outcomes fixed by a failed generation (see ``plan``)."""
    forfeit_counts = defaultdict(int)
    for _, index, _ in forfeits:
        forfeit_counts[index] += 1
    rows = [*rows, *forfeits]
    by_product = _aggregate(rows)
    products = sorted(by_product)
    edges = design["comparisons"]

    def totals_for(sample: list[str]) -> dict[int, list[float]]:
        totals: dict[int, list[float]] = defaultdict(lambda: [0, 0.0])
        for product in sample:
            for index, (n, score) in by_product[product].items():
                totals[index][0] += n
                totals[index][1] += score
        return dict(totals)

    totals = totals_for(products)
    point = fit(totals, design, pseudo)
    rng = random.Random(seed)
    draws = {name: [] for name in design["systems"]}
    edge_draws = {index: [] for index in range(len(edges))}
    for _ in range(iterations):
        sample_totals = totals_for([products[rng.randrange(len(products))] for _ in products])
        strengths = fit(sample_totals, design, pseudo, start=point)
        for name, value in strengths.items():
            draws[name].append(value)
        for index, (n, score) in sample_totals.items():
            if n:
                edge_draws[index].append(score / n)

    counts = defaultdict(lambda: [0, 0, 0])  # first-system wins, ties, losses
    for _, index, score in rows:
        counts[index][0 if score == 1 else 1 if score == 0.5 else 2] += 1
    comparison_rows = []
    for index, edge in enumerate(edges):
        a, b = edge["systems"]
        n, score = totals.get(index, [0, 0.0])
        observed = score / n if n else None
        predicted = _sigmoid(point[a] - point[b])
        ci = _interval(edge_draws[index])
        comparison_rows.append({
            "systems": [a, b], "pairs": n, "wins": counts[index][0], "ties": counts[index][1],
            "losses": counts[index][2], "forfeits": forfeit_counts[index], "win_rate": round(observed, 4) if observed is not None else None,
            "win_rate_ci95": ci, "bt_predicted_win_rate": round(predicted, 4),
            # None when the observed interval is degenerate (e.g. one side won every pair): not testable.
            "predicted_outside_ci": (None if not ci or ci[0] == ci[1] else not ci[0] <= predicted <= ci[1])})
    system_rows = sorted(
        ({"system": name, "score": round(point[name], 4), "score_ci95": _interval(draws[name]),
          "pairs": int(sum(totals.get(i, [0])[0] for i, e in enumerate(edges) if name in e["systems"]))}
         for name in design["systems"]), key=lambda row: -row["score"])
    return {"schema_version": METRICS_SCHEMA_VERSION, "design": design["name"], "anchor": design["anchor"],
            "scale": "natural-log odds relative to the anchor; a difference of 1 is a 73% expected win rate",
            "scoring": ("per pair, the mean over AB and BA: 1 won both, 0.5 orders disagree, 0 lost both; "
                        "a side without a usable ad loses the pair (both without: tie)"),
            "pseudo_tied_pairs_per_comparison": pseudo, "cycle_rank": design["cycle_rank"],
            "bootstrap": {"unit": "product", "iterations": iterations, "seed": seed, "products": len(products)},
            "systems": system_rows, "comparisons": comparison_rows,
            "fit_check": {"comparisons_with_prediction_outside_ci": sum(r["predicted_outside_ci"] is True
                                                                        for r in comparison_rows),
                          "comparisons_not_testable": sum(r["predicted_outside_ci"] is None for r in comparison_rows),
                          "note": ("With cycle_rank 0 the comparisons form a tree and the fit reproduces every "
                                   "observed rate, so this check only has power when cycle_rank > 0. A comparison "
                                   "whose observed interval is a single value is not testable.")}}


def validate_metrics(metrics: dict[str, Any]) -> None:
    """Check a complete metrics.json document (summarize output plus pair_counts) against BTMetrics v1."""
    _validator("bt_metrics.schema.json").validate(metrics)


def render_report(metrics: dict[str, Any], counts: dict[str, Any]) -> str:
    def ci(values):
        return f"[{values[0]:.2f}, {values[1]:.2f}]" if values else "–"
    lines = [f"# Bradley–Terry: {metrics['design']}", "",
             f"Anchor `{metrics['anchor']}` = 0. {metrics['scale'].capitalize()}. "
             f"95% intervals: product bootstrap, {metrics['bootstrap']['iterations']} iterations, "
             f"{metrics['bootstrap']['products']} products. Judged pairs scored: {counts['scored']}; forfeits: "
             f"{counts['forfeits']}; incomplete: {counts.get('incomplete', 0)}; not judged: {counts['not_judged']}.", "",
             "| System | Score | 95% CI | Pairs |", "|---|---:|---|---:|"]
    lines += [f"| {r['system']} | {r['score']:.2f} | {ci(r['score_ci95'])} | {r['pairs']} |" for r in metrics["systems"]]
    lines += ["", "Win rate is for the first system; ties are pairs whose two orders disagree. Forfeits are "
                  "pairs decided by a failed generation and are included in W / T / L.", "",
              "| Comparison | Pairs | W / T / L | Forfeits | Win rate [95% CI] | BT predicted |",
              "|---|---:|---|---:|---|---:|"]
    for r in metrics["comparisons"]:
        flag = " ⚠" if r["predicted_outside_ci"] else " (n/t)" if r["predicted_outside_ci"] is None else ""
        rate = f"{r['win_rate']:.3f} {ci(r['win_rate_ci95'])}" if r["win_rate"] is not None else "–"
        lines.append(f"| {r['systems'][0]} vs {r['systems'][1]} | {r['pairs']} | {r['wins']} / {r['ties']} / "
                     f"{r['losses']} | {r['forfeits']} | {rate} | {r['bt_predicted_win_rate']:.3f}{flag} |")
    check = metrics["fit_check"]
    lines += ["", f"Fit check: {check['comparisons_with_prediction_outside_ci']} comparison(s) with the BT "
                  f"prediction outside the observed interval (⚠); {check['comparisons_not_testable']} not "
                  f"testable (n/t). Cycle rank {metrics['cycle_rank']}. "
                  f"{check['note']}", ""]
    return "\n".join(lines)
