"""Rubric judge statistics for factual_support.

summarize: agreement with human labels; the positive class is a conflict (factual_support = false) and
bootstrap intervals resample items. generator_rates: judged conflict rate per generator, with product-level
bootstrap intervals and paired differences between generators judged on the same products.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Iterable

CRITERION = "factual_support"
QUOTE_MARKER = 'conflicts: ad "'


def criterion(record: dict[str, Any]) -> dict[str, Any] | None:
    result = record["rubric_result"]
    if result["status"] != "success":
        return None
    return next(c for c in result["criteria"] if c["criterion_id"] == CRITERION)


def confusion(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """pairs: (human_value, model_value). Returns counts, accuracy, Cohen's kappa, recall and precision."""
    pairs = list(pairs)
    n = len(pairs)
    tp = sum(h is False and m is False for h, m in pairs)
    fn = sum(h is False and m is True for h, m in pairs)
    fp = sum(h is True and m is False for h, m in pairs)
    tn = n - tp - fn - fp
    accuracy = (tp + tn) / n if n else None
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n ** 2 if n else None
    kappa = (accuracy - expected) / (1 - expected) if n and expected < 1 else None
    return {"n": n, "agree": tp + tn, "accuracy": accuracy, "kappa": kappa,
            "human_false": tp + fn, "model_false": tp + fp, "caught": tp, "missed": fn, "extra_false": fp,
            "recall": tp / (tp + fn) if tp + fn else None, "precision": tp / (tp + fp) if tp + fp else None}


def bootstrap(pairs: list[tuple[bool, bool]], keys: Iterable[str], iterations: int = 10000,
              seed: int = 20261124) -> dict[str, list[float] | None]:
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(iterations):
        stats = confusion([pairs[rng.randrange(len(pairs))] for _ in pairs])
        for key in draws:
            if stats[key] is not None:
                draws[key].append(stats[key])
    return {key: _interval(values) for key, values in draws.items()}


def summarize(results: list[dict[str, Any]], human: dict[str, bool], generator: dict[str, str],
              calls: list[dict[str, Any]] | None = None, iterations: int = 10000) -> dict[str, Any]:
    """results: model RubricResult records; human/generator: keyed by candidate_id."""
    judged = {r["candidate_id"]: criterion(r) for r in results}
    pairs, by_group = [], {"qwen_local": [], "other_generators": []}
    false_without_quote = 0
    for candidate_id, label in human.items():
        verdict = judged.get(candidate_id)
        if verdict is None:
            continue
        pairs.append((label, verdict["value"]))
        group = "qwen_local" if generator[candidate_id] == "qwen_local" else "other_generators"
        by_group[group].append((label, verdict["value"]))
        if verdict["value"] is False and QUOTE_MARKER not in (verdict["reason"] or ""):
            false_without_quote += 1
    summary = confusion(pairs)
    summary.update(items=len(human), failed_or_missing=len(human) - len(pairs),
                   ci95=bootstrap(pairs, ("accuracy", "kappa", "recall"), iterations),
                   by_generator={group: confusion(rows) for group, rows in by_group.items()},
                   false_without_quoted_conflict=false_without_quote)
    if calls is not None:
        summary["calls"] = _call_totals(calls)
    return summary


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values = sorted(values)
    return [round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values)) - 1], 4)]


def _call_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    usage = [row.get("usage") or {} for row in calls]
    return {"attempts": len(calls), "failed_attempts": sum(row["status"] != "success" for row in calls),
            "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in usage)}


def generator_rates(results: list[dict[str, Any]], items: list[dict[str, Any]],
                    calls: list[dict[str, Any]] | None = None, iterations: int = 10000,
                    seed: int = 20261124) -> dict[str, Any]:
    """results: model RubricResult records; items: rows with candidate_id, product_id and generator.

    A product's score for a generator is the share of its judged candidates on that product that conflict
    (0 or 1 with one sample per product). Judge failures are excluded and counted.
    """
    judged = {r["candidate_id"]: criterion(r) for r in results}
    scores: dict[str, dict[str, list[bool]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        generator = item["generator"]
        count = counts.setdefault(generator, {"candidates": 0, "judged": 0, "judge_failed": 0, "false": 0,
                                              "false_without_quoted_conflict": 0})
        count["candidates"] += 1
        verdict = judged.get(item["candidate_id"])
        if verdict is None:
            count["judge_failed"] += 1
            continue
        count["judged"] += 1
        conflict = verdict["value"] is False
        count["false"] += conflict
        count["false_without_quoted_conflict"] += conflict and QUOTE_MARKER not in (verdict["reason"] or "")
        scores.setdefault(generator, {}).setdefault(item["product_id"], []).append(conflict)
    per_product = {g: {p: sum(v) / len(v) for p, v in by_product.items()} for g, by_product in scores.items()}

    rng = random.Random(seed)
    by_generator = {}
    for generator in sorted(counts):
        values = list(per_product.get(generator, {}).values())
        draws = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                 for _ in range(iterations)] if values else []
        count = counts[generator]
        by_generator[generator] = {**count, "products": len(values),
                                   "false_rate": round(count["false"] / count["judged"], 4) if count["judged"] else None,
                                   "false_rate_ci95": _interval(draws)}

    paired = []
    for first, second in itertools.combinations(sorted(per_product), 2):
        shared = sorted(set(per_product[first]) & set(per_product[second]))
        if not shared:
            continue
        diffs = [per_product[first][p] - per_product[second][p] for p in shared]
        draws = [sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(iterations)]
        paired.append({"generators": [first, second], "shared_products": len(shared),
                       "false_rate_difference": round(sum(diffs) / len(diffs), 4),
                       "difference_ci95": _interval(draws)})

    summary = {"criterion": CRITERION, "positive_class": "conflict (factual_support = false)",
               "bootstrap": {"unit": "product", "iterations": iterations, "seed": seed},
               "by_generator": by_generator, "paired_differences": paired,
               "note": ("Judged conflict rates, not human labels. Judges miss some human-labelled conflicts "
                        "(docs/rubric_judge_agreement.md), so use these to compare generators judged by the same "
                        "run rather than as absolute conflict rates.")}
    if calls is not None:
        summary["calls"] = _call_totals(calls)
    return summary
