"""Agreement between rubric judge results and human factual_support labels.

The positive class is a conflict (factual_support = false). Bootstrap intervals resample items.
"""

from __future__ import annotations

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
    out: dict[str, list[float] | None] = {}
    for key, values in draws.items():
        values.sort()
        out[key] = ([round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values)) - 1], 4)]
                    if values else None)
    return out


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
        usage = [row.get("usage") or {} for row in calls]
        summary["calls"] = {"attempts": len(calls), "failed_attempts": sum(row["status"] != "success" for row in calls),
                            "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
                            "output_tokens": sum(u.get("output_tokens") or 0 for u in usage)}
    return summary
