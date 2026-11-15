"""Derived statistics; the two orders of one pair remain related observations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def decision(row: dict[str, Any]) -> dict[str, Any]:
    by_order = {item["prompt_order"]: item for item in row["pairwise_result"]}
    ab, ba = by_order["AB"], by_order["BA"]
    if ab["status"] != "success" or ba["status"] != "success":
        status, winner = "incomplete", None
    elif ab["winner_candidate_id"] == ba["winner_candidate_id"]:
        status, winner = "stable", ab["winner_candidate_id"]
    else:
        status, winner = "order_inconsistent", None
    return {"pair_id": row["pair_id"], "decision_status": status,
            "winner_candidate_id": winner,
            "ab_winner_candidate_id": ab["winner_candidate_id"],
            "ba_winner_candidate_id": ba["winner_candidate_id"]}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson(successes: int, total: int) -> dict[str, float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def summarize(results: list[dict[str, Any]], humans: dict[str, dict[str, Any]],
              calls: list[dict[str, Any]], expected_pairs: int) -> dict[str, Any]:
    decisions = {row["pair_id"]: decision(row) for row in results}
    counts = Counter(item["decision_status"] for item in decisions.values())
    successful = sum(item["status"] == "success"
                     for row in results for item in row["pairwise_result"])
    complete = counts["stable"] + counts["order_inconsistent"]
    labeled = {key: row["pairwise_result"]["winner_candidate_id"]
               for key, row in humans.items()
               if row["pairwise_result"]["status"] == "success"}
    compared = {key: item for key, item in decisions.items() if key in labeled}
    stable = {key: item for key, item in compared.items() if item["decision_status"] == "stable"}
    matches = sum(item["winner_candidate_id"] == labeled[key] for key, item in stable.items())
    per_order = {}
    for order in ("AB", "BA"):
        usable = [(row["pair_id"], item) for row in results if row["pair_id"] in labeled
                  for item in row["pairwise_result"]
                  if item["prompt_order"] == order and item["status"] == "success"]
        matched = sum(item["winner_candidate_id"] == labeled[key] for key, item in usable)
        per_order[order] = {"compared_count": len(usable), "agreement_count": matched,
                            "agreement_rate": _rate(matched, len(usable))}

    # Include every response, including retries and invalid/truncated outputs.
    # Missing usage is unknown, never a zero-cost inference.
    usage = {}
    for name in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
                 "cache_read_tokens", "cache_write_tokens"):
        values = [(call.get("response") or {}).get("usage", {}).get(name) for call in calls]
        known = [value for value in values if type(value) is int and value >= 0]
        usage[name] = {"known_sum": sum(known) if known else None,
                       "known_attempt_count": len(known),
                       "unknown_attempt_count": len(calls) - len(known)}
    latencies = [call["latency_seconds"] for call in calls]
    first_choices = sum(
        item["choice"] == ("A" if item["prompt_order"] == "AB" else "B")
        for row in results for item in row["pairwise_result"] if item["status"] == "success")
    return {
        "expected_pair_count": expected_pairs,
        "recorded_pair_count": len(results),
        "successful_order_count": successful,
        "failed_or_missing_order_count": 2 * expected_pairs - successful,
        "order_success_rate": _rate(successful, 2 * expected_pairs),
        "both_orders_successful_count": complete,
        "stable_winner_count": counts["stable"],
        "order_inconsistent_count": counts["order_inconsistent"],
        "incomplete_pair_count": expected_pairs - complete,
        "stable_coverage_rate": _rate(counts["stable"], expected_pairs),
        "order_consistency_rate": _rate(counts["stable"], complete),
        "order_consistency_wilson_ci95": _wilson(counts["stable"], complete),
        "first_presented_choice_rate": _rate(first_choices, successful),
        "human_agreement": {
            "annotated_pair_count": len(humans), "valid_label_count": len(labeled),
            "stable_compared_count": len(stable), "stable_agreement_count": matches,
            "stable_agreement_rate": _rate(matches, len(stable)),
            "stable_agreement_wilson_ci95": _wilson(matches, len(stable)),
            "stable_coverage_of_labeled_pairs": _rate(len(stable), len(labeled)),
            "stable_and_agree_rate_over_all_labels": _rate(matches, len(labeled)),
            "per_order": per_order,
        },
        "api_attempt_count": len(calls),
        "failed_attempt_count": sum(call["status"] == "failed" for call in calls),
        "attempt_error_counts": dict(Counter(call["normalized_result"]["error"]
                                             for call in calls if call["status"] == "failed")),
        "token_usage_all_attempts": usage,
        "latency_seconds": {"sum": sum(latencies),
                            "mean": sum(latencies) / len(latencies) if latencies else None},
        "uncertainty_note": (
            "The Wilson interval is descriptive for stable pairs only. AB/BA calls are "
            "not independent samples; repeated pairs for one product also share an input. "
            "Use product-level resampling for final model comparisons."
        ),
    }
