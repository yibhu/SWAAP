"""Exact existing Stage-1 Pareto/maximin selector with a 20-candidate cap."""
from __future__ import annotations

import math

from .io import finite, require


FIELDS = {"candidate_id", "checkpoint", "iteration", "lambda", "w_c",
          "mean_return", "mean_deviation", "mean_absolute_deviation", "run",
          "source"}


def _scores(rows):
    def normalize(key):
        values = [finite(row[key]) for row in rows]
        low, high = min(values), max(values)
        return ([0.0] * len(values) if low == high else
                [(value - low) / (high - low) for value in values])
    returns, deviations = normalize("mean_return"), normalize("mean_deviation")
    return {row["candidate_id"]: (left, right)
            for row, left, right in zip(rows, returns, deviations)}


def _key(row, scores):
    left, right = scores[row["candidate_id"]]
    return (max(left, right), left + right, row["mean_deviation"],
            row["mean_return"], row["candidate_id"])


def rank_targets(stage1, rule, candidate_limit=10):
    """Retain the existing ranking byte-for-byte in behavior; widen only 10→20."""
    require(type(candidate_limit) is int and 1 <= candidate_limit <= 20,
            "Finite one-through-twenty target limit")
    require(rule["name"] == "stage1_pareto_maximin",
            "Unknown pre-outcome priority rule")
    ceiling = finite(rule["deviation_ceiling"])
    require(ceiling >= 0, "Nonnegative deviation ceiling")
    baseline = stage1.get("baseline", {})
    gate = rule["return_gate"]
    require(gate in ("fresh_original", "none"),
            "Explicit fresh original Stage1 baseline policy required")
    base = finite(baseline.get("mean_return")) if gate == "fresh_original" else None
    if gate == "fresh_original":
        require(baseline.get("role") == "FRESH_ORIGINAL_STAGE1_BASELINE",
                "Old target/proxy is not the fresh baseline")
    eligible, excluded, seen = [], [], set()
    for row in stage1["targets"]:
        require(not (set(row) - FIELDS),
                "Non-Stage1 fields cannot enter priority ranking")
        identity = row.get("candidate_id")
        require(isinstance(identity, str) and identity and identity not in seen,
                "Unique Stage1 identity required")
        seen.add(identity)
        reward, deviation = (finite(row.get("mean_return")),
                             finite(row.get("mean_deviation")))
        require(deviation >= 0, "Nonnegative canonical relative deviation")
        reasons = []
        if deviation > ceiling:
            reasons.append("DEVIATION_ABOVE_CEILING")
        if base is not None and reward >= base:
            reasons.append("NO_STRICT_STAGE1_RETURN_REDUCTION_VS_FRESH_ORIGINAL")
        if reasons:
            excluded.append({"candidate_id": identity, "reasons": reasons})
        else:
            eligible.append(row)
    require(bool(eligible),
            "No eligible Stage1 target; do not relax the gate or invent candidates")
    remaining, selected, layer_sizes = list(eligible), [], []
    while remaining and len(selected) < candidate_limit:
        layer = [row for row in remaining if not any(
            other["mean_return"] <= row["mean_return"]
            and other["mean_deviation"] <= row["mean_deviation"]
            and (other["mean_return"] < row["mean_return"]
                 or other["mean_deviation"] < row["mean_deviation"])
            for other in remaining)]
        require(bool(layer), "Empty Pareto layer")
        layer_sizes.append(len(layer))
        scores = _scores(layer)
        count = min(candidate_limit - len(selected), len(layer))
        if count == len(layer):
            representatives = list(layer)
        elif count == 1:
            representatives = [min(layer, key=lambda row: _key(row, scores))]
        else:
            low = min(layer, key=lambda row: (
                row["mean_deviation"], row["mean_return"], row["candidate_id"]))
            high = min(layer, key=lambda row: (
                row["mean_return"], row["mean_deviation"], row["candidate_id"]))
            representatives = [low]
            if high != low:
                representatives.append(high)
            while len(representatives) < count:
                unpicked = [row for row in layer if row not in representatives]
                def coverage(row):
                    left, right = scores[row["candidate_id"]]
                    distance = min(math.hypot(
                        left - scores[saved["candidate_id"]][0],
                        right - scores[saved["candidate_id"]][1],
                    ) for saved in representatives)
                    return (-distance, row["mean_deviation"],
                            row["mean_return"], row["candidate_id"])
                representatives.append(min(unpicked, key=coverage))
        selected.extend(sorted(representatives,
                               key=lambda row: _key(row, scores)))
        layer_ids = {row["candidate_id"] for row in layer}
        remaining = [row for row in remaining
                     if row["candidate_id"] not in layer_ids]
    return {
        "candidates": selected, "preoutcome_only": True,
        "rule": dict(rule), "baseline": baseline,
        "eligible_count": len(eligible), "selected_count": len(selected),
        "sparse_candidate_set": len(selected) < candidate_limit,
        "pareto_layer_sizes": layer_sizes, "exclusions": excluded,
        "priority_locked_before_victim_assessment": True,
    }
