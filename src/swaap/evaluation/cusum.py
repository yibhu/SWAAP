"""Pinned incoming-buffer residual-CUSUM numerical kernels. No model imports."""
from __future__ import annotations
import math
from math import ceil
from statistics import NormalDist
from typing import Sequence, Mapping, Any
import numpy as np
TARGET_FPRS=(.05,.10)

def robust_stats(signal: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    return median, max(mad, eps)

def cusum_score(
    signal: np.ndarray,
    *,
    median: float,
    mad: float,
    drift_factor: float,
) -> float:
    """One-sided residual CUSUM score ``max_t S_t``."""

    kappa = drift_factor * mad
    running = 0.0
    maximum = 0.0
    for residual in signal - median:
        running = max(0.0, running + float(residual) - kappa)
        maximum = max(maximum, running)
    return float(maximum)

def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC as pairwise ranking probability, with half credit for ties."""

    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    total = 0.0
    for score in positive:
        total += float(np.sum(score > negative))
        total += 0.5 * float(np.sum(score == negative))
    return total / float(len(positive) * len(negative))

def percentile_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a two-sided percentile interval."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)

def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli rate."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    n = float(total)
    proportion = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)

def split_conformal_threshold(
    scores: Sequence[float], target_fpr: float
) -> tuple[float, int]:
    """Return the finite-sample split-conformal upper-tail threshold.

    With ``n`` threshold-calibration scores, ``k`` is
    ``ceil((n + 1) * (1 - target_fpr))`` (one based).  A detector alarms only
    for ``score > threshold``.  If ``k == n + 1``, the augmented order
    statistic is positive infinity, the conservative finite-sample choice.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("threshold calibration scores must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("threshold calibration scores must be finite")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between zero and one")
    k = int(ceil((values.size + 1) * (1.0 - target_fpr)))
    if k == values.size + 1:
        return float("inf"), k
    if not 1 <= k <= values.size:
        raise RuntimeError(f"invalid conformal rank k={k} for n={values.size}")
    return float(np.partition(values, k - 1)[k - 1]), k

def standardized_partial_auc(
    labels: Sequence[int], scores: Sequence[float], *, max_fpr: float = 0.05
) -> float:
    """McClish-standardized ROC partial AUC over ``FPR <= max_fpr``.

    The standardization matches ``sklearn.metrics.roc_auc_score(max_fpr=...)``:
    random ranking maps to 0.5 and perfect ranking maps to 1.0.
    """

    y = np.asarray(labels, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size or y.size == 0:
        raise ValueError("labels and scores must be non-empty and equally sized")
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must lie in (0, 1]")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0 or np.any((y != 0) & (y != 1)):
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order]
    distinct_end = np.r_[
        np.flatnonzero(np.diff(sorted_scores)), sorted_scores.size - 1
    ]
    tp = np.cumsum(sorted_labels == 1)[distinct_end]
    fp = np.cumsum(sorted_labels == 0)[distinct_end]
    tpr = np.r_[0.0, tp / positives]
    fpr = np.r_[0.0, fp / negatives]

    if max_fpr < 1.0:
        stop = int(np.searchsorted(fpr, max_fpr, side="right"))
        fpr_left, fpr_right = fpr[stop - 1], fpr[stop]
        tpr_left, tpr_right = tpr[stop - 1], tpr[stop]
        weight = (max_fpr - fpr_left) / (fpr_right - fpr_left)
        fpr = np.r_[fpr[:stop], max_fpr]
        tpr = np.r_[tpr[:stop], tpr_left + weight * (tpr_right - tpr_left)]

    integrate = getattr(np, "trapezoid", np.trapz)
    partial = float(integrate(tpr, fpr))
    if max_fpr == 1.0:
        return partial
    minimum = 0.5 * max_fpr**2
    maximum = max_fpr
    return float(0.5 * (1.0 + (partial - minimum) / (maximum - minimum)))

def _finite_vector(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result

def decode_threshold(record: Mapping[str, Any] | float) -> float:
    if isinstance(record, Mapping):
        if bool(record.get("threshold_is_positive_infinity")):
            return float("inf")
        value = record.get("threshold")
    else:
        value = record
    value = float(value)
    if math.isnan(value):
        raise ValueError("threshold cannot be NaN")
    return value

def _ranking_points(clean: np.ndarray, poison: np.ndarray) -> dict[str, Any]:
    labels = np.r_[np.zeros(clean.size, dtype=int), np.ones(poison.size, dtype=int)]
    scores = np.r_[clean, poison]
    return {
        "auc": float(binary_auc(labels, scores)),
        "standardized_pauc": {
            f"{level:.2f}": standardized_partial_auc(
                labels, scores, max_fpr=level
            )
            for level in TARGET_FPRS
        },
    }

def evaluate_score_samples(
    clean_scores: Sequence[float],
    poison_scores: Sequence[float],
    thresholds: Mapping[str, Mapping[str, Any] | float],
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 20260925,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate one frozen detector with independent episode resampling."""

    clean = _finite_vector(clean_scores, "clean-test scores")
    poison = _finite_vector(poison_scores, "poison scores")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    decoded = {
        f"{level:.2f}": decode_threshold(thresholds[f"{level:.2f}"])
        for level in TARGET_FPRS
    }
    ranking = _ranking_points(clean, poison)
    rng = np.random.default_rng(bootstrap_seed)
    auc_samples = np.empty(bootstrap_resamples, dtype=float)
    pauc_samples = {
        key: np.empty(bootstrap_resamples, dtype=float) for key in decoded
    }
    for index in range(bootstrap_resamples):
        clean_sample = clean[rng.integers(0, clean.size, size=clean.size)]
        poison_sample = poison[rng.integers(0, poison.size, size=poison.size)]
        points = _ranking_points(clean_sample, poison_sample)
        auc_samples[index] = points["auc"]
        for key in pauc_samples:
            pauc_samples[key][index] = points["standardized_pauc"][key]
    auc_interval = percentile_interval(auc_samples, confidence)
    ranking_intervals = {
        "auc": {
            "lower": auc_interval[0],
            "upper": auc_interval[1],
            "method": "independent_episode_percentile_bootstrap",
        },
        "standardized_pauc": {},
    }
    for key, samples in pauc_samples.items():
        interval = percentile_interval(samples, confidence)
        ranking_intervals["standardized_pauc"][key] = {
            "lower": interval[0],
            "upper": interval[1],
            "method": "independent_episode_percentile_bootstrap",
        }

    operating_points: dict[str, Any] = {}
    for key, threshold in decoded.items():
        clean_alarms = int(np.sum(clean > threshold))
        poison_alarms = int(np.sum(poison > threshold))
        fpr_interval = wilson_interval(clean_alarms, int(clean.size), confidence)
        tpr_interval = wilson_interval(poison_alarms, int(poison.size), confidence)
        operating_points[key] = {
            "target_fpr": float(key),
            "threshold": None if math.isinf(threshold) else threshold,
            "threshold_is_positive_infinity": bool(math.isinf(threshold)),
            "realized_fpr": clean_alarms / float(clean.size),
            "tpr": poison_alarms / float(poison.size),
            "num_clean_alarms": clean_alarms,
            "num_poison_alarms": poison_alarms,
            "intervals": {
                "realized_fpr": {
                    "lower": fpr_interval[0],
                    "upper": fpr_interval[1],
                    "method": "wilson_score",
                },
                "tpr": {
                    "lower": tpr_interval[0],
                    "upper": tpr_interval[1],
                    "method": "wilson_score",
                },
            },
        }
    return {
        **ranking,
        "num_clean_test_episodes": int(clean.size),
        "num_poison_episodes": int(poison.size),
        "operating_points": operating_points,
        "intervals": ranking_intervals,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "unit": "episode",
            "sampling": "independent_by_class",
            "detector_refit": False,
        },
    }

