"""Pure scalar deviation metrics for preserved or prospective evaluation arrays.

This module does not import a scientific runtime, fit a model, reset an
environment, draw random numbers, or alter its input arrays.  Callers bind the
arrays to their immutable task/checkpoint/root identities and persist the returned
scalars in a separate metric result or a prospective episode record.  Importing
this helper does not retrofit or mutate any historical receipt.
"""
from __future__ import annotations

import numpy as np


EPSILON = 1e-8
DELTA_M_DEFINITION = (
    "mean_t L2(z[t+1] - predicted_next_z[t]) / (L2(z[t+1]) + 1e-8)"
)
DELTA_D_DEFINITION = (
    "mean_rows L2(next_z - P_psi0(z,a)) / (L2(P_psi0(z,a)) + 1e-8)"
)


def _count(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive exact integer")
    return value


def _rows(value, expected_rows: int, name: str) -> np.ndarray:
    """Read finite real vector rows, flattening only non-row dimensions."""
    array = np.asarray(value)
    if array.ndim < 2 or array.shape[0] != expected_rows:
        raise ValueError(name + " has an unexpected row count or vector shape")
    if array.dtype.kind not in "iuf" or any(size == 0 for size in array.shape[1:]):
        raise ValueError(name + " must contain nonempty real numeric vectors")
    array = np.asarray(array, dtype=np.float64).reshape(expected_rows, -1)
    if not bool(np.isfinite(array).all()):
        raise ValueError(name + " contains nonfinite values")
    return array


def _normalized_mean(realized: np.ndarray, predicted: np.ndarray,
                     denominator: np.ndarray) -> float:
    if realized.shape != predicted.shape or realized.shape != denominator.shape:
        raise ValueError("latent vector shapes differ")
    # Scope error handling locally: do not modify a caller's NumPy error mode.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = np.linalg.norm(realized - predicted, axis=1) / (
            np.linalg.norm(denominator, axis=1) + EPSILON)
        mean = float(np.mean(values))
    if not bool(np.isfinite(values).all()) or not np.isfinite(mean):
        raise ValueError("deviation metric produced a nonfinite value")
    return mean


def episode_delta_m(latents, predicted_next_latents, *, expected_steps: int) -> float:
    """Frozen Block-2 per-episode δm, computed in float64.

    ``latents`` contains the T+1 realized states and predictions contains T
    one-step predictions.  Normalization uses the *realized next state*, not the
    prediction and not the original model's prediction.  Averaging episode
    scalars across a declared root panel is the caller's responsibility.
    """
    steps = _count(expected_steps, "expected_steps")
    states = _rows(latents, steps + 1, "latents")
    predictions = _rows(predicted_next_latents, steps, "predicted_next_latents")
    realized = states[1:]
    return _normalized_mean(realized, predictions, realized)


def buffer_delta_d(clean_next_latents, poison_next_latents,
                   predicted_next_latents, *, expected_rows: int) -> dict:
    """Paired clean/poison δd using frozen original-model predictions.

    This is a forward-only array reduction: callers obtain ``predicted`` from
    their frozen P_psi0 on the unchanged clean (z,a) inputs.  The helper never
    invokes a model.  Historical δd normalizes by *predicted* vector norm; that
    intentionally differs from δm.  Every row, not only poisoned support, is used.
    """
    rows = _count(expected_rows, "expected_rows")
    clean = _rows(clean_next_latents, rows, "clean_next_latents")
    poison = _rows(poison_next_latents, rows, "poison_next_latents")
    predictions = _rows(predicted_next_latents, rows, "predicted_next_latents")
    clean_mean = _normalized_mean(clean, predictions, predictions)
    poison_mean = _normalized_mean(poison, predictions, predictions)
    return {
        "definition": DELTA_D_DEFINITION,
        "row_count": rows,
        "clean_mean": clean_mean,
        "poison_mean": poison_mean,
        "poison_minus_clean_mean": poison_mean - clean_mean,
    }
