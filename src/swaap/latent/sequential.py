"""Trajectory-consistency primitives extracted without executing a notebook.

These are preparation helpers, not a scientific worker.  Callers must bind
frozen data, the ranked support, and explicit episode boundaries.  The stored
H-Walk terminal column does not encode time-limit resets; silently trusting
that column would join distinct episodes.  No trajectories are collected here.
"""
from __future__ import annotations

from .baselines import exact_support


def _need(value, message):
    if not value:
        raise ValueError(message)


def valid_adjacent_rows(current, successor, terminated, *, episode_end_indices):
    """Return rows with an exact same-episode successor, without editing data.

    Explicit reset boundaries supplement, never replace, stored termination
    flags.  Every remaining adjacency must be exactly present in the frozen
    buffer; a mismatch is not silently treated as a new trajectory boundary.
    """
    import numpy as np
    current, successor, terminated = map(np.asarray, (current, successor, terminated))
    _need(current.ndim == successor.ndim == 2 and current.shape == successor.shape
          and len(current) > 1 and current.shape[1] > 0,
          "paired latent matrices required")
    n = len(current)
    _need(terminated.size == n and bool(np.isfinite(current).all())
          and bool(np.isfinite(successor).all())
          and bool(np.isin(terminated, (0, 1)).all()), "invalid buffer/termination values")
    ends = tuple(episode_end_indices)
    _need(len(set(ends)) == len(ends)
          and all(type(i) is int and 0 <= i < n for i in ends)
          and n - 1 in ends, "explicit unique episode ends including final row required")
    mask = ~terminated.reshape(n).astype(bool)
    mask[list(ends)] = False
    indices = np.flatnonzero(mask)
    _need(bool(np.array_equal(successor[indices], current[indices + 1])),
          "undeclared episode boundary or nonsequential frozen buffer")
    return tuple(int(i) for i in indices)


def ranked_adjacent_support(ranked_indices, eligible_indices, *, rows, count):
    """Filter a frozen ranking; require the exact budget, never silently shrink."""
    ranking = tuple(ranked_indices)
    ranking = exact_support(ranking, rows=rows, count=len(ranking))
    eligible = tuple(eligible_indices)
    eligible = exact_support(eligible, rows=rows, count=len(eligible))
    _need(type(count) is int and 0 < count <= rows, "explicit positive pair count required")
    allowed = set(eligible)
    selected = tuple(i for i in ranking if i in allowed)[:count]
    return exact_support(selected, rows=rows, count=count)


def write_consistent_pairs(clean_current, clean_successor, proposed_successor,
                           selected, *, eligible_indices, count):
    """Write each selected intermediate state into both adjacent tuple fields.

    This mirrors the notebook's post-update consistency assignment.  Adjacent
    selected pairs are handled simultaneously, and original arrays/flags are
    never mutated.  No projection, inference, or gradient matching occurs here.
    """
    import numpy as np
    current, successor, proposed = map(np.asarray,
        (clean_current, clean_successor, proposed_successor))
    _need(current.ndim == 2 and current.shape == successor.shape == proposed.shape,
          "paired current/successor proposal matrices required")
    n = len(current)
    support = exact_support(selected, rows=n, count=count)
    eligible = set(eligible_indices)
    _need(all(i in eligible and i + 1 < n for i in support),
          "selected pair crosses a declared episode boundary")
    _need(current.dtype == successor.dtype == proposed.dtype
          and current.dtype.kind == "f"
          and all(bool(np.isfinite(v).all()) for v in (current, successor, proposed)),
          "finite shape/dtype-preserving latent arrays required")
    outside = np.ones(n, dtype=bool)
    outside[list(support)] = False
    _need(bool(np.array_equal(proposed[outside], successor[outside])),
          "proposal changed successors outside frozen support")
    out_current, out_successor = current.copy(), proposed.copy()
    out_current[[i + 1 for i in support]] = out_successor[list(support)]
    touched = set(support) | {i + 1 for i in support}
    return out_current, out_successor, {
        "selected_underlying_states": count,
        "stored_state_field_writes": 2 * count,
        "distinct_touched_transition_rows": len(touched),
        "selected_adjacent_pair_overlaps": 2 * count - len(touched),
        "source_arrays_unchanged": True,
    }


def sequential_matching_loss(model, dynamics, current, actions, successor,
                             clean_successor, selected, *, sequential_weight):
    """Notebook SARSA-style poison-gradient loss, with explicit scalar weight.

    Ordinary full-buffer successor MSE is augmented by prediction of the clean
    next-next state from each poisoned intermediate state and following action.
    The caller owns model identity, support/boundary checks, cosine matching,
    projection, and the 5,000-step construction loop.  No optimizer is created.
    """
    import math
    import torch
    import torch.nn.functional as F
    _need(type(sequential_weight) in (int, float)
          and math.isfinite(sequential_weight) and sequential_weight >= 0,
          "explicit finite nonnegative sequential weight required")
    _need(current.ndim == successor.ndim == clean_successor.ndim == 2
          and current.shape == successor.shape == clean_successor.shape
          and actions.ndim == 2 and len(actions) == len(current),
          "paired sequential tensor shapes required")
    selected = tuple(selected)
    indices = exact_support(selected, rows=len(current), count=len(selected))
    _need(all(i + 1 < len(current) for i in indices), "successor row absent")
    ids = torch.as_tensor(indices, dtype=torch.long, device=successor.device)
    ordinary = F.mse_loss(successor, model.next(current, actions, dynamics))
    consistency = F.mse_loss(clean_successor[ids + 1],
        model.next(successor[ids], actions[ids + 1], dynamics))
    return ordinary + sequential_weight * consistency
