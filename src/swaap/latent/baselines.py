"""Pure Block-4 baseline primitives, not a scientific dispatcher.

These helpers validate explicit inputs or transform provided toy/array values.
They never load artifacts/models, choose a task recipe, construct a gradient-
matched poison, fit, evaluate, dispatch, or publish a result.  NumPy is imported
only when a caller explicitly asks for array/support operations.  In particular,
the caller must freeze support/noise seeds and all scientific recipes before
any later authorized execution.
"""
from __future__ import annotations

import hashlib
import math
from .io import require
import re
import struct


SHA256 = re.compile(r"[0-9a-f]{64}")
APPROVED_RANDOM_METHOD = {
    "noise_scale": .3, "alpha": .9, "epsilon": 1.0,
    "step_size": 1.0, "projected_steps": 5000,
}


def _need(value, message):
    if not value:
        raise ValueError(message)


def exact_support(indices, *, rows: int, count: int) -> tuple[int, ...]:
    """Validate an exact-count support without changing its frozen order."""
    _need(type(rows) is int and type(count) is int and 0 < count <= rows,
          "positive exact rows/count required")
    values = tuple(indices)
    _need(len(values) == count and all(type(index) is int for index in values),
          "support must contain the exact count of non-boolean Python integers")
    _need(len(set(values)) == count and all(0 <= index < rows for index in values),
          "support must be unique and within the frozen buffer")
    return values


def support_identity(indices, *, rows: int, count: int) -> dict:
    """Index-content identities, not a claim about any artifact file's bytes."""
    values = exact_support(indices, rows=rows, count=count)
    def digest(sequence):
        return hashlib.sha256(b"".join(struct.pack("<q", index) for index in sequence)).hexdigest()
    return {"row_count": rows, "support_count": count,
            "ordered_int64_sha256": digest(values),
            "sorted_int64_sha256": digest(sorted(values))}


def uniform_random_support(seed: int, *, rows: int, count: int) -> tuple[int, ...]:
    """Block-1-compatible exact-count random support using an isolated RNG.

    This is distinct from the legacy notebook's Bernoulli mask.  Nothing chooses
    a default seed or automatically replaces an adopted support artifact.
    """
    _need(type(seed) is int and 0 <= seed < 2**32,
          "explicit unsigned support seed required")
    _need(type(rows) is int and type(count) is int and 0 < count <= rows,
          "positive exact rows/count required")
    import numpy as np
    selected = np.sort(np.random.default_rng(seed).choice(rows, size=count, replace=False))
    return exact_support((int(index) for index in selected), rows=rows, count=count)


def check_support_only_random_pair(random_arm: dict, same_support_arm: dict,
                                   *, rows: int, count: int) -> dict:
    """Check scientific constructor inputs differ *only* in support indices.

    Pass scientific inputs, not operational output paths/attempt IDs.  The
    random-target hash binds the shared perturbed reference target tensor; it
    does not assert that optimized successor values on different supports are
    identical.  The noise tensor must be generated/adopted once elsewhere.
    """
    _need(type(random_arm) is dict and type(same_support_arm) is dict,
          "two explicit scientific-input dictionaries required")
    _need("support_indices" in random_arm and "support_indices" in same_support_arm,
          "both frozen support lists required")
    for arm in (random_arm, same_support_arm):
        _need(bool(SHA256.fullmatch(str(arm.get("random_target_sha256", "")))),
              "shared random target tensor SHA256 required")
        _need(type(arm.get("random_seed")) is int and 0 <= arm["random_seed"] < 2**32,
              "explicit frozen Random seed required")
        _need(type(arm.get("method")) is dict and bool(arm["method"]),
              "explicit frozen constructor method required")
        method = arm["method"]
        _need(type(method.get("support_rows")) is int and method["support_rows"] == count,
              "constructor support budget differs from the paired support count")
        _need(set(method) == set(APPROVED_RANDOM_METHOD) | {"support_rows"}
              and all(type(method.get(key)) in (int, float)
                      and method[key] == value
                      for key, value in APPROVED_RANDOM_METHOD.items())
              and type(method["projected_steps"]) is int,
              "Block-4 same-support Random must preserve the approved Block-1 recipe")
    left = {key: value for key, value in random_arm.items() if key != "support_indices"}
    right = {key: value for key, value in same_support_arm.items() if key != "support_indices"}
    _need(left == right, "Random arms differ in more than row selection")
    return {"random_support": support_identity(random_arm["support_indices"], rows=rows, count=count),
            "same_as_swaap_support": support_identity(same_support_arm["support_indices"], rows=rows, count=count),
            "shared_random_target_sha256": random_arm["random_target_sha256"],
            "only_support_changes": True}


def direct_target_successors(clean_next, target_predictions, indices,
                             *, rows: int, count: int):
    """Pure DMP-data direct substitution on an explicitly supplied support.

    Mirrors the notebook's direct-model-data branch: copy target predictions at
    selected rows, leave all other successors unchanged, and do not perform
    gradient matching or silently add projection/clipping.  Inputs remain
    untouched; a future operator separately binds Original/target/input hashes
    and validates the full five-column buffer.
    """
    support = exact_support(indices, rows=rows, count=count)
    import numpy as np
    clean = np.asarray(clean_next)
    predicted = np.asarray(target_predictions)
    _need(clean.ndim == predicted.ndim == 2 and clean.shape == predicted.shape
          and clean.shape[0] == rows and clean.shape[1] > 0,
          "paired nonempty latent successor matrices required")
    _need(clean.dtype.kind == "f" and clean.dtype == predicted.dtype,
          "shape/dtype-preserving floating-point target predictions required")
    _need(bool(np.isfinite(clean).all()) and bool(np.isfinite(predicted).all()),
          "nonfinite clean/target successor values")
    result = clean.copy()
    result[list(support)] = predicted[list(support)]
    return result


def multiplicative_noise_target(next_z, seed: int, scale: float, torch):
    """Materialize the panel-frozen baseline target on CPU deterministically."""
    require(next_z.device.type == 'cpu' and bool(torch.isfinite(next_z).all()), 'finite CPU reference next_z required')
    require(type(seed) is int and 0 <= seed < 2 ** 32, 'explicit unsigned Random seed required')
    require(type(scale) in (int, float) and (not isinstance(scale, bool)) and math.isfinite(scale) and (float(scale) == 0.3), 'Block-1 Random noise scale must be exactly 0.3')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    uniform = torch.rand(next_z.shape, dtype=next_z.dtype, device='cpu', generator=generator)
    target = torch.clamp(next_z * (1 + (uniform * 2 - 1) * float(scale)), 0, 1)
    require(target.shape == next_z.shape and target.dtype == next_z.dtype and bool(torch.isfinite(target).all()), 'finite shape/dtype-preserving Random target required')
    return target
