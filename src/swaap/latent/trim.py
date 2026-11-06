"""Versioned implementations of the TRIM baseline used in SWAAP.

The original notebook implementation and the later corrected implementation
are deliberately kept as two different modes.  They must not be conflated:

``legacy_notebook``
    Reproduces the implementation in cell 4 (ID
    ``22f04964-9519-4395-8e57-c285284c48a1``) of
    ``nips_code/docs/SWAAP_experiment_nips_code.ipynb``.  It performs an
    additional full-buffer warm-up before the iterative trimmed fits and has
    the original break-before-assignment behavior.

``corrected_matched_epochs``
    Reproduces the standard-TRIM path in cell 4 (the same cell ID) of
    ``experiment_40_nips_rebuttal.ipynb``.  It starts from ``psi_0`` and
    distributes exactly ``num_epochs`` over the TRIM iterations.

This module contains no TRIAD code and never synthesizes poison data.
"""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TrimMode = Literal["legacy_notebook", "corrected_matched_epochs"]


@dataclass(frozen=True)
class TrimConfig:
    mode: TrimMode
    num_epochs: int
    batch_size: int = 256
    learning_rate: float = 1e-4
    beta: float = 0.2
    max_iters: int = 10
    convergence_tolerance: float = 1e-6
    device: str = "cuda:0"


@dataclass
class TrimFitResult:
    psi: torch.nn.Module
    mode: TrimMode
    trim: bool
    requested_epochs: int
    completed_epochs: int
    optimizer_steps: int
    retained_indices: torch.Tensor | None
    history: list[dict[str, Any]]
    final_training_loss: float
    stopped_early: bool
    returned_last_candidate: bool


def seed_everything(seed: int) -> None:
    """Reset all RNGs used by the notebook training/evaluation code."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sequence_sha256(data: Sequence[torch.Tensor]) -> str:
    """Content hash for an in-memory transition-buffer tuple."""

    digest = hashlib.sha256()
    for index, tensor in enumerate(data):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(index).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_sha256(module: torch.nn.Module) -> str:
    """Stable content hash of a module state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if not isinstance(tensor, torch.Tensor):
            digest.update(type(tensor).__qualname__.encode("utf-8"))
            digest.update(repr(tensor).encode("utf-8"))
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit_buffer_pair(
    clean_data: Sequence[torch.Tensor],
    poison_data: Sequence[torch.Tensor],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check whether two buffers are row-aligned and locate changed rows.

    Poison labels are used only for retrospective analysis.  They are not used
    by the TRIM selection rule.
    """

    field_names = ("z", "action", "reward", "next_z", "terminated")
    if len(clean_data) != len(field_names) or len(poison_data) != len(field_names):
        raise ValueError("expected five transition-buffer tensors")

    num_rows = int(clean_data[0].shape[0])
    masks: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    for name, clean, poison in zip(field_names, clean_data, poison_data):
        if clean.shape != poison.shape:
            raise ValueError(
                f"{name} shape mismatch: clean={tuple(clean.shape)}, "
                f"poison={tuple(poison.shape)}"
            )
        if int(clean.shape[0]) != num_rows:
            raise ValueError(f"{name} has inconsistent first dimension")
        clean_cpu = clean.detach().cpu()
        poison_cpu = poison.detach().cpu()
        if torch.is_floating_point(clean_cpu):
            difference = (clean_cpu - poison_cpu).abs().reshape(num_rows, -1)
            mask = difference.amax(dim=1) > float(tolerance)
        else:
            mask = (clean_cpu != poison_cpu).reshape(num_rows, -1).any(dim=1)
        masks[name] = mask
        shapes[name] = list(clean_cpu.shape)

    any_changed = torch.zeros(num_rows, dtype=torch.bool)
    for mask in masks.values():
        any_changed |= mask

    changed_by_field = {name: int(mask.sum().item()) for name, mask in masks.items()}
    next_z_only = (
        changed_by_field["next_z"] == int(any_changed.sum().item())
        and all(changed_by_field[name] == 0 for name in field_names if name != "next_z")
    )
    return {
        "num_rows": num_rows,
        "shapes": shapes,
        "changed_by_field": changed_by_field,
        "num_changed_rows": int(any_changed.sum().item()),
        "next_z_only": bool(next_z_only),
        "row_aligned_for_retention_audit": bool(next_z_only),
        "changed_row_mask": any_changed,
    }


def _validate_config(config: TrimConfig) -> None:
    if config.mode not in ("legacy_notebook", "corrected_matched_epochs"):
        raise ValueError(f"unknown TRIM mode: {config.mode}")
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if not 0.0 <= config.beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    if config.mode == "legacy_notebook" and config.num_epochs // config.max_iters <= 0:
        raise ValueError("legacy mode requires num_epochs >= max_iters")
    if config.mode == "corrected_matched_epochs" and config.num_epochs < config.max_iters:
        raise ValueError("corrected mode requires num_epochs >= max_iters")


def _retention_record(
    *,
    iteration: int,
    epochs: int,
    indices: torch.Tensor,
    losses: torch.Tensor,
    poison_mask: torch.Tensor | None,
    previous_indices: torch.Tensor | None,
) -> dict[str, Any]:
    indices_cpu = indices.detach().cpu()
    record: dict[str, Any] = {
        "iteration": int(iteration),
        "epochs_this_iteration": int(epochs),
        "retained_total": int(indices_cpu.numel()),
        "ranking_loss_mean": float(losses.mean().item()),
        "ranking_loss_q50": float(torch.quantile(losses, 0.5).item()),
    }
    if previous_indices is not None:
        previous_cpu = previous_indices.detach().cpu()
        current_set = set(indices_cpu.tolist())
        previous_set = set(previous_cpu.tolist())
        overlap = len(current_set & previous_set)
        union = len(current_set | previous_set)
        record.update(
            {
                "overlap_with_previous": overlap,
                "jaccard_with_previous": overlap / union if union else 1.0,
            }
        )

    if poison_mask is not None:
        mask = poison_mask.detach().cpu().bool()
        kept = torch.zeros(len(mask), dtype=torch.bool)
        kept[indices_cpu] = True
        clean_mask = ~mask
        record.update(
            {
                "total_poison": int(mask.sum().item()),
                "total_clean": int(clean_mask.sum().item()),
                "kept_poison": int((kept & mask).sum().item()),
                "kept_clean": int((kept & clean_mask).sum().item()),
                "removed_poison": int((~kept & mask).sum().item()),
                "removed_clean": int((~kept & clean_mask).sum().item()),
            }
        )
    return record


def fit_world_model(
    *,
    system: Any,
    data: Sequence[torch.Tensor],
    config: TrimConfig,
    trim: bool,
    seed: int,
    poison_mask: torch.Tensor | None = None,
) -> TrimFitResult:
    """Fit a fresh copy of ``system.psi_0`` under one versioned TRIM mode."""

    _validate_config(config)
    seed_everything(seed)
    device = torch.device(config.device)
    tensors = [tensor.to(device).detach() for tensor in data]
    z, action, _reward, next_z, _terminated = tensors
    num_samples = int(z.shape[0])
    if any(int(tensor.shape[0]) != num_samples for tensor in tensors):
        raise ValueError("transition tensors have inconsistent first dimensions")
    retained_count = int((1.0 - config.beta) * num_samples)
    if retained_count <= 0:
        raise ValueError("TRIM retains zero transitions")
    if poison_mask is not None and len(poison_mask) != num_samples:
        raise ValueError("poison mask length does not match buffer")

    initial_indices = torch.arange(num_samples, device=device, dtype=torch.long)
    optimizer_steps = 0
    completed_epochs = 0

    def train_on_subset(
        indices: torch.Tensor,
        *,
        init_params: torch.nn.Module | None,
        epochs: int,
        legacy_cpu_permutation: bool,
    ) -> tuple[torch.nn.Module, float]:
        nonlocal optimizer_steps, completed_epochs
        model_params = deepcopy(system.psi_0 if init_params is None else init_params)
        optimizer = torch.optim.Adam(model_params.parameters(), lr=config.learning_rate)
        subset_size = int(indices.numel())
        for _ in range(int(epochs)):
            # This intentionally preserves the legacy notebook's CPU
            # randperm, while corrected mode samples on the indices' device.
            if legacy_cpu_permutation:
                permutation = torch.randperm(subset_size)
                shuffled = indices[permutation.to(indices.device)]
            else:
                permutation = torch.randperm(subset_size, device=indices.device)
                shuffled = indices[permutation]
            for start in range(0, subset_size, config.batch_size):
                batch_indices = shuffled[start : start + config.batch_size]
                prediction = system.model.next(
                    z[batch_indices], action[batch_indices], model_params
                )
                loss = F.mse_loss(prediction, next_z[batch_indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
            completed_epochs += 1
        with torch.no_grad():
            prediction = system.model.next(z[indices], action[indices], model_params)
            final_loss = float(F.mse_loss(prediction, next_z[indices]).item())
        return model_params, final_loss

    if not trim:
        psi, final_loss = train_on_subset(
            initial_indices,
            init_params=None,
            epochs=config.num_epochs,
            legacy_cpu_permutation=(config.mode == "legacy_notebook"),
        )
        return TrimFitResult(
            psi=psi,
            mode=config.mode,
            trim=False,
            requested_epochs=config.num_epochs,
            completed_epochs=completed_epochs,
            optimizer_steps=optimizer_steps,
            retained_indices=None,
            history=[],
            final_training_loss=final_loss,
            stopped_early=False,
            returned_last_candidate=True,
        )

    history: list[dict[str, Any]] = []
    previous_indices: torch.Tensor | None = None
    stopped_early = False
    returned_last_candidate = True

    if config.mode == "legacy_notebook":
        epochs_per_iteration = int(config.num_epochs / config.max_iters)
        # Legacy Step 1: extra full-buffer fit before any trimming.
        psi, previous_loss = train_on_subset(
            initial_indices,
            init_params=None,
            epochs=epochs_per_iteration,
            legacy_cpu_permutation=True,
        )
        top_indices: torch.Tensor | None = None
        for iteration in range(1, config.max_iters + 1):
            with torch.no_grad():
                predictions = system.model.next(z, action, psi)
                losses = F.mse_loss(
                    predictions, next_z, reduction="none"
                ).mean(dim=1)
                top_indices = torch.topk(-losses, k=retained_count).indices
            history.append(
                _retention_record(
                    iteration=iteration,
                    epochs=epochs_per_iteration,
                    indices=top_indices,
                    losses=losses,
                    poison_mask=poison_mask,
                    previous_indices=previous_indices,
                )
            )
            previous_indices = top_indices.detach().clone()
            psi_new, new_loss = train_on_subset(
                top_indices,
                init_params=psi,
                epochs=epochs_per_iteration,
                legacy_cpu_permutation=True,
            )
            # Preserve the original bug exactly: a converged candidate is not
            # assigned to psi before breaking.
            if abs(new_loss - previous_loss) < config.convergence_tolerance:
                stopped_early = True
                returned_last_candidate = False
                break
            psi, previous_loss = psi_new, new_loss
        assert top_indices is not None
        final_loss = float(previous_loss)

    else:
        base_epochs, remainder = divmod(config.num_epochs, config.max_iters)
        schedule = [
            base_epochs + (1 if index < remainder else 0)
            for index in range(config.max_iters)
        ]
        psi = deepcopy(system.psi_0)
        top_indices = None
        final_loss = float("nan")
        for iteration, epochs_this_iteration in enumerate(schedule, start=1):
            with torch.no_grad():
                predictions = system.model.next(z, action, psi)
                losses = F.mse_loss(
                    predictions, next_z, reduction="none"
                ).reshape(num_samples, -1).mean(dim=1)
                top_indices = torch.topk(
                    losses, k=retained_count, largest=False
                ).indices
            history.append(
                _retention_record(
                    iteration=iteration,
                    epochs=epochs_this_iteration,
                    indices=top_indices,
                    losses=losses,
                    poison_mask=poison_mask,
                    previous_indices=previous_indices,
                )
            )
            previous_indices = top_indices.detach().clone()
            psi, final_loss = train_on_subset(
                top_indices,
                init_params=psi,
                epochs=epochs_this_iteration,
                legacy_cpu_permutation=False,
            )
        assert top_indices is not None

    return TrimFitResult(
        psi=psi,
        mode=config.mode,
        trim=True,
        requested_epochs=config.num_epochs,
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        retained_indices=top_indices.detach().cpu(),
        history=history,
        final_training_loss=float(final_loss),
        stopped_early=stopped_early,
        returned_last_candidate=returned_last_candidate,
    )
