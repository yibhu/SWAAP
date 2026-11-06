"""Explicit-input Block-4 constructors; no loading, fitting, or dispatch.

The caller supplies frozen models, tensors, ordered support, seed, settings,
and artifact identities.  Gradient matching delegates every numerical update
to the unchanged ordinary public DataPoisoner.gradient_matching_step.  This
module adds no target search, top-k recomputation, simulator, scheduler, retry,
or task recipe, and does not import a scientific runtime at module load.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

from .baselines import exact_support, support_identity


FIXED_SUPPORT_CONDITIONS = ("swaap_fixed_support", "swaap_random_support",
                            "same_support_random")


def _need(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class FixedSupportSettings:
    rows: int
    support_rows: int
    projected_steps: int
    step_size: float
    alpha: float
    epsilon: float
    construction_seed: int

    def __post_init__(self):
        _need(type(self.rows) is int and type(self.support_rows) is int
              and 0 < self.support_rows <= self.rows, "explicit positive row/support budgets required")
        _need(type(self.projected_steps) is int and self.projected_steps > 0,
              "explicit positive projected-step budget required")
        _need(type(self.construction_seed) is int and 0 <= self.construction_seed < 2**32,
              "explicit unsigned construction seed required")
        for name in ("step_size", "alpha", "epsilon"):
            value = getattr(self, name)
            _need(type(value) in (int, float) and math.isfinite(value),
                  "explicit finite construction scalar required: " + name)
        _need(self.step_size > 0 and self.epsilon > 0 and 0 <= self.alpha <= 1,
              "invalid step size, epsilon, or alpha")


def _declared_identities(value):
    _need(type(value) is dict and bool(value), "explicit frozen input identities required")
    # This is a copy of caller-declared identities, not artifact-file admission.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _buffer(value, rows, torch):
    _need(isinstance(value, (tuple, list)) and len(value) == 5,
          "five-column canonical buffer required")
    _need(all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 1
              and len(tensor) == rows and bool(torch.isfinite(tensor).all())
              for tensor in value), "invalid/nonfinite buffer columns or row counts")
    return tuple(tensor.detach().cpu().clone() for tensor in value)


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@contextmanager
def _isolated_construction_rng(seed, device, torch, np):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def construct_fixed_support(*, system, original_dynamics, reference_buffer,
                            clean_buffer, target_next_latents, ordered_support,
                            settings, condition, input_identities):
    """Return a five-column poison and provenance using exact provided support.

    ``target_next_latents`` is the immutable target field on the reference
    buffer: either Stage-1 predictions or the shared frozen random target tensor.
    The caller must bind its source/hash and the complete settings to a frozen
    registry before execution.  This function never recomputes a support/noise
    field and never uses the ordinary constructor's native top-k branch.
    """
    if condition == "swaap_sequential":
        raise NotImplementedError("sequential consistency needs its own full construction loop")
    _need(condition in FIXED_SUPPORT_CONDITIONS, "unknown explicit fixed-support condition")
    _need(isinstance(settings, FixedSupportSettings), "explicit FixedSupportSettings required")
    identities = _declared_identities(input_identities)
    support = exact_support(ordered_support, rows=settings.rows, count=settings.support_rows)
    import numpy as np
    import torch
    from .stage2 import DataPoisoner

    clean = _buffer(clean_buffer, settings.rows, torch)
    _need(isinstance(reference_buffer, (tuple, list)) and len(reference_buffer) == 5
          and len(reference_buffer[0]) > 0, "explicit reference buffer required")
    reference = _buffer(reference_buffer, len(reference_buffer[0]), torch)
    _need(isinstance(target_next_latents, torch.Tensor)
          and target_next_latents.shape == reference[3].shape
          and target_next_latents.dtype == reference[3].dtype
          and bool(torch.isfinite(target_next_latents).all()),
          "finite shape/dtype-preserving frozen reference target required")
    target_next = target_next_latents.detach().clone()
    device = torch.device(system.model.device)
    parameter_before = {name: value.detach().clone()
                        for name, value in original_dynamics.state_dict().items()}
    grad_before = [(parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                   for parameter in original_dynamics.parameters()]
    support_np = np.asarray(support, dtype=np.int64)
    history = []
    try:
        with _isolated_construction_rng(settings.construction_seed, device, torch, np):
            constructor = DataPoisoner(system, original_dynamics, original_dynamics,
                reference, clean, step_size=settings.step_size, alpha=settings.alpha,
                epsilon=settings.epsilon, random_poison_idxs=False,
                sequential_dp=False, synthesize_poison_data=False)
            constructor.r_c = settings.support_rows / settings.rows
            target_transition = [reference[0].to(device), reference[1].to(device), target_next.to(device)]
            poison_transition = [deepcopy(clean[index].to(device)) for index in (0, 1, 3)]
            constructor.poison_transition_orig = [tensor.detach().clone() for tensor in poison_transition]
            ids = torch.as_tensor(support_np, dtype=torch.int64, device=device)
            poison_transition[2][ids] = poison_transition[2][ids].detach().clone().requires_grad_(True)
            for step in range(settings.projected_steps):
                constructor.step = step
                value = constructor.gradient_matching_step(original_dynamics,
                    target_transition, poison_transition, support_np, poison_next_idxs=None)
                _need(isinstance(value, (tuple, list)) and len(value) == 6
                      and all(bool(torch.isfinite(item).all()) for item in value[1:])
                      and all(bool(torch.isfinite(item).all()) for item in value[0]),
                      "nonfinite fixed-support constructor state/loss/gradient")
                poison_transition = value[0]
                history.append({"projected_step": step + 1,
                    "loss": float(value[1].detach()), "preupdate_cosine": 1. - float(value[2].detach()),
                    "regularization_loss": float(value[3].detach())})
                if (step + 1) % 500 == 0:
                    print(json.dumps({"BLOCK4_CONSTRUCTION_PROGRESS": condition,
                        "projected_steps": step + 1, **history[-1]}, allow_nan=False), flush=True)
            poisoned = (poison_transition[0].detach().cpu(), clean[1], clean[2],
                        poison_transition[2].detach().cpu(), clean[4])
    finally:
        # The reused gradient-matching step accumulates parameter .grad buffers;
        # no model update uses them. Restore caller-owned buffers on all exits.
        for parameter, grad in grad_before:
            parameter.grad = grad
    _need(all(torch.equal(value, original_dynamics.state_dict()[name])
              for name, value in parameter_before.items()), "constructor modified Original weights/buffers")
    _need(all(torch.equal(clean[index], poisoned[index]) for index in (0, 1, 2, 4)),
          "constructor changed a non-successor field")
    outside = torch.ones(settings.rows, dtype=torch.bool)
    outside[list(support)] = False
    _need(torch.equal(clean[3][outside], poisoned[3][outside]), "constructor changed successors outside frozen support")
    _need(float((poisoned[3] - clean[3]).abs().max()) <= settings.epsilon,
          "constructor exceeded explicit perturbation bound")
    return tuple(tensor.clone() for tensor in poisoned), {
        "condition": condition, "settings": asdict(settings), "input_identities": identities,
        "support": support_identity(support, rows=settings.rows, count=settings.support_rows),
        "ordered_support": list(support), "target_next_tensor_sha256": _tensor_sha(target_next),
        "projected_steps_completed": len(history), "history": history,
        "optimizer_steps": 0, "victim_fits": 0, "environment_steps": 0,
        "support_recomputed": False, "target_field_resampled": False,
        "caller_rng_preserved": True, "caller_parameter_grad_buffers_restored": True,
    }


def construct_direct_target_data(*, clean_buffer, target_predictions, ordered_support,
                                 rows, support_rows, input_identities):
    """Raw DMP-data target substitution: no gradient matching or projection."""
    identities = _declared_identities(input_identities)
    support = exact_support(ordered_support, rows=rows, count=support_rows)
    import torch
    clean = _buffer(clean_buffer, rows, torch)
    _need(isinstance(target_predictions, torch.Tensor)
          and target_predictions.shape == clean[3].shape
          and target_predictions.dtype == clean[3].dtype
          and bool(torch.isfinite(target_predictions).all()),
          "finite shape/dtype-preserving Stage1 target predictions required")
    target = target_predictions.detach().cpu()
    poisoned = list(clean)
    poisoned[3][list(support)] = target[list(support)]
    return tuple(poisoned), {
        "condition": "dmp_data", "input_identities": identities,
        "support": support_identity(support, rows=rows, count=support_rows),
        "ordered_support": list(support), "target_next_tensor_sha256": _tensor_sha(target),
        "projected_steps_completed": 0, "gradient_calls": 0,
        "optimizer_steps": 0, "victim_fits": 0, "environment_steps": 0,
        "projection": "none_raw_target_predictions", "support_recomputed": False,
    }


def _sequential_gradient_matching_step(constructor, original_dynamics,
                                      target_transition, poison_transition,
                                      support, next_support, torch):
    """Latest continuous notebook cell7, GM method lines731--864.

    The only extra loss over ordinary SWAAP is its weight-one SARSA term.
    Current states are not gradient-updated; detached consistency writes occur
    after the ordinary next-state projection.  Do not route this through the
    public ordinary constructor, which intentionally rejects sequential mode.
    """
    loss_fn = torch.nn.MSELoss()
    target_z, target_action, target_next = target_transition
    target_prediction = constructor.model.next(target_z, target_action, original_dynamics)
    target_loss = loss_fn(target_next, target_prediction)
    target_grads = torch.autograd.grad(target_loss,
        original_dynamics.parameters(), create_graph=True)
    g_t = torch.cat([g.view(-1) for g in target_grads if g is not None])

    prediction = constructor.model.next(poison_transition[0], poison_transition[1],
                                        original_dynamics)
    poison_loss = loss_fn(poison_transition[2], prediction)
    ids = torch.as_tensor(support, dtype=torch.long, device=constructor.device)
    next_ids = torch.as_tensor(next_support, dtype=torch.long, device=constructor.device)
    seq_prediction = constructor.model.next(poison_transition[2][ids],
        poison_transition[1][next_ids], original_dynamics)
    # This is the unmodified CLEAN next-next target, even when selected pairs
    # overlap.  It is not taken from the evolving poisoned successor tensor.
    seq_target = constructor.poison_transition_orig[2][next_ids]
    poison_loss = poison_loss + loss_fn(seq_target, seq_prediction)
    poison_grads = torch.autograd.grad(poison_loss,
        original_dynamics.parameters(), create_graph=True)
    g_p = torch.cat([g.view(-1) for g in poison_grads if g is not None])
    similarity = 1. - torch.dot(g_t, g_p) / (torch.norm(g_t) * torch.norm(g_p))
    clean_residual = prediction[support] - constructor.poison_transition_orig[2][support]
    poison_residual = poison_transition[2][support] - prediction[support]
    regularization = loss_fn(clean_residual.norm(p=2, dim=1, keepdim=True),
                             poison_residual.norm(p=2, dim=1, keepdim=True))
    loss = (1 - constructor.alpha) * similarity + constructor.alpha * regularization
    for value in poison_transition:
        if value is not None and value.requires_grad and value.grad is not None:
            value.grad.zero_()
    loss.backward(retain_graph=True)
    updated_transition = []
    for mode, value in zip(constructor.poison_mode, poison_transition):
        if mode:
            gradient = torch.autograd.grad(loss, value, retain_graph=True,
                                           create_graph=True)[0]
            with torch.no_grad():
                updated = value.clone()
                updated[support] = updated[support] - constructor.step_size * gradient[support]
                updated = constructor.project_lp(updated,
                    constructor.poison_transition_orig[2], constructor.eps, p="linf",
                    print_log=False)
            value = updated.detach().clone().requires_grad_(True)
        updated_transition.append(value)
    with torch.no_grad():
        current = updated_transition[0].clone()
        current[next_ids] = updated_transition[2][ids].detach()
    # In the notebook poison_mode[0] is False: no differentiation through a
    # previous consistency assignment is carried into the next update.
    updated_transition[0] = current.detach().clone()
    return updated_transition, loss, similarity, regularization, g_t.detach(), g_p.detach()


def construct_sequential_support(*, system, original_dynamics, reference_buffer,
                                 clean_buffer, target_next_latents, ordered_support,
                                 settings, episode_end_indices, input_identities):
    """Full notebook trajectory-consistent construction on explicit pair rows.

    The frozen Block4B invocation supplies 250 selected rows of 5,000 and reset
    ends499,999,...,4999.  Toy inputs may be smaller.  Selection/ranking is a
    separate frozen input, never recomputed here.  Each selected underlying
    state writes two fields: next[i] and current[i+1]; overlapping pairs do not
    reduce that 2*support_rows field-write budget.
    """
    _need(isinstance(settings, FixedSupportSettings), "explicit FixedSupportSettings required")
    identities = _declared_identities(input_identities)
    support = exact_support(ordered_support, rows=settings.rows, count=settings.support_rows)
    import numpy as np
    import torch
    from .stage2 import DataPoisoner
    from .sequential import valid_adjacent_rows

    clean = _buffer(clean_buffer, settings.rows, torch)
    _need(clean[0].ndim == 2 and clean[0].shape == clean[3].shape
          and clean[0].is_floating_point() and clean[3].dtype == clean[0].dtype,
          "paired floating latent matrices required")
    _need(all(bool(((tensor >= 0) & (tensor <= 1)).all()) for tensor in (clean[0], clean[3])),
          "notebook latent projection requires clean state values within [0,1]")
    episode_ends = tuple(episode_end_indices)
    eligible = set(valid_adjacent_rows(clean[0].numpy(), clean[3].numpy(), clean[4].numpy(),
                                      episode_end_indices=episode_ends))
    _need(all(index in eligible for index in support),
          "selected sequential pair crosses a frozen reset/termination boundary")
    next_support = tuple(index + 1 for index in support)
    _need(isinstance(reference_buffer, (tuple, list)) and len(reference_buffer) == 5
          and len(reference_buffer[0]) > 0, "explicit reference buffer required")
    reference = _buffer(reference_buffer, len(reference_buffer[0]), torch)
    _need(isinstance(target_next_latents, torch.Tensor)
          and target_next_latents.shape == reference[3].shape
          and target_next_latents.dtype == reference[3].dtype
          and bool(torch.isfinite(target_next_latents).all()),
          "finite shape/dtype-preserving frozen reference target required")
    target_next = target_next_latents.detach().clone()
    device = torch.device(system.model.device)
    weights_before = {name: value.detach().clone()
                      for name, value in original_dynamics.state_dict().items()}
    grads_before = [(parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                    for parameter in original_dynamics.parameters()]
    support_np = np.asarray(support, dtype=np.int64)
    history = []
    try:
        with _isolated_construction_rng(settings.construction_seed, device, torch, np):
            constructor = DataPoisoner(system, original_dynamics, original_dynamics,
                reference, clean, step_size=settings.step_size, alpha=settings.alpha,
                epsilon=settings.epsilon, random_poison_idxs=False,
                sequential_dp=False, synthesize_poison_data=False)
            constructor.r_c = settings.support_rows / settings.rows
            transition = [deepcopy(clean[index].to(device)) for index in (0, 1, 3)]
            constructor.poison_transition_orig = [value.detach().clone() for value in transition]
            transition[2][support_np] = transition[2][support_np].detach().clone().requires_grad_(True)
            target_transition = [reference[0].to(device), reference[1].to(device),
                                 target_next.to(device)]
            for step in range(settings.projected_steps):
                constructor.step = step
                value = _sequential_gradient_matching_step(constructor, original_dynamics,
                    target_transition, transition, support_np, next_support, torch)
                _need(all(bool(torch.isfinite(item).all()) for item in value[1:])
                      and all(bool(torch.isfinite(item).all()) for item in value[0]),
                      "nonfinite sequential constructor state/loss/gradient")
                transition = value[0]
                _need(torch.equal(transition[0][list(next_support)], transition[2][list(support)]),
                      "sequential post-update state identity failed")
                history.append({"projected_step": step + 1, "loss": float(value[1].detach()),
                    "preupdate_cosine": 1. - float(value[2].detach()),
                    "regularization_loss": float(value[3].detach())})
                if (step + 1) % 500 == 0:
                    print(json.dumps({"BLOCK4_CONSTRUCTION_PROGRESS": "sequential",
                        "projected_steps": step + 1, **history[-1]}, allow_nan=False), flush=True)
            poisoned = (transition[0].detach().cpu(), clean[1], clean[2],
                        transition[2].detach().cpu(), clean[4])
    finally:
        for parameter, gradient in grads_before:
            parameter.grad = gradient
    _need(all(torch.equal(value, original_dynamics.state_dict()[name])
              for name, value in weights_before.items()), "constructor modified Original weights/buffers")
    _need(all(torch.equal(clean[index], poisoned[index]) for index in (1, 2, 4)),
          "sequential constructor changed action/reward/termination")
    for field, permitted in ((0, next_support), (3, support)):
        outside = torch.ones(settings.rows, dtype=torch.bool)
        outside[list(permitted)] = False
        _need(torch.equal(clean[field][outside], poisoned[field][outside]),
              "sequential constructor changed an unselected latent field")
        _need(float((poisoned[field] - clean[field]).abs().max()) <= settings.epsilon
              and bool(((poisoned[field] >= 0) & (poisoned[field] <= 1)).all()),
              "sequential constructor exceeded notebook projection bounds")
    touched = set(support) | set(next_support)
    return tuple(tensor.clone() for tensor in poisoned), {
        "condition": "swaap_sequential", "settings": asdict(settings),
        "input_identities": identities, "sequential_weight": 1.,
        "support": support_identity(support, rows=settings.rows, count=settings.support_rows),
        "ordered_support": list(support), "episode_end_indices": list(episode_ends),
        "target_next_tensor_sha256": _tensor_sha(target_next),
        "selected_underlying_states": len(support),
        "stored_state_field_writes": 2 * len(support),
        "distinct_touched_transition_rows": len(touched),
        "selected_adjacent_pair_overlaps": 2 * len(support) - len(touched),
        "projected_steps_completed": len(history), "history": history,
        "optimizer_steps": 0, "victim_fits": 0, "environment_steps": 0,
        "support_recomputed": False, "target_field_resampled": False,
        "caller_rng_preserved": True, "caller_parameter_grad_buffers_restored": True,
    }
