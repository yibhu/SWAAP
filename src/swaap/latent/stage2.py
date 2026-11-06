"""Ordinary SWAAP next-state gradient-matching constructor.

The optimization, native top-k tie behavior, clean initialization, and
projection arithmetic are extracted from standalone_attack_core_v2's
DataPoisoner. Legacy synthesis, sequential, defense, evaluation, and alternative
attack modes are intentionally not part of this public implementation.
Defaults bind rp=.1, alpha=.9, epsilon=1, step_size=10 and 5000 steps for the
current protocol; explicit numeric overrides support ordinary ablations.
Importing this module does not import Torch or create a GPU.
"""

from copy import deepcopy
import math

__all__ = ["DataPoisoner", "constructor_settings", "load_core"]


def constructor_settings(*, rp=.1, alpha=.9, epsilon=1.0, step_size=10.0):
    """Map portable ordinary Stage-2 settings to DataPoisoner keyword args.

    ``rp`` is the caller-facing poison fraction, named ``r_c`` by the original
    implementation. Defaults bind the current release; numeric overrides reuse
    the same constructor for ablations. This helper never seeds, imports Torch,
    loads data, or runs optimization. Supply the iteration budget separately
    to ``learn_poison_data(steps=5000)`` for the current release.
    """
    values = {"rp": rp, "alpha": alpha, "epsilon": epsilon, "step_size": step_size}
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Finite non-boolean constructor scalar required: " + name)
    if not (0 < rp <= 1 and 0 <= alpha <= 1 and epsilon > 0 and step_size > 0):
        raise ValueError("Invalid ordinary Stage-2 fraction, weight, or update bounds")
    return {"r_c": float(rp), "alpha": float(alpha), "epsilon": float(epsilon),
            "step_size": float(step_size), "poison_mode": [False, False, True],
            "random_poison_idxs": False, "sequential_dp": False, "synthesize_poison_data": False}


def _load_tensor_runtime():
    global torch, nn, F
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


def print_stats(x, full_detail=False, title=None):
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, torch.Tensor) and x.is_cuda:
        x = x.cpu()
    if full_detail:
        print(f"x: shape = {x.shape}")
        print("type(x):", type(x))
        print("x.dtype:", x.dtype)
    if title is None:
        print(f"  mean: {x.mean():.4f}, std: {torch.sqrt(x.var()):.12f}, var: {x.var():.12f}, min: {x.min():.4f}, max: {x.max():.4f}, shape: {x.shape}")
    else:
        print(f"  {title}: mean: {x.mean():.4f}, std: {torch.sqrt(x.var()):.12f}, var: {x.var():.12f}, min: {x.min():.4f}, max: {x.max():.4f}, shape: {x.shape}")


class DataPoisoner:
    """Construct poison buffers without fitting or evaluating any policy.

    The original argument names are retained: ``target_psi`` is the original
    victim dynamics at which gradients are matched; ``poison_psi`` is the
    desired Stage-1 target dynamics. ``data_target`` is the reference buffer,
    and ``data_poison`` is the canonical clean adaptation buffer. The ordinary
    output keeps all five buffer columns, changing only supported next states.
    Use constructor_settings() for the current release's numeric defaults.
    """

    def __init__(
        self, system, target_psi, poison_psi, data_target, data_poison,
        step_size=10, random_poison_idxs=False, r_c=0.1, epsilon=1.0,
        poison_mode=None, alpha=0.9, result_desc="result_desc",
        sequential_dp=False, sequential_weight=1.0,
        synthesize_poison_data=False, surrogate_sampling="random",
        synthesis_seed=0, defense_surrogate_psi=None,
        defense_surrogate_checkpoint=None,
    ):
        # Keep the ordinary call interface; unsupported modes cannot silently
        # change the scientific method after removal of the legacy branches.
        if random_poison_idxs or sequential_dp or synthesize_poison_data:
            raise ValueError("Only native top-k, non-sequential real-buffer poisoning is supported")
        if defense_surrogate_psi is not None or defense_surrogate_checkpoint is not None:
            raise ValueError("Defense surrogates are not part of ordinary SWAAP")
        if poison_mode is None:
            poison_mode = [False, False, True]
        if list(poison_mode) != [False, False, True]:
            raise ValueError("Only next-z poisoning is supported")
        _load_tensor_runtime()
        self.target_psi = target_psi
        self.poison_psi = poison_psi
        self.data_target = data_target
        self.data_poison = data_poison
        self.step_size = step_size
        self.r_c = r_c
        self.eps = epsilon
        self.alpha = alpha
        self.poison_mode = list(poison_mode)
        self.model = system.model
        self.system = system
        self.device = self.model.device
        self.random_poison_idxs = False
        self.step = 0
        self.result_desc = result_desc
        self.sequential_dp = False
        self.sequential_weight = sequential_weight
        self.synthesize_poison_data = False

    def learn_poison_data(self, steps=5000, baseline=False, noise_scale=0.1, return_direct_model_poisoned_data=False, return_pgd_poisoned_data=False, return_clean_model_data=False):
        if baseline or return_direct_model_poisoned_data or return_pgd_poisoned_data or return_clean_model_data:
            raise ValueError('Only ordinary next-z gradient matching is supported')
        if self.random_poison_idxs or self.sequential_dp or self.synthesize_poison_data or self.poison_mode != [False, False, True]:
            raise ValueError('Ordinary next-z constructor settings were changed')
        if type(steps) is not int or steps <= 0:
            raise ValueError('Positive integral projected-step budget required')
        print(f'learn poison data with r_c={self.r_c} and epsilon={self.eps}')
        target_data = [x.to(self.device) for x in self.data_target]
        zs, action, _, next_z, _ = target_data
        target_psi, poison_psi, r_c = (self.target_psi, self.poison_psi, self.r_c)
        target_zs = deepcopy(zs)
        target_action = deepcopy(action)
        with torch.no_grad():
            target_next_z = self.model.next(zs, action, poison_psi)
        target_transition = [target_zs, target_action, target_next_z]
        print_stats(zs, title='zs')
        print_stats(next_z, title='next_z')
        print_stats(target_next_z, title='poison_next_z')
        original_poison_data = [x.to(self.device) for x in self.data_poison]
        poison_data = original_poison_data
        poison_idxs = None
        zs, action, reward, next_z, terminated = poison_data
        data_size = zs.shape[0]
        valid_seq_mask = None
        valid_seq_idxs = None
        print('finetune data_size:', data_size)
        print('initialize poison with self.random_poison_idxs:', self.random_poison_idxs)
        with torch.no_grad():
            target_next = self.model.next(zs, action, target_psi)
            poison_next = self.model.next(zs, action, poison_psi)
            diffs = F.mse_loss(target_next, poison_next, reduction='none').mean(dim=1)
        num_poison = int(data_size * self.r_c)
        poison_idxs = torch.topk(diffs, num_poison, largest=True).indices.cpu().numpy()
        poison_next_idxs = poison_idxs + 1 if self.sequential_dp else None
        poison_zs = deepcopy(zs)
        poison_action = deepcopy(action)
        poison_next_z = deepcopy(next_z)
        self.poison_transition_orig = [poison_zs, poison_action, poison_next_z]
        poison_transition_orig = [poison_zs, poison_action, poison_next_z]
        poison_transition = []
        poison_transition_orig_copy = deepcopy(poison_transition_orig)
        for mode, x in zip(self.poison_mode, poison_transition_orig_copy):
            if mode:
                x.requires_grad_(False)
            poison_transition.append(x)
        poison_transition[2][poison_idxs] = poison_transition[2][poison_idxs].detach().clone().requires_grad_(True)
        sims = []
        for step in range(steps):
            self.step = step
            poison_transition, loss, similarity_loss, regularization_loss, g_t, g_p = self.gradient_matching_step(target_psi, target_transition, poison_transition, poison_idxs, poison_next_idxs=poison_next_idxs)
            if (step + 1) % 100 == 0:
                print(f'r_c: {self.r_c}, epsilon: {self.eps}, alpha: {self.alpha}')
                print(f'Step [{step + 1}/{steps}], Loss: {loss.item():.4f}, Cosine Sim: {1.0 - similarity_loss.item():.4f}, Reg loss: {regularization_loss.item():.4f}')
            sims.append(1.0 - similarity_loss.item())
        data_cpu = [x.to('cpu') for x in poison_data]
        zs, action, reward, next_z, terminated = data_cpu
        poison_zs = poison_transition[0].detach().to('cpu')
        poison_next_z = poison_transition[2].detach().to('cpu')
        zs = poison_zs
        next_z = poison_next_z
        poison_data = (zs, action, reward, next_z, terminated)
        return (poison_data, sims)

    def project_lp(self, x, x_orig, eps, p='l2', print_log=False):
        delta = x - x_orig
        if p == 'l2':
            delta_flat = delta.view(delta.size(0), -1)
            norm = delta_flat.norm(p=2, dim=1, keepdim=True) + 1e-08
            factor = torch.clamp(norm, max=eps) / norm
            print_stats(x_orig, title='x_orig')
            print_stats(x, title='x')
            magnitude = x_orig.norm(p=2, dim=1, keepdim=True)
            print_stats(magnitude, title='x_orig l2 norm')
            print_stats(norm, title='delta l2 norm')
            print_stats(factor, title='factor')
            print_stats(abs(delta), title='|delta|')
            delta = delta_flat * factor
            delta = delta.view_as(x)
        elif p == 'linf':
            delta = torch.clamp(delta, -eps, eps)
            if print_log:
                print_stats(x_orig, title='x_orig')
                print_stats(x, title='x')
                print_stats(abs(delta), title='|delta|')
        x = x_orig + delta
        x = torch.clamp(x, 0, 1)
        return x.detach().clone().requires_grad_(True)

    def gradient_matching_step(self, target_psi, target_tranisition, poison_transition, poison_idxs, poison_next_idxs=None):
        if poison_next_idxs is not None:
            raise ValueError('Sequential poisoning is not supported')
        loss_fn = nn.MSELoss()
        target_zs, target_action, target_next_z = target_tranisition
        target_next_z_pred = self.model.next(target_zs, target_action, target_psi)
        target_loss = loss_fn(target_next_z, target_next_z_pred)
        target_grads = torch.autograd.grad(target_loss, target_psi.parameters(), create_graph=True)
        g_t = torch.cat([g.view(-1) for g in target_grads if g is not None])
        poison_next_z_pred = self.model.next(poison_transition[0], poison_transition[1], target_psi)
        poison_loss = loss_fn(poison_transition[2], poison_next_z_pred)
        poison_grads = torch.autograd.grad(poison_loss, target_psi.parameters(), create_graph=True)
        g_p = torch.cat([g.view(-1) for g in poison_grads if g is not None])
        similarity_loss = 1.0 - torch.dot(g_t, g_p) / (torch.norm(g_t) * torch.norm(g_p))
        next_z_pred = poison_next_z_pred
        delta_clean = next_z_pred[poison_idxs] - self.poison_transition_orig[2][poison_idxs]
        delta_norm_clean = delta_clean.norm(p=2, dim=1, keepdim=True)
        delta_poison = poison_transition[2][poison_idxs] - next_z_pred[poison_idxs]
        delta_norm_poison = delta_poison.norm(p=2, dim=1, keepdim=True)
        regularization_loss = loss_fn(delta_norm_clean, delta_norm_poison)
        loss = (1 - self.alpha) * similarity_loss + self.alpha * regularization_loss
        poison_data_params = [p for p in poison_transition if p is not None and p.requires_grad]
        for p in poison_data_params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward(retain_graph=True)
        poison_transition_new = []
        for mode, x in zip(self.poison_mode, poison_transition):
            if mode:
                grad_x = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)[0]
                with torch.no_grad():
                    updated = x.clone()
                    updated[poison_idxs] = updated[poison_idxs] - self.step_size * grad_x[poison_idxs]
                    print_log = False
                    updated = self.project_lp(updated, self.poison_transition_orig[2], self.eps, p='linf', print_log=print_log)
                x = updated.detach().clone().requires_grad_(True)
            poison_transition_new.append(x)
        return (poison_transition_new, loss, similarity_loss, regularization_loss, g_t.detach(), g_p.detach())

def load_core():
    """Return public constructor definitions; no notebook or result imports."""
    return {"DataPoisoner": DataPoisoner}
