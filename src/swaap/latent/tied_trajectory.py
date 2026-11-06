"""Exact tied-state constructor, distinct from legacy sequential SWAAP.

Both stored occurrences use the same differentiable variable. Full-buffer
victim MSE supplies matching gradients; no consistency surrogate is added.
"""
from copy import deepcopy
import math
import json
from .io import require as need
PROBES=(0,1,5,20,200,1000,5000)
def scalar(v):
    x = float(v.detach())
    need(math.isfinite(x), 'Nonfinite trajectory telemetry')
    return x

def cos(a, b, torch):
    a = a.reshape(-1)
    b = b.reshape(-1)
    d = scalar(a.norm() * b.norm())
    return scalar(torch.dot(a, b)) / d if d else None

def flatgrad(loss, ps, torch, **kwargs):
    return torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, ps, **kwargs)])

class ScientificFailure(RuntimeError):

    def __init__(self, reason, detail):
        super().__init__(reason)
        self.detail = detail

def stored_fields(clean, u, support, torch):
    """One u variable, two indexed field writes, one actual victim bank."""
    ids = torch.as_tensor(support, device=u.device, dtype=torch.long)
    current = clean[0].index_copy(0, ids + 1, u)
    successor = clean[3].index_copy(0, ids, u)
    return (current, successor)

def objective(model, original, target_reference, reference, clean, u, support, torch, alpha=0.9):
    ps = tuple(original.parameters())
    current, successor = stored_fields(clean, u, support, torch)
    prediction = model.next(current, clean[1], original)
    target_loss = torch.nn.functional.mse_loss(target_reference, model.next(reference[0], reference[1], original))
    gt = flatgrad(target_loss, ps, torch, create_graph=True)
    full_bank_mse = torch.nn.functional.mse_loss(prediction, successor)
    gp = flatgrad(full_bank_mse, ps, torch, create_graph=True, retain_graph=True)
    need(scalar(gt.norm()) > 0 and scalar(gp.norm()) > 0, 'Zero target/full-bank matching gradient; no favorable retry')
    cosine = 1 - torch.dot(gt, gp) / (gt.norm() * gp.norm())
    ids = torch.as_tensor(support, device=u.device, dtype=torch.long)
    regularization = torch.nn.functional.mse_loss((prediction[ids] - clean[3][ids]).norm(p=2, dim=1, keepdim=True), (successor[ids] - prediction[ids]).norm(p=2, dim=1, keepdim=True))
    return dict(loss=(1 - alpha) * cosine + alpha * regularization, cosine=cosine, regularization=regularization, full_bank_mse=full_bank_mse, target_loss=target_loss, gt=gt, gp=gp, current=current, successor=successor, prediction=prediction)

def tied_vjp_probe(parts, u, support, desired, torch, alpha=0.9):
    """Gather the ONE joint dense-field VJP; do not split GEMM contributions."""
    field_current, field_next = torch.autograd.grad(parts['loss'], (parts['current'], parts['successor']), retain_graph=True)
    ids = torch.as_tensor(support, device=u.device, dtype=torch.long)
    through_input = field_current[ids + 1]
    through_label = field_next[ids]
    gathered = through_input + through_label
    direct = torch.autograd.grad(parts['loss'], u, retain_graph=True)[0]
    joint = torch.autograd.grad((parts['cosine'], parts['regularization']), u, grad_outputs=(torch.full_like(parts['cosine'], 1 - alpha), torch.full_like(parts['regularization'], alpha)), retain_graph=True)[0]
    torch.testing.assert_close(gathered, direct, rtol=0.0002, atol=2e-07)
    torch.testing.assert_close(joint, direct, rtol=0.0002, atol=2e-07)
    return dict(full_bank_rows=len(parts['current']), selected_underlying_states=len(support), full_bank_training_mse=scalar(parts['full_bank_mse']), target_reference_mse=scalar(parts['target_loss']), target_parameter_gradient_l2=scalar(parts['gt'].norm()), actual_full_bank_parameter_gradient_l2=scalar(parts['gp'].norm()), parameter_gradient_cosine=cos(parts['gt'], parts['gp'], torch), outer_direct_u_gradient_l2=scalar(direct.norm()), outer_input_field_gradient_l2=scalar(through_input.norm()), outer_next_field_gradient_l2=scalar(through_label.norm()), input_vs_next_field_gradient_cosine=cos(through_input, through_label, torch), input_descent_cosine_to_clean_input_target_direction=cos(-through_input, desired, torch), next_descent_cosine_to_clean_input_target_direction=cos(-through_label, desired, torch), tied_descent_cosine_to_clean_input_target_direction=cos(-direct, desired, torch), full_buffer_gather_vjp_max_abs_error=scalar((gathered - direct).abs().max()), joint_objective_vjp_max_abs_error=scalar((joint - direct).abs().max()), identity='VJP through actual joint dense fields, then gather next[S]+current[S+1]; no post-GEMM component linearity assumption', separate_consistency_surrogate=False, partial_field_occurrences=2, poison_optimizer_updates=0)

def initialize(clean, support, family, seed_poison, torch):
    ids = torch.as_tensor(support, device=clean[3].device, dtype=torch.long)
    if family == 'EXACT_TIED_TRAJECTORY':
        u = clean[3][ids].clone()
    else:
        need(seed_poison is not None, 'Closed ordinary seed absent')
        need(all((torch.equal(clean[i], seed_poison[i]) for i in (0, 1, 2, 4))), 'Ordinary seed changed a non-next field')
        outside = torch.ones(len(clean[0]), device=ids.device, dtype=torch.bool)
        outside[ids] = False
        need(torch.equal(clean[3][outside], seed_poison[3][outside]), 'Ordinary seed has a different support')
        need(int((clean[3] != seed_poison[3]).any(dim=1).sum()) == len(support), 'Seed is not the exact completed250-row ordinary poison')
        u = seed_poison[3][ids].clone()
    need(bool(torch.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()) and scalar((u - clean[3][ids]).abs().max()) <= 1.0, 'Initial u outside unchanged projection')
    return u.detach().clone().requires_grad_(True)

def construct(C, agent, original, target, reference, clean, support, family, seed_poison, settings, torch, np, save_probe=None):
    from .stage2 import DataPoisoner
    device = torch.device(agent.device)
    reference = tuple((t.detach().to(device) for t in reference))
    clean = tuple((t.detach().to(device) for t in clean))
    seed_poison = tuple((t.detach().to(device) for t in seed_poison)) if seed_poison is not None else None
    ids = torch.as_tensor(support, device=device, dtype=torch.long)
    history = []
    probes = []
    completed = 0
    u = None
    before = {k: v.detach().clone() for k, v in original.state_dict().items()}
    gradients = [(p, None if p.grad is None else p.grad.detach().clone()) for p in original.parameters()]
    try:
        with C._isolated_construction_rng(settings.construction_seed, device, torch, np):
            constructor = DataPoisoner(agent, original, original, reference, clean, step_size=settings.step_size, alpha=settings.alpha, epsilon=settings.epsilon, random_poison_idxs=False, sequential_dp=False, synthesize_poison_data=False)
            with torch.no_grad():
                target_reference = agent.model.next(reference[0], reference[1], target).detach()
                desired = (agent.model.next(clean[0][ids], clean[1][ids], target) - agent.model.next(clean[0][ids], clean[1][ids], original)).detach()
            u = initialize(clean, support, family, seed_poison, torch)
            for step in range(settings.projected_steps):
                parts = objective(agent.model, original, target_reference, reference, clean, u, support, torch, settings.alpha)
                if step in PROBES:
                    row = dict(completed_steps=step, **tied_vjp_probe(parts, u, support, desired, torch, settings.alpha))
                    if save_probe is not None:
                        row['u_checkpointpin'] = save_probe(step, u)
                    probes.append(row)
                gradient = torch.autograd.grad(parts['loss'], u, retain_graph=True, create_graph=True)[0]
                need(bool(torch.isfinite(gradient).all()), 'Nonfinite tied update gradient')
                with torch.no_grad():
                    proposed = u - settings.step_size * gradient
                    updated = constructor.project_lp(proposed, clean[3][ids], settings.epsilon, p='linf', print_log=False)
                u = updated.detach().clone().requires_grad_(True)
                completed = step + 1
                history.append(dict(projected_step=completed, loss=scalar(parts['loss']), preupdate_cosine=1 - scalar(parts['cosine']), regularization_loss=scalar(parts['regularization']), actual_full_bank_mse=scalar(parts['full_bank_mse'])))
                if completed % 500 == 0:
                    print(json.dumps(dict(TIED_CONSTRUCTION_PROGRESS=family, **history[-1])), flush=True)
            parts = objective(agent.model, original, target_reference, reference, clean, u, support, torch, settings.alpha)
            row = dict(completed_steps=completed, **tied_vjp_probe(parts, u, support, desired, torch, settings.alpha))
            if save_probe is not None:
                row['u_checkpointpin'] = save_probe(completed, u)
            probes.append(row)
            current, successor = stored_fields(clean, u, support, torch)
            poison = tuple((t.detach().cpu() for t in (current, clean[1], clean[2], successor, clean[4])))
    except Exception as exc:
        failure = ScientificFailure(type(exc).__name__ + ': ' + str(exc), dict(completed_steps=completed, history=history, telemetry=probes, no_resume_or_retry=True))
        if u is not None:
            try:
                current, successor = stored_fields(clean, u, support, torch)
                failure.partial = tuple((t.detach().cpu() for t in (current, clean[1], clean[2], successor, clean[4])))
            except Exception as export_error:
                failure.detail['partial_export_error'] = str(export_error)
        raise failure from exc
    finally:
        for par, gradient in gradients:
            par.grad = gradient
    need(all((torch.equal(v, original.state_dict()[k]) for k, v in before.items())), 'Original tensor state changed')
    return (poison, dict(projected_steps_completed=completed, history=history, construction_probes=probes, objective='actual ALL-row victim MSE gradient, shared u in next[S] and current[S+1]; ordinary residual-norm regularizer on S', loss_mean_rows=len(clean[0]), separate_consistency_surrogate=False, separate_consistency_weight=None, initialization='unchanged Clean next[S]' if family == 'EXACT_TIED_TRAJECTORY' else 'exact closed ordinary250 next[S], then tie both fields', target_gradient_definition='grad_theta MSE(frozenStage1_reference_prediction,P_Original(reference_z,reference_a)); exact ordinary target-loss argument order', target_reference_rows=len(reference[0]), field_occurrences_differentiated=['next[S]', 'current[S+1]'], victim_fits=0, episodes=0))
