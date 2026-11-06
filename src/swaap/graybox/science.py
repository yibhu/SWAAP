"""Executed raw-observation SWAAP: source encoder, full-buffer gradients, relative box.

No simulation, checkpoint loading or optimization runs on import.
"""
from types import SimpleNamespace
from ._latent_base import DataPoisoner
class RawObservationSWAAP(DataPoisoner):
    def __init__(self, model, original_dynamics, target_dynamics, reference, raw_clean,
                 *, alpha=.1, epsilon=1., step_size=1.):
        import torch
        self.torch = torch
        assert model.multitask is False
        clean = [x.detach().to(model.device) for x in raw_clean]
        assert clean[0].shape[0] == clean[3].shape[0] == 5000
        with torch.no_grad():
            z, nz = model.encode(clean[0]), model.encode(clean[3])
        latent = (z, clean[1], clean[2], nz, clean[4])
        super().__init__(SimpleNamespace(model=model), original_dynamics, target_dynamics,
                         reference, latent, alpha=alpha, epsilon=epsilon,
                         step_size=step_size, r_c=.1)
        self.raw_clean = clean
        self.raw_value = clean[3].clone()
        self.poison_transition_orig = [z, clean[1], nz]
        with torch.no_grad():
            difference = (model.next(z, clean[1], original_dynamics)
                          - model.next(z, clean[1], target_dynamics)).square().mean(1)
            self.support = torch.topk(difference, 500, largest=True).indices
            ref_z, ref_a = reference[0].to(model.device), reference[1].to(model.device)
            self.target_transition = [ref_z, ref_a, model.next(ref_z, ref_a, target_dynamics).detach()]
        self.target_residual = difference.detach()

    def project_lp(self, x, x_orig, eps, p='linf', print_log=False):
        """The latent proposal is unused: update the raw leaf via its full VJP."""
        torch = self.torch
        assert p == 'linf' and float(eps) == self.eps
        grad = self.raw_leaf.grad
        assert grad is not None and bool(torch.isfinite(grad).all())
        with torch.no_grad():
            new = self.raw_leaf.detach().clone()
            selected = self.support
            proposal = new[selected] - self.step_size * grad[selected]
            anchor = self.raw_clean[3][selected]
            new[selected] = anchor + torch.clamp(proposal - anchor, -self.eps, self.eps)
            self.raw_value = new
            self.last_raw_gradient = grad.detach().clone()
        # Shared code finishes its latent bookkeeping; the next call rebuilds E(raw).
        return x.detach()

    def step_raw(self, step):
        self.step = step
        self.target_psi.zero_grad(set_to_none=True)
        self.raw_leaf = self.raw_value.detach().clone().requires_grad_(True)
        encoded_next = self.model.encode(self.raw_leaf)
        encoded_next.retain_grad()
        transition = [self.poison_transition_orig[0], self.raw_clean[1], encoded_next]
        _, loss, geometry, regularizer, gt, gp = super().gradient_matching_step(
            self.target_psi, self.target_transition, transition, self.support)
        return dict(step=step+1, loss=float(loss.detach()), cosine=1.-float(geometry.detach()),
                    regularizer=float(regularizer.detach()), target_gradient_norm=float(gt.norm()),
                    poison_gradient_norm=float(gp.norm()), raw_gradient_norm=float(self.last_raw_gradient.norm()))

    def output(self):
        torch = self.torch
        mask = torch.zeros(5000, dtype=torch.bool, device=self.device)
        mask[self.support] = True
        assert torch.equal(self.raw_value[~mask], self.raw_clean[3][~mask])
        maximum = float((self.raw_value-self.raw_clean[3]).abs().max())
        assert maximum <= self.eps + 1e-6
        data = [x.detach().cpu().clone() for x in self.raw_clean]
        data[3] = self.raw_value.detach().cpu().clone()
        return tuple(data), dict(modifiable_rows=500,
                                changed_rows=int((data[3]!=self.raw_clean[3].cpu()).any(1).sum()),
                                maximum_absolute_raw_change=maximum,
                                outside_support_unchanged=True, latent_01_clipping_applied=False)

class CachedReferenceRawSWAAP(RawObservationSWAAP):
    def __init__(self, *args, frozen_support=None, **kwargs):
        super().__init__(*args, **kwargs)
        torch = self.torch
        if frozen_support is not None:
            # Reuse the already committed mask; no new scientific selection.
            assert torch.equal(self.support.cpu(), frozen_support.cpu())
            self.support = frozen_support.to(self.device)
        z, a, target = self.target_transition
        prediction = self.model.next(z, a, self.target_psi)
        loss = torch.nn.functional.mse_loss(target, prediction)
        gradients = torch.autograd.grad(loss, self.target_psi.parameters(), create_graph=True)
        self.fixed_target_gradient = torch.cat([g.reshape(-1) for g in gradients]).detach()
        assert bool(torch.isfinite(self.fixed_target_gradient).all())
        del gradients, loss, prediction

    def step_raw(self, step):
        torch = self.torch
        self.step = step
        self.raw_leaf = self.raw_value.detach().clone().requires_grad_(True)
        encoded_next = self.model.encode(self.raw_leaf)
        prediction = self.model.next(self.poison_transition_orig[0], self.raw_clean[1], self.target_psi)
        poison_loss = torch.nn.functional.mse_loss(encoded_next, prediction)
        gradients = torch.autograd.grad(poison_loss, self.target_psi.parameters(), create_graph=True)
        gp = torch.cat([g.reshape(-1) for g in gradients])
        gt = self.fixed_target_gradient
        geometry = 1.-torch.dot(gt, gp)/(gt.norm()*gp.norm())
        support = self.support
        clean_norm = (prediction[support]-self.poison_transition_orig[2][support]).norm(dim=1,keepdim=True)
        poison_norm = (encoded_next[support]-prediction[support]).norm(dim=1,keepdim=True)
        regularizer = torch.nn.functional.mse_loss(clean_norm, poison_norm)
        loss = (1.-self.alpha)*geometry+self.alpha*regularizer
        gradient = torch.autograd.grad(loss, self.raw_leaf)[0]
        self.raw_leaf.grad = gradient.detach()
        self.project_lp(encoded_next, self.poison_transition_orig[2], self.eps)
        return dict(step=step+1,loss=float(loss.detach()),cosine=1.-float(geometry.detach()),
                    regularizer=float(regularizer.detach()),target_gradient_norm=float(gt.norm()),
                    poison_gradient_norm=float(gp.detach().norm()),raw_gradient_norm=float(gradient.norm()))

class GridRelativeSWAAP(CachedReferenceRawSWAAP):
    """Only generalize the already-qualified relative projector's fixed constants."""
    def project_lp(self, x, x_orig, eps, p='linf', print_log=False):
        torch = self.torch
        assert p == 'linf' and float(eps) == self.eps
        assert self.eps in (.01, .005, .0025) and self.alpha in (0., .5, .9)
        grad = self.raw_leaf.grad
        assert grad is not None and bool(grad.isfinite().all())
        with torch.no_grad():
            value = self.raw_leaf.detach().clone()
            selected = self.support
            anchor = self.raw_clean[3][selected]
            proposal = value[selected] - self.step_size * grad[selected]
            radius = eps * anchor.abs()
            value[selected] = torch.maximum(torch.minimum(proposal, anchor + radius), anchor - radius)
            self.raw_value, self.last_raw_gradient = value, grad.detach().clone()
        return x.detach()

    def output(self):
        torch = self.torch
        anchor, value = self.raw_clean[3], self.raw_value
        mask = torch.zeros(5000, dtype=torch.bool, device=self.device)
        mask[self.support] = True
        change, radius = (value-anchor).abs(), self.eps*anchor.abs()
        assert torch.equal(value[~mask], anchor[~mask])
        assert torch.equal(value[anchor == 0], anchor[anchor == 0])
        assert not bool(((value < anchor-radius) | (value > anchor+radius)).any())
        data = [v.detach().cpu().clone() for v in self.raw_clean]
        data[3] = value.detach().cpu().clone()
        relative = change[self.support][anchor[self.support] != 0] / anchor[self.support][anchor[self.support] != 0].abs()
        return tuple(data), dict(modifiable_rows=500, changed_rows=int(change.any(1).sum()),
            changed_coordinates=int((change != 0).sum()), outside_support_unchanged=True,
            non_successor_fields_unchanged=True, exact_zero_coordinates_changed=0,
            projection='coordinate_relative_linf', epsilon=self.eps, minimum_radius_floor=False,
            raw_01_clipping=False, exact_interval_violations=0,
            maximum_absolute_change=float(change.max()), mean_absolute_change=float(change.mean()),
            selected_absolute_change_mean=float(change[self.support].mean()),
            selected_absolute_change_p50=float(change[self.support].quantile(.5)),
            selected_absolute_change_p95=float(change[self.support].quantile(.95)),
            selected_nonzero_relative_change_mean=float(relative.mean()),
            selected_nonzero_relative_change_max=float(relative.max()))

class RefinementSWAAP(GridRelativeSWAAP):
    def project_lp(self, x, x_orig, eps, p='linf', print_log=False):
        torch = self.torch
        assert p == 'linf' and float(eps) == self.eps
        assert self.eps in (.0045, .0040) and self.alpha == 0.
        grad = self.raw_leaf.grad
        assert grad is not None and bool(grad.isfinite().all())
        with torch.no_grad():
            value = self.raw_leaf.detach().clone()
            selected = self.support
            anchor = self.raw_clean[3][selected]
            proposal = value[selected] - self.step_size * grad[selected]
            radius = eps * anchor.abs()
            value[selected] = torch.maximum(torch.minimum(proposal, anchor + radius), anchor - radius)
            self.raw_value, self.last_raw_gradient = value, grad.detach().clone()
        return x.detach()
