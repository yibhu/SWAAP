from .science import RawObservationSWAAP
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
