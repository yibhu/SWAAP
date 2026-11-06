"""Unmodified executed numerical method reference; CPU parity fixtures only."""
import torch
import torch.nn as nn
from .stage2 import print_stats
class ExecutedStage2Reference:
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
