"""Executed bridge methods for CPU parity tests only; no runtime loader."""
import torch
import torch.nn.functional as F
TensorDict = dict
def print_cuda_stats(ignore=False):
    return None
class ExecutedStage1Reference:
    def backprop_theta(self, batch, theta, task=None, make_update=False):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
        action, info = self.model.pi(zs, theta, task=None)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.q_scale.update(qs[0])
        qs = self.q_scale(qs)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        return_objective = ((self.cfg.entropy_coef * info['scaled_entropy'] + qs).mean(dim=(1, 2)) * rho).mean()
        if make_update:
            loss = -return_objective
            loss.backward()
            torch.nn.utils.clip_grad_norm_(theta.parameters(), self.cfg.grad_clip_norm)
            self.theta_star_optimizer.step()
            self.theta_star_optimizer.zero_grad(set_to_none=True)
            return (return_objective, None)
        grads = torch.autograd.grad(return_objective, theta.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        return (return_objective, flat_grads)
    def backprop_psi(self, batch, task=None):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
            masks = torch.ones_like(reward)
        next_z_pred = self.model.next(zs, action, self.psi)
        next_action, log_prob = self.model.pi(next_z_pred, self.theta, task=None, return_log_prob=True)
        qs = self.model.Q(next_z_pred, next_action, task, return_type='avg', detach=True)
        self.q_scale.update(qs[0])
        qs = self.q_scale(qs)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        critical_qs = qs * masks
        return_objective = (+critical_qs.mean(dim=(1, 2)) * rho).mean()
        grads = torch.autograd.grad(return_objective, self.psi.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        if False:
            print_stats(qs)
            print_stats(critical_qs)
            plot_hists([qs.view(-1), critical_qs.view(-1)], titles=['qs', 'critical_qs'])
        return (return_objective, flat_grads)
    def consistency_loss(self, batch, task=None, make_update=False):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
            masks = torch.ones_like(reward)
        next_z_pred = self.model.next(zs, action, self.psi)
        if make_update:
            next_z_projected = self.project_next_z_pred(next_z_pred, next_z, eps=0.12, norm='l2').detach()
            loss = F.mse_loss(next_z_pred, next_z_projected) + F.mse_loss(next_z_pred, next_z)
        else:
            loss = F.mse_loss(next_z_pred, next_z)
        if make_update:
            self.psi_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.cfg.grad_clip_norm)
            self.psi_optimizer.step()
            self.psi_optimizer.zero_grad(set_to_none=True)
            return (loss.detach(), None)
        grads = torch.autograd.grad(loss, self.psi.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        return (loss, flat_grads)
    def update(self, apply_reachability=False):
        print('collect trajectory from psi_k theta_k')
        data = self.collect(self.psi, self.theta, self.cfg.num_collect_trajs, self.cfg.num_step)
        buffer_kk = self.make_buffer(data)
        self.theta_star.load_state_dict(self.theta.state_dict())
        for _ in range(self.cfg.T_step):
            self.process_batch(buffer_kk, lambda batch: self.backprop_theta(batch, theta=self.theta_star, make_update=True))
        print('collect trajectory from psi_k theta_T')
        data = self.collect(self.psi, self.theta_star, self.cfg.num_collect_trajs, self.cfg.num_step)
        buffer_kT = self.make_buffer(data)
        print('collect trajectory from env theta_k')
        data = self.collect(self.psi_0, self.theta, self.cfg.num_collect_trajs, self.cfg.num_step, use_env=True)
        buffer_0k = self.make_buffer(data)
        print('collect trajectory from env theta_0')
        data = self.collect(self.psi_0, self.theta_0, self.cfg.num_collect_trajs, self.cfg.num_step, use_env=True)
        buffer_gt = self.make_buffer(data)
        psi_losses = []
        theta_losses = []
        lambda_ks = []
        for i in range(self.cfg.update_batch_size):
            loss_psi_kT, grads_psi_kT = self.process_batch(buffer_kT, lambda batch: self.backprop_psi(batch))
            loss_theta_0k, grads_theta_0k = self.process_batch(buffer_0k, lambda batch: self.backprop_theta(batch, self.theta))
            loss_psi_consistency, grads_psi_consistency = self.process_batch(buffer_gt, lambda batch: self.consistency_loss(batch))
            loss_psi_regularization, grads_psi_regularization = self.process_batch(buffer_gt, lambda batch: self.consistency_loss(batch))
            if self.cfg.use_consistency:
                loss_psi_constraint, grads_psi_constraint = (loss_psi_consistency, grads_psi_consistency)
            else:
                loss_psi_constraint, grads_psi_constraint = (loss_psi_regularization, grads_psi_regularization)
            loss_psi_kk, grads_psi_kk = self.process_batch(buffer_kk, lambda batch: self.backprop_psi(batch))
            loss_theta_kk, grads_theta_kk = self.process_batch(buffer_kk, lambda batch: self.backprop_theta(batch, self.theta))
            eta = 0.5
            grads_f = (self.cfg.lambda_coef * grads_psi_constraint, grads_theta_0k)
            grads_psi_kT_minus_grads_psi_kk = grads_psi_kT - grads_psi_kk
            grads_q = (grads_psi_kT_minus_grads_psi_kk, -grads_theta_kk)
            grads_q_norm = grads_q[0].square().sum() + grads_q[1].square().sum()
            dot_product = torch.dot(grads_f[0], grads_q[0]) + torch.dot(grads_f[1], grads_q[1])
            lambda_k = ((eta * grads_q_norm - dot_product) / (grads_q_norm + 1e-08)).detach()
            lambda_k = lambda_k.clamp(min=0.0)
            loss_psi = self.cfg.lambda_coef * loss_psi_constraint + lambda_k * (loss_psi_kT - loss_psi_kk)
            loss_theta = loss_theta_0k - lambda_k * loss_theta_kk
            psi_losses.append(loss_psi)
            theta_losses.append(loss_theta)
            lambda_ks.append(lambda_k)
        print_cuda_stats()
        loss_psi_base = sum(psi_losses) / len(psi_losses)
        loss_psi_total = loss_psi_base
        reachability_metrics = {'reachability_applied': False, 'reachability_ratio': float('nan'), 'reachability_short_rho_f': float('nan'), 'reachability_penalty': 0.0, 'reachability_effective_weight': 0.0, 'reachability_base_grad_norm': float('nan'), 'reachability_constraint_grad_norm': float('nan'), 'reachability_clean_target_mse': float('nan'), 'reachability_poison_target_mse': float('nan')}
        if apply_reachability and bool(self.cfg.reachability_enabled) and (self._reachability_poison_proxy_psi is not None):
            reachability_penalty, current_reachability_metrics = self.reachability_constraint_loss()
            psi_parameters = [parameter for parameter in self.psi.parameters() if parameter.requires_grad]
            base_gradients = torch.autograd.grad(loss_psi_base, psi_parameters, retain_graph=True, create_graph=False, allow_unused=True)
            reachability_gradients = torch.autograd.grad(reachability_penalty, psi_parameters, retain_graph=True, create_graph=False, allow_unused=True)
    
            def gradient_norm(gradients):
                squared_terms = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
                if len(squared_terms) == 0:
                    return torch.zeros((), device=self.device)
                return torch.sqrt(torch.stack(squared_terms).sum())
            base_gradient_norm = gradient_norm(base_gradients)
            reachability_gradient_norm = gradient_norm(reachability_gradients)
            if reachability_gradient_norm.item() > 0.0:
                effective_weight = float(self.cfg.reachability_grad_ratio) * base_gradient_norm / (reachability_gradient_norm + 1e-12)
                effective_weight = effective_weight.clamp(max=float(self.cfg.reachability_max_weight))
            else:
                effective_weight = torch.zeros((), device=self.device)
            loss_psi_total = loss_psi_base + effective_weight.detach() * reachability_penalty
            reachability_metrics.update(current_reachability_metrics)
            reachability_metrics.update({'reachability_applied': True, 'reachability_effective_weight': float(effective_weight.detach().item()), 'reachability_base_grad_norm': float(base_gradient_norm.item()), 'reachability_constraint_grad_norm': float(reachability_gradient_norm.item())})
        self.psi_optimizer.zero_grad(set_to_none=True)
        loss_psi_total.backward()
        torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.cfg.grad_clip_norm)
        self.psi_optimizer.step()
        loss_theta = sum(theta_losses) / len(theta_losses)
        self.theta_optimizer.zero_grad()
        loss_theta.backward()
        torch.nn.utils.clip_grad_norm_(self.theta.parameters(), self.cfg.grad_clip_norm)
        self.theta_optimizer.step()
        print('lambda_k, loss_psi_base, loss_psi_total, loss_theta:', lambda_k.detach().cpu().item(), loss_psi_base.detach().cpu().item(), loss_psi_total.detach().cpu().item(), loss_theta.detach().cpu().item())
        info = TensorDict({'lambda_k': lambda_k.detach().cpu().item(), 'loss_psi': loss_psi_total.detach().cpu().item(), 'loss_psi_base': loss_psi_base.detach().cpu().item(), 'loss_psi_total': loss_psi_total.detach().cpu().item(), 'loss_theta': loss_theta.detach().cpu().item(), 'loss_psi_consistency': loss_psi_consistency.detach().cpu().item(), 'loss_psi_regularization': loss_psi_regularization.detach().cpu().item(), 'reachability_applied': bool(reachability_metrics['reachability_applied']), 'reachability_ratio': float(reachability_metrics['reachability_ratio']), 'reachability_short_rho_f': float(reachability_metrics['reachability_short_rho_f']), 'reachability_penalty': float(reachability_metrics['reachability_penalty']), 'reachability_effective_weight': float(reachability_metrics['reachability_effective_weight']), 'reachability_base_grad_norm': float(reachability_metrics['reachability_base_grad_norm']), 'reachability_constraint_grad_norm': float(reachability_metrics['reachability_constraint_grad_norm']), 'reachability_clean_target_mse': float(reachability_metrics['reachability_clean_target_mse']), 'reachability_poison_target_mse': float(reachability_metrics['reachability_poison_target_mse'])})
        return info
