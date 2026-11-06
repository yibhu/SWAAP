"""Exact online latent-PGD method copied from notebook cell4, lines309–382.

Original notebook SHA256:
079667cad73eb0b4572496939b60a1e03336006bfa9bc3b9c5a27fa465fa221f
Method normalized AST SHA256:
70b8d51962ba2f7bf9f6aeadbdc1b4dcba38b221f434b45773a92311c7f0bb02
The worker passes epsilon .01/.02 explicitly; all other method semantics remain.
"""
import torch

def pgd_latent_low_value(
        self,
        z,
        theta=None,
        num_steps=10,
        epsilon=0.02,
        step_size=1,
        norm="linf",
    ):
        """
        Adversarially perturb a single latent state z at test time
        to minimize Q(z, pi(z)).
        
        z: shape [latent_dim] or [1, latent_dim]
        returns: perturbed z with same shape
        """
        if theta is None:
            theta = self.theta
    
        single = (z.dim() == 1)
        if single:
            z = z.unsqueeze(0)   # [1, d]
    
        z0 = z.detach().clone()
        z_adv = z0.clone()
        best_z = z0.clone()
        best_q = float("inf")
    
        # IMPORTANT: PGD needs gradients even if caller is in no_grad
        with torch.enable_grad():
            for _ in range(num_steps):
                z_adv.requires_grad_(True)
    
                a_adv, _ = self.model.pi(z_adv, theta)
                q_adv = self.model.Q(z_adv, a_adv, return_type='avg')  # [1] or scalar
                q_scalar = q_adv.mean()
    
                grad = torch.autograd.grad(
                    q_scalar, z_adv, retain_graph=False, create_graph=False
                )[0]
    
                with torch.no_grad():
                    if norm == "linf":
                        z_adv = z_adv - step_size * grad.sign()
                        delta = (z_adv - z0).clamp(-epsilon, epsilon)
                        z_adv = z0 + delta
    
                    elif norm == "l2":
                        g_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        g_unit = grad / g_norm
                        z_adv = z_adv - step_size * g_unit
    
                        delta = z_adv - z0
                        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        factor = torch.clamp(epsilon / delta_norm, max=1.0)
                        delta = delta * factor
                        z_adv = z0 + delta
    
                    else:
                        raise ValueError(f"Unsupported norm: {norm}")
    
                    # evaluate after projection
                    a_eval, _ = self.model.pi(z_adv, theta)
                    q_eval = self.model.Q(z_adv, a_eval, return_type='avg').mean().item()
    
                    if q_eval < best_q:
                        best_q = q_eval
                        best_z = z_adv.clone()
    
                z_adv = z_adv.detach()
    
        if single:
            return best_z.squeeze(0).detach()
        return best_z.detach()
