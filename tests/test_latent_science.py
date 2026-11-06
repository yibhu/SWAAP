"""CPU comparisons with retained executed-source functions, not toy assertions."""
from copy import deepcopy
from types import SimpleNamespace
import ast
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from swaap.latent.stage2 import DataPoisoner, _load_tensor_runtime
from swaap.latent._reference_stage2 import ExecutedStage2Reference
from swaap.latent import kernels
from swaap.latent._reference_stage1 import ExecutedStage1Reference


class Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
    def forward(self, z, a):
        return torch.sigmoid(self.linear(torch.cat([z, a], dim=-1)))


def fixture():
    torch.manual_seed(71)
    original = Dynamics()
    target = deepcopy(original)
    with torch.no_grad():
        target.linear.weight.add_(.13)
        target.linear.bias.sub_(.07)
    model = SimpleNamespace(device=torch.device('cpu'), next=lambda z,a,d:d(z,a))
    z = torch.rand(17, 3); a = torch.rand(17, 2)
    nz = (original(z,a).detach() + .07*torch.randn(17,3)).clamp(.1,.9)
    bank = (z, a, torch.zeros(17,1), nz, torch.zeros(17,1))
    c = DataPoisoner(SimpleNamespace(model=model), original, target, bank, bank, r_c=.3, alpha=.3)
    return original,target,bank,c


def test_exact_loss_gradients_projection_and_one_update():
    original,target,bank,c = fixture()
    reference = ExecutedStage2Reference()
    reference.__dict__.update(c.__dict__)
    idx = torch.topk(torch.nn.functional.mse_loss(original(bank[0],bank[1]),target(bank[0],bank[1]),reduction='none').mean(1),5).indices.numpy()
    c.poison_transition_orig = [bank[0].clone(),bank[1].clone(),bank[3].clone()]
    reference.poison_transition_orig = deepcopy(c.poison_transition_orig)
    target_transition = [bank[0],bank[1],target(bank[0],bank[1]).detach()]
    p = deepcopy(c.poison_transition_orig);p[2].requires_grad_(True)
    q = deepcopy(c.poison_transition_orig);q[2].requires_grad_(True)
    before = deepcopy(original.state_dict())
    actual = c.gradient_matching_step(original,target_transition,p,idx)
    expected = reference.gradient_matching_step(original,target_transition,q,idx)
    for x,y in zip(actual[0],expected[0]):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(actual[1:],expected[1:]):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    untouched = torch.ones(17,dtype=torch.bool);untouched[idx]=False
    assert torch.equal(actual[0][2][untouched],bank[3][untouched])
    assert torch.equal(actual[0][0],bank[0]) and torch.equal(actual[0][1],bank[1])
    assert all(torch.equal(v,original.state_dict()[k]) for k,v in before.items())
    proposal = bank[3] + 3
    for pnorm in ('linf','l2'):
        torch.testing.assert_close(c.project_lp(proposal,bank[3],.2,pnorm), reference.project_lp(proposal,bank[3],.2,pnorm),rtol=0,atol=0)


def test_top_support_and_poison_constructor_match_native_tie_behavior():
    original,target,bank,c=fixture()
    from swaap.latent.pipeline import _support
    agent=SimpleNamespace(device=torch.device('cpu'),model=c.model,psi_0=original)
    expected=torch.topk(torch.nn.functional.mse_loss(original(bank[0],bank[1]),target(bank[0],bank[1]),reduction='none').mean(1),5).indices.cpu()
    assert torch.equal(_support(agent,target,bank,5),expected)
    # End-to-end synthetic construction executes the public scientific path.
    poison,history=c.learn_poison_data(steps=2)
    assert len(history)==2
    assert all(torch.equal(poison[k],bank[k]) for k in (0,1,2,4))
    outside=torch.ones(17,dtype=torch.bool);outside[expected]=False
    assert torch.equal(poison[3][outside],bank[3][outside])


def test_effective_update_ast_and_policy_binding_are_exact():
    base=Path(kernels.__file__).parent
    public=ast.parse((base/'stage1.py').read_text())
    cls=next(n for n in public.body if isinstance(n,ast.ClassDef) and n.name=='ThetaPsi')
    reference=ast.parse((base/'_reference_stage1.py').read_text())
    ref=next(n for n in reference.body if isinstance(n,ast.ClassDef))
    for name in ('backprop_theta','backprop_psi','consistency_loss','update'):
        a=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in ref.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backprop_psi')
    assert 'self.theta,' in ast.unparse(method) and 'self.theta_star' not in ast.unparse(method)
    update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    assignment=next(n for n in ast.walk(update) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='grads_q_norm' for t in n.targets))
    assert ast.unparse(assignment.value)=='grads_q[0].square().sum() + grads_q[1].square().sum()'


def test_immutable_scale_preserves_stateful_update_and_backward():
    from swaap.latent._vendor.common.scale import RunningScale
    scale=RunningScale.__new__(RunningScale)
    torch.nn.Module.__init__(scale)
    scale.cfg=SimpleNamespace(tau=.3)
    scale.value=torch.tensor([2.])
    scale._percentiles=torch.tensor([5.,95.])
    x=torch.tensor([[2.],[6.]],requires_grad=True)
    y=scale(x)
    scale.update(torch.tensor([[0.],[10.],[20.]]))
    assert scale.value.item()!=2
    y.sum().backward()
    torch.testing.assert_close(x.grad,torch.full_like(x,.5),rtol=0,atol=0)


class ToyStage1:
    def __init__(self):
        self.cfg=SimpleNamespace(num_collect_trajs=1,num_step=3,T_step=2,update_batch_size=2,
            use_consistency=True,lambda_coef=1.,grad_clip_norm=20.,reachability_enabled=False)
        self.device=torch.device('cpu')
        self.psi=torch.nn.Linear(1,1);self.theta=torch.nn.Linear(1,1);self.theta_star=deepcopy(self.theta)
        self.psi_0=deepcopy(self.psi);self.theta_0=deepcopy(self.theta)
        self.psi_optimizer=torch.optim.Adam(self.psi.parameters(),lr=.01)
        self.theta_optimizer=torch.optim.Adam(self.theta.parameters(),lr=.01)
        self.theta_star_optimizer=torch.optim.Adam(self.theta_star.parameters(),lr=.01)
    def collect(self,*args,**kwargs):return torch.tensor([[.2],[.7]])
    def make_buffer(self,x):return x
    def process_batch(self,b,fn):return fn(b)
    def backprop_theta(self,b,theta,make_update=False):
        loss=theta(b).square().mean()
        if make_update:
            (-loss).backward();self.theta_star_optimizer.step();self.theta_star_optimizer.zero_grad(set_to_none=True)
            return loss,None
        return loss,torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,theta.parameters(),retain_graph=True)])
    def backprop_psi(self,b):
        loss=(self.psi(b)-.4).square().mean()
        return loss,torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,self.psi.parameters(),retain_graph=True)])
    def consistency_loss(self,b):
        loss=(self.psi(b)-.8).square().mean()
        return loss,torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,self.psi.parameters(),retain_graph=True)])


def test_full_effective_stage1_update_matches_bridge_and_freezes_original(monkeypatch):
    monkeypatch.setattr(kernels,'TensorDict',dict)
    torch.manual_seed(19);a=ToyStage1()
    torch.manual_seed(19);b=ToyStage1()
    initial=deepcopy(a.psi_0.state_dict());theta0=deepcopy(a.theta_0.state_dict())
    out=kernels.effective_update(a)
    ref=ExecutedStage1Reference.update(b)
    for module in ('psi','theta','theta_star'):
        for p,q in zip(getattr(a,module).parameters(),getattr(b,module).parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
    for key in ('lambda_k','loss_psi','loss_theta','loss_psi_consistency'):
        assert out[key]==ref[key]
    assert all(torch.equal(v,a.psi_0.state_dict()[k]) for k,v in initial.items())
    assert all(torch.equal(v,a.theta_0.state_dict()[k]) for k,v in theta0.items())


def test_native_fit_exact_orders_and_extended_schedule_freeze_model():
    from swaap.latent.native import fit_exact_orders
    from swaap.latent.fitting import continue_exact_orders
    torch.manual_seed(5)
    dynamics=torch.nn.Linear(2,1)
    reference=deepcopy(dynamics)
    model=SimpleNamespace(next=lambda z,a,d:d(torch.cat([z,a],dim=-1)))
    agent=SimpleNamespace(device=torch.device('cpu'),model=model)
    bank=(torch.rand(5000,1),torch.rand(5000,1),torch.zeros(5000,1),torch.rand(5000,1),torch.zeros(5000,1))
    schedule={'permutations':[torch.randperm(5000) for _ in range(10)]}
    a=torch.optim.Adam(dynamics.parameters(),lr=1e-4)
    b=torch.optim.Adam(reference.parameters(),lr=1e-4)
    history,n=fit_exact_orders(agent,dynamics,a,bank,schedule,torch)
    for order in schedule['permutations']:
        for i in range(0,5000,256):
            ix=order[i:i+256]
            loss=torch.nn.functional.mse_loss(model.next(bank[0][ix],bank[1][ix],reference),bank[3][ix])
            b.zero_grad(set_to_none=True);loss.backward();b.step()
    assert n==200 and len(history)==10
    for p,q in zip(dynamics.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    _,n=continue_exact_orders(fit_exact_orders,agent,dynamics,a,bank,schedule,torch,start_updates=200,target_updates=500)
    assert n==500
