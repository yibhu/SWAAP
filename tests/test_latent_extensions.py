"""Bounded CPU checks for retained controls; no simulator/model assets used."""
from copy import deepcopy
from types import SimpleNamespace
import json
import math
import numpy as np
import torch
from swaap.latent.fixed_support import FixedSupportSettings,construct_fixed_support,construct_sequential_support
from swaap.latent import tied_trajectory as tied
from swaap.latent import fixed_support as C
from swaap.latent.online_pgd import pgd_latent_low_value


class D(torch.nn.Module):
    def __init__(self):
        super().__init__();self.f=torch.nn.Linear(5,3)
    def forward(self,z,a):return torch.sigmoid(self.f(torch.cat((z,a),-1)))


def data_fixture():
    torch.manual_seed(8)
    d=D();target=deepcopy(d)
    with torch.no_grad():target.f.weight.add_(.1)
    z=torch.rand(18,3);a=torch.rand(18,2)
    nz=z.roll(-1,0).clone()
    data=(z,a,torch.zeros(18,1),nz,torch.zeros(18,1))
    model=SimpleNamespace(device=torch.device('cpu'),next=lambda z,a,d:d(z,a))
    system=SimpleNamespace(model=model,device=torch.device('cpu'),psi_0=d)
    return system,d,target,data


def test_fixed_support_and_legacy_sequential_keep_exact_budgets():
    system,d,target,data=data_fixture();support=(1,4,7)
    settings=FixedSupportSettings(rows=18,support_rows=3,projected_steps=2,step_size=10.,alpha=.9,epsilon=1.,construction_seed=0)
    targets=target(data[0],data[1]).detach()
    ordinary,meta=construct_fixed_support(system=system,original_dynamics=d,reference_buffer=data,clean_buffer=data,
        target_next_latents=targets,ordered_support=support,settings=settings,condition='swaap_fixed_support',input_identities={'fixture':'synthetic'})
    seq,sm=construct_sequential_support(system=system,original_dynamics=d,reference_buffer=data,clean_buffer=data,
        target_next_latents=targets,ordered_support=support,settings=settings,episode_end_indices=(17,),input_identities={'fixture':'synthetic'})
    assert meta['projected_steps_completed']==sm['projected_steps_completed']==2
    assert sm['stored_state_field_writes']==6
    assert torch.equal(seq[0][[i+1 for i in support]],seq[3][list(support)])
    assert all(torch.equal(seq[i],data[i]) for i in (1,2,4))
    assert all(torch.equal(ordinary[i],data[i]) for i in (0,1,2,4))


def test_exact_tied_objective_is_actual_full_buffer_victim_gradient():
    system,d,target,data=data_fixture();support=(1,4,7)
    u=data[3][list(support)].clone().requires_grad_(True)
    target_reference=target(data[0],data[1]).detach()
    parts=tied.objective(system.model,d,target_reference,data,data,u,support,torch,.9)
    current=data[0].index_copy(0,torch.tensor(support)+1,u)
    successor=data[3].index_copy(0,torch.tensor(support),u)
    mse=torch.nn.functional.mse_loss(system.model.next(current,data[1],d),successor)
    expected=torch.cat([g.reshape(-1) for g in torch.autograd.grad(mse,d.parameters(),create_graph=True)])
    torch.testing.assert_close(parts['gp'],expected,rtol=0,atol=0)
    desired=(target(data[0][list(support)],data[1][list(support)])-d(data[0][list(support)],data[1][list(support)])).detach()
    trace=tied.tied_vjp_probe(parts,u,support,desired,torch,.9)
    assert trace['separate_consistency_surrogate'] is False
    assert trace['full_buffer_gather_vjp_max_abs_error']<1e-6
    settings=FixedSupportSettings(rows=18,support_rows=3,projected_steps=2,step_size=10.,alpha=.9,epsilon=1.,construction_seed=0)
    poison,metadata=tied.construct(C,system,d,target,data,data,support,'EXACT_TIED_TRAJECTORY',None,settings,torch,np)
    assert metadata['projected_steps_completed']==2
    assert torch.equal(poison[0][torch.tensor(support)+1],poison[3][list(support)])


def test_exact_online_pgd_is_iterative_and_respects_budget():
    class Model:
        def pi(self,z,theta):return z[:,:2],{}
        def Q(self,z,a,return_type):return z.square().sum(-1)+a.square().sum(-1)
    agent=SimpleNamespace(model=Model(),theta=None)
    z=torch.tensor([.2,.3,.4])
    with torch.no_grad():out=pgd_latent_low_value(agent,z,num_steps=10,epsilon=.02,step_size=1,norm='linf')
    torch.testing.assert_close(out,z-.02,rtol=0,atol=1e-7)
    assert (out-z).abs().max()<=.020001


def test_retained_trim_is_matched_epochs_without_warmup():
    from swaap.latent.trim import fit_world_model,TrimConfig
    system,d,target,data=data_fixture()
    before=deepcopy(d.state_dict())
    result=fit_world_model(system=system,data=data,config=TrimConfig(mode='corrected_matched_epochs',num_epochs=2,
        batch_size=256,learning_rate=1e-4,beta=.2,max_iters=2,device='cpu'),trim=True,seed=3)
    assert result.completed_epochs==2 and result.optimizer_steps==2
    assert all(torch.equal(v,d.state_dict()[k]) for k,v in before.items())


def test_buffer_native_dimension_parameterization_and_aggregation(tmp_path):
    from swaap.latent.buffer_fitting import fit_buffer_orders
    from swaap.latent.pipeline import aggregate
    system,d,target,data=data_fixture()
    large=tuple(x.repeat((556,)+(1,)*(x.ndim-1))[:10000].clone() for x in data)
    schedule={'permutations':[torch.arange(10000) for _ in range(10)]}
    optimizer=torch.optim.Adam(d.parameters(),lr=1e-4)
    history,steps=fit_buffer_orders(system,d,optimizer,large,schedule,torch,10000)
    assert steps==400 and len(history)==10
    rows=[dict(task='synthetic',method='swaap',stage1_run=run,fit_order=order,returns=[base+order,base+order+2])
          for run,base in [('one',0),('two',10)] for order in (1,2,3)]
    inputs=tmp_path/'input.json';inputs.write_text(json.dumps(rows))
    result=aggregate(dict(output_dir=str(tmp_path/'out'),assets={'results':str(inputs)}))['results'][0]
    assert result['mean']==8. and result['independent_stage1_runs']==2
    assert abs(result['sample_sd']-math.sqrt(50))<1e-10


def test_maniskill_fixed_horizon_reward_accumulation_without_episodes():
    from swaap.latent.maniskill import FixedHorizonEnvironment
    class Space:
        shape=(42,)
        def seed(self,x):pass
        def sample(self):return np.zeros(4)
    class Env:
        observation_space=Space();action_space=SimpleNamespace(shape=(4,),seed=lambda x:None)
        def __init__(self):self.steps=0
        def reset(self,**kw):self.steps=0;return np.zeros(42),{}
        def step(self,a):self.steps+=1;return np.zeros(42),1.,True,False,{'success':True}
        def close(self):pass
    env=Env();wrapper=FixedHorizonEnvironment(env,task='lift-cube',seed=1,torch=torch)
    wrapper.reset();total=0.
    for i in range(100):
        _,r,done,info=wrapper.step(torch.zeros(4));total+=float(r)
        assert done==(i==99)
    assert total==200 and env.steps==200 and info['native_terminated'] is True


def test_random_target_generation_preserves_published_cpu_rng_algorithm():
    from swaap.latent.baselines import multiplicative_noise_target
    z=torch.tensor([[.2,.4],[.3,.6]])
    before=torch.get_rng_state().clone()
    actual=multiplicative_noise_target(z,17,.3,torch)
    generator=torch.Generator(device='cpu').manual_seed(17)
    expected=(z*(1+(torch.rand(z.shape,generator=generator)*2-1)*.3)).clamp(0,1)
    assert torch.equal(actual,expected) and torch.equal(before,torch.get_rng_state())


def test_fixed500_append_preserves_poison_bytes_and_all_extra_rows():
    from swaap.latent.buffer_fitting import append_pair
    base=tuple(torch.zeros(5000,1) for _ in range(5))
    extra=tuple(torch.ones(20000,1) for _ in range(5))
    poison=tuple(x.clone() for x in base);support=torch.arange(500,dtype=torch.int64)
    poison[3][support]=.25
    clean,out=append_pair(base,extra,poison,support,25000,torch)
    assert all(torch.equal(out[i][:5000],poison[i]) for i in range(5))
    assert all(torch.equal(out[i][5000:],clean[i][5000:]) for i in range(5))
    assert int((out[3]!=clean[3]).any(1).sum())==500
