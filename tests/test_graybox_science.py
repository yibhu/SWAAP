"""CPU-only comparisons with frozen executed-source definitions."""
from copy import deepcopy
import torch
from swaap.graybox.science import CachedReferenceRawSWAAP, RefinementSWAAP
from swaap.graybox._reference_cached import CachedReferenceRawSWAAP as ExecutedCached
from swaap.graybox.pipeline import SingleModel,TaskModel


def fixture():
    torch.manual_seed(12)
    class Raw(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Linear(3,2)
        def encode(self,x,task):return self.encoder(x)
    raw=Raw().requires_grad_(False)
    model=SingleModel(raw,torch.device('cpu'))
    dynamics=torch.nn.Linear(3,2);target=deepcopy(dynamics)
    with torch.no_grad():target.weight.add_(.1)
    x=torch.randn(5000,3);x[0,0]=0
    data=(x,torch.randn(5000,1),torch.randn(5000,1),torch.randn(5000,3),torch.zeros(5000,1))
    ref=(torch.randn(19,2),torch.randn(19,1),torch.zeros(19,1),torch.zeros(19,2),torch.zeros(19,1))
    return model,dynamics,target,data,ref


def test_cached_full_buffer_loss_gradient_and_update_matches_executed_source():
    model,dynamics,target,data,ref=fixture()
    a=CachedReferenceRawSWAAP(model,dynamics,target,ref,data,alpha=.1,epsilon=1,step_size=1)
    b=ExecutedCached(model,deepcopy(dynamics),target,ref,data,alpha=.1,epsilon=1,step_size=1)
    assert torch.equal(a.support,b.support)
    ar,br=a.step_raw(0),b.step_raw(0)
    assert ar==br
    assert torch.equal(a.last_raw_gradient,b.last_raw_gradient)
    assert torch.equal(a.raw_value,b.raw_value)
    mask=torch.ones(5000,dtype=torch.bool);mask[a.support]=False
    assert torch.equal(a.raw_value[mask],data[3][mask])
    assert torch.equal(a.model.raw.encoder.weight,model.raw.encoder.weight)


def test_final_relative_projector_has_no_absolute_floor_or_raw_clamp():
    model,dynamics,target,data,ref=fixture()
    a=RefinementSWAAP(model,dynamics,target,ref,data,alpha=0,epsilon=.0045,step_size=1)
    a.raw_leaf=data[3].clone().requires_grad_(True);a.raw_leaf.grad=torch.ones_like(a.raw_leaf)*1e4
    a.raw_clean[3][a.support[0],0]=0;a.raw_leaf.data[a.support[0],0]=0
    a.project_lp(torch.zeros(1),torch.zeros(1),.0045)
    expected=data[3].clone();expected[a.support]=data[3][a.support]-.0045*data[3][a.support].abs()
    assert torch.equal(a.raw_value,expected)
    assert a.raw_value[a.support[0],0]==0
    assert bool((a.raw_value<0).any())


def test_mt80_uses_its_encoder_and_pads_only_action_tail():
    class Raw:
        _dynamics=object()
        def encode(self,x,index):assert index==68;return x+10
        def next(self,z,a,index):
            assert index==68 and a.shape[-1]==6
            assert torch.equal(a[...,4:],torch.zeros_like(a[...,4:]))
            return z
    m=TaskModel(Raw(),torch.device('cpu'),68)
    assert torch.equal(m.encode(torch.ones(2,3)),torch.nn.functional.pad(torch.ones(2,3),(0,36))+10)
    assert torch.equal(m.next(torch.ones(2,3),torch.ones(2,4),m.raw._dynamics),torch.ones(2,3))


def test_transfer_aggregation_pairs_roots_and_does_not_promote_episodes(tmp_path):
    import json
    from swaap.graybox.pipeline import aggregate
    roots=[{'environment_seed':1,'planner_seed':2},{'environment_seed':3,'planner_seed':4}]
    clean={'task':'mw-soccer','episodes':2,'returns':[100,200],'roots':roots}
    poison={'task':'mw-soccer','episodes':2,'returns':[50,100],'roots':roots}
    c=tmp_path/'clean.json';c.write_text(json.dumps(clean))
    p=tmp_path/'attack.json';p.write_text(json.dumps(poison))
    r=aggregate({'output_dir':str(tmp_path/'summary'),'assets':{'evaluation':str(p),'clean_evaluation':str(c)}})
    assert r['independent_attack_count']==1 and r['reduction_pct']==50
    assert r['paired_return_harm']==[50,100]


def test_final_native_target_packet_matches_executed_loader(tmp_path):
    from types import SimpleNamespace
    from swaap.graybox.pipeline import _target
    from swaap.graybox._reference_target import target_model
    torch.manual_seed(7)
    original=SimpleNamespace(model=torch.nn.Module())
    original.model._dynamics=torch.nn.Linear(3,2)
    target=deepcopy(original.model._dynamics)
    with torch.no_grad():target.weight.add_(.2)
    path=tmp_path/'native_target.pt'
    torch.save({'psi_k':target.state_dict(),'theta_k':{'unused_actor':torch.ones(1)}},path)
    item={'checkpoint':path.name,'checkpoint_format':'psi_k_theta_k','id':'fixture'}
    executed=target_model(tmp_path,item,original,torch)
    public=_target({'assets':{'target':str(path)}},original)
    assert all(torch.equal(v,executed.state_dict()[k]) for k,v in public.state_dict().items())
    assert not any(p.requires_grad for p in public.parameters())


def test_frozen_guard_preserves_tensordict_structural_metadata():
    from swaap.graybox.pipeline import _frozen_state,_check_frozen_state
    import pytest
    class Model:
        def __init__(self):self.size=torch.Size([5]);self.fixed=torch.ones(2)
        def state_dict(self):return {'_Qs.__batch_size':self.size,'_Qs.__device':torch.device('cuda:0'),'_encoder.weight':self.fixed,'_dynamics.weight':torch.randn(2)}
    model=Model();before=_frozen_state(model);_check_frozen_state(model,before)
    model.fixed.add_(1)
    with pytest.raises(ValueError,match='Non-dynamics'):_check_frozen_state(model,before)


def test_mt80_rollout_adapter_batches_flat_latent_and_action_like_source():
    class Raw:
        _dynamics=object()
        def next(self,z,a,index):
            assert index==68 and z.shape==(1,512) and a.shape==(1,6)
            assert torch.equal(a[:,4:],torch.zeros(1,2))
            return z+1
    model=TaskModel(Raw(),torch.device('cpu'),68)
    result=model.next(torch.zeros(512),torch.ones(4),model.raw._dynamics)
    assert result.shape==(512,) and torch.equal(result,torch.ones(512))
