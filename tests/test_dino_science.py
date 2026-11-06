"""CPU fixtures; no backbone, policy, GPU or environment is loaded."""
from copy import deepcopy
import torch
from swaap.dino.science import rms_regularizer,cosine_vector,project,latent_next
from swaap.dino import _reference_geometry as executed
from swaap.dino.orders import make_source_slate
from swaap.dino.fitting import update


def test_rms_regularizer_loss_gradient_and_dimension_normalization():
    torch.manual_seed(13)
    c=torch.randn(4,7,9);p=torch.randn_like(c);x=torch.randn_like(c).requires_grad_(True)
    a=rms_regularizer(c,p,x)[0];b=executed.rms_regularizer(c,p,x)[0]
    assert torch.equal(a,b)
    ga=torch.autograd.grad(a,x,retain_graph=True)[0];gb=torch.autograd.grad(b,x)[0]
    assert torch.equal(ga,gb)
    # Repeating every dimension leaves the RMS statistic unchanged, unlike L2.
    repeated=rms_regularizer(c.repeat(1,1,2),p.repeat(1,1,2),x.repeat(1,1,2))[0]
    assert torch.allclose(a,repeated)


def test_cosine_derivative_matches_source_and_direct_autograd():
    t=torch.tensor([1.,-2.,3.]);g=torch.tensor([3.,4.,-1.])
    a=cosine_vector(t,g);b=executed.cosine_vector(t,g)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    v=g.clone().requires_grad_(True);loss=1-torch.dot(t,v)/(t.norm()*v.norm())
    assert torch.equal(a[2],torch.autograd.grad(loss,v)[0])


def test_quantization_aware_projection_matches_source_and_consumed_box():
    torch.manual_seed(4);clean=torch.randn(3,7,394)*40
    proposal=clean+torch.randn_like(clean)*100
    actual=project(proposal,clean);expected=executed.project(proposal,clean)
    assert torch.equal(actual,expected)
    assert float((actual.double()-clean.double()).abs().max())<=1
    assert float((actual.bfloat16().double()-clean.double()).abs().max())<=1
    assert torch.equal(project(actual,clean),actual)


def test_native_orders_restore_rng_and_cover_every_row():
    torch.manual_seed(145);state=torch.get_rng_state().clone()
    a=make_source_slate(torch,99);b=make_source_slate(torch,99)
    assert a==b and torch.equal(torch.get_rng_state(),state)
    assert len(a)==10 and all(sorted(e)==list(range(320)) for e in a)
    assert a!=make_source_slate(torch,100)


def test_history_action_replacement_and_one_adamw_update_frozen_fields():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.predictor=torch.nn.Linear(6,6);self.encoder=torch.nn.Linear(6,6).requires_grad_(False)
        def replace_actions_from_z(self,z,a):
            z=z.clone();z[...,-2:]=a[:,:,None,:];return z
        def predict(self,z):return self.predictor(z)
    torch.manual_seed(3);a=Model();b=deepcopy(a)
    z=torch.randn(2,3,4,6);actions=torch.randn(2,2);labels=torch.randn(2,4,6)
    before=z.clone();frozen=deepcopy(a.encoder.state_dict())
    oa=torch.optim.AdamW(a.predictor.parameters(),lr=2e-5);ob=torch.optim.AdamW(b.predictor.parameters(),lr=2e-5)
    public=update(a,z,actions,labels,oa)
    # Literal source fit core against its actual final-frame full-feature MSE.
    ob.zero_grad();prediction=latent_next(b,z,actions);reference=torch.nn.functional.mse_loss(prediction,labels);reference.backward();ob.step()
    assert torch.equal(public,reference)
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    assert all(torch.equal(frozen[k],v) for k,v in a.encoder.state_dict().items())
    assert torch.equal(z,before)


def test_goal_aggregation_preserves_nested_statistical_unit(tmp_path):
    import json
    from swaap.dino.pipeline import aggregate
    paths=[]
    for j,flags in enumerate(([1,0],[1,1],[0,0])):
        p=tmp_path/('order'+str(j)+'.json')
        p.write_text(json.dumps({'records':[{'pair_identity':str(i),'success':bool(v),'state_dist':i+1} for i,v in enumerate(flags)]}))
        paths.append(str(p))
    result=aggregate({'output_dir':str(tmp_path/'summary'),'result_files':paths})
    assert result['independent_attack_count']==1 and result['fitting_runs']==3
    assert result['mean_success_fraction']==.5
    assert result['sample_SD_over_nested_fit_runs']==.5


def test_paper_native_fp32_deviation_uses_no_bf16_cast_or_action_features():
    from swaap.dino.metrics import normalized_deviation,residuals_from_predictions
    import pytest
    prediction=torch.ones(8,196,404,dtype=torch.float32)*2
    labels=prediction.clone();labels[...,:394]+=1e-5;labels[...,394:]=1000
    actual=normalized_deviation(labels,prediction)
    assert actual['mean']>0  # BF16 would erase this perturbation.
    expected=((labels[...,:394]-prediction[...,:394]).flatten(1).norm(dim=1)/(prediction[...,:394].flatten(1).norm(dim=1)+1e-8)).mean()
    assert actual['mean']==float(expected)
    origins=[{'record_index':13,'local_window':i,'last_current_grouped_step':i+2,'successor_grouped_step':i+3} for i in range(8)]
    residuals,ids=residuals_from_predictions(labels,prediction,origins)
    assert residuals.shape==(1,8) and ids==[13]
    assert float(residuals.max())<.01
    with pytest.raises(AssertionError,match='Canonical FP32'):
        normalized_deviation(labels.bfloat16(),prediction)


def test_direct_predictor_replacement_preserves_mixed_original_modes_and_flags():
    from types import SimpleNamespace
    from swaap.dino.direct import predictor_replacement
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.predictor=torch.nn.Module()
            self.predictor.register_parameter('weight',torch.nn.Parameter(torch.ones(1),requires_grad=False))
            for i in range(68):self.predictor.register_buffer('buffer'+str(i),torch.ones(1))
            self.predictor.dropout=torch.nn.Dropout(.1)
            self.encoder=torch.nn.Linear(1,1,bias=False).requires_grad_(False)
            for i in range(225):self.register_buffer('fixed'+str(i),torch.ones(1))
    model=Model();model.train();model.predictor.dropout.eval()
    target=deepcopy(model.predictor.state_dict());target['weight']+=.1
    modes={k:m.training for k,m in model.named_modules()}
    result,proof=predictor_replacement(SimpleNamespace(torch=torch,original=model),target)
    assert {k:m.training for k,m in result.named_modules()}==modes
    assert proof['nonpredictor_state_tensors_exact']==226
    assert proof['BF16_conversion'] is False and proof['optimizer_updates']==0
    assert all(torch.equal(result.state_dict()[k],v) for k,v in model.state_dict().items() if not k.startswith('predictor.'))


def test_retained_world_model_eval_none_return_does_not_erase_model_reference():
    from swaap.dino.pipeline import _eval_frozen
    class RetainedEvalModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.predictor=torch.nn.Linear(2,2)
        def eval(self):
            # The executed VWorldModel implements mode changes without returning self.
            super().eval()
    model=RetainedEvalModel()
    result=_eval_frozen(model)
    assert result is model and not result.predictor.training
    assert not any(p.requires_grad for p in model.parameters())
