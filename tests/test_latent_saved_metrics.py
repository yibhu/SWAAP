"""CPU-only saved-array metrics and a synthetic Original-forward kernel."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from swaap.latent.deviation import episode_delta_m,buffer_delta_d
from swaap.latent.saved_metrics import original_on_matched_clean,predict_original_episode


def manifest(tmp_path,include_original=True):
    records=[]
    for i in range(2):
        states=np.ones((4,2),dtype=np.float32)*(i+1)
        values=dict(latents=states,predicted_next_latents=states[1:].copy())
        if include_original:values['original_predicted_next_latents']=np.zeros((3,2),dtype=np.float32)
        path=tmp_path/f'episode{i}.npz';np.savez(path,**values)
        records.append(dict(task='synthetic',stage1_run='run1',fit_order='order1',episode_index=i,
            environment_seed=i,planner_seed=i+20,expected_steps=3,predictor_role='Original',rollout_role='matched Clean',
            original_checkpoint_sha256='a'*64,clean_fitted_checkpoint_sha256='b'*64,arrays=path.name))
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(episodes=records)))
    return dict(output_dir=str(tmp_path/'out'),assets={'saved_rollouts':str(p)},evaluation={'episodes':2},aggregation={'fit_orders':1})


def test_original_on_clean_uses_original_saved_predictions_not_own(tmp_path):
    config=manifest(tmp_path)
    result=original_on_matched_clean(config)
    expected=episode_delta_m(np.ones((4,2),np.float32),np.zeros((3,2),np.float32),expected_steps=3)
    assert abs(result['episodes'][0]['delta_m_ref']-expected)<1e-15
    assert result['tasks'][0]['mean']>.99
    assert result['new_model_forwards']==result['new_rollouts']==0


def test_missing_original_predictions_never_substitute_own(tmp_path):
    with pytest.raises(ValueError,match='MISSING_MEASUREMENT'):
        original_on_matched_clean(manifest(tmp_path,include_original=False))


def test_exact_original_forward_kernel_without_simulator():
    calls=[]
    def predict(z,a,d):
        calls.append((z.dtype,a.dtype,torch.is_grad_enabled(),d))
        return z*.5+a
    original=object();agent=SimpleNamespace(device=torch.device('cpu'),psi_0=original,model=SimpleNamespace(next=predict))
    z=np.ones((4,2),dtype=np.float32);a=np.ones((3,2),dtype=np.float32)*.25
    actual=predict_original_episode(agent,z,a,torch)
    np.testing.assert_array_equal(actual,np.ones((3,2),np.float32)*.75)
    assert calls==[(torch.float32,torch.float32,False,original)]


def test_constructor_delta_d_uses_all5000_original_prediction_denominators():
    from swaap.latent.pipeline import _delta_d
    n=5000;pred=torch.ones(n,2)*2
    labels=pred.clone();labels[:500]+=1
    bank=(torch.zeros(n,2),torch.zeros(n,1),torch.zeros(n,1),labels,torch.zeros(n,1))
    agent=SimpleNamespace(device=torch.device('cpu'),psi_0=object(),model=SimpleNamespace(next=lambda z,a,d:pred))
    expected=buffer_delta_d(labels.numpy(),labels.numpy(),pred.numpy(),expected_rows=n)['poison_mean']
    assert _delta_d(agent,bank)==expected
    assert abs(expected-.05)<1e-8
