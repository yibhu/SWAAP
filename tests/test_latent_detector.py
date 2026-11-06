"""Detector adapter schema/reference checks on CPU fixtures, without simulators."""
from types import SimpleNamespace
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
import torch
from swaap.latent import detector as D
from swaap.evaluation.detector import load,disjoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5))
    def next(self,z,a,dynamics=None):return z*self.weight


def fake_agent():
    model=Model()
    return SimpleNamespace(device=torch.device('cpu'),model=model,psi_0=object(),theta_0=object(),cfg=SimpleNamespace(),env=SimpleNamespace(close=lambda:None))


def roots(count):
    return [dict(episode_index=i,environment_seed=100+i,planner_seed=200+i)for i in range(count)]


def write_roots(tmp_path,count):
    p=tmp_path/'roots.json';p.write_text(json.dumps(dict(roots=roots(count))))
    return p


def test_role_independent_ids_detect_overlap():
    a=D.trajectory_id('humanoid-walk','a'*64,roots(1)[0])
    b=D.trajectory_id('humanoid-walk','a'*64,dict(roots(1)[0],role='threshold'))
    assert a==b
    with pytest.raises(ValueError,match='disjoint'):disjoint([a],[b])


def test_reference_raw_l2_preserves_full_episode_float32():
    agent=fake_agent();z=torch.ones(500,2);a=torch.zeros(500,1)
    data=(z,a,torch.zeros(500,1),z*2,torch.zeros(500,1))
    actual=D.residual_episode(data,agent,family='tdmpc2',expected_steps=500)
    expected=(data[3]-agent.model.next(z,a)).norm(p=2,dim=1).detach().numpy()
    np.testing.assert_array_equal(actual,expected)
    assert actual.dtype==np.float32


def test_collector_wires_original_and_outputs_detector_schema(tmp_path,monkeypatch):
    from swaap.latent import runtime,native
    agent=fake_agent();calls=[]
    monkeypatch.setattr(runtime,'load_agent',lambda config:agent)
    def rollout(**kwargs):
        calls.append(kwargs)
        return dict(latents=np.ones((501,2),np.float32),actions=np.zeros((500,1),np.float32),
            rewards=np.ones(500,np.float32),dones=np.zeros(500,dtype=bool)),dict(return_value=500.,steps=500)
    monkeypatch.setattr(native,'rollout_episode',rollout)
    original=tmp_path/'original.pt';original.write_bytes(b'fixture')
    cfg=dict(task='humanoid-walk',family='tdmpc2',output_dir=str(tmp_path/'out'),
        assets={'original':str(original),'roots':str(write_roots(tmp_path,2))},detector={'role':'smoke'})
    result=D.collect_original(cfg);res=load(result['residuals'])
    assert res['residuals'].shape==(2,500) and result['smoke_only']
    assert len(calls)==2 and all(c['dynamics'] is agent.psi_0 for c in calls)
    assert result['new_environment_episodes']==2 and result['new_fits']==0


def test_incoming_scores_all_trajectories_after_lock(tmp_path,monkeypatch):
    from swaap.latent import runtime
    agent=fake_agent();monkeypatch.setattr(runtime,'load_agent',lambda config,environment=False:agent)
    original=tmp_path/'original.pt';original.write_bytes(b'fixture');sha=hashlib.sha256(b'fixture').hexdigest()
    from swaap.evaluation.detector import calibrate
    rng=np.random.default_rng(42)
    def cohort(name,n):
        path=tmp_path/(name+'.npz')
        np.savez_compressed(path,residuals=rng.uniform(.9,1.1,(n,500)).astype(np.float32),
            trajectory_ids=np.asarray([name+str(i)for i in range(n)]),original_sha256=np.asarray(sha),
            task=np.asarray('humanoid-walk'))
        return path
    lock=tmp_path/'lock.json'
    calibrate(cohort('statistics',500),cohort('threshold',500),lock,family='tdmpc2',task='humanoid-walk')
    clean_test=cohort('clean_test',1000)
    clean=(torch.ones(5000,2),torch.zeros(5000,1),torch.zeros(5000,1),torch.ones(5000,2),torch.zeros(5000,1))
    poison=tuple(x.clone()for x in clean);poison[3][:500]+=.1
    cp=tmp_path/'clean.pt';pp=tmp_path/'poison.pt';torch.save(clean,cp);torch.save(poison,pp)
    cfg=dict(task='humanoid-walk',family='tdmpc2',output_dir=str(tmp_path/'out'),assets=dict(original=str(original),
        calibration=str(lock),clean_test=str(clean_test),trajectory_roots=str(write_roots(tmp_path,10)),clean=str(cp),poison=str(pp)))
    result=D.score_buffer(cfg)
    incoming=load(result['poison_residuals']);matched=load(result['matched_source_clean_residuals'])
    assert incoming['residuals'].shape==(10,500) and result['all_stored_trajectories_scored']
    assert np.array_equal(incoming['residuals'][1:],matched['residuals'][1:])
    assert result['new_environment_episodes']==0
    assert result['cusum']['num_clean_test_episodes']==1000 and Path(result['cusum_file']).is_file()


def test_model_only_geometry_comes_from_original_weights(tmp_path):
    from swaap.latent.runtime import checkpoint_geometry
    path=tmp_path/'original.pt'
    torch.save(dict(model={'_encoder.state.0.weight':torch.zeros(256,67),'_dynamics.0.weight':torch.zeros(512,533)}),path)
    assert checkpoint_geometry(path)==dict(obs_shape={'state':[67]},action_dim=21)


def test_frozen_snapshot_handles_tensordict_structural_metadata():
    from types import SimpleNamespace
    from swaap.latent.pipeline import _state_snapshot,_unchanged
    state={'weight':torch.ones(2),'params.__batch_size':torch.Size([5]),'params.__device':None}
    agent=SimpleNamespace(model=SimpleNamespace(state_dict=lambda:state))
    before=_state_snapshot(agent);_unchanged(agent,before)
    state['params.__batch_size']=torch.Size([6])
    with pytest.raises(ValueError,match='frozen'):_unchanged(agent,before)


def test_explicit_fresh_collection_handoff_without_historical_claim(tmp_path,monkeypatch):
    from swaap.latent import pipeline,runtime,collection
    agent=fake_agent();agent.cfg=SimpleNamespace(task='mw-push');observed=[]
    # The test uses the real native-call boundary orchestration but no simulator.
    class Boundary:
        def __init__(self,*a):self.active=None
        def arm(self,r,role,offset,*,use_environment):self.active=dict(r,role=role,uses_environment=use_environment,environment_steps=100 if use_environment else 0)
        def finish(self,horizon):return dict(self.active,rows=horizon)
        def close_scope(self):pass
    def collect(**kw):
        observed.append(kw)
        return (torch.ones(100,2),torch.zeros(100,1),torch.zeros(100,1),torch.ones(100,2),torch.zeros(100,1))
    agent.collect=collect
    monkeypatch.setattr(runtime,'load_agent',lambda c:agent)
    monkeypatch.setattr(collection,'ExplicitRootCollectionBoundary',Boundary)
    panel=tmp_path/'panels.json';panel.write_text(json.dumps(dict(clean_roots=roots(1),reference_roots=[dict(episode_index=0,environment_seed=500,planner_seed=600)])))
    original=tmp_path/'original.pt';original.write_bytes(b'original')
    cfg=dict(task='mw-push',output_dir=str(tmp_path/'out'),family='tdmpc2',
        assets=dict(original=str(original),collection_roots=str(panel)),
        data=dict(mode='explicit_roots',rows=100,reference_rows=100,reference_use_environment=False,
            clean_random_action_chance=.03,reference_random_action_chance=.03))
    result=pipeline.prepare_data(cfg)
    manifest=json.loads(Path(result['files']['clean_trajectory_roots']).read_text())
    assert Path(result['files']['clean_trajectory_roots']).name=='clean_trajectory_roots.json'
    assert manifest['roots']==roots(1) and manifest['clean_bank_sha256']==hashlib.sha256(Path(result['files']['clean']).read_bytes()).hexdigest()
    assert result['new_environment_episodes']==1 and result['new_environment_steps']==100
    assert not result['reproduces_historical_bank_bytes'] and len(observed)==2
    assert all(x['psi']is agent.psi_0 and x['batch_size']==1 for x in observed)


def test_explicit_preparation_rejects_root_overlap_before_loading():
    from swaap.latent.collection import validate_collection_panels
    with pytest.raises(ValueError,match='disjoint'):
        validate_collection_panels(dict(clean_roots=roots(1),reference_roots=roots(1)),100,100,100)
