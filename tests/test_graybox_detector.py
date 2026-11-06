import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def test_breadth_rejects_construction_and_fitting_before_model_loading():
    from swaap.graybox.pipeline import construct,fit
    c={'evaluation':{'scope':'breadth'}}
    for function in (construct,fit):
        with pytest.raises(ValueError,match='evaluation-only'):function(c)


def test_detector_collector_keeps_single_task_policy_and_mt80_reference(tmp_path,monkeypatch):
    from swaap.graybox import pipeline,detection
    from swaap.latent import native
    from swaap.evaluation.detector import load
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self._dynamics=torch.nn.Linear(2,2);self._pi=torch.nn.Linear(2,1)
        def encode(self,x,index):
            assert index==68
            return x[:,:2]
        def next(self,z,a,index):
            assert index==68 and a.shape==(100,6)
            assert torch.equal(a[:,4:],torch.zeros(100,2))
            return z+1
    def agent():
        return SimpleNamespace(model=Model(),device=torch.device('cpu'),_prev_mean=torch.zeros(1),cfg=SimpleNamespace())
    source,victim=agent(),agent()
    monkeypatch.setattr(pipeline,'_agent',lambda c,multitask=False:victim if multitask else source)
    observed={}
    def rollout(**kwargs):
        observed.update(kwargs)
        assert kwargs['agent'].raw is source and kwargs['dynamics'] is source.model._dynamics
        return {'observations':np.zeros((101,39),np.float32),'actions':np.zeros((100,4),np.float32),'rewards':np.zeros(100,np.float32)}, {'return':0}
    monkeypatch.setattr(native,'rollout_episode',rollout)
    roots=tmp_path/'roots.json';roots.write_text(json.dumps([{'environment_seed':12,'planner_seed':25}]))
    original=tmp_path/'reference.pt';original.write_bytes(b'exact-reference-fixture')
    output=tmp_path/'collected'
    result=detection.collect({'output_dir':str(output),'assets':{'roots':str(roots),'victim':str(original)},'detector':{'role':'statistics'},'collection':{'maximum_trajectories':1}})
    matrix=load(output/'statistics.npz')
    assert matrix['residuals'].shape==(1,100)
    assert np.allclose(matrix['residuals'],np.sqrt(2))
    assert result['status']=='COLLECTION_SMOKE_ONLY' and result['calibration_frozen'] is False
    assert observed['expected_episode_length']==100


def test_released_cohort_conversion_preserves_values_and_exact_seed_pairs(tmp_path):
    from swaap.graybox.detection import prepare_saved
    from swaap.evaluation.detector import load
    counts={'statistics':300,'threshold':500,'clean_test':1000}
    assets={};roots={};offset=0
    for role,count in counts.items():
        matrix=np.arange(count*100,dtype=np.float32).reshape(count,100)/100
        path=tmp_path/(role+'.npy');np.save(path,matrix)
        assets['detector_'+role]=str(path)
        roots[role]=[{'environment_seed':offset+2*i,'planner_seed':offset+2*i+1} for i in range(count)]
        offset+=2*count
    reference=tmp_path/'original.pt';reference.write_bytes(b'frozen-original')
    manifest=tmp_path/'cohort_roots.json';manifest.write_text(json.dumps(roots))
    assets.update(victim=str(reference),detector_roots=str(manifest))
    result=prepare_saved({'assets':assets,'output_dir':str(tmp_path/'converted')})
    for role in counts:
        packet=load(result[role])
        assert np.array_equal(packet['residuals'],np.load(assets['detector_'+role]))
        assert packet['ids']==[f"{r['environment_seed']}:{r['planner_seed']}" for r in roots[role]]
