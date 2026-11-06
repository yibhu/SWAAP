import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from swaap.artifacts import AssetStore, AssetError
from swaap.config import load_config, resolve_assets
from swaap.evaluation.aggregation import hierarchy, buffer_hierarchy, paired_reduction
from swaap.evaluation.cusum import (robust_stats, cusum_score, binary_auc,
    standardized_partial_auc, split_conformal_threshold)
from swaap.evaluation.detector import disjoint
from swaap.reproduce.saved import main_table,cusum_table,summarize_grid,verify_inputs,per_order_ablations,further_retained_tables

ROOT=Path(__file__).resolve().parents[1]


def test_asset_import_integrity_and_no_overwrite(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'public deterministic fixture\n')
    manifest=tmp_path/'assets.json'
    manifest.write_text(json.dumps({'assets':[dict(id='fixture',filename='model/test.bin',
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),bytes=source.stat().st_size,urls=[])]}))
    store=AssetStore(tmp_path/'cache',manifest)
    with pytest.raises(AssetError,match='Missing'):store.verify('fixture')
    result=store.import_file('fixture',source)
    assert result['status']=='VERIFIED'
    before=store.destination('fixture').read_bytes()
    source.write_bytes(b'wrong')
    with pytest.raises(AssetError):store.import_file('fixture',source)
    assert store.destination('fixture').read_bytes()==before
    assert store.download('fixture')['status']=='VERIFIED'


def test_asset_path_traversal_rejected(tmp_path):
    m=tmp_path/'m.json';m.write_text(json.dumps({'assets':[{'id':'bad','filename':'../escape'}]}))
    with pytest.raises(AssetError,match='Unsafe'):AssetStore(tmp_path/'cache',m).destination('bad')


def test_config_overrides_and_relative_inputs(tmp_path):
    p=tmp_path/'test.yaml';p.write_text('family: tdmpc2\nstage2: {alpha: 0.9}\nassets: {roots: "config:roots.json"}\n')
    cfg=load_config(p,['stage2.alpha=.1','seed=43'])
    cfg,_=resolve_assets(cfg,AssetStore(tmp_path/'artifacts',ROOT/'assets/asset_manifest.json'))
    assert cfg['stage2']['alpha']==.1 and cfg['seed']==43
    assert cfg['assets']['roots']==str(tmp_path/'roots.json')


def test_correct_statistical_unit_and_missing_values():
    rows=[dict(task='a',stage1_run=r,fit_order=f,episode_index=e,value=v)
          for r,v in [('r1',1),('r2',3),('r3',5)] for f in range(3) for e in range(2)]
    got=hierarchy(rows,expected_orders=3,expected_episodes=2)[0]
    assert got['mean']==3 and got['sample_sd']==2 and got['independent_stage1_runs']==3
    with pytest.raises(ValueError,match='Incomplete'):hierarchy(rows[:-1],expected_episodes=2)
    with pytest.raises(ValueError,match='Duplicate'):hierarchy(rows+[rows[0]])
    with pytest.raises(ValueError,match='missing'):hierarchy([dict(rows[0],value='')])
    with pytest.raises(ValueError,match='once per fit'):buffer_hierarchy([
        dict(task='a',stage1_run='one',value=.4),dict(task='a',stage1_run='one',value=.4)])
    assert paired_reduction(100,105)==-5


def test_cusum_pinned_conventions():
    # Hand-computable source-fixture checks, absolute tolerance 1e-12.
    median,mad=robust_stats(np.array([0.,1.,2.]))
    assert (median,mad)==(1.,1.)
    assert cusum_score(np.array([1.,9.,10.,0.]),median=1,mad=1,drift_factor=6)==5
    y=np.array([0,0,1,1]);s=np.array([0.,0.,1.,1.])
    assert binary_auc(y,s)==1 and standardized_partial_auc(y,s,max_fpr=.05)==1
    tied=np.zeros(4)
    assert binary_auc(y,tied)==.5 and standardized_partial_auc(y,tied,max_fpr=.1)==.5
    threshold,rank=split_conformal_threshold(range(500),.05)
    assert rank==476 and threshold==475
    assert np.isinf(split_conformal_threshold([0,1],.05)[0])
    with pytest.raises(ValueError,match='disjoint'):disjoint(['one'],['one'])


def test_saved_inputs_144_manuscript_cells_and_units():
    data=Path(os.environ.get('SWAAP_SAVED_DATA',ROOT/'data/saved'))
    if not (data/'MANIFEST.json').is_file():
        pytest.skip('Install the saved-results bundle and set SWAAP_SAVED_DATA to its results/saved directory.')
    assert len(verify_inputs(data))>=72
    cells=main_table(data)
    assert len(cells)==144
    expected=list(csv.DictReader((ROOT/'provenance/main_table_rounding.csv').open()))
    for q in expected:
        r=next(r for r in cells if all(r[k]==q[k] for k in ('task','method','metric')))
        n=0 if q['metric']=='Return' else 3
        assert round(r['mean'],n)==float(q['displayed_mean']),q
        assert round(r['sample_sd'],n)==float(q['displayed_sample_sd']),q
        assert r['independent_stage1_runs']==(2 if q['task']=='lift-cube' else 3)
    assert len(cusum_table(data))==21
    assert len(summarize_grid(data))==166
    ablations=per_order_ablations(data)
    assert len(ablations['pgd'])==4 and len(ablations['trim'])==3
    assert len(ablations['defense_and_mpc'])==19
    additional=further_retained_tables(data)
    assert len(additional['buffer25k'])==2 and len(additional['trajectory_consistency'])==5
    assert all(r['independent_stage1_runs']==1 for r in additional['matched_deviation_dmp'])


def test_cli_import_is_inert_and_output_protection(tmp_path):
    env=dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    run=subprocess.run([sys.executable,'-c',
        "import sys;import swaap.cli,swaap.reproduce.saved;assert 'torch' not in sys.modules;assert 'gym' not in sys.modules"],
        cwd=tmp_path,capture_output=True,text=True,env=env)
    assert run.returncode==0,run.stderr
    from swaap.reproduce.saved import reproduce
    with pytest.raises(ValueError,match='exists'):reproduce(ROOT/'data/saved',tmp_path)


def test_detector_freeze_and_cohort_separation(tmp_path):
    from swaap.evaluation.detector import calibrate
    original='a'*64
    def fixture(name,count,offset):
        path=tmp_path/(name+'.npz')
        np.savez(path,residuals=np.ones((count,3),dtype=np.float32),
                 trajectory_ids=np.array([str(offset+i) for i in range(count)]),
                 original_sha256=np.array(original))
        return path
    stat=fixture('statistics',300,0);threshold=fixture('threshold',500,300)
    target=tmp_path/'detector.json'
    result=calibrate(stat,threshold,target,'graybox')
    frozen=json.loads(target.read_text())
    assert result['statistics_trajectories']==300 and frozen['drift_factor']==6
    assert frozen['alarm_rule']=='score > threshold' and frozen['original_sha256']==original
    assert frozen['thresholds']['0.05']['rank_one_based']==476
    with pytest.raises(FileExistsError):calibrate(stat,threshold,target,'graybox')
    with pytest.raises(ValueError,match='disjoint'):calibrate(stat,stat,tmp_path/'invalid.json','graybox')


def test_fixed_target_script_handoff_uses_matched_order(tmp_path, monkeypatch):
    import runpy
    from swaap.latent import pipeline
    calls=[]
    def construction(cfg):
        calls.append(('construct',cfg))
        return {'files':{'poison':'constructed-poison.pt'}}
    def fit(cfg):
        calls.append(('fit',cfg))
        return {'checkpoint':str(Path(cfg['output_dir'])/'fitted.pt'),'order_sha256':'same-order'}
    def evaluate(cfg):
        calls.append(('evaluate',cfg))
        return {'episodes':[dict(episode_index=0,environment_seed=1,planner_seed=2,
                                **{'return':2.0 if cfg['method']=='clean' else 1.0})]}
    monkeypatch.setattr(pipeline,'construct',construction)
    monkeypatch.setattr(pipeline,'fit',fit)
    monkeypatch.setattr(pipeline,'evaluate',evaluate)
    entry=runpy.run_path(str(ROOT/'scripts/run_attack.py'))['main']
    output=tmp_path/'new-run'
    inputs=[]
    for key in ['original','clean','reference','target']:
        path=tmp_path/(key+'.pt');path.write_bytes(b'engineering fixture')
        inputs+=['--set','assets.'+key+'='+str(path)]
    assert entry(['--config',str(ROOT/'configs/tdmpc2/humanoid-walk.yaml'),
        '--output',str(output),'--manifest',str(ROOT/'assets/asset_manifest.json')]+inputs)==0
    assert len(calls)==5
    assert calls[2][1]['assets']['order']==str(output/'clean_fit/order.pt')
    assert calls[2][1]['assets']['data']=='constructed-poison.pt'
    assert calls[3][1]['assets']['checkpoint'].endswith('clean_fit/fitted.pt')
    assert calls[4][1]['assets']['checkpoint'].endswith('attack_fit/fitted.pt')
    result=json.loads((output/'result.json').read_text())
    assert result['target_selection_performed'] is False and result['relative_reduction_percent']==50


def test_poison_fit_cannot_silently_fall_back_to_clean(tmp_path):
    pytest.importorskip('torch')
    from swaap.latent.pipeline import fit
    with pytest.raises(ValueError,match='requires assets.data'):
        fit(dict(output_dir=str(tmp_path/'refused'),method='swaap',assets={'poison':'some-poison.pt'}))


def test_development_and_final_order_bindings_stay_disjoint(tmp_path):
    import runpy
    entry=runpy.run_path(str(ROOT/'scripts/bind_order.py'))['main']
    for order in ['dev1','hold1']:
        assert entry(['--config',str(ROOT/'configs/tdmpc2/myo-pen-twirl-hard.yaml'),
                      '--order',order,'--output',str(tmp_path/(order+'.yaml'))])==0
    dev=load_config(tmp_path/'dev1.yaml');hold=load_config(tmp_path/'hold1.yaml')
    assert dev['stage2']['alpha']==.5 and hold['stage2']['alpha']==.9
    a=json.loads(Path(dev['assets']['roots']).read_text())
    b=json.loads(Path(hold['assets']['roots']).read_text())
    assert len(a)==len(b)==100
    assert not {x['environment_seed'] for x in a}&{x['environment_seed'] for x in b}
    assert not {x['planner_seed'] for x in a}&{x['planner_seed'] for x in b}
    assert dev['fit_seed']!=hold['fit_seed']
    assert dev['reproduction_binding']['new_seed_policy'] is False
