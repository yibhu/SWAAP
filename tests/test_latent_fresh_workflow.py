"""Serial workflow handoffs with real selectors and synthetic stage fixtures.

No model/simulator/GPU executes. This checks finite orchestration and disjoint
selection/final orders, not qualification of a whole fresh scientific run.
"""
from pathlib import Path
import importlib.util,json
from copy import deepcopy
import pytest
from swaap.config import load_config
from swaap.latent import pipeline
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fresh_task',ROOT/'scripts/run_fresh_task.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


def stages(tmp_path,monkeypatch,count=5):
    events=[];endpoints={};constructed=[];fits=[]
    def save(c,name,record):
        p=Path(c['output_dir']);p.mkdir(parents=True,exist_ok=True);q=p/(name+'.json')
        record=dict(record,result_file=str(q));q.write_text(json.dumps(record));return record
    def prepare(c):
        p=Path(c['output_dir']);p.mkdir(parents=True)
        for name in ['clean','reference']:(p/(name+'.pt')).write_bytes(name.encode())
        return save(c,'prepare_data',{'files':{k:str(p/(k+'.pt'))for k in ['clean','reference']}})
    def stage1(c):
        p=Path(c['output_dir']);p.mkdir(parents=True);rows=[]
        for i in range(count):
            q=p/(str(i)+'.pt');q.write_bytes(('unique target'+str(i)).encode())
            rows.append(dict(candidate_id='target'+str(i),checkpoint=str(q),iteration=(i+1)*10,
                mean_return=80-i,mean_deviation=.1,mean_absolute_deviation=.1))
        return save(c,'stage1',dict(baseline={'mean_return':100},targets=rows))
    original_select=pipeline.select
    def select(c):events.append(c['selection']['phase']);return original_select(c)
    def construct(c):
        p=Path(c['output_dir']);p.mkdir(parents=True);p.joinpath('poison.pt').write_bytes(b'fixture poison')
        rank=int(p.name)if p.name.isdigit()else 5
        constructed.append(c.get('alpha',c['stage2']['alpha']))
        return save(c,'construct',dict(files={'poison':str(p/'poison.pt')},fixture_rank=rank))
    def fit(c):
        p=Path(c['output_dir']);p.mkdir(parents=True);q=p/'fitted.pt';q.write_bytes(b'fixture endpoint')
        order=c['assets'].get('order')or str(p/'order.pt')
        if not Path(order).exists():Path(order).write_bytes(b'fixture saved order')
        rank=int(p.parent.name)if p.parent.name.isdigit()else 5
        endpoints[str(q)]=dict(method=c['method'],rank=rank);fits.append(deepcopy(c))
        return save(c,'fit',dict(checkpoint=str(q),order=order))
    def evaluate(c):
        roots=json.loads(Path(c['assets']['roots']).read_text())['roots']
        if 'hold'in Path(c['assets']['roots']).name:assert 'winner'in events
        e=endpoints[c['assets']['checkpoint']]
        value=100. if e['method']=='clean'else 100.-e['rank']
        return save(c,'evaluate',dict(episodes=[dict(r,**{'return':value})for r in roots]))
    for name,fn in [('prepare_data',prepare),('stage1',stage1),('select',select),('construct',construct),('fit',fit),('evaluate',evaluate)]:monkeypatch.setattr(pipeline,name,fn)
    config=load_config(ROOT/'configs/tdmpc2/myo-reach-hard.yaml')
    original=tmp_path/'original.pt';original.write_bytes(b'fixture Original');config['assets']={'original':str(original)}
    config['seed']=17
    return config,events,constructed,fits


def test_one_declared_run_builds_development_json_without_final_selection(tmp_path,monkeypatch):
    config,events,constructed,fits=stages(tmp_path,monkeypatch)
    result=F.run(config,tmp_path/'fresh',panel_directory=ROOT/'configs/tdmpc2')
    assert result['status']=='COMPLETE_ONE_STAGE1_RUN' and result['independent_stage1_runs']==1
    assert events==['targets','screen','winner']
    assert constructed==[.5]*5+[.9]
    assert len(fits)==20 and sum(c['method']=='clean'for c in fits)==6
    summary=json.loads((tmp_path/'fresh/workflow.json').read_text())
    assert summary['status']=='COMPLETE_ONE_STAGE1_RUN' and not summary['automatic_retry']
    final=json.loads((tmp_path/'fresh/final_results.json').read_text())['results']
    assert len(final)==6 and {r['fit_order']for r in final}=={'hold1','hold2','hold3'}
    assert result['aggregate']['results'][0]['sample_sd']is None
    assert all(r['independent_stage1_runs']==1 for r in result['aggregate']['results'])


def test_rejected_run_stops_without_another_attempt_or_fit(tmp_path,monkeypatch):
    config,events,constructed,fits=stages(tmp_path,monkeypatch,count=4)
    result=F.run(config,tmp_path/'rejected',panel_directory=ROOT/'configs/tdmpc2')
    assert result['status']=='REJECTED_STAGE1' and result['eligible_count']==4
    assert not constructed and not fits and events==['targets']
    assert json.loads((tmp_path/'rejected/workflow.json').read_text())['status']=='REJECTED_STAGE1'


def test_maniskill_cannot_silently_use_nine_task_selector(tmp_path):
    with pytest.raises(ValueError,match='ManiSkill'):
        F.run({'family':'maniskill','method':'swaap'},tmp_path/'must-not-exist')
    assert not(tmp_path/'must-not-exist').exists()
