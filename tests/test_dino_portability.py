"""Configuration/CPU-only smoke from outside the checkout, without weights."""
import json
import os
from pathlib import Path
import subprocess
import sys

RELEASE=Path(__file__).resolve().parents[1]


def test_dino_named_configs_resolve_outside_source_cwd(tmp_path):
    env=dict(os.environ,PYTHONPATH=str(RELEASE/'src'))
    for recipe in ['pusht_faithful.yaml','pusht_artifact.yaml','pusht_fresh.yaml']:
        child=subprocess.run([sys.executable,'-m','swaap.cli','construct','--config',str(RELEASE/'configs/dino'/recipe),'--manifest',str(RELEASE/'assets/asset_manifest.json'),'--artifact-root',str(tmp_path/'artifacts'),'--output',str(tmp_path/'not_started'),'--dry-run'],cwd=tmp_path,env=env,text=True,capture_output=True)
        assert child.returncode==0,child.stderr
        result=json.loads(child.stdout)
        assert result['entrypoint']=='swaap.dino.pipeline.construct'
        assert result['configuration']['construction']['editable_features']==394
        assert result['configuration']['construction']['eligible_rows']==80
        assert not (tmp_path/'not_started').exists()


def test_public_cpu_smoke_runs_outside_source_cwd(tmp_path):
    env=dict(os.environ,PYTHONPATH=str(RELEASE/'src'))
    output=tmp_path/'synthetic_smoke'
    child=subprocess.run([sys.executable,'-m','swaap.cli','smoke','--output',str(output)],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert child.returncode==0,child.stderr
    assert output.is_dir()
    result=json.loads((output/'smoke.json').read_text())
    assert result['environment_episodes']==0 and result['device']=='cpu'
    assert result.get('status') in ('PASS','PASSED','COMPLETE','PASS_CPU_SYNTHETIC'),result


def test_retained_relative_diagnostics_stay_in_goal_output(tmp_path):
    from swaap.dino.pipeline import _working_directory
    caller=Path.cwd()
    goal=tmp_path/'goal_diagnostics'
    try:
        with _working_directory(goal):
            Path('logs.json').write_text('{"complete":false}')
            raise RuntimeError('engineering failure')
    except RuntimeError:
        pass
    assert Path.cwd()==caller
    assert (goal/'logs.json').is_file()
    assert not (tmp_path/'logs.json').exists()
