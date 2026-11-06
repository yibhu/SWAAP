"""Standalone public interfaces: no model, environment, GPU or remote service."""
import json
import os
from pathlib import Path
import subprocess
import sys

RELEASE=Path(__file__).resolve().parents[1]


def _env():
    # The source location is explicit, rather than the child's working directory.
    return dict(os.environ,PYTHONPATH=str(RELEASE/'src'))


def test_graybox_named_config_and_bundled_inputs_outside_source_cwd(tmp_path,monkeypatch):
    from swaap.config import load_config
    from swaap.graybox.pipeline import asset
    config=load_config(RELEASE/'configs/graybox/soccer_transfer.yaml')
    monkeypatch.chdir(tmp_path)
    roots=asset(config,'roots');task_map=asset(config,'task_map')
    assert roots.is_absolute() and task_map.is_absolute()
    assert len(json.loads(roots.read_text())['roots'])==100
    assert len(json.loads(task_map.read_text()))==80
    child=subprocess.run([sys.executable,'-m','swaap.cli','construct','--config',str(RELEASE/'configs/graybox/soccer_transfer.yaml'),'--manifest',str(RELEASE/'assets/asset_manifest.json'),'--artifact-root',str(tmp_path/'artifacts'),'--output',str(tmp_path/'not_started'),'--dry-run'],cwd=tmp_path,env=_env(),text=True,capture_output=True)
    assert child.returncode==0,child.stderr
    result=json.loads(child.stdout)
    assert result['entrypoint']=='swaap.graybox.pipeline.construct'
    assert result['configuration']['construction']['epsilon']==.0045
    assert not (tmp_path/'not_started').exists()


def test_public_pipeline_imports_are_inert_outside_source_cwd(tmp_path):
    code="import sys; import swaap.graybox.pipeline; import swaap.dino.pipeline; assert 'torch' not in sys.modules; assert 'gym' not in sys.modules; assert 'mani_skill2' not in sys.modules; print('INERT')"
    child=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=_env(),text=True,capture_output=True)
    assert child.returncode==0,child.stderr
    assert child.stdout.strip()=='INERT'
