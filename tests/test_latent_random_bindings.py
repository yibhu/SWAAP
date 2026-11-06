"""Published Random construction seeds are independent of victim fitting seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]

def test_published_nine_task_legacy_random_bindings():
    d=json.loads((ROOT/'configs/tdmpc2/published_orders.json').read_text())
    rows=d['runs'];assert len(rows)==27
    seeds=[]
    for r in rows:
        binding=r['construction_rng']['swaap_legacy_random'];seeds.append(binding['seed'])
        assert r['construction_rng']['swaap']['seed']==0
        assert binding['status']=='executed' and 0<binding['support_rows']<5000
        assert len(binding['poison_sha256'])==len(binding['support_sha256'])==64
    assert len(set(seeds))==27 and 700008301 in seeds

def test_final_maniskill_slot_seed_inventory_stays_bounded():
    d=json.loads((ROOT/'configs/maniskill/published_orders.json').read_text())
    assert len(d['runs'])==9 and {k.replace('_','-'):v for k,v in d['contributing_runs'].items()}=={'lift-cube':2,'pick-cube':3,'stack-cube':3}
    assert len({r['construction_rng']['swaap_legacy_random']['seed']for r in d['runs']})==9
    missing=next(r for r in d['runs']if r['task']=='lift-cube'and r['stage1_run']=='N01')
    assert missing['construction_rng']['swaap_legacy_random']['status'].startswith('reserved_only')


def test_shared_support_random_target_order_binding(tmp_path):
    output = tmp_path / 'condition.yaml'
    process = subprocess.run([sys.executable, str(ROOT/'scripts/bind_order.py'),
        '--config', str(ROOT/'configs/tdmpc2/humanoid-walk_random_target_shared_support.yaml'),
        '--order', 'hold1', '--run-index', '1', '--output', str(output)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
        capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    cfg = yaml.safe_load(output.read_text())
    assert cfg['random_target_seed'] == 700001905
    assert cfg['random_target_sha256'] == 'cc6124506233ead0cc85a0fc326e910885140072425d5d57fd33543cf1e47580'
    assert cfg['assets']['support'] == 'asset:tdmpc2.humanoid-walk.main-run1.support'
    assert cfg['stage2']['step_size'] == 10


def test_published_random_support_mask_is_bound_without_redrawing(tmp_path):
    output = tmp_path/'random-support.yaml'
    process = subprocess.run([sys.executable, str(ROOT/'scripts/bind_order.py'),
        '--config', str(ROOT/'configs/tdmpc2/humanoid-walk_swaap_random_support.yaml'),
        '--order', 'hold2', '--run-index', '2', '--output', str(output)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(ROOT/'src')),
        capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    cfg = yaml.safe_load(output.read_text())
    assert cfg['construction_seed'] == 0
    assert cfg['random_support_seed'] == 700011805
    assert cfg['assets']['target'] == 'asset:tdmpc2.humanoid-walk.main-run2.target'
    support = json.loads(Path(cfg['assets']['support']).read_text())
    assert len(support) == len(set(support)) == 500
