#!/usr/bin/env python3
"""Bind one recorded common order/panel to a task recipe. No RNG or models.

Development and final panels are read from the executed, verified inventories;
this helper never generates replacement seeds or selects an order by efficacy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from swaap.config import load_config


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--order',choices=['dev1','dev2','dev3','hold1','hold2','hold3'],required=True)
    p.add_argument('--run-index',type=int,choices=[1,2,3],default=1)
    p.add_argument('--output',type=Path,required=True,help='New YAML file; companion root JSON is saved beside it.')
    a=p.parse_args(argv);cfg=load_config(a.config)
    if cfg['family']!='tdmpc2':
        raise ValueError('This helper binds the nine-task protocol; use the final ManiSkill recipe and its separate selection gates.')
    folder=a.config.resolve().parent
    final_path=folder/'published_orders.json';dev_path=folder/'development_orders.json'
    final=json.loads(final_path.read_text());development=json.loads(dev_path.read_text())
    rows=[r for r in final['runs'] if r['task']==cfg['task']]
    if len(rows)!=3:raise ValueError('Expected three recorded independent Stage-1 positions.')
    record=rows[a.run_index-1]
    earlier=next(r for r in development['runs'] if r['stage1_run']==record['stage1_run'])
    options=earlier['development_orders'] if a.order.startswith('dev') else record['fit_orders']
    selected=next(r for r in options if r['fit_order']==a.order)
    # Resolve recipe-local files before moving the configuration elsewhere.
    for key,value in cfg.get('assets',{}).items():
        if isinstance(value,str) and value.startswith('config:'):
            cfg['assets'][key]=str((folder/value[7:]).resolve())
    output=a.output.resolve();roots=output.with_name(output.stem+'_roots.json')
    if output.exists() or roots.exists():raise ValueError('Output YAML or root file already exists.')
    output.parent.mkdir(parents=True,exist_ok=True)
    cfg['fit_seed']=selected['fit_seed']
    rng=record.get('construction_rng',{})
    if rng.get('swaap_legacy_random',{}).get('seed') is not None:
        cfg['legacy_random_seed']=rng['swaap_legacy_random']['seed']
    method=cfg.get('method','swaap')
    if rng.get(method,{}).get('seed') is not None:
        cfg['construction_seed']=rng[method]['seed']
    if method == 'random_target_shared_support':
        random_target = rng.get(method, {})
        if 'random_seed' not in random_target:
            raise ValueError('This task/run has no recorded shared-support random-target condition.')
        cfg['construction_seed'] = random_target['construction_seed']
        cfg['random_target_seed'] = random_target['random_seed']
        cfg['random_target_sha256'] = random_target['random_target_sha256']
        cfg['assets']['target'] = f"asset:tdmpc2.{cfg['task']}.main-run{a.run_index}.target"
        cfg['assets']['support'] = f"asset:tdmpc2.{cfg['task']}.main-run{a.run_index}.support"
    if method == 'swaap_random_support':
        condition = rng.get(method, {})
        if 'support_file' not in condition:
            raise ValueError('This task/run has no recorded random-support comparison.')
        support = folder/condition['support_file']
        if hashlib.sha256(support.read_bytes()).hexdigest() != condition['support_file_sha256']:
            raise ValueError('The frozen random-support mask has changed.')
        cfg['assets']['target'] = 'asset:' + condition['target_asset']
        cfg['assets']['support'] = str(support)
        cfg['construction_seed'] = condition['construction_seed']
        cfg['random_support_seed'] = condition['random_support_seed']
        cfg['stage2'].update(alpha=condition['alpha'], rp=condition['poison_fraction'],
                             steps=condition['steps'], step_size=condition['step_size'],
                             epsilon=condition['epsilon'])
    if method == 'exact_tied_trajectory':
        conditions = json.loads((folder/'trajectory_conditions.json').read_text())['conditions']
        conditions = [row for row in conditions if row['stage1_run'] == record['stage1_run']]
        if len(conditions) != 1 or a.order not in conditions[0]['fitting_orders']:
            raise ValueError('This exact tied run/order is not a retained condition.')
        condition = conditions[0]
        support = folder/condition['support_file']
        if hashlib.sha256(support.read_bytes()).hexdigest() != condition['support_file_sha256']:
            raise ValueError('The exact ordered trajectory support has changed.')
        cfg['assets']['target'] = 'asset:' + condition['target_asset']
        cfg['assets']['support'] = str(support)
        cfg['construction_seed'] = condition['construction_seed']
        cfg['stage2'].update(alpha=condition['alpha'], rp=condition['poison_fraction'],
                             steps=condition['steps'], step_size=condition['step_size'],
                             epsilon=condition['epsilon'])
        cfg.setdefault('data', {})['episode_end_indices'] = condition['episode_end_indices']
    cfg.setdefault('assets',{})['roots']=str(roots)
    cfg.setdefault('stage2',{})['alpha']=record['selection_alpha'] if a.order.startswith('dev') else record['main_alpha']
    cfg.setdefault('stage1',{})['evaluation_seeds']=earlier['stage1_direct_seeds']
    cfg['reproduction_binding']=dict(stage1_position=a.run_index,stage1_identity=record['stage1_run'],
        order=a.order,target_selection_uses_final_results=False,new_seed_policy=False,
        development_index_sha256=hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        final_index_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest())
    cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
    with roots.open('x') as f:json.dump(selected['roots'],f,indent=2)
    with output.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    print(json.dumps(dict(status='RECORDED_ORDER_BOUND',config=str(output),roots=str(roots),
        fit_seed=cfg['fit_seed'],alpha=cfg['stage2']['alpha'],new_random_draws=0)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
