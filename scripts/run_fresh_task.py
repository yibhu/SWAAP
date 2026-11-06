#!/usr/bin/env python3
"""One finite nine-task SWAAP run, from data/Stage1 to matched final results.

No scheduler, retries, outcome-dependent replacement or automatic second run.
The corrected ManiSkill developmental/stealth gates are separate; this script
never substitutes the nine-task selector for them.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback

from swaap.latent import pipeline
from swaap.latent.io import digest,require,write_json


def common_panels(folder,task,position):
    final=json.loads((folder/'published_orders.json').read_text())
    development=json.loads((folder/'development_orders.json').read_text())
    runs=[r for r in final['runs']if r['task']==task]
    require(len(runs)==3 and 1<=position<=3,'Choose one of the three declared panel positions')
    record=runs[position-1]
    dev=next(r for r in development['runs']if r['stage1_run']==record['stage1_run'])
    orders=dev['development_orders']+record['fit_orders']
    require([r['fit_order']for r in orders]==['dev1','dev2','dev3','hold1','hold2','hold3'],'Six exact declared orders are required')
    used_env=set();used_planner=set();fit_seeds=set()
    for row in orders:
        roots=row['roots'];es={r['environment_seed']for r in roots};ps={r['planner_seed']for r in roots}
        require(len(roots)==len(es)==len(ps)==100,'Every common panel must contain exactly100 unique roots')
        require(not used_env&es and not used_planner&ps and row['fit_seed']not in fit_seeds,'Development/final panels and orders must be disjoint')
        used_env|=es;used_planner|=ps;fit_seeds.add(row['fit_seed'])
    return record,dev,{r['fit_order']:r for r in orders}


def run(config,output,*,panel_position=1,panel_directory=None,reuse_banks=False):
    cfg=deepcopy(config)
    require(cfg.get('family')in('tdmpc2','latent'),'This serial workflow covers the nine DMControl/MyoSuite/MetaWorld tasks. ManiSkill uses its explicit developmental and detector-gated recipes.')
    require(cfg.get('method','swaap')=='swaap','Fresh primary target selection is defined for ordinary SWAAP only')
    folder=Path(panel_directory or cfg.get('_config_dir','configs/tdmpc2')).resolve()
    record,dev,panels=common_panels(folder,cfg['task'],panel_position)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    for key in ('target','support','data','checkpoint','order','stage1_results','development_results','results'):
        cfg.setdefault('assets',{}).pop(key,None)
    cfg.setdefault('stage1',{})['evaluation_seeds']=dev['stage1_direct_seeds']
    cfg['construction_seed']=0
    run_identity=f"{cfg['task']}-seed{int(cfg.get('seed',0))}"
    request=dict(task=cfg['task'],generation_seed=cfg.get('seed',0),panel_position=panel_position,
        panel_source_identity=record['stage1_run'],panel_inventory_sha256=digest(folder/'published_orders.json'),
        development_inventory_sha256=digest(folder/'development_orders.json'),reuse_input_banks=reuse_banks,
        new_stochastic_run=True,reproduces_selected_historical_target_bytes=False,
        automatic_second_attempt=False)
    write_json(root/'request.json',request)
    completed={};final_status='FAILED_PRESERVED';error=None
    def record_stage(name,value):
        completed[name]={key:value[key]for key in ('result_file','status','classification','eligible_count','iterations','selected')if key in value}
        if name=='target_selection':completed[name]['selected']=[r['candidate_id']for r in value['selected']]
    def invoke(stage,path,**changes):
        local=deepcopy(cfg);local['output_dir']=str(root/path)
        for key,value in changes.items():
            if key=='assets':local['assets'].update(value)
            else:local[key]=value
        return getattr(pipeline,stage)(local)
    try:
        if not reuse_banks:
            prepared=invoke('prepare_data','data');record_stage('prepare_data',prepared)
            cfg['assets'].update(clean=prepared['files']['clean'],reference=prepared['files']['reference'])
        else:
            require(all(Path(cfg['assets'][k]).is_file()for k in('clean','reference')),'Artifact-assisted fresh run needs verified clean/reference banks')
        generated=invoke('stage1','stage1');record_stage('stage1',generated)
        targets=invoke('select','target_selection',assets={'stage1_results':generated['result_file']},
            selection={**cfg.get('selection',{}),'phase':'targets'})
        record_stage('target_selection',targets)
        if targets['classification']=='REJECTED':
            final_status='REJECTED_STAGE1';return dict(status=final_status,eligible_count=targets['eligible_count'],automatic_retry=False)
        candidates={r['candidate_id']:r for r in targets['selected']}
        selection_alpha=float(cfg.get('selection',{}).get('construction_alpha',cfg.get('stage2',{}).get('alpha',.9)))
        final_alpha=float(cfg.get('alpha',cfg.get('stage2',{}).get('alpha',.9)))
        constructions={};clean={};develop=[]
        def bind_order(name):
            row=panels[name];path=root/'orders'/(name+'_roots.json')
            if not path.exists():write_json(path,dict(roots=row['roots']))
            return row,path
        def clean_order(name):
            if name not in clean:
                row,roots=bind_order(name)
                fitted=invoke('fit',f'clean/{name}/fit',method='clean',fit_seed=row['fit_seed'],
                    assets={'data':cfg['assets']['clean']})
                order=fitted.get('order')or str(root/f'clean/{name}/fit/order.pt')
                evaluated=invoke('evaluate',f'clean/{name}/evaluate',method='clean',
                    assets={'checkpoint':fitted['checkpoint'],'roots':str(roots)})
                clean[name]=dict(fit=fitted,evaluation=evaluated,order=order)
            return clean[name]
        def assess(candidate,name,poison):
            shared=clean_order(name);row,roots=bind_order(name)
            fitted=invoke('fit',f"development/{name}/{candidate['frozen_rank']:02d}/fit",fit_seed=row['fit_seed'],
                assets={'data':poison,'order':shared['order']})
            if 'order_sha256'in fitted or 'order_sha256'in shared['fit']:
                require(fitted.get('order_sha256')==shared['fit'].get('order_sha256'),'Development Clean/attack fitting orders differ')
            evaluated=invoke('evaluate',f"development/{name}/{candidate['frozen_rank']:02d}/evaluate",
                assets={'checkpoint':fitted['checkpoint'],'roots':str(roots)})
            c,a=shared['evaluation']['episodes'],evaluated['episodes']
            identity=lambda values:[(r['episode_index'],r['environment_seed'],r['planner_seed'])for r in values]
            require(identity(c)==identity(a),'Development Clean/attack roots differ')
            return dict(candidate_id=candidate['candidate_id'],checkpoint_sha256=candidate['checkpoint_sha256'],
                frozen_rank=candidate['frozen_rank'],fit_order=name,matched_clean=True,
                clean_returns=[r['return']for r in c],attack_returns=[r['return']for r in a],
                clean_evaluation=shared['evaluation']['result_file'],attack_evaluation=evaluated['result_file'])
        for identity,candidate in candidates.items():
            constructed=invoke('construct',f"construction/{candidate['frozen_rank']:02d}",alpha=selection_alpha,
                assets={'target':candidate['checkpoint']})
            constructions[identity]=constructed
            develop.append(assess(candidate,'dev1',constructed['files']['poison']))
        screened_path=write_json(root/'development_screen.json',dict(results=develop))
        screen=invoke('select','screen',selection={**cfg.get('selection',{}),'phase':'screen'},assets={'development_results':screened_path})
        record_stage('screen',screen)
        finalists=screen['selected'];race=[r for r in develop if r['candidate_id']in finalists]
        for identity in finalists:
            for order in ('dev2','dev3'):
                race.append(assess(candidates[identity],order,constructions[identity]['files']['poison']))
        race_path=write_json(root/'development_three_orders.json',dict(results=race))
        lock=invoke('select','winner',selection={**cfg.get('selection',{}),'phase':'winner'},assets={'development_results':race_path})
        record_stage('winner',lock);winner=candidates[lock['selected'][0]]
        final_poison=constructions[winner['candidate_id']]
        if final_alpha!=selection_alpha:
            final_poison=invoke('construct','final_construction',alpha=final_alpha,assets={'target':winner['checkpoint']})
        final_rows=[]
        for name in ('hold1','hold2','hold3'):
            shared=clean_order(name);row,roots=bind_order(name)
            fitted=invoke('fit',f'final/{name}/fit',fit_seed=row['fit_seed'],
                assets={'data':final_poison['files']['poison'],'order':shared['order']})
            if 'order_sha256'in fitted or 'order_sha256'in shared['fit']:
                require(fitted.get('order_sha256')==shared['fit'].get('order_sha256'),'Final Clean/attack fitting orders differ')
            evaluated=invoke('evaluate',f'final/{name}/evaluate',assets={'checkpoint':fitted['checkpoint'],'roots':str(roots)})
            identity=lambda values:[(r['episode_index'],r['environment_seed'],r['planner_seed'])for r in values]
            require(identity(shared['evaluation']['episodes'])==identity(evaluated['episodes']),'Final Clean/attack roots differ')
            for method,value in [('clean',shared['evaluation']),('swaap',evaluated)]:
                final_rows.append(dict(task=cfg['task'],method=method,stage1_run=run_identity,fit_order=name,
                    returns=[r['return']for r in value['episodes']],evaluation_file=value['result_file'],
                    winner_checkpoint_sha256=winner['checkpoint_sha256']))
        final_path=write_json(root/'final_results.json',dict(results=final_rows))
        aggregate=invoke('aggregate','aggregate',assets={'results':final_path})
        record_stage('aggregate',aggregate)
        final_status='COMPLETE_ONE_STAGE1_RUN'
        return dict(status=final_status,selected_target=winner,selection_alpha=selection_alpha,final_alpha=final_alpha,
            final_poison_files=final_poison['files'],aggregate=aggregate,independent_stage1_runs=1,
            manuscript_replications_not_claimed=True,automatic_follow_on=False)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
    finally:
        write_json(root/'workflow.json',dict(status=final_status,request=request,completed=completed,error=error,
            preserves_partial_products=True,automatic_retry=False,new_order_or_target_selected_after_final_results=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact-root',type=Path,default=Path('artifacts'));parser.add_argument('--seed',type=int)
    parser.add_argument('--panel-position',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--panel-directory',type=Path)
    parser.add_argument('--reuse-input-banks',action='store_true',help='Start with verified released banks; Stage1 remains a new run.')
    a=parser.parse_args(argv)
    from swaap.config import load_config,resolve_assets
    from swaap.artifacts import AssetStore
    config=load_config(a.config,[])
    if a.seed is not None:config['seed']=a.seed
    for key in ('target','support','data','checkpoint','order'):
        config.get('assets',{}).pop(key,None)
    if not a.reuse_input_banks:
        for key in ('clean','reference'):
            config.get('assets',{}).pop(key,None)
    config,_=resolve_assets(config,AssetStore(a.artifact_root))
    result=run(config,a.output,panel_position=a.panel_position,panel_directory=a.panel_directory,
        reuse_banks=a.reuse_input_banks)
    print(json.dumps(result,indent=2,allow_nan=False));return 0

if __name__=='__main__':raise SystemExit(main())
