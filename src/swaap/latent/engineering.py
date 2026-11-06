"""Finite real-model integration check, separate from paper experiment recipes.

One unchanged Stage-1 outer update is NOT an accepted target. The following
constructor uses an explicitly provided released target and its bound banks.
All products are engineering fixtures and never enter paper aggregations.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import json
import math
import os
import platform
import time
import traceback

from .io import asset,digest,load_buffer,output,require,save_tensor,write_json


def _scope(config,path):
    value=deepcopy(config);value['output_dir']=str(path);return value


def one_stage1_update(config):
    """Execute the real fixed-setting outer update, without checkpoint admission."""
    import torch
    from .runtime import load_agent
    root=output(config);agent=load_agent(config,stage1_context=True)
    from .pipeline import _state_snapshot
    before=_state_snapshot(agent)
    started=time.perf_counter()
    try:
        observation=agent.env.reset()
        require(bool(torch.isfinite(observation).all()),'Nonfinite simulator reset')
        result=agent.update(apply_reachability=False)
        torch.cuda.synchronize()
        after=agent.model.state_dict()
        allowed=('psi.','theta.','theta_star.')
        frozen=[k for k in before if not k.startswith(allowed)]
        equal=lambda k: torch.equal(before[k],after[k].detach().cpu()) if isinstance(before[k],torch.Tensor) else type(after[k])is type(before[k]) and after[k]==before[k]
        require(all(equal(k) for k in frozen),
                'Stage-1 update changed a component outside dynamics/policy/inner policy')
        changed=[k for k in before if not equal(k)]
        require(any(k.startswith('psi.') for k in changed),'Stage-1 target dynamics did not move')
        checkpoint=save_tensor(root/'one_update_target.pt',dict(model={'psi.'+k:v.detach().cpu().clone() for k,v in agent.psi.state_dict().items()}))
        def diagnostic(v):
            if isinstance(v,torch.Tensor):v=v.detach().cpu().tolist()
            if isinstance(v,float)and not math.isfinite(v):return None
            if isinstance(v,list):return [diagnostic(x)for x in v]
            return v
        info={k:diagnostic(v)for k,v in result.items()}
        record=dict(status='PASS_ONE_OUTER_UPDATE',checkpoint=checkpoint,checkpoint_sha256=digest(checkpoint),
            elapsed_seconds=time.perf_counter()-started,changed_components=changed,frozen_component_count=len(frozen),
            settings=dict(inner_policy_updates=agent.cfg.T_step,update_batches=agent.cfg.update_batch_size,
                collect_trajectories=agent.cfg.num_collect_trajs,steps_per_trajectory=agent.cfg.num_step),
            update_info=info,unmeasured_diagnostic_values='null; inactive optional reachability diagnostics are not zero',selection_assessed=False,scientifically_accepted_target=False,
            subsequent_target='separate explicit released target')
        write_json(root/'stage1_update.json',record);return record
    finally:agent.env.close()


def short_fit(config):
    """First complete native epoch:20unchanged Adam updates on5000rows."""
    import torch
    from .runtime import load_agent
    from .pipeline import _state_snapshot,_unchanged
    from .native import fit_exact_orders
    from .fitting import continue_exact_orders
    root=output(config);data=load_buffer(asset(config,'data'),rows=5000)
    agent=load_agent(config);snapshot=_state_snapshot(agent)
    dynamics=deepcopy(agent.psi_0).to(agent.device).train(True)
    optimizer=torch.optim.Adam(dynamics.parameters(),lr=float(config['fit']['learning_rate']))
    order_path=asset(config,'order',required=False)
    fit_seed=int(config['fit_seed'])
    if order_path:
        schedule=torch.load(order_path,map_location='cpu',weights_only=False)
    else:
        generator=torch.Generator(device=agent.device).manual_seed(fit_seed)
        schedule=dict(rows=5000,fit_seed=fit_seed,
            permutations=[torch.randperm(5000,generator=generator,device=agent.device).cpu() for _ in range(10)])
        order_path=save_tensor(root/'orders.pt',schedule)
    require(schedule['rows']==5000 and schedule['fit_seed']==fit_seed and len(schedule['permutations'])==10,'Short-fit frozen order metadata mismatch')
    expected=torch.arange(5000)
    require(all(p.dtype==torch.int64 and torch.equal(p.sort().values,expected) for p in schedule['permutations']),'Invalid short-fit order')
    started=time.perf_counter()
    try:
        history,updates=continue_exact_orders(fit_exact_orders,agent,dynamics,optimizer,data,schedule,torch,start_updates=0,target_updates=20)
        torch.cuda.synchronize();_unchanged(agent,snapshot)
        state={k:v.detach().cpu().clone() for k,v in dynamics.state_dict().items()}
        path=save_tensor(root/'fitted.pt',dict(dynamics_state_dict=state,optimizer_state_dict=optimizer.state_dict(),
            metadata=dict(engineering_only=True,optimizer_updates=updates,fit_seed=fit_seed)))
        reloaded=torch.load(path,map_location='cpu',weights_only=False)
        check=deepcopy(agent.psi_0).cpu();check.load_state_dict(reloaded['dynamics_state_dict'],strict=True)
        require(all(torch.equal(state[k],check.state_dict()[k])for k in state),'Fitted checkpoint reload differs')
        record=dict(status='PASS_NATIVE_SHORT_FIT',checkpoint=path,checkpoint_sha256=digest(path),
            order=str(order_path),order_sha256=digest(order_path),updates=updates,history=history,
            data_sha256=digest(asset(config,'data')),learning_rate=float(config['fit']['learning_rate']),
            elapsed_seconds=time.perf_counter()-started,matched_clean_or_poison=config['method'],
            frozen_model_unchanged=True,save_reload_exact=True,engineering_only=True)
        write_json(root/'fit.json',record);return record
    finally:agent.env.close()


def run(config):
    """Run exactly the declared finite engineering workload, retaining failures."""
    import torch
    import numpy as np
    from . import pipeline,detector
    from .runtime import require_cuda
    root=output(config);started=time.time();completed={}
    record=dict(status='RUNNING',engineering_only=True,started_unix=started,
        host=platform.node(),platform=platform.platform(),python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        budget_gpu_minutes=float(config.get('engineering',{}).get('budget_gpu_minutes',20)),
        source_import=str(Path(__file__).resolve()),paper_efficacy_claim=False,
        original_sha256=digest(asset(config,'original')),
        target_sha256=digest(asset(config,'target')),new_stage1_proposals=0)
    require(config.get('engineering',{}).get('enabled') is True,'Use an explicitly named engineering configuration')
    require_cuda(config);record['gpu']=torch.cuda.get_device_name(0)
    record['gpu_uuid']=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable'))
    write_json(root/'started.json',record)
    try:
        completed['stage1']=one_stage1_update(_scope(config,root/'stage1'))
        c=_scope(config,root/'construct');c['method']='swaap';c.setdefault('stage2',{})['steps']=2
        completed['construct']=pipeline.construct(c)
        fit_paths={}
        for method,data in [('clean',asset(config,'clean')),('swaap',completed['construct']['files']['poison'])]:
            c=_scope(config,root/f'fit_{method}');c['method']=method;c['assets']['data']=str(data)
            if fit_paths:c['assets']['order']=completed['fit_clean']['order']
            completed[f'fit_{method}']=short_fit(c);fit_paths[method]=completed[f'fit_{method}']['checkpoint']
        # Config roots contain exactly one full-horizon root per compared endpoint.
        for method,checkpoint in fit_paths.items():
            c=_scope(config,root/f'evaluate_{method}');c['method']=method;c['assets']['checkpoint']=checkpoint
            c['evaluation']={'episodes':1,'save_trajectories':True}
            completed[f'evaluate_{method}']=pipeline.evaluate(c)
        c=_scope(config,root/'detector_original');c['assets']['roots']=str(asset(config,'detector_smoke_roots'))
        c['detector']={'role':'smoke','save_trajectories':True}
        completed['detector']=detector.collect_original(c)
        require(completed['detector']['trajectories']==2,'Exactly two Original detector smoke trajectories are declared')
        record['status']='PASS_FINITE_INTEGRATION'
    except BaseException as error:
        record.update(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        record.update(completed=completed,ended_unix=time.time(),allocated_gpu_seconds_conservative=time.time()-started,
            result_scope='Engineering plumbing only; reduced iteration endpoints are not paper results')
        write_json(root/'result.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--artifact-root',default='artifacts')
    a=p.parse_args()
    from swaap.config import load_config,resolve_assets
    from swaap.artifacts import AssetStore
    config=load_config(Path(a.config),[])
    config,_=resolve_assets(config,AssetStore(a.artifact_root))
    config['output_dir']=str(Path(a.output).resolve())
    print(json.dumps(run(config),indent=2))

if __name__=='__main__':main()
