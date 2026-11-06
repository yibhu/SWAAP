"""Single-task Original collection with encoder-distinct MT80 residual scoring.

The collector follows the retained gb-c2 numerical path. No random-action/model
mixture is added, and no fitted endpoint supplies incoming-buffer residuals.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from swaap.latent.io import output,write_json,require,digest,save_tensor


def _residuals(victim, arrays):
    import torch
    with torch.inference_mode():
        x=torch.as_tensor(arrays['observations'][:-1],device=victim.device)
        nx=torch.as_tensor(arrays['observations'][1:],device=victim.device)
        a=torch.nn.functional.pad(torch.as_tensor(arrays['actions'],device=victim.device),(0,2))
        z=victim.model.encode(x,68);nz=victim.model.encode(nx,68)
        prediction=victim.model.next(z,a,68)
        residual=(nz-prediction).norm(dim=1)
        return residual.cpu().numpy(),(residual/(prediction.norm(dim=1)+1e-8)).cpu().numpy()


def collect(config):
    import torch
    from .pipeline import _agent,AgentView,asset
    from swaap.latent.native import rollout_episode
    from swaap.evaluation.detector import calibrate
    root=output(config);source=_agent(config);victim=_agent(config,multitask=True)
    source.model.eval().requires_grad_(False);victim.model.eval().requires_grad_(False)
    roots=json.loads(asset(config,'roots' if config.get('detector',{}).get('role') and config.get('assets',{}).get('roots') else 'detector_roots').read_text())
    if isinstance(roots,list):
        role=config.get('detector',{}).get('role')
        require(role is not None,'A root list requires detector.role')
        roots={role:roots}
    require(isinstance(roots,dict),'Detector roots must be separated by statistics/threshold/clean_test')
    roots={('statistics' if k=='stats' else k):v for k,v in roots.items()}
    seed_ids=[(int(r['environment_seed']),int(r['planner_seed'])) for rows in roots.values() for r in rows]
    require(len(seed_ids)==len(set(seed_ids)),'Detector seed/root pairs overlap between roles')
    primitive_seeds=[seed for pair in seed_ids for seed in pair]
    require(len(primitive_seeds)==len(set(primitive_seeds)),'Detector environment/planner primitive seeds overlap')
    smoke=int(config.get('collection',{}).get('maximum_trajectories',0))
    counts={'statistics':300,'threshold':500,'clean_test':1000};summary={};used=0
    reference=digest(asset(config,'victim'))
    requested=config.get('detector',{}).get('role')
    if requested=='stats':requested='statistics'
    require(requested is None or requested in counts,'Unknown detector role')
    roles=[requested] if requested else list(counts)
    for role in roles:
        total=counts[role]
        require(role in roots and (len(roots[role])==total or bool(smoke and roots[role])),'Frozen gray-box cohort has incorrect role count')
        if role=='clean_test' and not smoke:
            calibration=asset(config,'calibration') if requested else root/'calibration.json'
            require(calibration.is_file(),'Thresholds must be frozen before independent testing')
            require(json.loads(calibration.read_text())['original_sha256']==reference,'Frozen detector and MT80 Original differ')
        residuals=[];norms=[];ids=[]
        for i,r in enumerate(roots[role]):
            arrays,row=rollout_episode(agent=AgentView(source),cfg=source.cfg,task='mw-soccer',dynamics=source.model._dynamics,environment_seed=r['environment_seed'],planner_seed=r['planner_seed'],expected_episode_length=100,torch=torch,np=np)
            raw,norm=_residuals(victim,arrays)
            require(raw.shape==(100,) and raw.dtype==np.float32,'MT80 incoming residual layout/precision changed')
            residuals.append(raw);norms.append(norm);ids.append(f"{r['environment_seed']}:{r['planner_seed']}")
            save_tensor(root/(role+'_'+str(i).zfill(4)+'.pt'),{'roots':r,'observations':arrays['observations'],'actions':arrays['actions'],'rewards':arrays['rewards'],'residuals':raw,'normalized_residuals':norm,'Original_policy':'single-task Soccer Original','reference':'MT80 Original Soccer68','record':row})
            used+=1
            if smoke and used>=smoke:break
        path=root/(role+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream,residuals=np.stack(residuals),trajectory_ids=np.asarray(ids),original_sha256=np.asarray(reference))
        summary[role]={'trajectories':len(ids),'residual_file':str(path),'normalized_residual_mean':float(np.mean(norms))}
        write_json(root/(role+'.json'),summary[role])
        if smoke and used>=smoke:break
        if role=='threshold' and not requested:calibrate(root/'statistics.npz',root/'threshold.npz',root/'calibration.json',family='graybox')
    result={'status':'COLLECTION_SMOKE_ONLY' if smoke else 'COMPLETE_CLEAN_COHORTS','roles':summary,'environment_episodes':used,'reference_model':'MT80 Original Soccer68','collection_policy':'single-task Soccer Original MPC eval_mode=False; zero random action/model mixture','calibration_frozen':(root/'calibration.json').is_file()}
    write_json(root/'detector_collection.json',result);return result


def incoming(config):
    import torch
    from .pipeline import _agent,TaskModel,SingleModel,asset
    root=output(config)
    clean=torch.load(asset(config,'clean'),map_location='cpu',weights_only=False)
    poison=torch.load(asset(config,'poison'),map_location='cpu',weights_only=False)
    victim=_agent(config,multitask=True);source=_agent(config)
    results={}
    for name,agent,model,index,key in [('victim',victim,TaskModel(victim.model,victim.device,68),68,'victim'),('surrogate',source,SingleModel(source.model,source.device),None,'original')]:
        values={};reference=digest(asset(config,key))
        for label,data in [('clean',clean),('poison',poison)]:
            signals=[];normalized=[]
            with torch.no_grad():
                for start in range(0,5000,100):
                    z=model.encode(data[0][start:start+100].to(agent.device));nz=model.encode(data[3][start:start+100].to(agent.device))
                    a=data[1][start:start+100].to(agent.device);p=model.next(z,a,agent.model._dynamics)
                    r=(nz-p).norm(dim=1);signals.append(r.cpu().numpy());normalized.append((r/(p.norm(dim=1)+1e-8)).cpu().numpy())
            path=root/(name+'_'+label+'.npz')
            with path.open('xb') as stream:np.savez_compressed(stream,residuals=np.stack(signals),trajectory_ids=np.asarray([f'incoming:{i}' for i in range(50)]),original_sha256=np.asarray(reference))
            values[label]={'residuals':str(path),'delta_d':float(np.mean(normalized)),'trajectories':50,'transitions':5000}
        results[name]=values
    write_json(root/'incoming.json',results);return results


def prepare_saved(config):
    """Convert released, closed residual arrays into the common scorer schema."""
    from .pipeline import asset
    root=output(config);reference=digest(asset(config,'victim'));result={}
    roots=None
    if config.get('assets',{}).get('detector_roots'):
        roots=json.loads(asset(config,'detector_roots').read_text())
        roots={('statistics' if k=='stats' else k):v for k,v in roots.items()}
        primitive=[int(row[key]) for rows in roots.values() for row in rows for key in ('environment_seed','planner_seed')]
        require(len(primitive)==len(set(primitive)),'Released detector seed cohorts overlap')
    for role in ('statistics','threshold','clean_test'):
        path=asset(config,'detector_'+role)
        if path.suffix=='.npy':matrix=np.load(path,allow_pickle=False)
        else:
            with np.load(path,allow_pickle=False) as packet:matrix=packet['residuals'].copy()
        require(matrix.shape==({'statistics':300,'threshold':500,'clean_test':1000}[role],100),'Released gray-box cohort shape changed')
        if roots is not None:
            require(len(roots[role])==len(matrix),'Released residuals and frozen seed manifest differ')
            ids=[str(r['environment_seed'])+':'+str(r['planner_seed']) for r in roots[role]]
        else:
            ids=[role+':'+str(i) for i in range(len(matrix))]
        destination=root/(role+'.npz')
        with destination.open('xb') as stream:np.savez_compressed(stream,residuals=matrix,trajectory_ids=np.asarray(ids),original_sha256=np.asarray(reference))
        result[role]=str(destination)
    write_json(root/'saved_cohorts.json',{'derived_schema_only':True,'residual_values_unchanged':True,'cohorts':result});return result
