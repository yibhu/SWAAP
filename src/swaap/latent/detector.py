"""Original-policy detector collection and incoming-buffer residual adapters.

Statistics, threshold selection and independent scoring stay in
swaap.evaluation.detector. This module supplies their complete trajectory arrays;
it never calibrates against poisoned buffers or substitutes adapted rollouts.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any,Sequence
import hashlib
import json
import time
import numpy as np
import torch
from .io import asset,digest,finite,load_buffer,output,read_json,require,write_json
from .families import expected_episode_length


def reference_residual_signal(data:Sequence[torch.Tensor],trainer:Any,start:int,end:int,*,device:str)->np.ndarray:
    """Canonical V4 function: one full-trajectory Original prediction batch."""
    z=data[0][start:end].to(device).float()
    action=data[1][start:end].to(device).float()
    next_z=data[3][start:end].to(device).float()
    with torch.no_grad():
        predicted_next_z=trainer.model.next(z,action)
    return (next_z-predicted_next_z).norm(p=2,dim=1).detach().cpu().numpy()


def residual_episode(data,agent,*,family,expected_steps):
    """Preserve the adopted norm location: MS CPU, nine-task native device."""
    require(all(len(x)==expected_steps for x in data),'Only complete stored trajectories are scored')
    require(all(data[i].dtype==torch.float32 for i in (0,1,3)),'Canonical latent/action inputs must be float32')
    if family=='maniskill':
        z=data[0].to(agent.device).float();action=data[1].to(agent.device).float()
        with torch.no_grad():predicted=agent.model.next(z,action)
        prediction=predicted.detach().cpu().float()
        require(tuple(prediction.shape)==tuple(data[3].shape) and bool(torch.isfinite(prediction).all()),
                'Nonfinite/incorrect Original prediction geometry')
        raw=(data[3].float()-prediction).norm(p=2,dim=1).numpy().astype(np.float32)
    else:
        raw=reference_residual_signal(data,agent,0,expected_steps,device=str(agent.device)).astype(np.float32)
    require(raw.shape==(expected_steps,) and bool(np.isfinite(raw).all()) and bool((raw>=0).all()),
            'Canonical raw residual vector is incomplete or nonfinite')
    return raw


def root_panel(config,*,name='roots'):
    path=asset(config,name);payload=read_json(path)
    roots=payload['roots'] if isinstance(payload,dict) else payload
    require(isinstance(roots,list) and roots,'Explicit nonempty trajectory roots are required')
    require([r['episode_index'] for r in roots]==list(range(len(roots))),'Root indices must preserve complete episode order')
    pairs=[]
    for row in roots:
        require(all(type(row[k]) is int and 0<=row[k]<2**32 for k in ('environment_seed','planner_seed')),
                'Unsigned environment/planner seeds are required')
        pairs.append((row['environment_seed'],row['planner_seed']))
    require(len(pairs)==len(set(pairs)),'Duplicate trajectory roots')
    return path,roots,payload


def trajectory_id(task,original_sha256,root):
    """Role-independent IDs prevent relabeling reused roots as a disjoint cohort."""
    value=dict(task=task,original_sha256=original_sha256,
               environment_seed=int(root['environment_seed']),planner_seed=int(root['planner_seed']))
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _write_residuals(path,residuals,ids,*,original_sha,task,role,roots):
    require(not Path(path).exists(),'Refusing to overwrite detector residuals')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,residuals=np.asarray(residuals,dtype=np.float32),
            trajectory_ids=np.asarray(ids,dtype='U64'),original_sha256=np.asarray(original_sha),
            task=np.asarray(task),role=np.asarray(role),
            environment_seeds=np.asarray([r['environment_seed']for r in roots],dtype=np.uint64),
            planner_seeds=np.asarray([r['planner_seed']for r in roots],dtype=np.uint64))
    return str(Path(path).resolve())


def collect_original(config):
    """Collect an explicit Original-policy cohort through the family adapter.

    `detector.role=smoke` may contain only a few trajectories and is not a
    calibration. Full roles use the same trajectory extraction and residual
    batching. Calibration/scoring enforces its own published sample counts.
    """
    from .runtime import load_agent
    from .native import rollout_episode
    from .pipeline import _state_snapshot,_unchanged
    role=config.get('detector',{}).get('role','smoke')
    aliases={'stats':'statistics','clean-test':'clean_test'};role=aliases.get(role,role)
    require(role in ('statistics','threshold','clean_test','smoke'),'Unknown Original detector cohort role')
    path,roots,manifest=root_panel(config)
    original_sha=digest(asset(config,'original'))
    ids=[trajectory_id(config['task'],original_sha,r) for r in roots]
    root=output(config);expected=expected_episode_length(config['task'])
    if isinstance(manifest,dict) and manifest.get('original_sha256'):
        require(manifest['original_sha256']==original_sha,'Root manifest binds a different Original')
    agent=load_agent(config);before=_state_snapshot(agent)
    residuals=[];episodes=[];started=time.perf_counter()
    try:
        for declared in roots:
            arrays,record=rollout_episode(agent=agent,cfg=agent.cfg,task=config['task'],dynamics=agent.psi_0,
                environment_seed=declared['environment_seed'],planner_seed=declared['planner_seed'],
                expected_episode_length=expected,torch=torch,np=np)
            # Predictions are deliberately recomputed as ONE T-row batch.
            # Per-step rollout predictions are not a replacement for this batch.
            data=(torch.from_numpy(arrays['latents'][:-1]),torch.from_numpy(arrays['actions']),
                  torch.from_numpy(arrays['rewards'][:,None]),torch.from_numpy(arrays['latents'][1:]),
                  torch.from_numpy(arrays['dones'][:,None]).float())
            raw=residual_episode(data,agent,family=config.get('family','tdmpc2'),expected_steps=expected)
            residuals.append(raw);record.update(episode_index=declared['episode_index'],trajectory_id=ids[len(episodes)],
                residual_definition='raw L2(next_z-P_Original(z,a))',prediction_batch_rows=expected)
            episodes.append(record)
            if config.get('detector',{}).get('save_trajectories',False):
                with (root/f"episode_{declared['episode_index']:04d}.npz").open('xb') as stream:np.savez_compressed(stream,**arrays)
            _unchanged(agent,before)
        file=_write_residuals(root/'residuals.npz',np.stack(residuals),ids,
            original_sha=original_sha,task=config['task'],role=role,roots=roots)
        result=dict(status='COMPLETE_ORIGINAL_POLICY_COHORT',role=role,task=config['task'],
            residuals=file,residuals_sha256=digest(file),original_sha256=original_sha,roots_sha256=digest(path),
            trajectories=len(roots),steps_per_trajectory=expected,prediction_batch_rows=expected,
            episodes=episodes,new_environment_episodes=len(roots),new_fits=0,new_poison_constructions=0,
            policy='frozen Original actor and Original planning dynamics; MPC; no random action/model replacement',
            residual_precision='float32, CPU norm' if config.get('family')=='maniskill' else 'float32, native-device norm',
            smoke_only=role=='smoke',statistically_qualified_calibration=False,model_unchanged=True,
            elapsed_seconds=time.perf_counter()-started)
        result['result_file']=write_json(root/'collection.json',result)
        return result
    finally:
        agent.env.close()


def score_buffer(config):
    """Produce every incoming trajectory residual after calibration is locked.

    This is inference-only. All stored trajectories, including entirely
    unmodified trajectories, contribute. No simulator or fitting is required.
    """
    from .runtime import load_agent
    from .pipeline import _state_snapshot,_unchanged
    from swaap.evaluation.detector import disjoint,load as load_residuals,score as score_residuals
    calibration_path=asset(config,'calibration');lock=read_json(calibration_path)
    require('thresholds' in lock and set(lock['thresholds'])=={'0.05','0.10'},'A frozen nominal5%/10% threshold lock is required before poison access')
    original_sha=digest(asset(config,'original'))
    require(lock['original_sha256']==original_sha,'Calibration and incoming scorer require identical frozen Original')
    path,roots,manifest=root_panel(config,name='trajectory_roots')
    ids=[trajectory_id(config['task'],original_sha,r) for r in roots]
    clean_test_path=asset(config,'clean_test')
    clean_test=load_residuals(clean_test_path)
    require(clean_test['original']==original_sha,'Independent clean test uses a different Original')
    require(len(clean_test['ids'])>=1000,'Independent clean-FPR cohort needs at least1000 complete trajectories')
    for key,value in lock.get('reference_metadata',{}).items():
        require(clean_test['metadata'].get(key)==value,'Independent clean test reference metadata differs: '+key)
    disjoint(lock['statistics_ids'],lock['threshold_ids'],clean_test['ids'],ids)
    expected=expected_episode_length(config['task']);rows=len(roots)*expected
    clean=load_buffer(asset(config,'clean'),rows=rows)
    poison=load_buffer(asset(config,'poison'),rows=rows)
    if isinstance(manifest,dict) and manifest.get('clean_bank_sha256'):
        require(manifest['clean_bank_sha256']==digest(asset(config,'clean')),'Trajectory manifest belongs to a different source bank')
    agent=load_agent(config,environment=False);before=_state_snapshot(agent)
    clean_raw=[];poison_raw=[];started=time.perf_counter()
    for i in range(len(roots)):
        lo,hi=i*expected,(i+1)*expected
        clean_raw.append(residual_episode(tuple(x[lo:hi]for x in clean),agent,family=config.get('family','tdmpc2'),expected_steps=expected))
        poison_raw.append(residual_episode(tuple(x[lo:hi]for x in poison),agent,family=config.get('family','tdmpc2'),expected_steps=expected))
    _unchanged(agent,before)
    root=output(config)
    file=_write_residuals(root/'poison_residuals.npz',np.stack(poison_raw),ids,
        original_sha=original_sha,task=config['task'],role='incoming_poison',roots=roots)
    clean_file=_write_residuals(root/'matched_clean_residuals.npz',np.stack(clean_raw),ids,
        original_sha=original_sha,task=config['task'],role='matched_source_clean',roots=roots)
    cusum=score_residuals(calibration_path,clean_test_path,file,root/'cusum.json')
    result=dict(status='COMPLETE_INCOMING_BUFFER_RESIDUALS',task=config['task'],poison_residuals=file,
        matched_source_clean_residuals=clean_file,trajectories=len(roots),rows=rows,
        all_stored_trajectories_scored=True,prediction_batch_rows=expected,original_sha256=original_sha,
        calibration_sha256=digest(calibration_path),clean_bank_sha256=digest(asset(config,'clean')),
        independent_clean_test_sha256=digest(clean_test_path),cusum=cusum,cusum_file=str(root/'cusum.json'),
        poison_bank_sha256=digest(asset(config,'poison')),new_environment_episodes=0,new_fits=0,
        new_poison_constructions=0,model_unchanged=True,elapsed_seconds=time.perf_counter()-started)
    result['result_file']=write_json(root/'buffer_residuals.json',result)
    return result
