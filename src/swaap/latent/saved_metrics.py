"""CPU reference scoring and a separate explicit Original-inference entry point."""
from __future__ import annotations
from pathlib import Path
import math
import statistics
from .deviation import episode_delta_m
from .io import asset,digest,output,read_json,require,write_json


def original_on_matched_clean(config):
    """Score saved Original predictions on exact matched Clean rollouts.

    The manifest has one record per episode. Each record explicitly identifies
    the predictor as Original and the rollout as matched Clean, and supplies a
    trusted NPZ containing both realized latents and saved Original predictions.
    A condition-model prediction is not an interchangeable substitute.
    """
    import numpy as np
    root=output(config)
    manifest_path=asset(config,'saved_rollouts')
    manifest=read_json(manifest_path)
    records=manifest['episodes']
    require(isinstance(records,list) and records,'A nonempty saved episode manifest is required')
    episodes=[];seen=set();identity={}
    for row in records:
        require(row.get('predictor_role')=='Original' and row.get('rollout_role')=='matched Clean',
                'delta_m_ref requires Original predictions on matched Clean rollouts; no own-rollout substitution')
        for key in ('task','stage1_run','fit_order','episode_index','environment_seed','planner_seed','expected_steps',
                    'original_checkpoint_sha256','clean_fitted_checkpoint_sha256','arrays'):
            require(key in row,'Missing saved-rollout identity field: '+key)
        require(all(isinstance(row[k],str) and len(row[k])==64 for k in ('original_checkpoint_sha256','clean_fitted_checkpoint_sha256')),
                'Explicit Original and matched Clean checkpoint SHA256 identities are required')
        key=(row['task'],str(row['stage1_run']),str(row['fit_order']),int(row['episode_index']))
        require(key not in seen,'Duplicate saved Clean episode');seen.add(key)
        order=key[:3]
        binding=(row['original_checkpoint_sha256'],row['clean_fitted_checkpoint_sha256'])
        require(order not in identity or identity[order]==binding,'A Clean fitting order mixes checkpoint identities')
        identity[order]=binding
        path=Path(row['arrays']).expanduser()
        if not path.is_absolute():path=manifest_path.parent/path
        require(path.is_file(),'Missing saved rollout array: '+str(path))
        sha=digest(path)
        require(not row.get('arrays_sha256') or row['arrays_sha256']==sha,'Saved rollout bytes differ')
        with np.load(path,allow_pickle=False) as data:
            require('latents' in data.files,'Missing realized Clean rollout latents')
            require('original_predicted_next_latents' in data.files,
                    'MISSING_MEASUREMENT: Original predictions on these exact matched Clean transitions were not saved. Model forwards would be necessary and are not run by this command.')
            value=episode_delta_m(data['latents'],data['original_predicted_next_latents'],expected_steps=int(row['expected_steps']))
        episodes.append(dict(task=row['task'],stage1_run=str(row['stage1_run']),fit_order=str(row['fit_order']),
            episode_index=int(row['episode_index']),environment_seed=int(row['environment_seed']),planner_seed=int(row['planner_seed']),
            original_checkpoint_sha256=binding[0],clean_fitted_checkpoint_sha256=binding[1],delta_m_ref=value,
            arrays=str(path.resolve()),arrays_sha256=sha))
    grouped={}
    for row in episodes:grouped.setdefault((row['task'],row['stage1_run'],row['fit_order']),[]).append(row)
    fits=[];by_run={}
    expected_episodes=int(config.get('evaluation',{}).get('episodes',100))
    for key,rows in sorted(grouped.items()):
        require(len(rows)==expected_episodes and sorted(r['episode_index'] for r in rows)==list(range(expected_episodes)),
                'Incomplete matched Clean episode panel; no missing-value imputation')
        mean=math.fsum(r['delta_m_ref'] for r in rows)/len(rows)
        fits.append(dict(task=key[0],stage1_run=key[1],fit_order=key[2],episodes=len(rows),delta_m_ref=mean))
        by_run.setdefault(key[:2],[]).append(mean)
    runs=[];by_task={};expected_orders=int(config.get('aggregation',{}).get('fit_orders',3))
    for key,values in sorted(by_run.items()):
        require(len(values)==expected_orders,'Incomplete nested fitting-order panel')
        mean=math.fsum(values)/len(values)
        runs.append(dict(task=key[0],stage1_run=key[1],fitting_orders=len(values),delta_m_ref=mean))
        by_task.setdefault(key[0],[]).append(mean)
    tasks=[dict(task=task,independent_stage1_runs=len(values),mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None) for task,values in sorted(by_task.items())]
    result=dict(status='COMPLETE_SAVED_ARRAY_METRIC',metric='Original-reference delta_m on matched Clean rollouts',
        predictor_role='Original',rollout_role='matched Clean',episodes=episodes,fit_orders=fits,stage1_runs=runs,tasks=tasks,
        manifest_sha256=digest(manifest_path),new_model_forwards=0,new_rollouts=0,missing_values_imputed=False)
    result['result_file']=write_json(root/'aggregate.json',result)
    return result


def predict_original_episode(agent,latents,actions,torch):
    """The adopted score_entry forward, one float32 call for the full episode.

    No environment call, actor call, fitting, projection or metric normalization
    occurs here. The returned array is the frozen Original's prediction on the
    saved current states/actions, not a prediction on a different rollout.
    """
    z,action=latents,actions
    with torch.no_grad():
        prediction=agent.model.next(torch.as_tensor(z[:-1],device=agent.device).float(),torch.as_tensor(action,device=agent.device).float(),agent.psi_0)
    return prediction.detach().cpu().numpy()


def produce_original_predictions(config):
    """Explicit optional inference-only producer for saved matched Clean arrays.

    This function is not called by saved-result reproduction or aggregation.
    It requires the frozen Original checkpoint and a qualified scientific GPU
    environment. It never makes a simulator or launches a rollout.
    """
    import numpy as np
    from copy import deepcopy
    from .runtime import load_agent
    root=output(config)
    manifest_path=asset(config,'saved_rollouts')
    manifest=read_json(manifest_path)
    records=manifest['episodes']
    require(isinstance(records,list) and records,'Saved matched Clean episodes required')
    original_sha=digest(asset(config,'original'))
    first_path=Path(records[0]['arrays'])
    if not first_path.is_absolute():first_path=manifest_path.parent/first_path
    with np.load(first_path,allow_pickle=False) as arrays:
        require('observations' in arrays.files and 'actions' in arrays.files,
                'Model-only loading requires retained observation/action dimensions')
        obs_shape=list(arrays['observations'].shape[1:])
        action_dim=int(arrays['actions'].shape[-1])
    bound=deepcopy(config);bound.setdefault('model',{})
    if 'obs_shape' in bound['model']:
        require(tuple(bound['model']['obs_shape']['state'])==tuple(obs_shape),'Configured observation geometry differs from saved rollout')
    if 'action_dim' in bound['model']:
        require(bound['model']['action_dim']==action_dim,'Configured action geometry differs from saved rollout')
    bound['model']['obs_shape']={'state':obs_shape};bound['model']['action_dim']=action_dim
    agent=load_agent(bound,environment=False)
    calls=0;rows_predicted=0;scoring=[]
    for index,row in enumerate(records):
        require(row.get('rollout_role')=='matched Clean' and row.get('task')==config['task'],
                'Only saved matched Clean rollouts for the declared task are admitted')
        require(row.get('original_checkpoint_sha256')==original_sha,'Frozen Original checkpoint differs from saved-rollout provenance')
        path=Path(row['arrays'])
        if not path.is_absolute():path=manifest_path.parent/path
        require(path.is_file(),'Missing saved Clean arrays')
        before=digest(path)
        require(not row.get('arrays_sha256') or row['arrays_sha256']==before,'Saved Clean rollout bytes differ')
        with np.load(path,allow_pickle=False) as arrays:
            values={key:np.asarray(arrays[key]) for key in arrays.files}
        require('original_predicted_next_latents' not in values,
                'Original predictions already survive; reuse them through CPU aggregation instead of replaying inference')
        z,action=values['latents'],values['actions']
        require(z.dtype==action.dtype==np.dtype('float32') and z.ndim==action.ndim==2 and len(z)==len(action)+1,
                'Exact float32 saved Original-encoder latent/action geometry required')
        require(len(action)==int(row['expected_steps']) and bool(np.isfinite(z).all()) and bool(np.isfinite(action).all()),
                'Incomplete/nonfinite saved Clean trajectory')
        values['original_predicted_next_latents']=predict_original_episode(agent,z,action,__import__('torch'))
        calls+=1;rows_predicted+=len(action)
        require(digest(path)==before,'Saved Clean arrays changed while scoring')
        destination=root/f'clean_reference_{index:04d}.npz'
        with destination.open('xb') as stream:np.savez_compressed(stream,**values)
        item=dict(row,predictor_role='Original',arrays=str(destination.resolve()),arrays_sha256=digest(destination),
                  source_clean_arrays_sha256=before)
        scoring.append(item)
    manifest_file=write_json(root/'saved_rollouts.json',dict(episodes=scoring))
    result=dict(status='COMPLETE_INFERENCE_ONLY_SAVED_CLEAN',manifest=manifest_file,
        original_checkpoint_sha256=original_sha,original_forward_calls=calls,predicted_transition_rows=rows_predicted,
        new_environment_episodes=0,new_victim_fits=0,new_poison_constructions=0,source_clean_arrays_modified=False)
    result['result_file']=write_json(root/'evaluate.json',result)
    return result
