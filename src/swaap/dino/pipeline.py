"""Explicit DINO-WM workflow with retained history/BF16/planner semantics.

The separately installed, pinned upstream DINO-WM runtime supplies architecture,
PushT and planning modules. No private notebook, controller, host or path is used.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from swaap.latent.io import asset, output, save_tensor, write_json, require


def _source(config):
    import sys
    root=Path(config.get('runtime',{}).get('dino_wm_source',Path(__file__).with_name('_vendor'))).expanduser().resolve()
    require((root/'models/visual_world_model.py').is_file(),'Missing bundled DINO-WM runtime sources; install the complete release source distribution')
    sys.path.insert(0,str(root))
    from . import _executed_runtime as source
    return source


def _runtime(config, *, planning=False):
    import torch
    import numpy as np
    import hydra
    from omegaconf import OmegaConf
    if not config.get('runtime',{}).get('allow_unqualified_software',False):
        require(str(torch.__version__)=='2.9.0.dev20250723+cu128' and str(torch.version.cuda)=='12.8','Recorded DINO execution requires Torch2.9.0.dev20250723+cu128; another stack must be explicitly labeled exploratory')
    require(torch.cuda.is_available(),'The executed DINO model path requires CUDA; CPU tests use synthetic tensors only')
    require(torch.cuda.device_count()==1,'Expose one GPU per independent DINO run')
    require(config.get('task')=='pusht','This family preserves the executed PushT runtime; a task change needs matching family assets and an explicit adapter')
    source=_source(config)
    cfg=OmegaConf.load(asset(config,'planning_config'))
    cfg.saved_folder=str(output(config));model_cfg=OmegaConf.load(asset(config,'model_config'))
    # Use an explicit public cache and the verified backbone asset, not a lab cache.
    # The architecture code is fetched by the retained, commit-pinned torch.hub call.
    import shutil
    cache=Path(config.get('runtime',{}).get('cache_dir','artifacts/cache/torch')).expanduser().resolve()
    torch.hub.set_dir(str(cache/'hub'))
    weights=cache/'hub/checkpoints/dinov2_vits14_pretrain.pth'
    weights.parent.mkdir(parents=True,exist_ok=True)
    backbone=asset(config,'backbone')
    from swaap.latent.io import digest
    if weights.exists():
        require(digest(weights)==digest(backbone),'Cached backbone differs from verified asset')
    else:
        shutil.copyfile(backbone,weights)
    source.seed(99)
    model=source.get_cached_model(asset(config,'original'),model_cfg,model_cfg.num_action_repeat,torch.device('cuda:0'))
    model.requires_grad_(False)
    rt=SimpleNamespace(torch=torch,np=np,source=source,cfg=cfg,model_cfg=model_cfg,original=model,seed=source.seed)
    if planning:
        from . import stats
        rt.dataset=SimpleNamespace(action_dim=2,action_mean=stats.ACTION_MEAN,action_std=stats.ACTION_STD,state_mean=stats.STATE_MEAN,state_std=stats.STATE_STD,proprio_mean=stats.PROPRIO_MEAN,proprio_std=stats.PROPRIO_STD,transform=hydra.utils.instantiate(model_cfg.env.dataset.transform))
        rt.env=source.get_cached_env(model_cfg.env.name,1,list(model_cfg.env.args),dict(model_cfg.env.kwargs),True)
        c=source.get_config();c.update(over_all_element=True,rep_dim=10)
        rt.policy=source.HIQLAgent(torch.rand(1024,3,64,64,device='cuda'),torch.rand(1024,10,device='cuda'),c).cuda()
        saved=torch.load(asset(config,'hiql'),map_location='cuda',weights_only=False)
        rt.policy.load_state_dict(saved['theta_state_dict'],strict=True)
        for actor in (rt.policy.low_actor,rt.policy.high_actor):actor.const_std,actor.heuristic_std=True,.3
        rt.policy.requires_grad_(False)
    return rt


def _bank(config):
    import torch
    data=torch.load(asset(config,'clean'),map_location='cpu',weights_only=False)
    require('fitting' in data and 'reference' in data,'DINO clean packet must contain fitting/reference five-tuples')
    train,reference=data['fitting'],data['reference']
    require(train[0].shape==(320,3,196,404) and train[3].shape==(320,196,404),'Retained DINO N320/history3/token196/full404 layout mismatch')
    require(reference[0].shape==(80,3,196,404),'Retained DINO reference80 mismatch')
    return train,reference


def _eval_frozen(model):
    """The executed VWorldModel.eval() returns None; never chain its result."""
    model.eval()
    model.requires_grad_(False)
    return model


def _target(config,rt):
    value=rt.torch.load(asset(config,'target'),map_location='cpu',weights_only=False)
    from .direct import predictor_replacement
    model,verification=predictor_replacement(rt,value['predictor'])
    return model


def _predict(model,data):
    import torch
    from .science import latent_next
    dtype=next(model.parameters()).dtype;device=next(model.parameters()).device
    values=[]
    with torch.no_grad():
        for i in range(0,len(data[0]),4):
            values.append(latent_next(model,data[0][i:i+4].to(device=device,dtype=dtype),data[1][i:i+4].to(device=device,dtype=dtype)).detach().cpu().float())
    return torch.cat(values)


from contextlib import contextmanager


@contextmanager
def _working_directory(directory):
    """Contain legacy relative diagnostic writes; restore the caller on exit."""
    import os
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    previous=Path.cwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)


def _goal(rt,row,model,output_directory):
    torch,np,source=rt.torch,rt.np,rt.source
    cfg=deepcopy(rt.cfg);cfg.seed=row['global_seed']
    ws=source.SimulationWorkspace(cfg,model,rt.dataset,rt.env,rt.model_cfg.env.name,rt.model_cfg.frameskip,None,None)
    source.workspace=ws
    ws.obs_0,ws.obs_g=row['obs_0'],row['obs_g']
    ws.state_0,ws.state_g,ws.eval_seed=row['state_0'],row['state_g'],row['environment_roots']
    rt.seed(row['global_seed'])
    observed,_=rt.env.prepare(ws.eval_seed,ws.state_0)
    require(np.array_equal(observed['visual'],ws.obs_0['visual'][:,0]),'Goal visual/reset join mismatch')
    require(np.allclose(observed['proprio'],ws.obs_0['proprio'][:,0],rtol=0,atol=1e-6),'Goal proprio/reset join mismatch')
    # The source preview consumes policy/planner random numbers; retain it.
    with torch.no_grad():
        obs=source.resize_obs(ws.obs_0).cuda()
        target=source.resize_obs(ws.obs_g['visual'].squeeze(0).squeeze(0)).cuda()
        action=rt.policy.sample_actions(obs,target)
        raw=source.move_to_device(ws.preprocessor.transform_obs(ws.obs_0),'cuda')
        latent=model.encode(raw,action.unsqueeze(1))
        require(bool(model.predict(latent[:,-cfg.num_hist:]).isfinite().all()),'Nonfinite planning preview')
        # Retained planning writes logs.json/plots by relative filename. Keep
        # those optional diagnostics beside this goal without changing actions,
        # random streams, rewards, or planning inputs.
        with _working_directory(output_directory):
            ws.interact_mpc(ws.obs_0,ws.obs_g,ws.state_0,ws.state_g,return_success_only=True)
    logs,success,actual_obs,actual_states,_=ws.last_evaluation
    actions=torch.cat(ws.planner.planned_actions,dim=1).detach().cpu()
    packet={'record':row,'normalized_grouped_actions':actions,'observed':actual_obs,'physical_states':actual_states,'final_metrics':{k:float(v) for k,v in logs.items()},'success':bool(success[0]),'action_len':ws.planner.action_len}
    result={'record_index':row['record_index'],'pair_identity':row['pair_identity'],'global_seed':row['global_seed'],'environment_roots':row['environment_roots'],'success':bool(success[0]),'state_dist':float(logs['mean_state_dist']),'final_native_steps':int(actual_states.shape[1]-1),'metrics':packet['final_metrics']}
    return result,packet


def _fitting_origins(collection):
    """Same controlled-then-untouched concatenation as the fitting tensors."""
    return [{'record_index':int(row['record_index']),'local_window':local,
             'last_current_grouped_step':local+2,'successor_grouped_step':local+3}
            for role in ('controlled','untouched') for row in collection if row['role']==role
            for local in range(8)]


def prepare_data(config):
    mode=config.get('data',{}).get('mode','')
    if mode in ('detector_collect','detector_saved'):
        from .detection import collect,prepare_saved
        return collect(config) if mode=='detector_collect' else prepare_saved(config)
    import torch
    root=output(config)
    if config.get('data',{}).get('mode','published_bank')=='published_bank':
        train,ref=_bank(config)
        return {'status':'VERIFIED_BANK_LAYOUT','fitting_rows':len(train[0]),'reference_rows':len(ref[0]),'note':'Reuses the published observed-transition bank; no environment collection'}
    # Exactly ten genuine grouped actions plus the source next-action preview.
    # Each trajectory contributes eight history3 windows, including its true tail.
    from .collection import collect
    rt=_runtime(config,planning=True)
    records=torch.load(asset(config,'records'),map_location='cpu',weights_only=False)
    require(len(records)==90,'Retained record roles must include 50 collection and 40 evaluation pairs')
    parts={name:[] for name in ('reference','controlled','untouched')};inventory=[]
    for row in records[:50]:
        values,metadata=collect(rt,row,root,root)
        require(values[0].shape==(8,3,196,404),'Fixed10 collection window count mismatch')
        parts[row['role']].append(values);inventory.append(metadata)
    roles={role:tuple(torch.cat([p[i] for p in values],dim=0) for i in range(5)) for role,values in parts.items()}
    require([len(roles[k][0]) for k in ('reference','controlled','untouched')]==[80,80,240],'Role quotas changed')
    fitting=tuple(torch.cat([roles['controlled'][i],roles['untouched'][i]],dim=0) for i in range(5))
    origins=_fitting_origins(inventory)
    require(len(origins)==320,'Fresh fitting trajectory/window identity count changed')
    save_tensor(root/'clean.pt',{'fitting':fitting,'reference':roles['reference'],'collection':inventory,'roles_disjoint':True,'window_origins':origins})
    from swaap.latent.io import digest
    reference_model=_eval_frozen(deepcopy(rt.original))
    require(all(p.dtype==torch.float32 for p in reference_model.parameters()),'Incoming-bank Original predictions must remain FP32')
    save_tensor(root/'reference_predictions.pt',{'Original_FP32_fitting':_predict(reference_model,fitting),
        'original_sha256':digest(asset(config,'original')),'clean_sha256':digest(root/'clean.pt'),'window_origins':origins})
    write_json(root/'collection.json',{'trajectories':inventory,'cross_boundary_windows':0,'padded_or_duplicated_transitions':0})
    return {'bank':str(root/'clean.pt'),'reference_predictions':str(root/'reference_predictions.pt'),'fitting_rows':320,'reference_rows':80}


def stage1(config):
    import json
    rt=_runtime(config,planning=True);root=output(config);source=rt.source
    cfg=deepcopy(rt.cfg);cfg.lambda_coef=1000;cfg.n_evals=1;cfg.fix_seed=False;cfg.seed=99
    records=rt.torch.load(asset(config,'sampler_records'),map_location='cpu',weights_only=False)
    class SavedRecordWorkspace(source.SimulationWorkspace):
        def refresh_targets(self):
            row=records[int(self.record_rng.randint(len(records)))]
            self.obs_0,self.obs_g=row['obs_0'],row['obs_g']
            self.state_0,self.state_g,self.eval_seed=row['state_0'],row['state_g'],row['environment_roots'];self.gt_actions=None
    ws=SavedRecordWorkspace(cfg,rt.original,rt.dataset,rt.env,rt.model_cfg.env.name,5,None,None)
    ws.record_rng=rt.np.random.RandomState(99);source.workspace=ws
    policy=deepcopy(rt.policy).requires_grad_(True);rt.seed(99)
    learner=source.ThetaPsi(cfg,rt.env,policy,ws)
    frozen={k:v.detach().cpu().clone() for k,v in learner.psi.state_dict().items() if not k.startswith('predictor.')}
    iterations=int(config.get('stage1',{}).get('iterations',50))
    with (root/'stage1.jsonl').open('x') as stream:
        for completed in range(1,iterations+1):
            source.step=completed-1;info=learner.update()
            require(all(rt.torch.equal(learner.psi.state_dict()[k].cpu(),v) for k,v in frozen.items()),'Stage1 changed a frozen DINO component')
            values={k:float(v) if not hasattr(v,'item') else v.item() for k,v in info.items()}
            stream.write(json.dumps({'update':completed,'source_info':values},allow_nan=False)+'\n')
            if completed in (1,10,25,50) or completed==iterations:
                save_tensor(root/('target_'+str(completed)+'.pt'),{'completed_updates':completed,'predictor':learner.psi.predictor.state_dict(),'theta':learner.theta.state_dict(),'theta_star':learner.theta_star.state_dict(),'psi_optimizer':learner.psi_optimizer.state_dict()})
    return {'iterations':iterations,'snapshots':[str(p) for p in root.glob('target_*.pt')],'stage1_precision':'FP32','non_predictor_components_unchanged':True,'retained_target_eligible':iterations>=25}


def select(config):
    """Bind the prospectively retained update25; do not invent K20 selection."""
    import torch
    from swaap.latent.io import digest
    root=output(config);path=asset(config,'target')
    value=torch.load(path,map_location='cpu',weights_only=False)
    require('predictor' in value,'Expected retained DINO predictor checkpoint')
    update=int(value.get('completed_updates',config.get('stage1',{}).get('retained_target_update',25)))
    require(update==25,'The adopted DINO target is prospectively fixed at update25')
    result={'selected_target':str(path),'sha256':digest(path),'update':update,'selection':'prospectively fixed update25; no K20 or outcome target selection'}
    write_json(root/'target.json',result);return result


def construct(config):
    import json
    import torch
    from .science import GradientEngine,project,deviation
    root=output(config);rt=_runtime(config);train,ref=_bank(config)
    original=_eval_frozen(deepcopy(rt.original).bfloat16())
    target=_eval_frozen(_target(config,rt).bfloat16())
    predictions=_predict(original,train);target_train=_predict(target,train);target_ref=_predict(target,ref)
    scores=(target_train[:80,:,:394]-predictions[:80,:,:394]).square().flatten(1).mean(1)
    support=torch.argsort(scores,descending=True,stable=True)[:32]
    save_tensor(root/'support.pt',{'indices':support,'scores':scores,'rule':'top32 target discrepancy restricted to first80'})
    train_gpu=[v.cuda().bfloat16() for v in train];ref_gpu=[v.cuda().bfloat16() for v in ref]
    reference_labels=torch.cat([target_ref[...,:394],ref[3][...,394:]],dim=-1)
    original.predictor.requires_grad_(True)
    engine=GradientEngine(original,train_gpu,ref_gpu,reference_labels,support,predictions,train[3]);engine.cache()
    alpha=float(config.get('alpha',.9));require(alpha in (0.,.9),'Retained ordinary DINO alpha configurations are 0 and .9')
    require(float(config.get('poison_fraction',.1))==.1,'Retained DINO editable support is32/320 with first80 restriction')
    settings=config['construction'];require(float(settings.get('epsilon',1))==1,'Retained DINO projector uses epsilon1')
    master=engine.clean.clone()
    with (root/'construction.jsonl').open('x') as stream:
        for step in range(int(settings.get('steps',5000))):
            gradient,row=engine.objective(master,alpha)
            master=project(master-float(settings.get('step_size',10))*gradient,engine.clean,epsilon=1).detach()
            stream.write(json.dumps(dict(row,update=step+1),allow_nan=False)+'\n')
    labels=train[3].clone();labels[support,:,:394]=master.detach().cpu()
    fixed=torch.ones(320,dtype=torch.bool);fixed[support]=False
    require(torch.equal(labels[fixed],train[3][fixed]) and torch.equal(labels[...,394:],train[3][...,394:]),'DINO frozen labels changed')
    require(float((labels.double()-train[3].double()).abs().max())<=1,'Canonical epsilon exceeded')
    save_tensor(root/'poison.pt',{'labels':labels,'support':support,'alpha':alpha,'K':settings.get('steps',5000),'all_other_tuple_fields_unchanged':True})
    result={'poison':str(root/'poison.pt'),'support':support.tolist(),'construction_BF16_delta_d_diagnostic':deviation(labels,predictions),'construction_BF16_clean_delta_d_diagnostic':deviation(train[3],predictions)}
    write_json(root/'construction.json',result);engine.release_graphs();return result


def fit(config):
    import torch
    from .fitting import update
    from .orders import make_source_slate,slate_hash
    recipe=config.get('fitting',{})
    require(float(recipe.get('learning_rate',2e-5))==2e-5 and int(recipe.get('epochs',10))==10 and int(recipe.get('batch_size',16))==16,'Retained DINO fit is fixed AdamW2e-5/E10/B16; changed settings are not silently ignored')
    root=output(config);rt=_runtime(config);data,_=_bank(config)
    z,a,y=(data[i].detach().cpu().clone() for i in (0,1,3))
    if config.get('method','swaap')!='clean':
        value=torch.load(asset(config,'poison'),map_location='cpu',weights_only=False);y=value['labels'].detach().cpu().clone()
    seed=int(config.get('seed',99));slate=make_source_slate(torch,seed)
    prepared=deepcopy(rt.original).to(dtype=torch.bfloat16)
    rt.seed(seed)
    dataset=torch.utils.data.TensorDataset(z,a,y,torch.arange(320))
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
    model=deepcopy(prepared).cuda();model.requires_grad_(False);model.predictor.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.predictor.parameters(),lr=2e-5)
    limit=int(config.get('smoke',{}).get('fit_updates',200))
    require(1<=limit<=200 and (limit==200 or config.get('smoke',{}).get('enabled') is True),'Reduced fitting is an explicitly labeled integration prefix only')
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('predictor.')};updates=0
    import json
    with (root/'fitting.jsonl').open('x') as stream:
        model.train()
        for epoch in range(10):
            seen=[]
            for zz,aa,yy,idx in loader:
                seen.extend(idx.tolist())
                loss=update(model,zz.cuda().bfloat16(),aa.cuda().bfloat16(),yy.cuda().bfloat16(),optimizer);updates+=1
                stream.write(json.dumps({'update':updates,'epoch':epoch+1,'loss':float(loss.detach()),'indices':idx.tolist()})+'\n')
                if updates==limit:break
            require(seen==slate[epoch][:len(seen)],'Native DataLoader order diverged from prospective slate')
            if updates==limit:break
    require(updates==limit,'DINO update count mismatch')
    require(all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in frozen.items()),'Frozen DINO component changed')
    path=save_tensor(root/'fit.pt',{'full_source_fit_state_dict':model.state_dict(),'fit_seed':seed,'fitting_slate':slate,'updates':updates,'source_schedule_updates':200,'engineering_prefix':updates<200,'lr':2e-5,'epochs':10})
    write_json(root/'fit.json',{'seed':seed,'slate_sha256':slate_hash(slate),'updates':updates,'engineering_prefix':updates<200,'BF16_training_FP32_deployment':True})
    return {'checkpoint':path,'updates':updates,'engineering_prefix':updates<200}


def evaluate(config):
    require(config.get('method') in ('original','direct') or bool(config.get('assets',{}).get('checkpoint')),'Fitted DINO evaluation requires assets.checkpoint; Original must be explicitly method=original')
    root=output(config);rt=_runtime(config,planning=True);model=rt.original
    if config.get('method')=='direct':model=_target(config,rt)
    elif config.get('assets',{}).get('checkpoint'):
        value=rt.torch.load(asset(config,'checkpoint'),map_location='cpu',weights_only=False)
        model=deepcopy(rt.original);model.load_state_dict(value['full_source_fit_state_dict'],strict=True)
        modules=dict(model.named_modules())
        for name,module in rt.original.named_modules():modules[name].training=module.training
        model.requires_grad_(False)
    records=rt.torch.load(asset(config,'goals'),map_location='cpu',weights_only=False)
    maximum=int(config.get('smoke',{}).get('evaluation_goals',len(records)))
    require(maximum==len(records) or config.get('smoke',{}).get('enabled') is True,'Reduced evaluation is integration-only')
    records=records[:maximum];rows=[]
    for record in records:
        row,packet=_goal(rt,record,model,root/('goal_'+str(record['record_index'])+'_diagnostics'));rows.append(row)
        save_tensor(root/('goal_'+str(record['record_index'])+'.pt'),packet)
    result={'goals':len(rows),'records':rows,'success_fraction':sum(r['success'] for r in rows)/len(rows),'mean_state_dist':sum(r['state_dist'] for r in rows)/len(rows),'statistical_unit':'one goal pair; fit seeds are nested on the same target/bank/poison'}
    write_json(root/'evaluation.json',result);return result


def aggregate(config):
    """Summarize nested fitting orders without inventing independent attacks."""
    import json
    import numpy as np
    if config.get('aggregation',{}).get('mode')=='incoming_metrics':
        return _incoming_metrics(config)
    root=output(config)
    paths=config.get('result_files',[])
    require(bool(paths),'Provide result_files as an explicit list of evaluation.json paths')
    values=[json.loads(Path(p).read_text()) for p in paths]
    identities=[[r['pair_identity'] for r in v['records']] for v in values]
    require(all(ids==identities[0] for ids in identities),'DINO goal identities/order differ across fitting results')
    rates=[float(np.mean([r['success'] for r in v['records']])) for v in values]
    distances=[float(np.mean([r['state_dist'] for r in v['records']])) for v in values]
    result={'fitting_runs':len(values),'goal_pairs_per_fit':len(identities[0]),'success_fraction_per_fit':rates,
        'mean_success_fraction':float(np.mean(rates)),'sample_SD_over_nested_fit_runs':float(np.std(rates,ddof=1)) if len(rates)>1 else None,
        'mean_state_dist_per_fit':distances,'independent_attack_count':1,
        'statistical_unit':'one fixed target/bank/support/poison; fitting runs are nested variations, not independent attacks',
        'source_files':[str(Path(p).resolve()) for p in paths]}
    write_json(root/'summary.json',result);return result


def _incoming_metrics(config):
    """Recompute paper-native incoming metrics from saved FP32 arrays only."""
    import numpy as np
    import torch
    from .metrics import normalized_deviation,residuals_from_predictions
    from swaap.latent.io import digest
    root=output(config)
    packet=torch.load(asset(config,'clean'),map_location='cpu',weights_only=False)
    clean=packet['fitting']
    reference_packet=torch.load(asset(config,'reference_predictions'),map_location='cpu',weights_only=False)
    if reference_packet.get('clean_sha256'):
        require(reference_packet['clean_sha256']==digest(asset(config,'clean')),'Saved Original predictions belong to a different fitting bank')
    if reference_packet.get('original_sha256'):
        require(reference_packet['original_sha256']==digest(asset(config,'original')),'Saved predictions belong to a different Original')
    predictions=reference_packet['Original_FP32_fitting']
    poison=torch.load(asset(config,'poison'),map_location='cpu',weights_only=False)
    labels=poison['labels'];support=poison['support']
    result={'domain':'Canonical FP32 incoming-bank labels/reference predictions; no BF16 cast; no rollouts',
        'clean_delta_d_ref':normalized_deviation(clean[3],predictions),
        'poison_delta_d':normalized_deviation(labels,predictions),
        'selected32_delta_d':normalized_deviation(labels,predictions,support),
        'model_forwards':0,'environment_episodes':0}
    origins=packet.get('window_origins')
    if origins is not None:
        clean_residuals,ids=residuals_from_predictions(clean[3],predictions,origins)
        poison_residuals,poison_ids=residuals_from_predictions(labels,predictions,origins)
        require(ids==poison_ids and len(ids)==40,'Incoming trajectories must retain all40 eight-window sequences')
        original_sha=digest(asset(config,'original'))
        for name,matrix in [('matched_clean',clean_residuals),('poison',poison_residuals)]:
            path=root/(name+'_incoming_residuals.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream,residuals=matrix,trajectory_ids=np.asarray(ids,dtype=str),original_sha256=np.asarray(original_sha))
        result.update(incoming_trajectories=len(ids),windows_per_trajectory=8,
            raw_residual_files={'matched_clean':str(root/'matched_clean_incoming_residuals.npz'),'poison':str(root/'poison_incoming_residuals.npz')},
            detector_rule='Use shared incoming-buffer detector with frozen200/200/400 clean cohorts; DINO bootstrapseed20261104')
    else:
        result['incoming_residual_disposition']='Missing window_origins; normalized deviation available but trajectory CUSUM requires the exact saved eight-window identity join'
    write_json(root/'incoming_metrics.json',result);return result
