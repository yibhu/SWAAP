"""Raw-observation surrogate construction and encoder-distinct MT80 fitting.

All paths are supplied in configuration. No model or simulator starts on import.
The paper gray-box study reuses a retained single-task Stage-1 target; fresh
single-task target generation belongs to the common TD-MPC2 Stage-1 command.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from swaap.latent.io import asset as _ordinary_asset, output, save_tensor, write_json, load_buffer, require


def asset(config, name, *, required=True):
    """Resolve bundled recipe inputs independently of the caller's directory."""
    value=config.get('assets',{}).get(name)
    if isinstance(value,str) and value.startswith('config:'):
        require(bool(config.get('_config_dir')),'config: inputs require a loaded configuration file')
        base=Path(config['_config_dir']).resolve()
        path=(base/value[7:]).resolve()
        require(path==base or base in path.parents,'config: inputs cannot escape their configuration directory')
        amended=dict(config,assets=dict(config['assets'],**{name:str(path)}))
        return _ordinary_asset(amended,name,required=required)
    return _ordinary_asset(config,name,required=required)


class SingleModel:
    multitask = False
    def __init__(self, model, device):
        self.raw, self.device = model, device
    def encode(self, obs):
        flat = obs.ndim == 1
        z = self.raw.encode(obs.unsqueeze(0) if flat else obs, None)
        return z[0] if flat else z
    def next(self, z, action, dynamics):
        import torch
        return dynamics(torch.cat([z, action], dim=-1))


class TaskModel:
    def __init__(self, model, device, task_index):
        self.raw, self.device, self.task_index = model, device, task_index
    def encode(self, observation):
        flat = observation.ndim == 1
        x = observation.unsqueeze(0) if flat else observation
        import torch.nn.functional as F
        require(x.shape[-1]<=39, "MT80 observation width exceeds39")
        x=F.pad(x,(0,39-x.shape[-1]))
        value = self.raw.encode(x, self.task_index)
        return value[0] if flat else value
    def next(self, z, action, dynamics):
        import torch.nn.functional as F
        require(dynamics is self.raw._dynamics,'MT80 next-state reference must use the bound dynamics')
        flat=z.ndim==1
        if flat:z,action=z.unsqueeze(0),action.unsqueeze(0)
        require(action.shape[-1] <= 6, 'MT80 actions exceed checkpoint width')
        value=self.raw.next(z, F.pad(action, (0, 6-action.shape[-1])), self.task_index)
        return value[0] if flat else value


class AgentView:
    def __init__(self, agent, task_index=None, action_dim=4):
        self.raw, self.device = agent, agent.device
        self.model = SingleModel(agent.model, agent.device) if task_index is None else TaskModel(agent.model, agent.device, task_index)
        self.theta, self._prev_mean = agent.model._pi, agent._prev_mean
        self.task_index, self.action_dim = task_index, action_dim
    def act(self, observation, *, psi, theta, t0, eval_mode, task, input_obs):
        require(psi is self.raw.model._dynamics and theta is self.theta and input_obs, 'Gray-box planner binding changed')
        if self.task_index is not None:
            import torch.nn.functional as F
            observation=F.pad(observation,(0,39-observation.shape[-1]))
        value = self.raw.act(observation, t0=t0, eval_mode=eval_mode, task=self.task_index)
        return value if self.task_index is None else value[:self.action_dim]


def _agent(config, multitask=False):
    from swaap.latent.runtime import require_cuda, seed_runtime, make_configuration
    from swaap.latent._vendor.common.layers import api_model_conversion
    from ._agent import TDMPC2
    torch = require_cuda()
    require(config.get('task')=='mw-soccer','Gray-box construction/fitting source is Soccer; choose the explicit evaluation.task for transfer')
    key = 'victim' if multitask else 'original'
    checkpoint = torch.load(asset(config, key), map_location='cpu', weights_only=False)
    source = checkpoint['model']
    base = dict(config, task='mw-soccer')
    cfg = make_configuration(base, asset(config, key))
    cfg.task, cfg.multitask = ('mt80', True) if multitask else ('mw-soccer', False)
    cfg.tasks = checkpoint['metadata']['tasks'] if multitask else ['mw-soccer']
    cfg.task_dim = 96 if multitask else 0
    cfg.obs_shape, cfg.action_dim, cfg.episode_length = {'state':[39]}, 6 if multitask else 4, 100
    if multitask:
        require(cfg.tasks[68]=='mw-soccer' and cfg.tasks[63]=='mw-push', 'MT80 task map mismatch')
        if config.get('assets',{}).get('task_map'):
            import json
            task_map=json.loads(asset(config,'task_map').read_text())
            require([r['embedding'] for r in task_map]==list(range(80)) and [r['task'] for r in task_map]==list(cfg.tasks),'Frozen MT80 task-map mismatch')
            cfg.action_dims=[r['action_dim'] for r in task_map]
            cfg.episode_lengths=[r['outer_steps'] for r in task_map]
        else:
            cfg.action_dims=[4 if i in (63,68) else 0 for i in range(80)]
            cfg.episode_lengths=[100]*80
    seed_runtime(int(config.get('seed',0)))
    agent = TDMPC2(cfg)
    missing, unexpected = agent.model.load_state_dict(api_model_conversion(agent.model.state_dict(), deepcopy(source)), strict=False)
    require(not unexpected and all(k.startswith('_termination.') for k in missing), 'Checkpoint architecture mismatch')
    for name,value in source.items():
        if name.startswith(('_encoder.','_dynamics.','_reward.','_pi.','_task_emb.')):
            require(torch.equal(agent.model.state_dict()[name].cpu(), value), 'Checkpoint conversion changed '+name)
    agent.model.eval().requires_grad_(False)
    return agent


def _target(config, agent):
    import torch
    value = torch.load(asset(config,'target'), map_location='cpu', weights_only=False)
    state = value.get('model',value)
    if 'psi_k' in value:
        # Final gb-e1 consumes target_model from graybox_target_frontier.py,
        # including its native dynamics-only target format. The historical
        # encoder is not embedded in that file; the prospective source encoder
        # stays the supplied Original, exactly as in the executed construction.
        dynamics_state=value['psi_k']
    elif value.get('metadata',{}).get('original_sha256'):
        from swaap.latent.io import digest
        require(value['metadata'].get('task')=='mw-soccer' and value['metadata']['original_sha256']==digest(asset(config,'original')),'Fresh shared-executor target belongs to a different Original/task')
        require(all(k.startswith('psi.') for k in state),'Unexpected fields in a public shared-executor target')
        dynamics_state={k[4:]:v for k,v in state.items() if k.startswith('psi.')}
    else:
        for tp,op in [('_encoder.','_encoder.'),('psi_0.','_dynamics.')]:
            a={k[len(tp):]:v for k,v in state.items() if k.startswith(tp) and isinstance(v,torch.Tensor)}
            b={k[len(op):]:v for k,v in agent.model.state_dict().items() if k.startswith(op)}
            require(a.keys()==b.keys() and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a), 'Surrogate Original/target binding mismatch '+tp)
        dynamics_state={k[4:]:v for k,v in state.items() if k.startswith('psi.')}
    dynamics = deepcopy(agent.model._dynamics)
    dynamics.load_state_dict(dynamics_state,strict=True)
    return dynamics.eval().requires_grad_(False)


def prepare_data(config):
    mode=config.get('data',{}).get('mode','')
    if mode in ('detector_collect','detector_saved'):
        from .detection import collect,prepare_saved
        return collect(config) if mode=='detector_collect' else prepare_saved(config)
    require(config.get('evaluation',{}).get('scope') not in ('breadth','detailed_frozen'),'Breadth reproduction is evaluation-only; load released endpoints with evaluate')
    import torch
    root=output(config)
    raw=load_buffer(asset(config,'clean'),rows=5000)
    reference=load_buffer(asset(config,'reference'),rows=50000)
    require(raw[0].shape==raw[3].shape==(5000,39) and raw[1].shape==(5000,4),'Raw bank shape mismatch')
    require(reference[0].shape==reference[3].shape==(50000,512),'Latent reference mismatch')
    return {'status':'VALIDATED_INPUTS','raw_rows':5000,'reference_rows':50000,'output':write_json(root/'data.json',{'raw_shapes':[list(x.shape) for x in raw],'reference_shapes':[list(x.shape) for x in reference]})}


def stage1(config):
    if config.get('reproduction_path')=='fresh_surrogate':
        from swaap.latent.pipeline import stage1 as common_stage1
        return common_stage1(dict(config,family='tdmpc2'))
    raise ValueError('The adopted gray-box experiment starts from the explicitly retained single-task Soccer target. Use family=tdmpc2 Stage-1 for a new surrogate target; that is a new experiment, not the retained gray-box result.')


def select(config):
    """Bind the retained surrogate target without victim-guided reselection."""
    if config.get('reproduction_path')=='fresh_surrogate':
        from swaap.latent.pipeline import select as common_select
        return common_select(dict(config,family='tdmpc2'))
    import torch
    from swaap.latent.io import digest
    root=output(config);path=asset(config,'target')
    value=torch.load(path,map_location='cpu',weights_only=False);state=value.get('model',value)
    require('psi_k' in value or any(k.startswith('psi.') for k in state),'A bound single-task target dynamics checkpoint is required')
    result={'selected_target':str(path),'sha256':digest(path),'selection':'explicit retained single-task surrogate target; MT80 outcomes/gradients unused',
        'checkpoint_format':'psi_k_theta_k' if 'psi_k' in value else 'shared_full_target',
        'historical_encoder_embedded':'psi_k' not in value}
    write_json(root/'target.json',result);return result


def construct(config):
    require(config.get('evaluation',{}).get('scope') not in ('breadth','detailed_frozen'),'The80-task breadth workflow is evaluation-only; it never constructs or fits')
    import torch
    from .science import RefinementSWAAP, GridRelativeSWAAP
    from . import _latent_base
    _latent_base._load_tensor_runtime()
    root=output(config)
    agent=_agent(config)
    target=_target(config,agent)
    clean=load_buffer(asset(config,'clean'),rows=5000)
    reference=load_buffer(asset(config,'reference'),rows=50000)
    agent.model._dynamics.requires_grad_(True)
    require(float(config.get('poison_fraction',.1))==.1,'The retained raw-space constructor has exactly500/5000 editable rows')
    c=config['construction']; eps=float(c['epsilon']); alpha=float(config.get('alpha',c.get('alpha',0.)))
    cls=RefinementSWAAP if eps in (.0045,.0040) else GridRelativeSWAAP
    frozen=None
    if config.get('assets',{}).get('support'):
        item=torch.load(asset(config,'support'),map_location='cpu',weights_only=False)
        frozen=item['indices'] if isinstance(item,dict) else item
    ctor=cls(SingleModel(agent.model,agent.device),agent.model._dynamics,target,reference,clean,
             alpha=alpha,epsilon=eps,step_size=float(c.get('step_size',1.)),frozen_support=frozen)
    save_tensor(root/'support.pt',{'indices':ctor.support.cpu(),'target_residual':ctor.target_residual.cpu()})
    import json
    with (root/'construction.jsonl').open('x') as stream:
        for step in range(int(c.get('steps',5000))):
            row=ctor.step_raw(step);stream.write(json.dumps(row,allow_nan=False)+'\n')
    poison,integrity=ctor.output()
    require(all(torch.equal(clean[i],poison[i]) for i in (0,1,2,4)),'Frozen field changed')
    path=save_tensor(root/'poison.pt',poison)
    write_json(root/'integrity.json',integrity)
    return {'poison':path,'support':str(root/'support.pt'),'integrity':integrity}


def _frozen_state(model):
    import torch
    return {k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else deepcopy(v)
            for k,v in model.state_dict().items() if not k.startswith('_dynamics.')}


def _check_frozen_state(model,expected):
    import torch
    current=_frozen_state(model)
    require(current.keys()==expected.keys() and all(torch.equal(current[k],v) if isinstance(v,torch.Tensor)
            else type(current[k]) is type(v) and current[k]==v for k,v in expected.items()),'Non-dynamics tensor or structural metadata changed')


def fit(config):
    require(config.get('evaluation',{}).get('scope') not in ('breadth','detailed_frozen'),'The80-task breadth workflow is evaluation-only; it never constructs or fits')
    import torch
    from swaap.latent.runtime import seed_runtime
    root=output(config);agent=_agent(config,multitask=True)
    data=load_buffer(asset(config,'poison' if config.get('method','swaap')!='clean' else 'clean'),rows=5000)
    with torch.no_grad():
        z=agent.model.encode(data[0].cuda(),68).detach()
        nz=agent.model.encode(data[3].cuda(),68).detach()
        a=torch.nn.functional.pad(data[1].cuda(),(0,2))
    baseline=_frozen_state(agent.model)
    agent.model._dynamics.requires_grad_(True).train()
    recipe=config['fitting'];require(recipe['epochs']==10 and recipe['batch_size']==256,'Adopted gray-box E10/B256 required')
    limit=int(config.get('smoke',{}).get('fit_updates',200))
    require(1<=limit<=200 and (limit==200 or config.get('smoke',{}).get('enabled') is True),'Reduced fitting is an explicitly labeled integration prefix only')
    opt=torch.optim.Adam(agent.model._dynamics.parameters(),lr=float(recipe['learning_rate']))
    seed_runtime(int(config.get('seed',0)));history=[];updates=0
    for epoch in range(10):
        losses=[]
        for start in range(0,5000,256):
            loss=torch.nn.functional.mse_loss(agent.model.next(z[start:start+256],a[start:start+256],68),nz[start:start+256])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()));updates+=1
            if updates==limit:break
        history.append({'epoch':epoch+1,'updates':updates,'mean_batch_MSE':sum(losses)/len(losses)})
        if updates==limit:break
    _check_frozen_state(agent.model,baseline)
    path=save_tensor(root/'fit.pt',{'dynamics_state_dict':agent.model._dynamics.state_dict(),'metadata':{'source_task_index':68,'epochs':10,'updates':updates,'source_schedule_updates':200,'engineering_prefix':updates<200,'batch_size':256,'learning_rate':recipe['learning_rate'],'sequential_row_order':True}})
    write_json(root/'fit.json',{'history':history,'non_dynamics_unchanged':True})
    return {'checkpoint':path,'updates':updates,'engineering_prefix':updates<200}


def evaluate(config):
    if config.get('evaluation',{}).get('scope') in ('breadth','detailed_frozen'):
        return _evaluate_frozen(config)
    require(config.get('method') in ('original','direct') or bool(config.get('assets',{}).get('checkpoint')),'Fitted endpoint evaluation requires assets.checkpoint; Original must be explicitly method=original')
    import json
    import numpy as np
    import torch
    from swaap.latent.native import rollout_episode
    root=output(config);mt=bool(config.get('evaluation',{}).get('multitask',True));agent=_agent(config,multitask=mt)
    if config.get('assets',{}).get('checkpoint'):
        state=torch.load(asset(config,'checkpoint'),map_location='cpu',weights_only=False)
        agent.model._dynamics.load_state_dict(state['dynamics_state_dict'],strict=True)
    elif config.get('method')=='direct':
        require(not mt,'Direct surrogate is not an MT80 transfer fit')
        agent.model._dynamics.load_state_dict(_target(config,agent).state_dict(),strict=True)
    task=config.get('evaluation',{}).get('task',config['task'])
    if mt:
        task_map=json.loads(asset(config,'task_map').read_text())
        binding=next((r for r in task_map if r['task']==task),None)
        require(binding is not None,'Evaluation task absent from frozen MT80 map')
        index=binding['embedding'];action_dim=binding['action_dim'];steps=binding['outer_steps']
    else:
        require(task=='mw-soccer','Retained surrogate is Soccer')
        index=None;action_dim=4;steps=100
    cfg=deepcopy(agent.cfg);cfg.multitask=False;cfg.task=task;cfg.task_dim=0;cfg.tasks=[task];cfg.action_dim=action_dim;cfg.episode_length=steps
    view=AgentView(agent,index,action_dim)
    roots=json.loads(asset(config,'roots').read_text())
    if isinstance(roots,dict):roots=roots['roots'] if 'roots' in roots else roots[task]
    maximum=int(config.get('smoke',{}).get('evaluation_episodes',len(roots)))
    require(maximum==len(roots) or config.get('smoke',{}).get('enabled') is True,'Reduced evaluation is integration-only')
    roots=roots[:maximum]
    rows=[]
    with (root/'episodes.jsonl').open('x') as stream:
        for r in roots:
            arrays,row=rollout_episode(agent=view,cfg=cfg,task=task,dynamics=agent.model._dynamics,environment_seed=r['environment_seed'],planner_seed=r['planner_seed'],expected_episode_length=steps,torch=torch,np=np)
            with torch.no_grad():
                z=view.model.encode(torch.as_tensor(arrays['observations'],device=agent.device))
                residual=(torch.as_tensor(arrays['predicted_next_latents'],device=agent.device)-z[1:]).norm(dim=1)/(z[1:].norm(dim=1)+1e-8)
            row.update(root=r,rollout_delta_m=float(residual.mean()));rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
    result={'task':task,'task_index':index,'episodes':len(rows),'returns':[r['return'] for r in rows],'roots':roots,'rollout_delta_m':[r['rollout_delta_m'] for r in rows]}
    write_json(root/'evaluation.json',result)
    return result


def aggregate(config):
    """One transfer endpoint/panel; episode variation is not independent attacks."""
    import json
    import numpy as np
    root=output(config)
    value=json.loads(asset(config,'evaluation').read_text())
    if 'tasks' in value:
        return _aggregate_frozen(value,root)
    returns=np.asarray(value['returns'],dtype=float)
    require(len(returns)==value['episodes'] and bool(np.isfinite(returns).all()),'Incomplete/nonfinite evaluation panel')
    result={'task':value['task'],'episodes':len(returns),'return_mean':float(returns.mean()),
        'return_episode_SD':float(returns.std(ddof=1)) if len(returns)>1 else None,
        'independent_attack_count':1,'statistical_unit':'one retained source target/poison and victim fit; episodes are repeated evaluation'}
    if config.get('assets',{}).get('clean_evaluation'):
        clean=json.loads(asset(config,'clean_evaluation').read_text())
        require(clean['task']==value['task'] and clean['episodes']==value['episodes'],'Matched Clean panel mismatch')
        require(clean['roots']==value['roots'],'Evaluation roots differ from matched Clean')
        c=np.asarray(clean['returns'],dtype=float);require(float(c.mean())>0,'Positive matched Clean reference required')
        result.update(matched_clean_return=float(c.mean()),paired_return_harm=(c-returns).tolist(),reduction_pct=100*float((c-returns).mean()/c.mean()))
    write_json(root/'summary.json',result);return result


def _evaluate_frozen(config):
    """Evaluate five explicitly frozen MT80 endpoints; never constructs or fits."""
    import json
    import numpy as np
    import torch
    from swaap.latent.native import rollout_episode
    panel_size=50 if config['evaluation']['scope']=='breadth' else 100
    root=output(config);agent=_agent(config,multitask=True)
    task_map=json.loads(asset(config,'task_map').read_text())
    panels=json.loads(asset(config,'roots').read_text())
    chosen=config.get('evaluation',{}).get('tasks') or [config.get('evaluation',{}).get('task','mw-soccer')]
    require(set(chosen)<=set(panels),'Requested task absent from frozen50-root breadth panels')
    definitions=config.get('evaluation',{}).get('endpoints')
    require(definitions=={'Original':None,'Clean_lr25':'clean_lr25','SWAAP_lr25':'swaap_lr25','Clean_lr50':'clean_lr50','SWAAP_lr50':'swaap_lr50'},'Breadth requires all five named frozen endpoints')
    original={k:v.detach().cpu().clone() for k,v in agent.model._dynamics.state_dict().items()}
    states={'Original':original}
    for label,key in definitions.items():
        if key is not None:
            payload=torch.load(asset(config,key),map_location='cpu',weights_only=False)
            require(payload.get('metadata',{}).get('source_task_index')==68,'Breadth fit must use Soccer task conditioning')
            states[label]=payload['dynamics_state_dict']
    result={}
    with (root/'episodes.jsonl').open('x') as stream:
        for task in chosen:
            binding=next(r for r in task_map if r['task']==task);rows={}
            cfg=deepcopy(agent.cfg);cfg.multitask=False;cfg.task=task;cfg.tasks=[task];cfg.task_dim=0
            cfg.action_dim=binding['action_dim'];cfg.episode_length=binding['outer_steps']
            roots=panels[task];require(len(roots)==panel_size,'Frozen root-panel size changed')
            for label,state in states.items():
                agent.model._dynamics.load_state_dict(state,strict=True);agent.model.eval().requires_grad_(False)
                view=AgentView(agent,binding['embedding'],binding['action_dim']);values=[]
                for declared in roots:
                    arrays,row=rollout_episode(agent=view,cfg=cfg,task=task,dynamics=agent.model._dynamics,environment_seed=declared['environment_seed'],planner_seed=declared['planner_seed'],expected_episode_length=binding['outer_steps'],torch=torch,np=np)
                    with torch.no_grad():
                        z=view.model.encode(torch.as_tensor(arrays['observations'],device=agent.device))
                        residual=(torch.as_tensor(arrays['predicted_next_latents'],device=agent.device)-z[1:]).norm(dim=1)/(z[1:].norm(dim=1)+1e-8)
                    row.update(task=task,method=label,root=declared,rollout_delta_m=float(residual.mean()))
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();values.append(row)
                rows[label]={'returns':[r['return'] for r in values],'rollout_delta_m':[r['rollout_delta_m'] for r in values],'initial_observation_hashes':[r['initial_observation_hash'] for r in values],'roots':roots}
            originals=rows['Original']['initial_observation_hashes']
            require(all(r['initial_observation_hashes']==originals for r in rows.values()),'Paired breadth environment resets differ')
            result[task]=rows
            write_json(root/(task+'.json'),{'task':task,'episodes_per_endpoint':panel_size,'endpoints':rows,'source_task':'mw-soccer','new_fits':0,'new_constructions':0,'new_targets':0})
    final={'tasks':result,'new_fits':0,'new_constructions':0,'new_targets':0,'evaluation_only':True,'episodes':len(chosen)*5*panel_size}
    write_json(root/('breadth.json' if panel_size==50 else 'frozen_evaluation.json'),final);return final


def _aggregate_frozen(value,root):
    import numpy as np
    summary=[]
    for task,methods in value['tasks'].items():
        for suffix in ('lr25','lr50'):
            clean=np.asarray(methods['Clean_'+suffix]['returns'],dtype=float)
            poison=np.asarray(methods['SWAAP_'+suffix]['returns'],dtype=float)
            require(clean.shape==poison.shape and float(clean.mean())>0,'Invalid matched frozen panel')
            require(methods['Clean_'+suffix]['roots']==methods['SWAAP_'+suffix]['roots'],'Frozen Clean/SWAAP roots differ')
            row={'task':task,'learning_rate':2.5e-5 if suffix=='lr25' else 5e-5,'episodes':len(clean),
                'Clean_return':float(clean.mean()),'SWAAP_return':float(poison.mean()),
                'paired_harm':(clean-poison).tolist(),'reduction_pct':100*float((clean-poison).mean()/clean.mean()),
                'Clean_episode_SD':float(clean.std(ddof=1)),'SWAAP_episode_SD':float(poison.std(ddof=1))}
            summary.append(row)
    result={'rows':summary,'task_count':len(value['tasks']),'independent_source_attacks':1,
        'statistical_unit':'one fixed source target/poison; one victim fit per learning rate and condition; paired episode variation',
        'by_learning_rate':{str(lr):{'task_reductions_pct':[r['reduction_pct'] for r in summary if r['learning_rate']==lr],
            'median_reduction_pct':float(np.median([r['reduction_pct'] for r in summary if r['learning_rate']==lr])),
            'tasks_ge5pct':sum(r['reduction_pct']>=5 for r in summary if r['learning_rate']==lr),
            'tasks_ge10pct':sum(r['reduction_pct']>=10 for r in summary if r['learning_rate']==lr)} for lr in (2.5e-5,5e-5)}}
    write_json(root/'summary.json',result);return result
