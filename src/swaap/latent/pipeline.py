"""Ordinary-file Stage-1 → selection → poison → matched fit → evaluation API.

Each call performs one explicitly requested stage. No scheduler, notification
controller, background process, automatic retry, or experiment launches on import.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time
from .io import asset, digest, finite, load_buffer, load_support, output, read_json, require, save_tensor, write_json


def _summary(root, stage, value):
    value = dict(stage=stage, **value)
    value['result_file'] = str(root / f'{stage}.json')
    write_json(value['result_file'], value)
    return value


def _settings(config):
    settings = dict(rp=.1, alpha=.9, epsilon=1., step_size=10., steps=5000)
    settings.update(config.get('stage2', {}))
    for key, public in [('rp', 'poison_fraction'), ('alpha', 'alpha')]:
        if public in config:
            settings[key] = config[public]
    return settings


def _state_snapshot(agent):
    import torch
    # Retained TensorDictParams serializes torch.Size/device structural entries
    # alongside tensors. They are immutable metadata, not numerical parameters.
    return {k: v.detach().cpu().clone() if isinstance(v,torch.Tensor) else deepcopy(v)
            for k,v in agent.model.state_dict().items()}


def _unchanged(agent, state):
    import torch
    current = agent.model.state_dict()
    require(set(current) == set(state) and all(
            torch.equal(current[k].detach().cpu(),v) if isinstance(v,torch.Tensor)
            else type(current[k]) is type(v) and current[k]==v
            for k,v in state.items()),
            'A frozen Original model component changed')


def prepare_data(config):
    """Collect canonical latent five-column banks under Original's planner.

    A public artifact import may instead provide existing exact banks. This call
    creates NEW collection episodes; it never masquerades as saved-result replay.
    """
    from .runtime import load_agent
    root = output(config)
    if config.get('data',{}).get('mode') == 'append_fixed500':
        return _append_fixed500(config,root)
    if config.get('data',{}).get('mode') == 'explicit_roots':
        return _prepare_explicit_roots(config,root)
    agent = load_agent(config)
    rows = int(config.get('data', {}).get('rows', 5000))
    ref_rows = int(config.get('data', {}).get('reference_rows', 50000))
    require(rows % agent.cfg.num_step == ref_rows % agent.cfg.num_step == 0, 'Banks must contain complete trajectories')
    started = time.perf_counter()
    try:
        if config.get('family') == 'maniskill':
            import numpy as np
            import torch
            from . import native
            from .collection import CollectionBoundary
            panels = read_json(asset(config,'collection_roots'))
            require(len(panels['clean_roots'])*100 == rows and len(panels['reference_roots'])*100 == ref_rows,
                    'Collection root counts must bind complete100-step trajectories')
            env = agent.env
            observer = CollectionBoundary(env,native,torch,np,agent.device,config['task'])
            agent.env = observer
            chunks = {'clean':[],'reference':[]}
            records = []
            try:
                for role in ('clean','reference'):
                    for declared in panels[role+'_roots']:
                        observer.arm(declared,role,len(chunks[role])*100)
                        data = agent.collect(psi=agent.psi_0,theta=agent.theta_0,batch_size=1,num_step=100,
                            use_env=role=='clean',random_model_perturb_chance=0,random_action_chance=0. if role=='clean' else .03)
                        records.append(observer.finish())
                        chunks[role].append(data)
            finally:
                observer.close_scope();agent.env = env
            clean = tuple(torch.cat([x[i] for x in chunks['clean']],dim=0) for i in range(5))
            reference = tuple(torch.cat([x[i] for x in chunks['reference']],dim=0) for i in range(5))
            write_json(root/'collection_trajectories.json',records)
        else:
            recipe = config.get('data',{})
            clean = agent.collect(agent.psi_0,agent.theta_0,rows//agent.cfg.num_step,agent.cfg.num_step,use_env=True,
                random_action_chance=float(recipe.get('clean_random_action_chance',.03)))
            reference = agent.collect(agent.psi_0,agent.theta_0,ref_rows//agent.cfg.num_step,agent.cfg.num_step,
                use_env=bool(recipe.get('reference_use_environment',False)),
                random_action_chance=float(recipe.get('reference_random_action_chance',.03)))
        files = dict(clean=save_tensor(root / 'clean.pt', tuple(clean)), reference=save_tensor(root / 'reference.pt', tuple(reference)))
        return _summary(root, 'prepare_data', dict(task=config['task'], files=files, rows=rows, reference_rows=ref_rows,
            new_environment_episodes=rows // agent.cfg.num_step + (ref_rows // agent.cfg.num_step if config.get('data',{}).get('reference_use_environment',False) else 0), seed=config.get('seed', 0), elapsed_seconds=time.perf_counter()-started,
            original_sha256=digest(asset(config, 'original'))))
    finally:
        agent.env.close()


def _prepare_explicit_roots(config,root):
    """Opt-in fresh collection with complete trajectory/root handoffs."""
    import numpy as np
    import torch
    from .runtime import load_agent
    from .families import expected_episode_length
    from . import native
    from .collection import ExplicitRootCollectionBoundary,validate_collection_panels
    recipe=config.get('data',{});rows=int(recipe.get('rows',5000));ref_rows=int(recipe.get('reference_rows',50000))
    horizon=expected_episode_length(config['task'])
    require(rows>0 and ref_rows>0 and rows%horizon==ref_rows%horizon==0,'Explicit collection requires whole trajectories')
    roots_path=asset(config,'collection_roots');panels=validate_collection_panels(read_json(roots_path),rows,ref_rows,horizon)
    original_sha=digest(asset(config,'original'))
    require(not panels.get('task')or panels['task']==config['task'],'Collection roots bind a different task')
    require(not panels.get('original_sha256')or panels['original_sha256']==original_sha,'Collection roots bind a different Original')
    agent=load_agent(config);original_env=agent.env
    observer=ExplicitRootCollectionBoundary(agent,native,torch,np);agent.env=observer
    before=_state_snapshot(agent)
    chunks={'clean':[],'reference':[]};records={'clean':[],'reference':[]};started=time.perf_counter()
    try:
        for role in ('clean','reference'):
            use_env=role=='clean' or bool(recipe.get('reference_use_environment',False))
            chance=float(recipe.get(role+'_random_action_chance',.03))
            for declared in panels[role+'_roots']:
                observer.arm(declared,role,len(chunks[role])*horizon,use_environment=use_env)
                data=agent.collect(psi=agent.psi_0,theta=agent.theta_0,batch_size=1,num_step=horizon,
                    use_env=use_env,random_model_perturb_chance=0,random_action_chance=chance)
                require(all(len(x)==horizon for x in data),'Native collection returned a partial trajectory')
                records[role].append(observer.finish(horizon));chunks[role].append(data)
        _unchanged(agent,before)
        files={}
        for role in ('clean','reference'):
            data=tuple(torch.cat([chunk[i]for chunk in chunks[role]],dim=0)for i in range(5))
            files[role]=save_tensor(root/(role+'.pt'),data)
            manifest=dict(task=config['task'],original_sha256=original_sha,
                roots=panels[role+'_roots'],source_roots_sha256=digest(roots_path),
                trajectory_records=records[role],bank_sha256=digest(files[role]),
                clean_bank_sha256=digest(files[role])if role=='clean'else None,
                collection_mode='explicit fresh trajectory roots; different RNG batching from historical batch collector',
                reproduces_historical_bank_bytes=False)
            filename=role+'_trajectory_roots.json'
            files[role+'_trajectory_roots']=write_json(root/filename,manifest)
        return _summary(root,'prepare_data',dict(task=config['task'],files=files,rows=rows,reference_rows=ref_rows,
            original_sha256=original_sha,collection_mode='explicit_roots',model_unchanged=True,
            new_environment_episodes=sum(1 for values in records.values()for r in values if r['uses_environment']),
            new_environment_steps=sum(r['environment_steps']for values in records.values()for r in values),
            elapsed_seconds=time.perf_counter()-started,reproduces_historical_bank_bytes=False))
    finally:
        observer.close_scope();agent.env=original_env;original_env.close()


def _stage1_assessment(agent, seeds):
    """Closed dense evaluator: isolated agent/environment and restored global RNG."""
    import numpy as np
    import torch
    import random
    from .stage1 import ThetaPsi
    from .families import make_environment
    from .runtime import seed_runtime
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    cfg = deepcopy(agent.cfg)
    cfg.seed = seeds[0]
    env = None
    rows = []
    try:
        seed_runtime(cfg.seed)
        env = make_environment(cfg)
        evaluator = ThetaPsi(cfg, env)
        evaluator.model.load_state_dict(agent.model.state_dict(), strict=True)
        evaluator.theta_0.load_state_dict(agent.theta_0.state_dict(), strict=True)
        modes = dict(agent.model.named_modules())
        for name, module in evaluator.model.named_modules():
            module.training = modes[name].training
        for episode, seed in enumerate(seeds):
            seed_runtime(seed)
            if episode:
                env.close()
                episode_cfg = deepcopy(cfg)
                episode_cfg.seed = seed
                env = make_environment(episode_cfg)
                evaluator.env = env
            seed_runtime(seed)
            result = evaluator.test(psi=evaluator.psi, theta=evaluator.theta_0, num_test_episodes=1, output_extra=False)
            require(len(result) == (10 if cfg.has_sr else 9) and len(result[0]) == 1, 'Unexpected evaluator output layout')
            rows.append(dict(episode_index=episode, environment_seed=seed, planner_seed=seed,
                return_value=finite(result[0][0]), absolute_deviation=finite(result[3]), relative_deviation=finite(result[4])))
        return dict(episodes=rows, mean_return=math.fsum(r['return_value'] for r in rows)/len(rows),
                    mean_deviation=math.fsum(r['relative_deviation'] for r in rows)/len(rows),
                    mean_absolute_deviation=math.fsum(r['absolute_deviation'] for r in rows)/len(rows))
    finally:
        if env is not None:
            env.close()
        random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2]); torch.cuda.set_rng_state_all(saved[3])


def stage1(config):
    """One fixed-budget Stage-1 run, preserving every stride-ten checkpoint."""
    import torch
    from .runtime import load_agent
    root = output(config)
    settings = config.get('stage1', {})
    iterations = int(settings.get('iterations', 400))
    stride = int(settings.get('checkpoint_stride', 10))
    seeds = settings.get('evaluation_seeds')
    require(isinstance(seeds, list) and len(seeds) == 10 and len(set(seeds)) == 10,
            'stage1.evaluation_seeds must contain ten frozen common Direct roots')
    require(iterations > 0 and iterations % stride == 0 and stride == 10, 'Paper Stage 1 saves every ten iterations over the fixed task budget')
    agent = load_agent(config, stage1_context=True)
    # Native Stage-1 context retains constructor training flags. Evaluation and
    # fitting contexts explicitly enter eval mode instead.
    targets = []
    target_dir = root / 'targets'
    target_dir.mkdir(exist_ok=False)
    original_hash = digest(asset(config, 'original'))
    started = time.perf_counter()
    try:
        baseline = _stage1_assessment(agent, seeds)
        for iteration in range(1, iterations + 1):
            agent.update(apply_reachability=False)
            if iteration % stride:
                continue
            state = {'psi.' + k: v.detach().cpu().contiguous().clone() for k, v in agent.psi.state_dict().items()}
            checkpoint = save_tensor(target_dir / f'target_{iteration:04d}.pt', dict(model=state,
                metadata=dict(task=config['task'], iteration=iteration, original_sha256=original_hash,
                              generation_seed=config.get('seed', 0), includes_optimizer_or_simulator_resume=False)))
            assessment = _stage1_assessment(agent, seeds)
            row = dict(candidate_id=f'{iteration:04d}-{digest(checkpoint)}', checkpoint=checkpoint,
                checkpoint_sha256=digest(checkpoint), iteration=iteration, **assessment)
            targets.append(row)
            write_json(target_dir / f'target_{iteration:04d}.json', row)
        return _summary(root, 'stage1', dict(task=config['task'], seed=config.get('seed', 0), iterations=iterations,
            baseline=baseline, targets=targets, original_sha256=original_hash, elapsed_seconds=time.perf_counter()-started,
            early_stopping=False, complete=True))
    finally:
        agent.env.close()


def select(config):
    """Select pre-fit candidates or freeze a winner from matched saved outcomes.

    `selection.phase=targets` uses only completed Stage-1 data. `winner` accepts
    three common development-order outcomes per candidate; it never changes the
    target list after reading final evaluation outcomes.
    """
    from .selection import FIELDS, rank_targets
    root = output(config)
    params = config.get('selection', {})
    phase = params.get('phase', 'targets')
    if phase == 'targets':
        source = read_json(asset(config, 'stage1_results'))
        baseline = source['baseline']
        original = finite(baseline['mean_return'])
        require(original > 0, 'Relative Stage-1 admission requires a positive same-run Original return')
        eligible, excluded, shas = [], [], set()
        for row in source['targets']:
            path = Path(row['checkpoint'])
            sha = digest(path)
            reason = None
            if finite(row['mean_return']) > .9 * original:
                reason = 'DIRECT_ABOVE_90_PERCENT_ORIGINAL'
            elif finite(row['mean_deviation']) > .30:
                reason = 'RELATIVE_DEVIATION_ABOVE_030'
            elif sha in shas:
                reason = 'DUPLICATE_CHECKPOINT_BYTES'
            if reason:
                excluded.append(dict(candidate_id=row['candidate_id'], reason=reason))
                continue
            shas.add(sha)
            candidate = {k: v for k, v in row.items() if k in FIELDS}
            candidate['lambda'] = config.get('stage1', {}).get('lambda', 1)
            candidate['w_c'] = 0
            candidate['source'] = 'completed Stage-1 checkpoint'
            candidate['run'] = str(config.get('seed', 0))
            eligible.append(candidate)
        count = len(eligible)
        classification = 'FULL' if count >= 20 else 'STANDARD' if count >= 5 else 'REJECTED'
        ranking = None
        if count >= 5:
            ranking = rank_targets(dict(baseline=dict(role='FRESH_ORIGINAL_STAGE1_BASELINE', mean_return=original), targets=eligible),
                dict(name='stage1_pareto_maximin', deviation_ceiling=.30, return_gate='none'), candidate_limit=min(count, 20))
        if ranking is not None:
            for rank,candidate in enumerate(ranking['candidates'],1):
                candidate['frozen_rank'] = rank
                candidate['checkpoint_sha256'] = digest(candidate['checkpoint'])
        return _summary(root, 'select', dict(task=config['task'], phase=phase, classification=classification,
            eligible_count=count, exclusions=excluded, selected=[] if ranking is None else ranking['candidates'],
            ranking=ranking, stage1_results_sha256=digest(asset(config, 'stage1_results'))))
    require(phase in ('screen', 'winner'), 'selection.phase must be targets, screen, or winner')
    rows = read_json(asset(config, 'development_results'))
    if isinstance(rows, dict):
        rows = rows['results']
    stealth_exclusions = []
    if params.get('stealth_qualification'):
        qualified = []
        gate = params['stealth_qualification']
        for row in rows:
            require(all(k in row for k in ('delta_d','delta_d_ref','tpr5','clean_fpr5')),
                    'ManiSkill return selection requires previously frozen incoming-buffer stealth measurements')
            passes = finite(row['delta_d']) <= float(gate['delta_d_max_multiple'])*finite(row['delta_d_ref']) and finite(row['tpr5']) <= finite(row['clean_fpr5'])+float(gate['tpr5_max_above_clean_fpr5'])
            if passes:
                qualified.append(row)
            else:
                stealth_exclusions.append(row['candidate_id'])
        rows = qualified
        if not rows:
            return _summary(root,'select',dict(phase=phase,selected=[],status='NO_STEALTH_QUALIFIED_CANDIDATE',
                return_selection_performed=False,stealth_exclusions=sorted(set(stealth_exclusions))))
    candidates = {}
    for row in rows:
        require(row['matched_clean'] and len(row['attack_returns']) == len(row['clean_returns']), 'Matched development vectors required')
        harms = [finite(c) - finite(a) for a, c in zip(row['attack_returns'], row['clean_returns'])]
        key = row['candidate_id']
        candidates.setdefault(key, []).append((row, math.fsum(harms)/len(harms)))
    expected = 1 if phase == 'screen' else 3
    require(all(len(v) == expected and len({r['fit_order'] for r, _ in v}) == expected for v in candidates.values()),
            f'Each candidate requires exactly {expected} distinct common development orders')
    common_orders = [{r['fit_order'] for r, _ in v} for v in candidates.values()]
    require(all(v == common_orders[0] for v in common_orders), 'Candidates must share the exact development-order list')
    ranked = sorted(candidates, key=lambda k: (-math.fsum(h for _, h in candidates[k])/expected,
                    int(candidates[k][0][0]['frozen_rank']), candidates[k][0][0]['checkpoint_sha256']))
    chosen = ranked[:3] if phase == 'screen' else ranked[:1]
    return _summary(root, 'select', dict(phase=phase, selected=chosen, ranking=[dict(candidate_id=k,
        mean_matched_harm=math.fsum(h for _, h in candidates[k])/expected) for k in ranked],
        selection_uses_return_units=True, development_orders=sorted(common_orders[0]), final_results_used=False))


def _support(agent, target, clean, count):
    import torch
    with torch.no_grad():
        z, action = clean[0].to(agent.device), clean[1].to(agent.device)
        differences = torch.nn.functional.mse_loss(agent.model.next(z, action, agent.psi_0),
            agent.model.next(z, action, target), reduction='none').mean(dim=1)
        return torch.topk(differences, count, largest=True).indices.cpu()


def _delta_d(agent, data):
    """Adopted float64 all-row reduction, normalized by Original prediction."""
    import numpy as np
    import torch
    with torch.no_grad():
        pred = agent.model.next(data[0].to(agent.device), data[1].to(agent.device), agent.psi_0).cpu()
    from .deviation import buffer_delta_d
    values = data[3].detach().cpu().numpy()
    return buffer_delta_d(values,values,pred.detach().cpu().numpy(),expected_rows=len(values))['poison_mean']



def construct(config):
    """Construct one poison; constructor variants remain explicitly distinct."""
    import numpy as np
    import torch
    from .runtime import load_agent, load_dynamics, seed_runtime
    from .stage2 import DataPoisoner
    root = output(config)
    rows = int(config.get('data', {}).get('rows', 5000))
    require(rows in (5000,10000,15000,20000,25000), 'Supported paper buffer sizes are5k/10k/15k/20k/25k')
    clean = load_buffer(asset(config, 'clean'), rows=rows)
    reference = load_buffer(asset(config, 'reference'))
    method = config.get('method', 'swaap')
    require(method in ('clean', 'raw_dmp', 'swaap', 'swaap_legacy_random', 'random500', 'swaap_random_support', 'random_target_shared_support', 'swaap_fixed_support', 'trajectory_consistent', 'exact_tied_trajectory', 'ordinary_seeded_tied'),
            'Unknown latent construction arm; method names are not interchangeable')
    agent = load_agent(config)
    target = load_dynamics(asset(config, 'target'), agent) if method not in ('clean', 'swaap_legacy_random', 'random500') else agent.psi_0
    frozen = _state_snapshot(agent)
    s = _settings(config)
    count = int(rows * s['rp'])
    require(0 < count <= rows and 0 <= s['alpha'] <= 1 and int(s['steps']) > 0, 'Invalid construction settings')
    if method == 'swaap_legacy_random':
        # The published comparator reserves a distinct seed for each Stage-1
        # run. Its global CUDA/NumPy draw order is part of the experiment, so a
        # missing run binding must never silently become construction seed0.
        declared = config.get('legacy_random_seed', config.get('construction_seed'))
        require(type(declared) is int, 'SWAAP (Random) requires legacy_random_seed (published_orders.json), or an explicitly chosen construction_seed for a new experiment')
        seed = declared
    elif method=='random500':
        seed=int(config.get('random500_seed',config.get('construction_seed',config.get('seed',0))))
    else:
        seed = int(config.get('construction_seed', config.get('seed', 0)))
    started = time.perf_counter()
    try:
        seed_runtime(seed)
        if method in ('swaap', 'raw_dmp', 'random_target_shared_support'):
            saved_support=asset(config,'support',required=False)
            if method == 'random_target_shared_support':
                support = load_support(asset(config,'support'),rows=rows,count=count)
                require(support.dtype == torch.int64 and support.numel() == count, 'Exact frozen shared support required')
            elif method=='raw_dmp'and saved_support:
                support=load_support(saved_support,rows=rows,count=count)
            else:
                support=_support(agent,target,clean,count)
        elif method == 'swaap_legacy_random':
            # Draw order is part of the executed legacy baseline: CUDA target
            # noise first, then NumPy-global independent Bernoulli selection.
            ref_next = reference[3].to(agent.device)
            noise = (torch.rand_like(ref_next) * 2 - 1) * .3
            target_next = (ref_next * (1 + noise)).clamp(0, 1)
            support = torch.as_tensor(np.where(np.random.rand(rows) < s['rp'])[0], dtype=torch.int64)
            require(len(support) > 0, 'Empty legacy Bernoulli draw; no automatic redraw')
        elif method in ('random500', 'swaap_random_support'):
            saved_support=asset(config,'support',required=False) if method=='swaap_random_support'else None
            support_seed=int(config.get('random_support_seed',seed)) if method=='swaap_random_support'else seed
            support=load_support(saved_support,rows=rows,count=count) if saved_support else torch.as_tensor(np.sort(np.random.default_rng(support_seed).choice(rows,size=count,replace=False)),dtype=torch.int64)
        elif method in ('swaap_fixed_support', 'trajectory_consistent', 'exact_tied_trajectory', 'ordinary_seeded_tied'):
            support = load_support(asset(config,'support'),rows=rows,count=count)
            require(support.dtype == torch.int64 and support.ndim == 1 and support.numel() == count, 'Exact frozen support count/order required')
        else:
            support = torch.empty(0, dtype=torch.int64)
        is_tied = method in ('trajectory_consistent','exact_tied_trajectory','ordinary_seeded_tied')
        if method in ('swaap_fixed_support','trajectory_consistent','exact_tied_trajectory','ordinary_seeded_tied'):
            from . import fixed_support as C
            from .sequential import valid_adjacent_rows
            settings = C.FixedSupportSettings(rows=rows,support_rows=count,projected_steps=int(s['steps']),
                step_size=float(s['step_size']),alpha=float(s['alpha']),epsilon=float(s['epsilon']),construction_seed=seed)
            identities = dict(original_sha256=digest(asset(config,'original')),clean_sha256=digest(asset(config,'clean')),
                target_sha256=digest(asset(config,'target')),support_sha256=digest(asset(config,'support')))
            with torch.no_grad():
                reference_targets = agent.model.next(reference[0].to(agent.device),reference[1].to(agent.device),target).cpu()
            ends = config.get('data',{}).get('episode_end_indices',list(range(agent.cfg.num_step-1,rows,agent.cfg.num_step)))
            if is_tied:
                eligible = valid_adjacent_rows(clean[0].numpy(),clean[3].numpy(),clean[4].numpy(),episode_end_indices=ends)
                require(set(support.tolist()).issubset(eligible), 'Frozen support crosses a reset/termination boundary; no silent replacement')
            if method == 'swaap_fixed_support':
                poisoned,details = C.construct_fixed_support(system=agent,original_dynamics=agent.psi_0,reference_buffer=reference,
                    clean_buffer=clean,target_next_latents=reference_targets,ordered_support=tuple(support.tolist()),settings=settings,
                    condition='swaap_fixed_support',input_identities=identities)
            elif method == 'trajectory_consistent':
                poisoned,details = C.construct_sequential_support(system=agent,original_dynamics=agent.psi_0,reference_buffer=reference,
                    clean_buffer=clean,target_next_latents=reference_targets,ordered_support=tuple(support.tolist()),settings=settings,
                    episode_end_indices=ends,input_identities=identities)
            else:
                from . import tied_trajectory as tied
                seed_poison = load_buffer(asset(config,'ordinary_seed'),rows=rows) if method == 'ordinary_seeded_tied' else None
                family = 'ORDINARY_SEEDED_TIED' if seed_poison is not None else 'EXACT_TIED_TRAJECTORY'
                try:
                    poisoned,details = tied.construct(C,agent,agent.psi_0,target,reference,clean,tuple(support.tolist()),family,seed_poison,settings,torch,np)
                except tied.ScientificFailure as error:
                    write_json(root/'scientific_failure.json',dict(reason=str(error),detail=error.detail,automatic_retry=False))
                    if hasattr(error,'partial'):save_tensor(root/'partial_poison.pt',error.partial)
                    raise
            history = details['history']
            write_json(root/'construction_details.json',details)
        elif method == 'clean':
            poisoned = tuple(x.detach().cpu().clone() for x in clean)
            history = []
        elif method == 'raw_dmp':
            poisoned = [x.detach().cpu().clone() for x in clean]
            with torch.no_grad():
                predicted = agent.model.next(clean[0].to(agent.device), clean[1].to(agent.device), target).cpu()
            poisoned[3][support] = predicted[support]
            poisoned = tuple(poisoned)
            history = []
        elif method == 'swaap':
            constructor = DataPoisoner(agent, agent.psi_0, target, reference, clean, r_c=s['rp'],
                step_size=s['step_size'], alpha=s['alpha'], epsilon=s['epsilon'])
            poisoned, similarities = constructor.learn_poison_data(steps=int(s['steps']))
            history = [dict(projected_step=k+1, preupdate_cosine=finite(v)) for k, v in enumerate(similarities)]
        else:
            if method == 'random_target_shared_support':
                saved_targets = asset(config, 'random_reference_targets',required=False)
                if saved_targets:
                    target_next = torch.load(saved_targets, map_location='cpu', weights_only=False)
                else:
                    from .baselines import multiplicative_noise_target
                    require(type(config.get('random_target_seed')) is int,'Supply exact random_reference_targets or an explicit random_target_seed')
                    target_next = multiplicative_noise_target(reference[3],config['random_target_seed'],.3,torch)
                require(isinstance(target_next, torch.Tensor) and target_next.shape == reference[3].shape,
                        'Exact shared-support random-target tensor must match the reference bank')
                if config.get('random_target_sha256'):
                    from .fixed_support import _tensor_sha
                    require(_tensor_sha(target_next)==config['random_target_sha256'],'Prescribed random target tensor identity mismatch')
                target_next = target_next.to(agent.device)
            elif method == 'random500':
                gen = torch.Generator(device='cpu').manual_seed(seed)
                nz = reference[3].detach().cpu()
                noise = (torch.rand(nz.shape, generator=gen, dtype=nz.dtype)*2-1)*.3
                target_next = (nz * (1+noise)).clamp(0, 1).to(agent.device)
            elif method == 'swaap_random_support':
                with torch.no_grad():
                    target_next = agent.model.next(reference[0].to(agent.device), reference[1].to(agent.device), target)
            constructor = DataPoisoner(agent, agent.psi_0, target, reference, clean, r_c=s['rp'],
                step_size=1. if method == 'random500' else s['step_size'], alpha=s['alpha'], epsilon=s['epsilon'])
            z, action, _, next_z, _ = [x.to(agent.device) for x in clean]
            constructor.poison_transition_orig = [deepcopy(z), deepcopy(action), deepcopy(next_z)]
            transition = deepcopy(constructor.poison_transition_orig)
            transition[2].requires_grad_(True)
            target_transition = [reference[0].to(agent.device), reference[1].to(agent.device), target_next]
            history = []
            for step in range(int(s['steps'])):
                constructor.step = step
                result = constructor.gradient_matching_step(agent.psi_0, target_transition, transition, support.cpu().numpy())
                transition = result[0]
                require(all(bool(torch.isfinite(x).all()) for x in transition) and all(bool(torch.isfinite(x).all()) for x in result[1:]),
                        'Nonfinite constructor result; no automatic retry')
                history.append(dict(projected_step=step+1, loss=finite(result[1]), preupdate_cosine=finite(1-result[2]), regularizer=finite(result[3])))
            poisoned = (transition[0].detach().cpu(), clean[1].clone(), clean[2].clone(), transition[2].detach().cpu(), clean[4].clone())
        poisoned = tuple(x.detach().cpu() for x in poisoned)
        for index in ((1,2,4) if is_tied else (0,1,2,4)):
            require(torch.equal(poisoned[index], clean[index]), 'A frozen buffer field changed')
        if is_tied:
            current_outside = torch.ones(rows,dtype=torch.bool);current_outside[support+1] = False
            require(torch.equal(poisoned[0][current_outside],clean[0][current_outside]),'Current inputs changed outside tied support')
            require(torch.equal(poisoned[0][support+1],poisoned[3][support]),'Tied stored fields disagree')
        outside = torch.ones(rows, dtype=torch.bool); outside[support] = False
        require(torch.equal(poisoned[3][outside], clean[3][outside]), 'A label outside declared support changed')
        _unchanged(agent, frozen)
        files = dict(poison=save_tensor(root/'poison.pt', poisoned), support=save_tensor(root/'support.pt', support))
        return _summary(root, 'construct', dict(method=method, settings=s, files=files,
            actual_modified_support_rows=len(support), nominal_poison_fraction=s['rp'], support_sha256=digest(files['support']),
            poison_sha256=digest(files['poison']), clean_sha256=digest(asset(config,'clean')),
            target_sha256=None if target is agent.psi_0 else digest(asset(config,'target')),
            delta_d=_delta_d(agent, poisoned), clean_delta_d_reference=_delta_d(agent, clean),
            construction_history=history, elapsed_seconds=time.perf_counter()-started))
    finally:
        agent.env.close()


def fit(config):
    """Fresh Original dynamics and fresh Adam, exactly saved epoch permutations."""
    import torch
    from .runtime import load_agent
    from .native import fit_exact_orders
    from .fitting import continue_exact_orders
    root = output(config)
    rows = int(config.get('data', {}).get('rows', 5000))
    if config.get('method','swaap') != 'clean':
        require(bool(config.get('assets',{}).get('data')),
                'Poison fitting requires assets.data=POISON.pt. Use method=clean for matched Clean; assets.poison is not a latent fitting input.')
    data = load_buffer(asset(config, 'data', required=False) or asset(config, 'clean'), rows=rows)
    agent = load_agent(config)
    params = dict(epochs=10, batch_size=256, learning_rate=.0001)
    params.update(config.get('fit', {}))
    require(params['batch_size'] == 256 and params['epochs'] in (10, 25, 50), 'Paper fits use B256 and E10/E25/E50')
    require(rows in (5000,10000,15000,20000,25000) and (rows == 5000 or params['epochs'] == 10), 'Larger-buffer paper fits use exactly ten epochs')
    if params.get('defense') == 'trim':
        return _fit_trim(config,agent,data,params,root)
    seed = int(config.get('fit_seed', config.get('seed', 0)))
    order_path = asset(config, 'order', required=False)
    if order_path:
        schedule = torch.load(order_path, map_location='cpu', weights_only=False)
    else:
        generator = torch.Generator(device=agent.device).manual_seed(seed)
        schedule = dict(fit_seed=seed, rows=rows,
                        permutations=[torch.randperm(rows, generator=generator, device=agent.device).cpu() for _ in range(10)])
        order_path = Path(save_tensor(root/'order.pt', schedule))
    require(schedule['fit_seed'] == seed and schedule['rows'] == rows and len(schedule['permutations']) == 10, 'Order metadata mismatch')
    expected = torch.arange(rows)
    require(all(o.dtype == torch.int64 and o.shape == (rows,) and torch.equal(o.sort().values, expected) for o in schedule['permutations']), 'Invalid epoch permutation')
    frozen = _state_snapshot(agent)
    dynamics = deepcopy(agent.psi_0).to(agent.device).train(True)
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=float(params['learning_rate']))
    started = time.perf_counter()
    try:
        if rows != 5000:
            from .buffer_fitting import fit_buffer_orders
            history, updates = fit_buffer_orders(agent,dynamics,optimizer,data,schedule,torch,rows)
        elif params['epochs'] == 10:
            history, updates = fit_exact_orders(agent, dynamics, optimizer, data, schedule, torch)
        else:
            history, updates = continue_exact_orders(fit_exact_orders, agent, dynamics, optimizer, data, schedule, torch,
                start_updates=0, target_updates=params['epochs']*20)
        _unchanged(agent, frozen)
        metadata = dict(task=config['task'], original_sha256=digest(asset(config,'original')), data_sha256=digest(asset(config,'data',required=False) or asset(config,'clean')),
            order_sha256=digest(order_path), fit_seed=seed, epochs=params['epochs'], optimizer_updates=updates,
            learning_rate=params['learning_rate'], fresh_original=True, fresh_adam=True,
            epoch_order_policy='cycle same ten permutations' if params['epochs'] != 10 else 'ten saved permutations')
        checkpoint = save_tensor(root/'fitted.pt', dict(dynamics_state_dict={k:v.detach().cpu().clone() for k,v in dynamics.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(), metadata=metadata))
        return _summary(root, 'fit', dict(checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint), **metadata,
            history=history, elapsed_seconds=time.perf_counter()-started))
    finally:
        agent.env.close()


def evaluate(config):
    """Exact native planner and common root panel; no fitting or construction."""
    if config.get('evaluation',{}).get('mode') == 'saved_original_predictions':
        from .saved_metrics import produce_original_predictions
        return produce_original_predictions(config)
    import numpy as np
    import torch
    from .runtime import load_agent, load_dynamics
    from .native import rollout_episode
    from .families import expected_episode_length
    root = output(config)
    roots_path = asset(config, 'roots')
    roots = read_json(roots_path)
    if isinstance(roots, dict):
        roots = roots['roots']
    require(len(roots) == int(config.get('evaluation', {}).get('episodes', 100)), 'Root panel size differs from configured evaluation size')
    require([r['episode_index'] for r in roots] == list(range(len(roots))), 'Roots must retain episode order')
    require(len({(r['environment_seed'],r['planner_seed']) for r in roots}) == len(roots), 'Duplicate evaluation roots')
    agent = load_agent(config)
    method = config.get('method', 'swaap')
    checkpoint = None if method == 'original' else asset(config,'target') if method == 'direct' else asset(config,'checkpoint')
    dynamics = agent.psi_0 if checkpoint is None else load_dynamics(checkpoint, agent).eval()
    frozen = _state_snapshot(agent)
    episodes = []
    trajectory_files = []
    try:
        for r in roots:
            if method == 'online_pgd':
                from .online_evaluation import online_episode
                from .online_pgd import pgd_latent_low_value
                from . import native
                arrays, record = online_episode(N=native,agent=agent,cfg=agent.cfg,task=config['task'],dynamics=dynamics,
                    epsilon=float(config['pgd']['epsilon']),environment_seed=int(r['environment_seed']),planner_seed=int(r['planner_seed']),
                    expected_episode_length=expected_episode_length(config['task']),torch=torch,np=np,attack=pgd_latent_low_value)
            else:
                arrays, record = rollout_episode(agent=agent, cfg=agent.cfg, task=config['task'], dynamics=dynamics,
                environment_seed=int(r['environment_seed']), planner_seed=int(r['planner_seed']),
                expected_episode_length=expected_episode_length(config['task']), torch=torch, np=np)
            true = arrays['latents'][1:].astype(np.float64)
            pred = arrays.get('true_current_predicted_next_latents',arrays['predicted_next_latents']).astype(np.float64)
            record['delta_m'] = finite(np.mean(np.linalg.norm(true-pred,axis=-1)/(np.linalg.norm(true,axis=-1)+1e-8)))
            record['episode_index'] = r['episode_index']
            episodes.append(record)
            if config.get('evaluation', {}).get('save_trajectories', True):
                name = root/f"episode_{r['episode_index']:03d}.npz"
                require(not name.exists(), 'Refusing to replace an episode')
                np.savez_compressed(name, **arrays)
                trajectory_files.append(str(name))
        _unchanged(agent, frozen)
        return _summary(root, 'evaluate', dict(task=config['task'], method=method, episodes=episodes,
            return_mean=math.fsum(e['return'] for e in episodes)/len(episodes),
            delta_m_mean=math.fsum(e['delta_m'] for e in episodes)/len(episodes),
            roots_sha256=digest(roots_path), original_sha256=digest(asset(config,'original')),
            checkpoint_sha256=None if checkpoint is None else digest(checkpoint), trajectories=trajectory_files))
    finally:
        agent.env.close()


def aggregate(config):
    """Nested fitting-order means, then sample SD across independent Stage-1 runs."""
    if config.get('aggregation',{}).get('metric_mode') == 'original_on_matched_clean':
        from .saved_metrics import original_on_matched_clean
        return original_on_matched_clean(config)
    import statistics
    root = output(config)
    rows = read_json(asset(config, 'results'))
    if isinstance(rows, dict):
        rows = rows['results']
    grouped = {}
    for row in rows:
        key = (row['task'], row['method'], str(row['stage1_run']))
        require(isinstance(row['returns'], list) and row['returns'], 'Per-order episode returns required')
        grouped.setdefault(key, []).append(dict(fit_order=row['fit_order'],
            return_mean=math.fsum(finite(x) for x in row['returns'])/len(row['returns'])))
    by_task = {}
    for (task, method, run), orders in grouped.items():
        require(len({o['fit_order'] for o in orders}) == len(orders), 'Duplicate fitting-order result')
        expected = int(config.get('aggregation', {}).get('fit_orders', 3))
        require(len(orders) == expected, f'Incomplete Stage-1 run: expected {expected} fitting orders, got {len(orders)}')
        by_task.setdefault((task, method), []).append(dict(stage1_run=run, fit_order_means=orders,
            mean=math.fsum(o['return_mean'] for o in orders)/len(orders)))
    results = []
    for (task, method), runs in sorted(by_task.items()):
        values = [r['mean'] for r in runs]
        results.append(dict(task=task, method=method, independent_stage1_runs=len(values), fit_orders_per_run=expected,
            mean=statistics.mean(values), sample_sd=None if len(values)<2 else statistics.stdev(values), runs=runs,
            statistical_unit='independent Stage-1 run', missing_values_imputed=False))
    return _summary(root, 'aggregate', dict(results=results))


def smoke(config):
    """Two real Stage-2 steps on CPU synthetic tensors, without model assets."""
    import torch
    from .stage2 import DataPoisoner
    from types import SimpleNamespace
    root = output(config)
    torch.manual_seed(int(config.get('seed', 7)))
    class SmallDynamics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 3)
        def forward(self,z,a):
            return torch.sigmoid(self.linear(torch.cat([z,a],dim=-1)))
    original = SmallDynamics()
    target = deepcopy(original)
    with torch.no_grad():
        target.linear.weight.add_(.1)
    model = SimpleNamespace(device=torch.device('cpu'),next=lambda z,a,d:d(z,a))
    system = SimpleNamespace(model=model)
    z = torch.rand(20,3); a = torch.rand(20,2)
    labels = (original(z,a).detach()+.05*torch.randn(20,3)).clamp(.05,.95)
    bank = (z,a,torch.zeros(20,1),labels,torch.zeros(20,1))
    before = deepcopy(original.state_dict())
    constructor = DataPoisoner(system,original,target,bank,bank,r_c=.1,alpha=.9)
    poison,cosines = constructor.learn_poison_data(steps=2)
    require(all(torch.equal(poison[k],bank[k]) for k in (0,1,2,4)), 'Synthetic frozen-field check failed')
    require(all(torch.equal(v,original.state_dict()[k]) for k,v in before.items()), 'Synthetic frozen-model check failed')
    require(all(math.isfinite(x) for x in cosines), 'Synthetic construction became nonfinite')
    poison_path = save_tensor(root/'synthetic_poison.pt',poison)
    return _summary(root,'smoke',dict(status='PASS',device='cpu',synthetic=True,environment_episodes=0,
        scientific_kernel='executed ordinary SWAAP DataPoisoner',projected_steps=2,rows=20,support_rows=2,
        original_unchanged=True,frozen_fields_unchanged=True,preupdate_cosines=cosines,poison=poison_path,
        qualification='CPU numerical-kernel smoke only; no real model, environment, or GPU qualification'))


def _fit_trim(config,agent,data,params,root):
    """Frozen corrected TRIM: beta=.2, E10,160 Adam updates; both Clean/poison."""
    from .trim import TrimConfig,fit_world_model
    require(len(data[0]) == 5000 and params['epochs'] == 10,'Retained TRIM uses5000rows andE10')
    before = _state_snapshot(agent)
    seed = int(config.get('fit_seed',config.get('seed',0)))
    started = time.perf_counter()
    try:
        fit = fit_world_model(system=agent,data=data,config=TrimConfig(mode='corrected_matched_epochs',num_epochs=10,
            batch_size=256,learning_rate=float(params['learning_rate']),beta=.2,max_iters=10,device=str(agent.device)),trim=True,seed=seed)
        require(fit.optimizer_steps == 160 and fit.completed_epochs == 10 and not fit.stopped_early and fit.returned_last_candidate,
                'Frozen TRIM endpoint did not match its declared schedule')
        _unchanged(agent,before)
        metadata = dict(task=config['task'],defense='TRIM',mode=fit.mode,beta=.2,epochs=10,optimizer_updates=160,
            fit_seed=seed,learning_rate=params['learning_rate'],original_sha256=digest(asset(config,'original')))
        checkpoint = save_tensor(root/'fitted.pt',dict(dynamics_state_dict={k:v.detach().cpu().clone() for k,v in fit.psi.state_dict().items()},metadata=metadata))
        return _summary(root,'fit',dict(checkpoint=checkpoint,**metadata,history=fit.history,elapsed_seconds=time.perf_counter()-started))
    finally:
        agent.env.close()


def _append_fixed500(config,root):
    """Retain500 existing poisoned rows; append certified Clean rows unchanged."""
    import torch
    from .buffer_fitting import append_pair
    rows = int(config['data']['rows'])
    base = load_buffer(asset(config,'clean'),rows=5000)
    extra = load_buffer(asset(config,'extra_clean'),rows=20000)
    poison = load_buffer(asset(config,'poison'),rows=5000)
    support = load_support(asset(config,'support'),rows=5000,count=500)
    clean,current = append_pair(base,extra,poison,support,rows,torch)
    files = dict(clean=save_tensor(root/'clean.pt',clean),poison=save_tensor(root/'poison.pt',current),support=save_tensor(root/'support.pt',support))
    return _summary(root,'prepare_data',dict(files=files,rows=rows,fixed_poison_rows=500,
        reused_poison_sha256=digest(asset(config,'poison')),extra_clean_sha256=digest(asset(config,'extra_clean')),
        new_constructions=0,new_episodes=0,optimizer_updates=0,original5000_prefix_unchanged=True))
