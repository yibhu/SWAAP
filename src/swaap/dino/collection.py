"""Fixed-length observed collection for the isolated faithful DINO release.

The numerical planner/encoder definitions remain the recovered source.  This
adapter retains ten grouped actions even after success, then asks the same CEM
planner for the next action without appending it to the retained trajectory.
Its source-defined diagnostic environment replays remain enabled and are
counted separately.  Nothing runs merely by importing this module.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re


def _save(torch, path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cpu(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(torch, item) for item in value)
    if hasattr(value, 'copy'):
        return value.copy()
    return value


def _rng(torch, np):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=torch.cuda.get_rng_state().cpu().clone(),
                cuda_device=torch.cuda.current_device())


def collect(rt, row, log, data, guard=None):
    """Return ``(eight_window_dataset, metadata)`` for one saved record.

    ``metadata['packet']`` names the durable raw/encoding packet.  The caller
    supplies the explicit role aggregation and output directories.
    A started or completed record is never replayed automatically.
    """
    import hydra

    torch, np, source = rt.torch, rt.np, rt.source
    log, data = Path(log), Path(data)
    log.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    role = str(row['role'])
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', role)
    index = int(row['record_index'])
    prefix = f'collection_{role}_record_{index:04d}'
    started = log / (prefix + '_STARTED.json')
    completed = log / (prefix + '_COMPLETE.json')
    packet_path = data / (prefix + '.pt')
    partial_path = data / (prefix + '_PARTIAL.pt')
    assert not any(path.exists() for path in
                   [started, completed, packet_path, partial_path]), 'No partial/completed replay'
    with started.open('x') as stream:
        json.dump(dict(record_index=index, pair_identity=row['pair_identity'], role=role,
                       global_seed=row['global_seed'], automatic_replay=False,
                       retained_grouped_actions=10, retained_native_steps=50), stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

    def check(unit):
        if guard is not None:
            guard(prefix + '_' + unit)

    cfg = deepcopy(rt.cfg)
    cfg.seed = row['global_seed']
    assert int(cfg.num_hist) == 3 and int(rt.model_cfg.frameskip) == 5
    ws = source.SimulationWorkspace(cfg, rt.original, rt.dataset, rt.env,
                                    rt.model_cfg.env.name, 5, None, None)
    ws.obs_0, ws.obs_g = row['obs_0'], row['obs_g']
    ws.state_0, ws.state_g = row['state_0'], row['state_g']
    ws.eval_seed = row['environment_roots']
    assert len(ws.eval_seed) == 1 and ws.state_0.shape == ws.state_g.shape == (1, 7)
    had_workspace, old_workspace = hasattr(source, 'workspace'), getattr(source, 'workspace', None)
    source.workspace = ws
    old_rollout = rt.env.rollout
    calls, metrics = [], []
    phase = {'value': 'initialization'}

    def rollout(seeds, states, actions, *args, **kwargs):
        value = old_rollout(seeds, states, actions, *args, **kwargs)
        calls.append(dict(phase=phase['value'], roots=list(seeds),
                          actions=_cpu(torch, actions), states=_cpu(torch, value[1]),
                          native_action_steps=int(actions.shape[1])))
        return value

    class RecordingEnvironment:
        def __getattr__(self, name):
            return getattr(rt.env, name)
        def rollout(self, seeds, states, actions, *args, **kwargs):
            return rollout(seeds, states, actions, *args, **kwargs)
    recorded_env = RecordingEnvironment()
    try:
        check('prepare')
        rt.seed(row['global_seed'])
        observed, _ = rt.env.prepare(ws.eval_seed, ws.state_0)
        assert np.array_equal(observed['visual'], ws.obs_0['visual'][:, 0])
        assert np.allclose(observed['proprio'], ws.obs_0['proprio'][:, 0], rtol=0, atol=1e-6)

        # Preserve the qualified evaluator's pre-planner policy preview and RNG
        # consumption.  This action is diagnostic and is not executed.
        with torch.no_grad():
            obs = source.resize_obs(ws.obs_0).cuda()
            target = source.resize_obs(ws.obs_g['visual'].squeeze(0).squeeze(0)).cuda()
            preview = rt.policy.sample_actions(obs, target)
            raw = source.move_to_device(ws.preprocessor.transform_obs(ws.obs_0), 'cuda')
            latent = rt.original.encode(raw, preview.unsqueeze(1))
            assert rt.original.predict(latent[:, -cfg.num_hist:]).isfinite().all()
        preview = preview.detach().cpu()
        del obs, target, raw, latent

        class RecordingEvaluator(source.PlanEvaluator):
            def eval_actions(self, actions, *args, **kwargs):
                name = kwargs.get('filename', 'output')
                kwargs['filename'] = str(log / (prefix + '_' + str(name)))
                value = super().eval_actions(actions, *args, **kwargs)
                metrics.append(dict(phase=phase['value'],
                    values={key: float(item) for key, item in value[0].items()},
                    native_success=[bool(item) for item in value[1]]))
                return value
        evaluator = RecordingEvaluator(obs_0=ws.obs_0, obs_g=ws.obs_g,
            state_0=ws.state_0, state_g=ws.state_g, env=recorded_env, wm=rt.original,
            frameskip=5, seed=ws.eval_seed, preprocessor=ws.data_preprocessor,
            n_plot_samples=cfg.n_plot_samples)
        ws.evaluator = evaluator
        planner = hydra.utils.instantiate(cfg.planner, wm=rt.original, env=recorded_env,
            action_dim=ws.action_dim, objective_fn=ws.objective_fn,
            preprocessor=ws.data_preprocessor, evaluator=evaluator,
            wandb_run=None, log_filename=str(log / (prefix + '_planner.jsonl')))
        ws.planner = planner
        assert planner.n_taken_actions == 5
        cem = planner.sub_planner
        assert (cem.horizon, cem.opt_steps, cem.num_samples, cem.topk, cem.eval_every) == (5, 10, 100, 10, 100)
        assert cem.evaluator is evaluator

        current_obs, memo_actions = ws.obs_0, None
        executed = []
        observed_full = physical_full = None
        for block in range(2):
            check(f'plan_block_{block}')
            phase['value'] = f'CEM_block_{block}'
            cem.logging_prefix = f'plan_{block}'
            with torch.no_grad():
                planned, _ = cem.plan(obs_0=current_obs, obs_g=ws.obs_g, actions=memo_actions)
            assert planned.shape == (1, 5, 10) and planned.isfinite().all()
            taken = planned.detach()[:, :5].clone()
            memo_actions = planned.detach()[:, 5:].clone()
            executed.append(taken)
            planner.planned_actions = executed
            # Fixed-horizon collection intentionally does not apply the source
            # outer success mask or stop at native success.  CEM's own native
            # diagnostic callback/early-return semantics remain untouched.
            evaluator.assign_init_cond(obs_0=ws.obs_0, state_0=ws.state_0)
            phase['value'] = f'retained_prefix_{5 * (block + 1)}'
            with torch.no_grad():
                result = evaluator.eval_actions(torch.cat(executed, dim=1),
                    np.full(1, np.inf), filename=f'retained_prefix_{block}', save_video=False)
            observed_full = _cpu(torch, result[2])
            physical_full = _cpu(torch, result[3])
            assert physical_full.shape == (1, 25 * (block + 1) + 1, 7)
            current_obs = {key: value[:, -1:] for key, value in observed_full.items()}
            current_state = physical_full[:, -1]
            evaluator.assign_init_cond(obs_0=current_obs, state_0=current_state)
            planner.iter = block + 1
            _save(torch, partial_path, dict(record=row, completed_grouped_actions=5 * (block + 1),
                normalized_grouped_actions=_cpu(torch, torch.cat(executed, dim=1)),
                observed=observed_full, physical_states=physical_full,
                native_prefix_replays=calls, evaluator_metrics=metrics,
                no_automatic_replay=True))

        assert observed_full['visual'].shape == (1, 51, 224, 224, 3)
        assert observed_full['proprio'].shape == (1, 51, 4)
        check('boundary_decision')
        phase['value'] = 'boundary_CEM_preview'
        boundary_before = _rng(torch, np)
        cem.logging_prefix = 'plan_2'
        with torch.no_grad():
            boundary_plan, _ = cem.plan(obs_0=current_obs, obs_g=ws.obs_g, actions=memo_actions)
        boundary_after = _rng(torch, np)
        assert boundary_plan.shape == (1, 5, 10) and boundary_plan.isfinite().all()
        boundary_action = boundary_plan[:, :1].detach().clone()
        retained_actions = torch.cat(executed, dim=1)
        all_actions = torch.cat([retained_actions, boundary_action], dim=1)
        assert retained_actions.shape == (1, 10, 10) and all_actions.shape == (1, 11, 10)

        check('encode_observed')
        phase['value'] = 'Original_observed_encoding'
        raw = {key: value[:, ::5] for key, value in observed_full.items()}
        assert raw['visual'].shape[1] == 11
        with torch.no_grad():
            encoded = rt.original.encode(source.move_to_device(ws.preprocessor.transform_obs(raw), 'cuda'), all_actions).cpu()
        assert encoded.shape == (1, 11, 196, 404) and encoded.isfinite().all()
        sequential = dict(states=encoded[:, :-1], actions=retained_actions.cpu(), next_states=encoded[:, 1:])
        values = source.build_dino_transition_dataset(sequential, device='cpu', num_hist=3)
        assert tuple(values[0].shape) == (8, 3, 196, 404)
        assert tuple(values[1].shape) == (8, 10) and tuple(values[3].shape) == (8, 196, 404)
        assert torch.equal(values[1], retained_actions[:, 2:].cpu().squeeze(0))
        assert torch.equal(values[3], encoded[:, 3:].squeeze(0))
        native_steps = sum(item['native_action_steps'] for item in calls)
        metadata = dict(record_index=index, pair_identity=row['pair_identity'], role=role,
            global_seed=row['global_seed'], environment_roots=list(ws.eval_seed),
            retained_grouped_actions=10, retained_native_steps=50, observed_states=11,
            complete_transitions=10, fitting_windows=8, history=3,
            history_end_indices=list(range(2, 10)), successor_indices=list(range(3, 11)),
            packet=str(packet_path), native_prefix_replay_calls=len(calls),
            native_replay_action_steps=native_steps, independent_collection_trajectories=1,
            boundary_action='Genuine next source CEM decision a10 at observed s10; never appended to the retained action sequence',
            boundary_native_checks='Source CEM evaluator callback retained; internal reset/replays counted separately',
            fixed_horizon_deviation='Collection omits outer MPC success stop/mask to retain ten source actions; usual evaluation remains unchanged',
            action_tail='Original action_encoder(a_t) at every observed s_t including genuine unexecuted a10 at s10; no padding/duplication',
            window_boundaries='Every three-history window and its successor belongs to this same trajectory',
            rewards_terminals='Recovered transition helper zero reward/false terminal fields, unchanged',
            encoder='Frozen FP32 Original DINO/proprio/action encoders and existing preprocessing')
        packet = dict(record=row, metadata=metadata, observed=observed_full,
            physical_states=physical_full, normalized_grouped_actions=retained_actions.cpu(),
            retained_native_actions=next(item['actions'] for item in calls
                                         if item['phase'] == 'retained_prefix_10'),
            boundary_next_action=boundary_action.cpu(), boundary_plan=boundary_plan.detach().cpu(),
            state_aligned_actions=all_actions.cpu(), Original_observed_encodings=encoded,
            native_prefix_replays=calls, evaluator_metrics=metrics,
            boundary_rng_before=boundary_before, boundary_rng_after=boundary_after,
            qualified_HIQL_preview_action=preview)
        _save(torch, packet_path, packet)
        with completed.open('x') as stream:
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        return values, metadata
    finally:
        if had_workspace:
            source.workspace = old_workspace
        elif hasattr(source, 'workspace'):
            del source.workspace
