"""Numerical fitting/evaluation primitives extracted from the closed executor."""
from contextlib import contextmanager
from copy import deepcopy
import random
import hashlib
import time
import math
from .families import make_seeded_environment
def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    require(value is not None and type(value) is not bool, 'Missing/boolean value is not a scientific scalar')
    value = float(value)
    require(math.isfinite(value), 'Missing/nonfinite scientific scalar')
    return value

def close_environment(env):
    close = getattr(env, 'close', None)
    if callable(close):
        close()

def optimizer_finite(opt, torch):
    return all((not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all()) for state in opt.state.values() for value in state.values()))

def fit_exact_orders(agent, dynamics, optimizer, buffer, schedule, torch, *, epoch_callback=None):
    """The current frozen native update sequence, supplied exact epoch orders."""
    z, a, _, nz, _ = [x.detach().to(agent.device) for x in buffer]
    history, steps = ([], 0)
    for epoch in range(1, 11):
        ix = schedule['permutations'][epoch - 1].to(agent.device)
        weighted = 0.0
        for b in range(0, 5000, 256):
            rows = ix[b:b + 256]
            error = torch.nn.functional.mse_loss(agent.model.next(z[rows], a[rows], dynamics), nz[rows])
            require(bool(torch.isfinite(error)), 'Nonfinite fit loss; no clipping/retry')
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            require(all((p.grad is None or bool(torch.isfinite(p.grad).all()) for p in dynamics.parameters())), 'Nonfinite fit gradient; no clipping/retry')
            optimizer.step()
            steps += 1
            require(all((bool(torch.isfinite(p).all()) for p in dynamics.parameters())) and optimizer_finite(optimizer, torch), 'Nonfinite fitted parameter/Adam state; no clipping/retry')
            weighted += float(error.detach()) * len(rows)
        history.append({'epoch': epoch, 'optimizer_updates': steps, 'training_mse': weighted / 5000})
        if epoch_callback is not None:
            epoch_callback(epoch, steps, history[-1])
    require(steps == 200, 'Fixed endpoint must have exactly 200 updates')
    return (history, steps)

def observation_hash(observation, torch, np):
    array = np.asarray(observation.detach().cpu().numpy() if isinstance(observation, torch.Tensor) else observation, dtype=np.float32)
    return hashlib.sha256(array.shape.__repr__().encode() + array.tobytes(order='C')).hexdigest()

def _environment_chain(env):
    chain, seen = ([], set())
    while env is not None and id(env) not in seen:
        seen.add(id(env))
        chain.append(env)
        child = getattr(env, 'env', None)
        if child is env:
            break
        env = child
    return chain

def _seed_myo_environment(env, seed):
    deepest = _environment_chain(env)[-1]
    methods = []
    seed_method = getattr(deepest, 'seed', None)
    if callable(seed_method):
        seed_method(int(seed))
        methods.append(type(deepest).__name__ + '.seed')
    else:
        try:
            deepest.reset(seed=int(seed))
            methods.append(type(deepest).__name__ + '.reset(seed=)')
        except TypeError as exc:
            raise RuntimeError('Myo environment lacks seed/reset(seed=)') from exc
    for name in ('action_space', 'observation_space'):
        space = getattr(deepest, name, None)
        if space is not None and callable(getattr(space, 'seed', None)):
            space.seed(int(seed))
            methods.append(type(deepest).__name__ + '.' + name + '.seed')
    return methods

@contextmanager
def isolated_planner_rng(seed, device, torch, np):
    python_state, numpy_state = (random.getstate(), np.random.get_state())
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

def _as_numpy(value, torch, np):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

def _bool_info(info, name, torch, np):
    value = info.get(name, False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return bool(value.item())
    return bool(value)

def rollout_episode(*, agent, cfg, task, dynamics, environment_seed, planner_seed, expected_episode_length, torch, np):
    """Canonical full native rollout/reset/RNG semantics, without cache reuse."""
    env, observation, methods = make_seeded_environment(cfg, task, environment_seed, torch, np)
    initial_hash = observation_hash(observation, torch, np)
    observations = [_as_numpy(observation, torch, np).astype(np.float32, copy=True)]
    latents, predictions, actions, rewards, dones, terminated_flags, truncated_flags, success_flags = ([] for _ in range(8))
    first_action_seconds = None
    started = time.perf_counter()
    try:
        with isolated_planner_rng(planner_seed, agent.device, torch, np), torch.no_grad():
            agent._prev_mean.zero_()
            done, step_index = (False, 0)
            while not done:
                observation_device = observation.to(agent.device, non_blocking=True)
                latent = agent.model.encode(observation_device)
                action_started = time.perf_counter()
                action = agent.act(observation, psi=dynamics, theta=agent.theta, t0=step_index == 0, eval_mode=False, task=None, input_obs=True).detach().cpu()
                if first_action_seconds is None:
                    first_action_seconds = time.perf_counter() - action_started
                action = torch.clamp(action, -1.0, 1.0)
                prediction = agent.model.next(latent, action.to(agent.device, non_blocking=True), dynamics)
                next_observation, reward, done, info = env.step(action)
                if not isinstance(next_observation, torch.Tensor):
                    next_observation = torch.as_tensor(next_observation, dtype=torch.float32)
                terminated = _bool_info(info, 'terminated', torch, np)
                latents.append(_as_numpy(latent, torch, np).reshape(-1).astype(np.float32, copy=True))
                predictions.append(_as_numpy(prediction, torch, np).reshape(-1).astype(np.float32, copy=True))
                actions.append(_as_numpy(action, torch, np).astype(np.float32, copy=True))
                rewards.append(float(_as_numpy(reward, torch, np).item()))
                dones.append(bool(done))
                terminated_flags.append(terminated)
                truncated_flags.append(bool(done and (not terminated)))
                success_flags.append(_bool_info(info, 'success', torch, np))
                observation = next_observation
                observations.append(_as_numpy(observation, torch, np).astype(np.float32, copy=True))
                step_index += 1
                require(step_index <= expected_episode_length, 'Episode exceeded fixed native length')
            final_latent = agent.model.encode(observation.to(agent.device))
            latents.append(_as_numpy(final_latent, torch, np).reshape(-1).astype(np.float32, copy=True))
    finally:
        close_environment(env)
    require(len(rewards) == expected_episode_length, 'Incomplete native episode')
    arrays = {'observations': np.stack(observations), 'latents': np.stack(latents), 'predicted_next_latents': np.stack(predictions), 'actions': np.stack(actions), 'rewards': np.asarray(rewards, dtype=np.float32), 'dones': np.asarray(dones, dtype=np.bool_), 'terminated': np.asarray(terminated_flags, dtype=np.bool_), 'truncated': np.asarray(truncated_flags, dtype=np.bool_), 'success': np.asarray(success_flags, dtype=np.bool_)}
    require(all((bool(np.isfinite(a).all()) for a in arrays.values())), 'Nonfinite episode data')
    metadata = {'environment_seed': environment_seed, 'planner_seed': planner_seed, 'initial_observation_hash': initial_hash, 'steps': len(rewards), 'return': finite(np.sum(arrays['rewards'], dtype=np.float64)), 'elapsed_seconds': time.perf_counter() - started, 'first_action_seconds': float(first_action_seconds or 0), 'seed_methods': methods}
    return (arrays, metadata)
