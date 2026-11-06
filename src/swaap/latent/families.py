"""Explicit environment-family adapters; optimization is shared."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path


def expected_episode_length(task):
    return 100 if task.startswith(('myo-', 'mw-')) or task in ('lift-cube', 'pick-cube', 'stack-cube') else 500


def make_environment(cfg):
    """Use the retained wrapper for the requested family, without fallbacks."""
    if cfg.task in ('lift-cube', 'pick-cube', 'stack-cube'):
        import torch
        import mani_skill2
        from .maniskill import make_environment as create
        return create(cfg, torch=torch, package_root=Path(mani_skill2.__file__).resolve().parent)
    if cfg.task.startswith('mw-'):
        from ._vendor.envs.metaworld import make_env
        from ._vendor.envs.wrappers.legacy_tensor import TensorWrapper
    elif cfg.task.startswith('myo-'):
        from ._vendor.envs.myosuite import make_env
        from ._vendor.envs.wrappers.tensor import TensorWrapper
    else:
        from ._vendor.envs.dmcontrol import make_env
        from ._vendor.envs.wrappers.tensor import TensorWrapper
    env = TensorWrapper(make_env(cfg))
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except AttributeError:
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env


def make_seeded_environment(cfg, task, environment_seed, torch, np):
    episode_cfg = deepcopy(cfg)
    episode_cfg.seed = int(environment_seed)
    env = make_environment(episode_cfg)
    methods = ['cfg.seed before make_env']
    if task in ('lift-cube', 'pick-cube', 'stack-cube'):
        observation = env.reset(seed=int(environment_seed))
        methods.extend(['Gymnasium reset(seed=environment_seed)', 'normalized_dense', 'fixed100_outer_repeat2'])
    else:
        if task.startswith('myo-'):
            from .native import _seed_myo_environment
            methods.extend(_seed_myo_environment(env, environment_seed))
        if task.startswith('mw-'):
            state = np.random.get_state()
            np.random.seed(int(environment_seed) % 2**32)
            try:
                observation = env.reset()
            finally:
                np.random.set_state(state)
            methods.append('isolated numpy.random seed for MetaWorld reset')
        else:
            observation = env.reset()
    if not isinstance(observation, torch.Tensor):
        observation = torch.as_tensor(observation, dtype=torch.float32)
    return env, observation, methods
