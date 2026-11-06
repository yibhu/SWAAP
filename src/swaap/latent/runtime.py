"""Configurable model loading using the executed architecture and MPC code."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import random
import os
from .io import asset, require


def seed_runtime(seed):
    import numpy as np
    import torch
    require(type(seed) is int and 0 <= seed < 2**32, 'An unsigned 32-bit seed is required')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_configuration(config, checkpoint=None):
    """Build the retained architecture without creating an environment/model.

    Gray-box loaders may provide explicit model metadata for a multi-task model.
    State shape and action dimensions are filled by the requested family before
    constructing a single-task model.
    """
    from omegaconf import OmegaConf
    from .stage1 import parse_cfg
    cfg = OmegaConf.load(Path(__file__).with_name('model_defaults.yaml'))
    for key in list(cfg.keys()):
        if key.startswith('wandb_') or key == 'defaults':
            del cfg[key]
    cfg.task = config['task']
    cfg.model_size = config.get('model', {}).get('model_size', 5)
    cfg.seed = int(config.get('seed', 0))
    cfg.enable_wandb = False
    cfg.episodic = False
    cfg.num_eval_iteration = cfg.num_save_iteration = 10
    cfg.num_test_episodes = 10
    cfg.num_v = 1
    cfg.load_v_checkpoint = cfg.load_surrogate_agent = cfg.load_imitate_policy = False
    cfg.critical_state = cfg.reachability_enabled = False
    cfg.load_from_tdmpc2_checkpoint = True
    cfg.T_step = 30
    cfg.sample_batch_size = 256
    cfg.update_batch_size = 16
    cfg.rho = .5
    cfg.w_c = 0
    cfg.use_consistency = True
    cfg.checkpoint = str(checkpoint or asset(config, 'original'))
    cfg.compile = bool(config.get('runtime', {}).get('compile', True))
    cfg.maniskill_rendererless = bool(config.get('runtime',{}).get('maniskill_rendererless',True))
    cfg.mpc = True
    cfg.num_collect_trajs = int(config.get('stage1', {}).get('collect_trajectories', 1 if cfg.task[:3] not in ('myo', 'mw-') else 5))
    cfg.lambda_coef = float(config.get('stage1', {}).get('lambda', 1))
    cfg.lambda_coefs = [cfg.lambda_coef]
    cfg.num_iteration = int(config.get('stage1', {}).get('iterations', 400))
    cfg.num_step = cfg.episode_length = 100 if cfg.task.startswith(('mw-', 'myo-')) or cfg.task.endswith('-cube') else 500
    cfg.has_sr = cfg.num_step == 100
    cfg.checkpoint_number = config.get('checkpoint_number', 1)
    for key, value in config.get('model', {}).items():
        cfg[key] = value
    parsed = parse_cfg(cfg)
    if 'output_dir' in config:
        parsed.work_dir = Path(config['output_dir']).resolve()
    return parsed


def require_cuda(config=None):
    import torch
    if not (config or {}).get('runtime',{}).get('allow_unqualified_software',False):
        require(torch.__version__.split('+')[0] == '2.6.0.dev20241112' and torch.version.cuda == '12.4',
            'Paper scientific path requires retained Torch2.6.0.dev20241112/CUDA12.4. Obtain the frozen environment; a newer release is not silently substituted. Explicit unqualified-software runs must be labeled separately.')
    require(torch.cuda.is_available(), 'The scientific latent model path requires CUDA. The CPU smoke command uses synthetic tensors and does not load a simulator.')
    require(torch.cuda.device_count() == 1, 'Expose one GPU per independent run using your normal device selection')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    return torch


def checkpoint_geometry(checkpoint):
    """Read single-task architecture dimensions from exact checkpoint weights."""
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=state.get('model',state)
    require(isinstance(state,dict),'Original model state dictionary required')
    encoder=state.get('_encoder.state.0.weight')
    dynamics=state.get('_dynamics.0.weight')
    require(isinstance(encoder,torch.Tensor) and encoder.ndim==2 and isinstance(dynamics,torch.Tensor) and dynamics.ndim==2,
            'Model-only loading needs explicit model.obs_shape/action_dim or recognized single-task Original weights')
    # The released model-size5 architecture has512 latent coordinates. The
    # dynamics input concatenates z and a; single-task task_dim is exactly0.
    latent=512
    action=int(dynamics.shape[1])-latent
    require(0<action<128,'Single-task action geometry could not be established')
    return dict(obs_shape={'state':[int(encoder.shape[1])]},action_dim=action)


def load_agent(config, checkpoint=None, *, multitask=False, task_index=None, environment=True, stage1_context=False):
    """Load the exact scientific model/planner; no fitting or evaluation occurs."""
    original = checkpoint or asset(config, 'original')
    os.environ['LAZY_LEGACY_OP'] = '0'
    os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
    renderer = config.get('runtime',{}).get('mujoco_gl','egl')
    require(renderer in ('auto','egl','glfw','osmesa'),'Unknown MuJoCo renderer setting')
    if renderer != 'auto':
        os.environ['MUJOCO_GL'] = renderer
    torch = require_cuda(config)
    seed_runtime(int(config.get('seed', 0)))
    from .stage1 import ThetaPsi, validate_frozen_stage1_config
    from .families import make_environment
    if not environment and not all(k in config.get('model',{}) for k in ('obs_shape','action_dim')):
        config=deepcopy(config);config.setdefault('model',{}).update(checkpoint_geometry(original))
    cfg = make_configuration(config, checkpoint=original)
    if multitask:
        raise ValueError('Use the raw TD-MPC2 WorldModel loader for gray-box multi-task victims; the latent attack path is single-task')
    validate_frozen_stage1_config(cfg)
    env = make_environment(cfg) if environment else None
    if env is None:
        require(isinstance(getattr(cfg, 'obs_shape', None), dict) and isinstance(getattr(cfg, 'action_dim', None), int),
                'Model-only loading needs model.obs_shape and model.action_dim metadata')
    agent = ThetaPsi(cfg, env)
    if not stage1_context:
        agent.model.train(False)
    return agent


def load_dynamics(path, agent):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    state = value.get('dynamics_state_dict', value.get('state_dict', value.get('model', value)))
    require(isinstance(state, dict), 'Dynamics/checkpoint state dictionary required')
    prefixed = {k[len('psi.'):]: v for k, v in state.items() if k.startswith('psi.')}
    if prefixed:
        state = prefixed
    dynamics = deepcopy(agent.psi_0)
    expected = dynamics.state_dict()
    require(set(state) == set(expected), 'Target dynamics keys do not match Original')
    require(all(state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype for k in state),
            'Target dynamics shape/dtype mismatch; implicit conversion is not allowed')
    dynamics.load_state_dict(state, strict=True)
    return dynamics.to(agent.device)
