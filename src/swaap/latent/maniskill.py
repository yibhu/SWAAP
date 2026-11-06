"""Corrected fixed-horizon state-observation ManiSkill adapter."""
from collections import defaultdict
from pathlib import Path
TASK_ROWS = {'lift-cube': {'environment_id': 'LiftCube-v0', 'checkpoint_number': 1, 'obs_dim': 42, 'action_dim': 4, 'original_sha256': 'da231968f24a40dd12cbb9415cd2c2ca41c1f7ff964d09b521a14c8866ea2d6a'}, 'pick-cube': {'environment_id': 'PickCube-v0', 'checkpoint_number': 1, 'obs_dim': 51, 'action_dim': 4, 'original_sha256': '1781f0fa959ee51fe96450e5bf5bf24f1d87ce515a89730e6b964c560144a258'}, 'stack-cube': {'environment_id': 'StackCube-v0', 'checkpoint_number': 3, 'obs_dim': 55, 'action_dim': 4, 'original_sha256': 'cee3bfaf6b1e1a7e505b414ef0b0a619f4825d7ff64bdbb84271c8ae2f3f6e66'}}

TASKS = tuple(TASK_ROWS)

CONTROL_MODE = 'pd_ee_delta_pos'

REWARD_MODE = 'normalized_dense'

EPISODE_LENGTH = 100

ACTION_REPEAT = 2

def need(condition, message):
    if not condition:
        raise ValueError(message)

def expected_episode_length(task):
    return 100 if task in TASK_ROWS or task.startswith(('myo-', 'mw-')) else 500

class FixedHorizonEnvironment:
    """Exactly100 outer decisions, exactly2 native steps each; no autoreset.

    Native success/termination flags are observed, not used to shorten this
    declared finite-horizon task. Success remains native success; the outer
    stopping event is the horizon. This is explicit task semantics, not an AND
    bug copied from the notebook or a post-hoc scaling of old returns.
    """

    def __init__(self, env, *, task, seed, torch):
        need(task in TASK_ROWS, 'Known ManiSkill task required')
        need(type(seed) is int and 0 <= seed < 2 ** 32, 'Unsigned environment seed required')
        self.env, self.task, self._seed, self._torch = (env, task, seed, torch)
        self.action_space, self.observation_space = (env.action_space, env.observation_space)
        need(tuple(self.observation_space.shape) == (TASK_ROWS[task]['obs_dim'],) and tuple(self.action_space.shape) == (TASK_ROWS[task]['action_dim'],), 'Bound ManiSkill state/action geometry differs')
        self._t = 0
        self._native_steps = 0
        self._first_reset = True
        self.max_episode_steps = self._max_episode_steps = EPISODE_LENGTH
        self.action_repeat = ACTION_REPEAT
        for space in (self.action_space, self.observation_space):
            if callable(getattr(space, 'seed', None)):
                space.seed(seed)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _observation(self, value):
        if isinstance(value, self._torch.Tensor):
            return value.float()
        return self._torch.as_tensor(value, dtype=self._torch.float32)

    def reset(self, task_idx=None, *, seed=None, options=None):
        need(task_idx in (None, -1), 'Single-task environment only')
        actual = self._seed if self._first_reset and seed is None else seed
        kwargs = {}
        if actual is not None:
            kwargs['seed'] = int(actual)
        if options is not None:
            kwargs['options'] = options
        result = self.env.reset(**kwargs)
        observation = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        self._first_reset = False
        self._t = 0
        self._native_steps = 0
        return self._observation(observation)

    def step(self, action):
        need(self._t < EPISODE_LENGTH, 'Reset required after declared outer horizon')
        if isinstance(action, self._torch.Tensor):
            action = action.detach().cpu().numpy()
        total = 0.0
        terminated_any = truncated_any = success_any = False
        info = {}
        for _ in range(ACTION_REPEAT):
            result = self.env.step(action)
            need(isinstance(result, tuple) and len(result) == 5, 'Qualified installed Gymnasium ManiSkill five-tuple required')
            observation, reward, terminated, truncated, info = result
            total += float(reward)
            self._native_steps += 1
            terminated_any = terminated_any or bool(terminated)
            truncated_any = truncated_any or bool(truncated)
            success_any = success_any or bool(info.get('success', False))
        self._t += 1
        timeout = self._t == EPISODE_LENGTH
        result_info = defaultdict(float, info)
        result_info.update(success=float(success_any), terminated=self._torch.tensor(0.0), truncated=timeout, native_terminated=terminated_any, native_truncated=truncated_any, native_steps=self._native_steps, native_steps_this_decision=ACTION_REPEAT, outer_timeout=timeout, termination_contract='fixed100_outer_repeat2_native_flags_observed')
        return (self._observation(observation), self._torch.tensor(total, dtype=self._torch.float32), timeout, result_info)

    def rand_act(self):
        return self._torch.as_tensor(self.action_space.sample(), dtype=self._torch.float32)

    def close(self):
        return self.env.close()

    def render(self, *a, **k):
        return self.env.render(*a, **k)

def make_environment(cfg, *, torch, package_root):
    need(cfg.task in TASK_ROWS, 'ManiSkill task required')
    import gymnasium as gym
    import mani_skill2
    import mani_skill2.envs
    origin = Path(mani_skill2.__file__).resolve()
    need(Path(package_root).resolve() in origin.parents, 'Source-verified installed corrected package required')
    if getattr(cfg,'maniskill_rendererless',True):
        from .maniskill_rendererless import make_environment as create_state
        env=create_state(cfg.task,obs_mode='state',control_mode=CONTROL_MODE,reward_mode=REWARD_MODE)
    else:
        env = gym.make(TASK_ROWS[cfg.task]['environment_id'], obs_mode='state', control_mode=CONTROL_MODE, reward_mode=REWARD_MODE, render_camera_cfgs={'width': 384, 'height': 384})
    need(getattr(env.unwrapped, 'reward_mode', None) == REWARD_MODE, 'Explicit normalized_dense reward mode was not honored')
    wrapped = FixedHorizonEnvironment(env, task=cfg.task, seed=int(cfg.seed), torch=torch)
    cfg.has_sr = True
    cfg.num_step = cfg.episode_length = EPISODE_LENGTH
    cfg.obs_shape = {'state': wrapped.observation_space.shape}
    cfg.action_dim = wrapped.action_space.shape[0]
    cfg.seed_steps = 1000
    return wrapped
