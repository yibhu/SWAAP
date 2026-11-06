"""Pinned latent SWAAP model, planner and effective Stage-1 updates.

Mechanically extracted scientific definitions. The executed bridge replaces
only the barrier denominator with the joint squared gradient norm. The running
scale uses an immutable denominator snapshot at each forward, not a frozen scale.
The notebook-style outer loop and operational logging have been removed.
See SOURCE_PROVENANCE.json for executed source identities and exact changes.
"""
from copy import deepcopy
import dataclasses
import os
from pathlib import Path
import pickle
import random
import re
import sys
from typing import Any
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from tqdm import tqdm
from swaap.latent._vendor.common import MODEL_SIZE, TASK_SET, init, layers, math
from swaap.latent._vendor.common.layers import api_model_conversion
from swaap.latent._vendor.common.scale import RunningScale
from swaap.latent._vendor.common.seed import set_seed
from swaap.latent._vendor.common.world_model import WorldModel

def cfg_to_dataclass(cfg, frozen=False):
    """
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
    cfg_dict = OmegaConf.to_container(cfg)
    fields = []
    for key, value in cfg_dict.items():
        fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
    dataclass_name = 'Config'
    dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)

    def get(self, val, default=None):
        return getattr(self, val, default)
    dataclass.get = get
    return dataclass()

def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match('(\\d+)([+\\-*/])(\\d+)', v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass
    import os
    from pathlib import Path
    try:
        from hydra.utils import get_original_cwd
        cwd = get_original_cwd()
    except Exception:
        cwd = os.getcwd()
    cfg.work_dir = Path(cwd) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace('-', ' ').title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    if cfg.get('model_size', None) is not None:
        assert cfg.model_size in MODEL_SIZE.keys(), f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        if cfg.task == 'mt30' and cfg.model_size == 19:
            cfg.latent_dim = 512
    cfg.multitask = cfg.task in TASK_SET.keys()
    if cfg.multitask:
        cfg.task_title = cfg.task.upper()
        cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
    else:
        cfg.task_dim = 0
    cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])
    return cfg_to_dataclass(cfg)

class ThetaPsiModel(nn.Module):
    """trainer class for applying the algorithm to TD-MPC2."""

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.cfg = cfg
        if self.cfg.load_from_tdmpc2_checkpoint == True:
            self.model = WorldModel(cfg).to(self.device)
            self.load(cfg.checkpoint)
            self._encoder = self.model._encoder
            self._termination = self.model._termination
            if self.cfg.load_v_checkpoint == True:
                self._Vs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_v)])
                self._Vs.load_state_dict(torch.load(self.cfg.v_checkpoint))
                print('v checkpoint loaded')
            else:
                self._Vs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_v)])
                self._Vs.apply(init.weight_init)
                init.zero_([self._Vs.params['2', 'weight']])
            if self.cfg.load_surrogate_agent:
                checkpoint = torch.load(self.cfg.surrogate_ckpt_path)
                print('surrogate checkpoint keys:', checkpoint.keys())
                self.model.load_state_dict(checkpoint['model'])
            self.psi_0 = deepcopy(self.model._dynamics)
            self.psi = deepcopy(self.model._dynamics)
            self.theta = deepcopy(self.model._pi)
            self.theta_star = deepcopy(self.model._pi)
            self._reward = deepcopy(self.model._reward)
            self._Qs = deepcopy(self.model._Qs)
        else:
            self._encoder = layers.enc(cfg)
            self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)
            self.psi_0 = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
            self.psi = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
            self.theta = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
            self.theta_star = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
            self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
            self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
            self._Vs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
            self._Vs.apply(init.weight_init)
            init.zero_([self._Vs.params['2', 'weight']])
        for net in [self.psi_0, self.psi, self.theta, self.theta_star, self._reward, self._Qs, self._Vs]:
            net.to(self.device)
        self.register_buffer('log_std_min', torch.tensor(cfg.log_std_min))
        self.register_buffer('log_std_dif', torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()
        if hasattr(cfg, 'experiment_checkpoint') and cfg.load_from_tdmpc2_checkpoint == False:
            self.load(cfg.experiment_checkpoint)
            print(f'loaded {cfg.experiment_checkpoint}')

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
    
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        if self.cfg.load_from_tdmpc2_checkpoint:
            state_dict = api_model_conversion(self.model.state_dict(), state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        return

    def init(self):
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
        with self._detach_Qs_params.data.to('meta').to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)
        delattr(self._detach_Qs, 'params')
        self._detach_Qs.__dict__['params'] = self._detach_Qs_params
        delattr(self._target_Qs, 'params')
        self._target_Qs.__dict__['params'] = self._target_Qs_params
        self._detach_Vs_params = TensorDictParams(self._Vs.params.data, no_convert=True)
        self._target_Vs_params = TensorDictParams(self._Vs.params.data.clone(), no_convert=True)
        with self._detach_Vs_params.data.to('meta').to_module(self._Vs.module):
            self._detach_Vs = deepcopy(self._Vs)
            self._target_Vs = deepcopy(self._Vs)
        delattr(self._detach_Vs, 'params')
        self._detach_Vs.__dict__['params'] = self._detach_Vs_params
        delattr(self._target_Vs, 'params')
        self._target_Vs.__dict__['params'] = self._target_Vs_params

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions', 'V-functions']
        for i, m in enumerate([self._encoder, self.psi, self._reward, self._termination, self.theta, self._Qs, self._Vs]):
            if m == self._termination and (not self.cfg.episodic):
                continue
            repr += f'{modules[i]}: {m}\n'
        repr += 'Learnable parameters: {:,}'.format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        self._target_Vs.train(False)
        self.psi_0.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def soft_update_target_V(self):
        self._target_Vs_params.lerp_(self._detach_Vs_params, self.cfg.tau)

    def encode(self, obs, task=None):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, psi=None, task=None):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if psi == None:
            psi = self.psi_0
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return psi(z)

    def reward(self, z, a, task=None):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(self, z, task=None, unnormalized=False):
        """
        Predicts termination signal.
        """
        assert task is None
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, theta=None, task=None, return_log_prob=False):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if theta == None:
            theta = self.theta
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        mean, log_std = theta(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)
        if self.cfg.multitask:
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:
            action_dims = None
        log_prob = math.gaussian_logprob(eps, log_std)
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        entropy_scale = scaled_log_prob / (log_prob + 1e-08)
        info = TensorDict({'mean': mean, 'log_std': log_std, 'action_prob': 1.0, 'entropy': -log_prob, 'scaled_entropy': -log_prob * entropy_scale})
        if return_log_prob:
            return (action, log_prob)
        else:
            return (action, info)

    def Q(self, z, a, task=None, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)
        if return_type == 'all':
            return out
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == 'min':
            return Q.min(0).values
        return Q.sum(0) / 2

    def V(self, z, task=None, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled V-values.
            - `avg`: return the average of two randomly subsampled V-values.
            - `all`: return all V-values.
        `target` specifies whether to use the target V-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if target:
            vnet = self._target_Vs
        elif detach:
            vnet = self._detach_Vs
        else:
            vnet = self._Vs
        out = vnet(z)
        if return_type == 'all':
            return out
        qidx = torch.randperm(self.cfg.num_v, device=out.device)[:2]
        V = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == 'min':
            return V.min(0).values
        return V.sum(0) / 2

class SequenceReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size, batch_size, horizon, device, critical_state, r_c=None):
        self.max_size = max_size
        self.horizon = horizon
        self.device = device
        self.batch_size = batch_size
        self.critical_state = critical_state
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.bool_)
        self.masks = np.zeros((max_size,), dtype=np.bool_)
        if r_c is not None:
            self.random_masks = np.random.rand(max_size) < r_c
        else:
            self.random_masks = None
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, mask=None):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        if self.critical_state:
            self.masks[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_trajectory(self, data):
        """Add one full trajectory (arrays of length T)."""
        if len(data) == 6:
            states, actions, rewards, next_states, dones, masks = data
            for s, a, r, ns, d, m in zip(states, actions, rewards, next_states, dones, masks):
                self.add(s, a, r, ns, d, mask=m)
        else:
            states, actions, rewards, next_states, dones = data
            for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                self.add(s, a, r, ns, d)

    def sample(self, random_critical=False, horizon=None):
        if horizon is None:
            horizon = self.horizon
        batch_size = self.batch_size
        idxs = []
        while len(idxs) < batch_size:
            idx = np.random.randint(0, self.size - horizon)
            if self.dones[idx:idx + horizon - 1].any():
                continue
            idxs.append(idx)
        states = np.stack([self.states[i:i + horizon] for i in idxs])
        actions = np.stack([self.actions[i:i + horizon] for i in idxs])
        rewards = np.stack([self.rewards[i:i + horizon] for i in idxs])
        next_states = np.stack([self.next_states[i:i + horizon] for i in idxs])
        dones = np.stack([self.dones[i:i + horizon] for i in idxs])
        if self.critical_state:
            if random_critical:
                masks = np.stack([self.random_masks[i:i + horizon] for i in idxs])
            else:
                masks = np.stack([self.masks[i:i + horizon] for i in idxs])
        states = np.swapaxes(states, 0, 1)
        actions = np.swapaxes(actions, 0, 1)
        rewards = np.swapaxes(rewards, 0, 1)
        next_states = np.swapaxes(next_states, 0, 1)
        dones = np.swapaxes(dones, 0, 1)
        if self.critical_state:
            masks = np.swapaxes(masks, 0, 1)
            return (torch.tensor(states, device=self.device), torch.tensor(actions, device=self.device), torch.tensor(rewards, device=self.device).unsqueeze(-1), torch.tensor(next_states, device=self.device), torch.tensor(dones, device=self.device).float().unsqueeze(-1), torch.tensor(masks, device=self.device).float().unsqueeze(-1))
        return (torch.tensor(states, device=self.device), torch.tensor(actions, device=self.device), torch.tensor(rewards, device=self.device).unsqueeze(-1), torch.tensor(next_states, device=self.device), torch.tensor(dones, device=self.device).float().unsqueeze(-1))

class ThetaPsi(torch.nn.Module):

    def __init__(self, cfg, env, buffer=None, eval_vanilla=None, eval_proxy=None):
        super().__init__()
        self.cfg = cfg
        self.eval_vanilla = eval_vanilla
        self.eval_proxy = eval_proxy
        self.env = env
        self.device = torch.device('cuda:0')
        self.model = ThetaPsiModel(cfg).to(self.device)
        self.offline_buffer = buffer
        self._encoder = self.model._encoder
        self._termination = self.model._termination
        self.psi_0 = self.model.psi_0
        self.psi = self.model.psi
        self.theta_0 = deepcopy(self.model.theta)
        self.theta = self.model.theta
        self.theta_star = self.model.theta_star
        if self.cfg.load_imitate_policy:
            self.theta_imitate = deepcopy(self.model.theta)
            checkpoint = torch.load(self.cfg.imitate_policy_ckpt_path)
            self.theta_imitate.load_state_dict(checkpoint)
            print(f'loaded imitate policy from {self.cfg.imitate_policy_ckpt_path}')
        self.psi_optimizer = optim.Adam(self.model.psi.parameters(), lr=self.cfg.lr)
        self.theta_optimizer = optim.Adam(self.model.theta.parameters(), lr=self.cfg.lr)
        self.theta_star_optimizer = optim.Adam(self.model.theta_star.parameters(), lr=self.cfg.lr)
        self.optimizer = torch.optim.Adam([{'params': self.model._Qs.parameters()}, {'params': self.model._Vs.parameters()}], lr=self.cfg.lr, capturable=True)
        self.q_scale = RunningScale(cfg)
        self.v_scale = RunningScale(cfg)
        self.discount = torch.tensor([self._get_discount(ep_len) for ep_len in self.cfg.episode_lengths], device='cuda:0') if self.cfg.multitask else self._get_discount(self.cfg.episode_length)
        print('Episode length:', self.cfg.episode_length)
        print('Discount factor:', self.discount)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))

    def save(self, fp=None):
        """
        Save state dict of the agent to filepath.
    
        Args:
            fp (str): Filepath to save state dict to. If None, uses default.
        """
        if fp is None:
            fp = f'{self.cfg.task}_experiment.pt'
        torch.save({'model': self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
    
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict['model'] if 'model' in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return

    @torch.no_grad()
    def act(self, obs, psi=None, theta=None, t0=False, eval_mode=False, task=None, input_obs=True):
        if psi is None:
            psi = self.psi_0
        if theta is None:
            theta = self.theta
        obs = obs.to(self.device, non_blocking=True)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        z = self.model.encode(obs, task) if input_obs else obs
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            return self.plan(z, psi, theta, t0=t0, eval_mode=eval_mode, task=task)
        action, info = self.model.pi(z, theta, task)
        if eval_mode:
            action = info['mean']
        return action[0]

    @property
    def plan(self):
        _plan_val = getattr(self, '_plan_val', None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode='reduce-overhead')
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.
    
        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.
    
        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

    @torch.no_grad()
    def _estimate_value(self, z, actions, psi, theta, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = (0, 1)
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], psi, task)
            G = G + discount * (1 - termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.0)
        action, _ = self.model.pi(z, theta, task)
        return G + discount * (1 - termination) * self.model.Q(z, action, task, return_type='avg')

    @torch.no_grad()
    def _plan(self, z, psi, theta, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
    
        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).
    
        Returns:
            torch.Tensor: Action to take in the environment.
        """
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t], _ = self.model.pi(_z, theta, task)
                _z = self.model.next(_z, pi_actions[t], psi, task)
            pi_actions[-1], _ = self.model.pi(_z, theta, task)
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
        for _ in range(self.cfg.iterations):
            r = torch.randn(self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]
            value = self._estimate_value(z, actions, psi, theta, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = (value[elite_idxs], actions[:, elite_idxs])
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-09)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-09)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = (actions[0], std[0])
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def test(self, psi=None, theta=None, direct_perturb_action=False, random_model_perturb_chance=0, num_test_episodes=1, num_collect_episodes=2000, eval_mode=False, collect_mode=False, output_extra=False, visualize_details=False, have_print=False, no_tqdm=False, online=False, model_p_data_p_psi=None, seed=None):
        print(f'testing with mpc {self.cfg.mpc}')
        if not self.cfg.multitask:
            task_idx = None
        if collect_mode == True:
            output_extra = False
            visualize_details = False
            traj_num = num_collect_episodes
            traj_length = 500
            max_size = traj_num * traj_length
            buffer = SequenceReplayBuffer(state_dim=512, action_dim=21, max_size=max_size, horizon=3, device='cuda')
        with torch.no_grad():
            ep_rewards, ep_losses = ([], [])
            all_next_z = []
            all_next_z_pred = []
            all_next_z_pred_mp = []
            all_next_z_pred_dp = []
            per_episode_extra = []
            all_rewards = []
            offline_dataset = []
            dp_mp_mse = []
            clean_mp_mse = []
            if collect_mode == True:
                num_episodes = traj_num
            else:
                num_episodes = num_test_episodes
            success_count = 0
            if no_tqdm:
                iterator = range(num_episodes)
            else:
                iterator = tqdm(range(num_episodes), desc=f'[Testing] {self.cfg.task}', leave=False)
            for i in iterator:
                print(f'-----------------------------{i}th test episode-----------------------------')
                if seed is None:
                    obs, done, ep_reward, ep_loss, t = (self.env.reset(task_idx=task_idx), False, 0, 0, 0)
                else:
                    obs, done, ep_reward, ep_loss, t = (self.env.reset(task_idx=task_idx), False, 0, 0, 0)
                episode_next_z = []
                episode_next_z_pred = []
                episode_next_z_pred_mp = []
                episode_next_z_pred_dp = []
                states = []
                actions = []
                rewards = []
                next_states = []
                next_states_pred = []
                dones = []
                deltas = []
                ts = []
                success = False
                count = 0
                while not done:
                    if random_model_perturb_chance == 0:
                        action = self.act(obs, psi, theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=True).cpu()
                        if direct_perturb_action:
                            action = self.act(obs, self.psi_0, theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=True).cpu()
                            original_action = action
                            with torch.enable_grad():
                                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                                z = self.model.encode(obs_tensor)
                                w = self.compute_importance(z, action)
                                if w > self.cfg.w_c:
                                    action = self.act(obs, psi, theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=True).cpu()
                                    count += 1
                    elif random.random() < random_model_perturb_chance:
                        action = self.act(obs, psi, theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=True).cpu()
                    else:
                        action = self.act(obs, self.psi_0, theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=True).cpu()
                    action = torch.clamp(action, -1.0, 1.0)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    z = self.model.encode(obs_tensor)
                    next_z_pred = self.model.next(z, action.to(z.device), psi)
                    if model_p_data_p_psi:
                        model_poisoned_psi, data_poisoned_psi = model_p_data_p_psi
                        next_z_pred_mp = self.model.next(z, action.to(z.device), model_poisoned_psi)
                        next_z_pred_dp = self.model.next(z, action.to(z.device), data_poisoned_psi)
                        print('F.MSELoss(next_z_pred-next_z_pred_mp):', F.mse_loss(next_z_pred, next_z_pred_mp))
                        print('F.MSELoss(next_z_pred_mp-next_z_pred_dp):', F.mse_loss(next_z_pred_mp, next_z_pred_dp))
                        dp_mp_mse.append(F.mse_loss(next_z_pred_mp, next_z_pred_dp).cpu())
                        clean_mp_mse.append(F.mse_loss(next_z_pred, next_z_pred_mp).cpu())
                    obs, reward, done, info = self.env.step(action)
                    if self.cfg.has_sr:
                        if info['success'] and (not success):
                            print("info['success']:", info['success'])
                            success = True
                            success_count += 1
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    next_z = self.model.encode(obs_tensor)
                    if online:
                        print_stats(next_z, title='next_z')
                        print_stats(next_z_pred, title='next_z_pred')
                        delta = next_z - next_z_pred
                        delta_norm = delta.norm(p=2, dim=1, keepdim=True)
                        z_norm = next_z.norm(p=2, dim=1, keepdim=True)
                        relative_delta_norm = delta_norm / (z_norm + 1e-08)
                        deltas.append(relative_delta_norm)
                    loss = F.mse_loss(next_z_pred, next_z)
                    episode_next_z.append(next_z.cpu())
                    episode_next_z_pred.append(next_z_pred.cpu())
                    if model_p_data_p_psi:
                        episode_next_z_pred_mp.append(next_z_pred_mp.cpu())
                        episode_next_z_pred_dp.append(next_z_pred_dp.cpu())
                    ep_reward += reward
                    ep_loss += loss.item()
                    t += 1
                    states.append(z.unsqueeze(0))
                    actions.append(action.unsqueeze(0))
                    rewards.append(reward.unsqueeze(0))
                    next_states.append(next_z.unsqueeze(0))
                    next_states_pred.append(next_z_pred.unsqueeze(0))
                    dones.append(torch.tensor(float(done)).unsqueeze(0))
                    all_rewards.append(reward.unsqueeze(0))
                if direct_perturb_action:
                    print('total critical found:', count)
                print('float(ep_reward):', float(ep_reward))
                states = torch.cat(states, dim=0)
                actions = torch.cat(actions, dim=0)
                rewards = torch.cat(rewards, dim=0)
                next_states = torch.cat(next_states, dim=0)
                next_states_pred = torch.cat(next_states_pred, dim=0)
                dones = torch.cat(dones, dim=0)
                if collect_mode:
                    traj = {'states': states.cpu(), 'actions': actions.cpu(), 'rewards': rewards.cpu(), 'next_states': next_states.cpu(), 'dones': dones.cpu()}
                    offline_dataset.append(traj)
                if have_print:
                    print('states.shape, actions.shape, rewards.shape, next_states.shape, next_states_pred.shape, dones.shape:', states.shape, actions.shape, rewards.shape, next_states.shape, next_states_pred.shape, dones.shape)
                qs = self.model.Q(states.to(self.device), actions.to(self.device), return_type='avg', detach=True)
                dones_pred = self.model.termination(states.to(self.device), unnormalized=False)
                rewards_pred = math.two_hot_inv(self.model.reward(states.to(self.device), actions.to(self.device)), self.cfg)
                if have_print:
                    print('qs.shape, vs.shape, dones_pred.shape, rewards_pred.shape:', qs.shape, vs.shape, dones_pred.shape, rewards_pred.shape)
                if visualize_details == True:
                    print_stats(qs)
                    print_stats(rewards)
                    plt.figure(figsize=(12, 5))
                    plt.plot(qs.cpu().numpy(), label='Q-values')
                    plt.xlabel('Step')
                    plt.ylabel('Value')
                    plt.title('Q values over rollout steps')
                    plt.legend()
                    plt.show()
                    pred_loss = self.loss_fn(next_states_pred, next_states, dim=1)
                    plt.figure(figsize=(12, 3))
                    plt.plot(pred_loss.detach().cpu(), label='pred_loss')
                    plt.legend()
                    plt.show()
                    with torch.enable_grad():
                        ws = []
                        print(states.shape, actions.shape)
                        for i in range(states.shape[0]):
                            z = states[i]
                            a = actions[i]
                            w = self.compute_importance(z, a).detach().cpu()
                            ws.append(w)
                        plt.figure(figsize=(12, 3))
                        plt.plot(ws, label='ws')
                        plt.legend()
                        plt.show()
                    rewards = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
                    rewards = rewards.to(qs.device)
                    plt.figure(figsize=(12, 3))
                    plt.plot(rewards.cpu().numpy(), label='Rewards')
                    plt.xlabel('Step')
                    plt.ylabel('Rewards')
                    plt.title('Reward structure')
                    plt.legend()
                    plt.show()
                    rewards = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
                    reward_diff = (rewards.to(qs.device) - rewards_pred).abs()
                    print(f'Mean absolute error between rewards and rewards_pred: {reward_diff.mean().item():.4f}')
                    plt.figure(figsize=(12, 3))
                    plt.plot(reward_diff.cpu().numpy(), label='Reward absolute error')
                    plt.xlabel('Step')
                    plt.ylabel('Absolute error')
                    plt.title('Difference between rewards and predicted rewards')
                    plt.legend()
                    plt.show()
                ep_rewards.append(float(ep_reward))
                ep_losses.append(float(ep_loss))
                episode_next_z = torch.stack(episode_next_z)
                episode_next_z_pred = torch.stack(episode_next_z_pred)
                if model_p_data_p_psi:
                    episode_next_z_pred_mp = torch.stack(episode_next_z_pred_mp)
                    episode_next_z_pred_dp = torch.stack(episode_next_z_pred_dp)
                if output_extra:
                    magnitude_per_dim = episode_next_z.abs().mean(dim=0)
                    deviation_per_dim = (episode_next_z - episode_next_z_pred).abs().mean(dim=0)
                    mean_magnitude = magnitude_per_dim.mean().item()
                    mean_deviation = deviation_per_dim.mean().item()
                    relative_deviation = mean_deviation / (mean_magnitude + 1e-08)
                    per_episode_extra.append({'magnitude_per_dim': magnitude_per_dim, 'deviation_per_dim': deviation_per_dim, 'mean_magnitude': mean_magnitude, 'mean_deviation': mean_deviation, 'relative_deviation': relative_deviation, 'reward': ep_reward})
                all_next_z.append(episode_next_z)
                all_next_z_pred.append(episode_next_z_pred)
                if model_p_data_p_psi:
                    all_next_z_pred_mp.append(episode_next_z_pred_mp)
                    all_next_z_pred_dp.append(episode_next_z_pred_dp)
            if model_p_data_p_psi:
                print('np.mean(dp_mp_mse):', np.mean(dp_mp_mse))
                print('np.mean(clean_mp_mse):', np.mean(clean_mp_mse))
                return (torch.stack(all_next_z_pred), torch.stack(all_next_z_pred_mp), torch.stack(all_next_z_pred_dp))
            mean_devs = []
            mean_rel_devs = []
            for i in range(len(all_next_z)):
                ep_next_z = all_next_z[i]
                ep_next_z_pred = all_next_z_pred[i]
                magnitude_per_dim = ep_next_z.abs().mean(dim=0)
                deviation_per_dim = (ep_next_z - ep_next_z_pred).abs().mean(dim=0)
                mean_magnitude = magnitude_per_dim.mean().item()
                mean_dev = deviation_per_dim.mean().item()
                mean_rel_dev = mean_dev / (mean_magnitude + 1e-08)
                mean_devs.append(mean_dev)
                mean_rel_devs.append(mean_rel_dev)
            next_z = torch.cat(all_next_z, dim=0)
            next_z_pred = torch.cat(all_next_z_pred, dim=0)
            test_rel_devs = (next_z - next_z_pred).abs().mean(dim=0) / (next_z.abs().reshape(-1, 512).mean() + 1e-08)
            test_devs = (next_z - next_z_pred).abs().mean(dim=0)
            print_stats(test_rel_devs, title='test_rel_devs')
            print_stats(test_rel_devs, title='test_devs')
            delta = next_z - next_z_pred
            delta_norm = delta.norm(p=2, dim=1, keepdim=True)
            z_norm = next_z.norm(p=2, dim=1, keepdim=True)
            relative_delta_norm = delta_norm / (z_norm + 1e-08)
            print_stats(abs(delta).mean(dim=1), title='delta_abs')
            print_stats(delta_norm, title='delta_norm')
            print_stats(relative_delta_norm, title='relative_delta_norm')
            print_stats(ep_rewards, title='ep_rewards')
            relative_delta_norm = torch.tensor(relative_delta_norm)
            if self.cfg.has_sr:
                print(colored(f'  {self.cfg.task:<22}\tR: {np.mean(ep_rewards):.01f}\tsr: {success_count / num_episodes:.01f}\tΔl: {np.mean(ep_losses):.02f}\tΔz: {torch.mean(delta_norm):.2f}\tΔz/z: {torch.mean(relative_delta_norm):.2f}', 'yellow'))
            else:
                print(colored(f'  {self.cfg.task:<22}\tR: {np.mean(ep_rewards):.01f}\tΔl: {np.mean(ep_losses):.02f}\tΔz: {torch.mean(delta_norm):.2f}\tΔz/z: {torch.mean(relative_delta_norm):.2f}', 'yellow'))
            if output_extra:
                extra = {'all_next_z': all_next_z, 'all_next_z_pred': all_next_z_pred, 'magnitude_per_dim': magnitude_per_dim, 'deviation_per_dim': deviation_per_dim, 'mean_magnitude': mean_magnitude, 'mean_deviation': mean_deviation, 'mean_relative_deviation': mean_relative_deviation, 'per_episode_extra': per_episode_extra}
            if collect_mode == True:
                for traj in offline_dataset:
                    buffer.add_trajectory(traj['states'], traj['actions'], traj['rewards'], traj['next_states'], traj['dones'])
                return buffer
            if self.cfg.has_sr:
                return (ep_rewards, all_rewards, np.mean(ep_losses), torch.mean(delta_norm), torch.mean(relative_delta_norm), delta_norm, relative_delta_norm, success_count / num_episodes, next_z, next_z_pred)
            if not output_extra:
                return (ep_rewards, all_rewards, np.mean(ep_losses), torch.mean(delta_norm), torch.mean(relative_delta_norm), delta_norm, relative_delta_norm, next_z, next_z_pred)
            else:
                return (ep_rewards, all_rewards, np.mean(ep_losses), mean_devs, mean_rel_devs, extra)

    def collect(self, psi=None, theta=None, batch_size=1, num_step=500, use_env=False, random_model_perturb_chance=0, random_action_chance=0):
        if psi is None:
            psi = self.psi_0
        if theta is None:
            theta = self.theta_0
        if not self.cfg.multitask:
            task_idx = None
        trajs = []
        with torch.no_grad():
            for i in range(batch_size):
                if self.env is not None:
                    obs = self.env.reset()
                else:
                    obs = np.random.rand(psi.fc1.in_features)
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                z = self.model.encode(obs).squeeze(0)
                termination = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
                traj = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
                for t in range(self.cfg.num_step):
                    if random_action_chance == 0:
                        if random_model_perturb_chance == 0:
                            action = self.act(z, psi=psi, theta=theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=False).clone()
                        else:
                            action = self.act(z, psi=self.psi_0, theta=theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=False).clone()
                    elif random.random() < random_action_chance:
                        action = 2 * torch.rand(self.cfg.action_dim, device=self.device) - 1
                    else:
                        action = self.act(z, psi=psi, theta=theta, t0=t == 0, eval_mode=False, task=task_idx, input_obs=False).clone()
                    if use_env:
                        obs, reward, done, info = self.env.step(action.cpu())
                        reward = reward.unsqueeze(0)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        next_z = self.model.encode(obs_tensor)
                    else:
                        reward = math.two_hot_inv(self.model.reward(z, action), self.cfg)
                        if random_model_perturb_chance == 0:
                            next_z = self.model.next(z, action, psi)
                        elif random.random() < random_model_perturb_chance:
                            next_z = self.model.next(z, action, psi)
                        else:
                            next_z = self.model.next(z, action, self.psi_0)
                    traj['states'].append(z.unsqueeze(0))
                    traj['actions'].append(action.unsqueeze(0))
                    traj['rewards'].append(reward.unsqueeze(0))
                    traj['next_states'].append(next_z.unsqueeze(0))
                    traj['dones'].append(termination.clone())
                    z = next_z
                    is_done = (self.model.termination(z) > 0.5).float()
                    termination = torch.clip(termination + is_done, max=1.0)
                    if use_env and done:
                        break
                trajs.append({k: torch.cat(v, dim=0).unsqueeze(1) for k, v in traj.items()})
            states = torch.cat([traj['states'] for traj in trajs], dim=1)
            actions = torch.cat([traj['actions'] for traj in trajs], dim=1)
            rewards = torch.cat([traj['rewards'] for traj in trajs], dim=1)
            next_states = torch.cat([traj['next_states'] for traj in trajs], dim=1)
            dones = torch.cat([traj['dones'] for traj in trajs], dim=1)
            batch = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        dones = torch.zeros_like(dones)
        data = (states, actions, rewards, next_states, dones)
        data = [t.transpose(0, 1).reshape(-1, *t.shape[2:]).cpu() for t in data]
        if self.cfg.critical_state:
            ws = self.pgd(data)
            q = 1 - self.cfg.w_c
            w_c = torch.quantile(ws, q)
            masks = ws >= w_c
            print(f'w_c: {self.cfg.w_c}, number of critical states: {sum(masks).item()}/{self.cfg.num_step * self.cfg.num_collect_trajs} ({sum(masks).item() / self.cfg.num_step * self.cfg.num_collect_trajs})')
            data.append(masks)
        return data

    def backprop_theta(self, batch, theta, task=None, make_update=False):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
        action, info = self.model.pi(zs, theta, task=None)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.q_scale.update(qs[0])
        qs = self.q_scale(qs)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        return_objective = ((self.cfg.entropy_coef * info['scaled_entropy'] + qs).mean(dim=(1, 2)) * rho).mean()
        if make_update:
            loss = -return_objective
            loss.backward()
            torch.nn.utils.clip_grad_norm_(theta.parameters(), self.cfg.grad_clip_norm)
            self.theta_star_optimizer.step()
            self.theta_star_optimizer.zero_grad(set_to_none=True)
            return (return_objective, None)
        grads = torch.autograd.grad(return_objective, theta.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        return (return_objective, flat_grads)

    def backprop_psi(self, batch, task=None):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
            masks = torch.ones_like(reward)
        next_z_pred = self.model.next(zs, action, self.psi)
        next_action, log_prob = self.model.pi(next_z_pred, self.theta, task=None, return_log_prob=True)
        qs = self.model.Q(next_z_pred, next_action, task, return_type='avg', detach=True)
        self.q_scale.update(qs[0])
        qs = self.q_scale(qs)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        critical_qs = qs * masks
        return_objective = (+critical_qs.mean(dim=(1, 2)) * rho).mean()
        grads = torch.autograd.grad(return_objective, self.psi.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        if False:
            print_stats(qs)
            print_stats(critical_qs)
            plot_hists([qs.view(-1), critical_qs.view(-1)], titles=['qs', 'critical_qs'])
        return (return_objective, flat_grads)

    def project_next_z_pred(self, next_z_pred, next_z, eps=None, norm='linf'):
        """
        Project next_z_pred onto an epsilon-ball centered at next_z.
        """
        delta = next_z_pred - next_z
        if norm == 'linf':
            delta_proj = delta.clamp(min=-eps, max=eps)
            return next_z + delta_proj
        elif norm == 'l2':
            delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(eps / delta_norm, max=1.0)
            delta_proj = delta * scale
            return next_z + delta_proj
        else:
            raise ValueError(f'Unsupported norm: {norm}')

    def consistency_loss(self, batch, task=None, make_update=False):
        if self.cfg.critical_state:
            zs, action, reward, next_z, terminated, masks = batch
        else:
            zs, action, reward, next_z, terminated = batch
            masks = torch.ones_like(reward)
        next_z_pred = self.model.next(zs, action, self.psi)
        if make_update:
            next_z_projected = self.project_next_z_pred(next_z_pred, next_z, eps=0.12, norm='l2').detach()
            loss = F.mse_loss(next_z_pred, next_z_projected) + F.mse_loss(next_z_pred, next_z)
        else:
            loss = F.mse_loss(next_z_pred, next_z)
        if make_update:
            self.psi_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.cfg.grad_clip_norm)
            self.psi_optimizer.step()
            self.psi_optimizer.zero_grad(set_to_none=True)
            return (loss.detach(), None)
        grads = torch.autograd.grad(loss, self.psi.parameters(), retain_graph=True, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        return (loss, flat_grads)

    def make_buffer(self, data, r_c=None):
        buffer = SequenceReplayBuffer(state_dim=self.cfg.latent_dim, action_dim=self.cfg.action_dim, max_size=self.cfg.buffer_size, batch_size=self.cfg.sample_batch_size, horizon=self.cfg.horizon, device=self.device, critical_state=self.cfg.critical_state, r_c=r_c)
        print(f'make data buffer with critical mask = {self.cfg.critical_state}')
        buffer.add_trajectory(data)
        return buffer

    def process_batch(self, buffer, fn):
        batch = buffer.sample()
        ret = fn(batch)
        return ret

    def update(self, apply_reachability=False):
        print('collect trajectory from psi_k theta_k')
        data = self.collect(self.psi, self.theta, self.cfg.num_collect_trajs, self.cfg.num_step)
        buffer_kk = self.make_buffer(data)
        self.theta_star.load_state_dict(self.theta.state_dict())
        for _ in range(self.cfg.T_step):
            self.process_batch(buffer_kk, lambda batch: self.backprop_theta(batch, theta=self.theta_star, make_update=True))
        print('collect trajectory from psi_k theta_T')
        data = self.collect(self.psi, self.theta_star, self.cfg.num_collect_trajs, self.cfg.num_step)
        buffer_kT = self.make_buffer(data)
        print('collect trajectory from env theta_k')
        data = self.collect(self.psi_0, self.theta, self.cfg.num_collect_trajs, self.cfg.num_step, use_env=True)
        buffer_0k = self.make_buffer(data)
        print('collect trajectory from env theta_0')
        data = self.collect(self.psi_0, self.theta_0, self.cfg.num_collect_trajs, self.cfg.num_step, use_env=True)
        buffer_gt = self.make_buffer(data)
        psi_losses = []
        theta_losses = []
        lambda_ks = []
        for i in range(self.cfg.update_batch_size):
            loss_psi_kT, grads_psi_kT = self.process_batch(buffer_kT, lambda batch: self.backprop_psi(batch))
            loss_theta_0k, grads_theta_0k = self.process_batch(buffer_0k, lambda batch: self.backprop_theta(batch, self.theta))
            loss_psi_consistency, grads_psi_consistency = self.process_batch(buffer_gt, lambda batch: self.consistency_loss(batch))
            loss_psi_regularization, grads_psi_regularization = self.process_batch(buffer_gt, lambda batch: self.consistency_loss(batch))
            if self.cfg.use_consistency:
                loss_psi_constraint, grads_psi_constraint = (loss_psi_consistency, grads_psi_consistency)
            else:
                loss_psi_constraint, grads_psi_constraint = (loss_psi_regularization, grads_psi_regularization)
            loss_psi_kk, grads_psi_kk = self.process_batch(buffer_kk, lambda batch: self.backprop_psi(batch))
            loss_theta_kk, grads_theta_kk = self.process_batch(buffer_kk, lambda batch: self.backprop_theta(batch, self.theta))
            eta = 0.5
            grads_f = (self.cfg.lambda_coef * grads_psi_constraint, grads_theta_0k)
            grads_psi_kT_minus_grads_psi_kk = grads_psi_kT - grads_psi_kk
            grads_q = (grads_psi_kT_minus_grads_psi_kk, -grads_theta_kk)
            grads_q_norm = grads_q[0].square().sum() + grads_q[1].square().sum()
            dot_product = torch.dot(grads_f[0], grads_q[0]) + torch.dot(grads_f[1], grads_q[1])
            lambda_k = ((eta * grads_q_norm - dot_product) / (grads_q_norm + 1e-08)).detach()
            lambda_k = lambda_k.clamp(min=0.0)
            loss_psi = self.cfg.lambda_coef * loss_psi_constraint + lambda_k * (loss_psi_kT - loss_psi_kk)
            loss_theta = loss_theta_0k - lambda_k * loss_theta_kk
            psi_losses.append(loss_psi)
            theta_losses.append(loss_theta)
            lambda_ks.append(lambda_k)
        print_cuda_stats()
        loss_psi_base = sum(psi_losses) / len(psi_losses)
        loss_psi_total = loss_psi_base
        reachability_metrics = {'reachability_applied': False, 'reachability_ratio': float('nan'), 'reachability_short_rho_f': float('nan'), 'reachability_penalty': 0.0, 'reachability_effective_weight': 0.0, 'reachability_base_grad_norm': float('nan'), 'reachability_constraint_grad_norm': float('nan'), 'reachability_clean_target_mse': float('nan'), 'reachability_poison_target_mse': float('nan')}
        if apply_reachability and bool(self.cfg.reachability_enabled) and (self._reachability_poison_proxy_psi is not None):
            reachability_penalty, current_reachability_metrics = self.reachability_constraint_loss()
            psi_parameters = [parameter for parameter in self.psi.parameters() if parameter.requires_grad]
            base_gradients = torch.autograd.grad(loss_psi_base, psi_parameters, retain_graph=True, create_graph=False, allow_unused=True)
            reachability_gradients = torch.autograd.grad(reachability_penalty, psi_parameters, retain_graph=True, create_graph=False, allow_unused=True)

            def gradient_norm(gradients):
                squared_terms = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
                if len(squared_terms) == 0:
                    return torch.zeros((), device=self.device)
                return torch.sqrt(torch.stack(squared_terms).sum())
            base_gradient_norm = gradient_norm(base_gradients)
            reachability_gradient_norm = gradient_norm(reachability_gradients)
            if reachability_gradient_norm.item() > 0.0:
                effective_weight = float(self.cfg.reachability_grad_ratio) * base_gradient_norm / (reachability_gradient_norm + 1e-12)
                effective_weight = effective_weight.clamp(max=float(self.cfg.reachability_max_weight))
            else:
                effective_weight = torch.zeros((), device=self.device)
            loss_psi_total = loss_psi_base + effective_weight.detach() * reachability_penalty
            reachability_metrics.update(current_reachability_metrics)
            reachability_metrics.update({'reachability_applied': True, 'reachability_effective_weight': float(effective_weight.detach().item()), 'reachability_base_grad_norm': float(base_gradient_norm.item()), 'reachability_constraint_grad_norm': float(reachability_gradient_norm.item())})
        self.psi_optimizer.zero_grad(set_to_none=True)
        loss_psi_total.backward()
        torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.cfg.grad_clip_norm)
        self.psi_optimizer.step()
        loss_theta = sum(theta_losses) / len(theta_losses)
        self.theta_optimizer.zero_grad()
        loss_theta.backward()
        torch.nn.utils.clip_grad_norm_(self.theta.parameters(), self.cfg.grad_clip_norm)
        self.theta_optimizer.step()
        print('lambda_k, loss_psi_base, loss_psi_total, loss_theta:', lambda_k.detach().cpu().item(), loss_psi_base.detach().cpu().item(), loss_psi_total.detach().cpu().item(), loss_theta.detach().cpu().item())
        info = TensorDict({'lambda_k': lambda_k.detach().cpu().item(), 'loss_psi': loss_psi_total.detach().cpu().item(), 'loss_psi_base': loss_psi_base.detach().cpu().item(), 'loss_psi_total': loss_psi_total.detach().cpu().item(), 'loss_theta': loss_theta.detach().cpu().item(), 'loss_psi_consistency': loss_psi_consistency.detach().cpu().item(), 'loss_psi_regularization': loss_psi_regularization.detach().cpu().item(), 'reachability_applied': bool(reachability_metrics['reachability_applied']), 'reachability_ratio': float(reachability_metrics['reachability_ratio']), 'reachability_short_rho_f': float(reachability_metrics['reachability_short_rho_f']), 'reachability_penalty': float(reachability_metrics['reachability_penalty']), 'reachability_effective_weight': float(reachability_metrics['reachability_effective_weight']), 'reachability_base_grad_norm': float(reachability_metrics['reachability_base_grad_norm']), 'reachability_constraint_grad_norm': float(reachability_metrics['reachability_constraint_grad_norm']), 'reachability_clean_target_mse': float(reachability_metrics['reachability_clean_target_mse']), 'reachability_poison_target_mse': float(reachability_metrics['reachability_poison_target_mse'])})
        return info

def print_cuda_stats(ignore=False):
    if ignore == False:
        free_mem = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f'CUDA memory - Free: {free_mem:.2f} GB / Total: {total_mem:.2f} GB')

def print_stats(x, full_detail=False, title=None):
    if isinstance(x, list):
        x = torch.tensor(x)
    if isinstance(x, torch.Tensor) and x.is_cuda:
        x = x.cpu()
    if full_detail:
        print(f'x: shape = {x.shape}')
        print('type(x):', type(x))
        print('x.dtype:', x.dtype)
    if title is None:
        print(f'  mean: {x.mean():.4f}, std: {torch.sqrt(x.var()):.12f}, var: {x.var():.12f}, min: {x.min():.4f}, max: {x.max():.4f}, shape: {x.shape}')
    else:
        print(f'  {title}: mean: {x.mean():.4f}, std: {torch.sqrt(x.var()):.12f}, var: {x.var():.12f}, min: {x.min():.4f}, max: {x.max():.4f}, shape: {x.shape}')

def validate_frozen_stage1_config(cfg):
    """Reject omitted experimental branches before any Stage-1 unit begins."""
    if getattr(cfg, 'reachability_enabled', None) is not False:
        raise ValueError('Released Stage 1 requires explicit reachability_enabled=False')
    if getattr(cfg, 'critical_state', None) is not False or getattr(cfg, 'w_c', None) != 0:
        raise ValueError('Released Stage 1 requires critical_state=False and w_c=0')
    if getattr(cfg, 'multitask', None) is not False or getattr(cfg, 'obs', None) != 'state':
        raise ValueError('Released Stage 1 supports single-task state observations only')
    if any((getattr(cfg, flag, None) is not False for flag in ('load_v_checkpoint', 'load_surrogate_agent', 'load_imitate_policy'))):
        raise ValueError('Released Stage 1 does not load experimental auxiliary policies')
    if getattr(cfg, 'load_from_tdmpc2_checkpoint', None) is not True:
        raise ValueError('Released Stage 1 must start from the original TD-MPC2 checkpoint')
    return cfg


def make_env(cfg):
    from .families import make_environment
    return make_environment(cfg)

use_consistency = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
