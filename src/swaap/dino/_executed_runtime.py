"""Executed DINO architecture, HIQL, planning and Stage1 definitions.

Extracted Python definitions, never notebook execution. Public upstream DINO-WM
runtime modules must be installed/configured before this optional module is used.
Unused reward/value training routines were omitted; retained methods are unchanged.
"""
import os,random
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal,Categorical
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import gym,hydra
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange,repeat
from omegaconf import OmegaConf,open_dict
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from env.venv import SubprocVectorEnv
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict,seed,move_to_device,aggregate_dct
GLOBAL_MODEL_CACHE={}
GLOBAL_ENV_CACHE={}
GLOBAL_DATASET_CACHE={}
ALL_MODEL_KEYS=['encoder','predictor','decoder','proprio_encoder','action_encoder']
class HierarchicalGoalDataset:
    """
    Dataset class for Hierarchical Goal-Conditioned RL (HIQL).

    This class takes a flat dataset of transitions and performs the complex on-the-fly
    goal sampling required by the HIQL algorithm. It requires trajectory boundary
    information (the 'terminals' array) to function correctly.
    """
    def __init__(self, transitions: dict, terminals: np.ndarray, config: dict, env, objective_fn=None, z_dicts=None):
        """
        Args:
            transitions (dict): A dictionary of tensors, each of shape (total_transitions, ...).
                                Should contain at least 'observations' and 'actions'.
                                'observations' should be the raw, single-frame observations.
            terminals (np.ndarray): A boolean array of shape (total_transitions,)
                                    marking the last state of each trajectory.
            config (dict): Configuration dictionary with sampling parameters.
        """
        self.transitions = {k: v.cpu().numpy() for k, v in transitions.items()}
        self.config = config
        self.size = len(next(iter(self.transitions.values())))
        self.env = env
        self.objective_fn = objective_fn
        self.z_dicts = z_dicts

        # Pre-compute trajectory boundaries from the terminals array
        (self.terminal_locs,) = np.nonzero(terminals)
        # The first trajectory starts at index 0
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        
        # Ensure the last transition is marked as terminal
        assert self.terminal_locs[-1] == self.size - 1, "The last transition in the dataset must be terminal."

    def __len__(self):
        return self.size

    def _sample_goals(self, idxs: np.ndarray, p_curgoal: float, p_trajgoal: float, geom_sample: bool, final_as_goal=False) -> np.ndarray:
        """Helper function to sample goals based on the HIQL strategy."""
        batch_size = len(idxs)
        
        # Find the final state index for each transition in the batch
        # np.searchsorted finds where each idx would be inserted into terminal_locs to maintain order
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # --- Sample goals from the same trajectory ---
        if geom_sample:
            # Geometric sampling of future states
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling of future states
            # Sample a random offset between the current state (+1) and the final state
            offsets = np.round(np.random.rand(batch_size) * (final_state_idxs - idxs)).astype(int)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)

        # --- Sample random goals from the entire dataset ---
        random_goal_idxs = np.random.randint(self.size, size=batch_size)

        # --- Combine goals based on probabilities ---
        # Decide whether to use trajectory goals or random goals
        p_random = 1.0 - p_curgoal - p_trajgoal
        goal_type = np.random.choice(
            ['traj', 'random'],
            size=batch_size,
            p=[p_trajgoal / (1 - p_curgoal + 1e-8), p_random / (1 - p_curgoal + 1e-8)]
        )
        
        sampled_goal_idxs = np.where(goal_type == 'traj', traj_goal_idxs, random_goal_idxs)
        
        # Decide whether to use the just-sampled goal or the current state as the goal
        is_curgoal = np.random.rand(batch_size) < p_curgoal
        final_goal_idxs = np.where(is_curgoal, idxs, sampled_goal_idxs)

        if final_as_goal:
            final_goal_idxs = final_state_idxs
        
        return final_goal_idxs.clip(0, self.size - 1)

    def sample(self, batch_size: int, have_state=True, have_e_state=False, have_z_dict=False, final_as_goal=False, return_idx=False) -> dict:
        """Sample a batch of transitions and construct all necessary goals for HIQL."""
        # Sample valid transition indices
        idxs = np.random.randint(self.size, size=batch_size)

        # Get terminal index for the trajectory that contains each idx
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        #training phase
        # Get base transitions: (s, a, s')
        states = self.transitions['states'][idxs] if have_state else None
        next_states = self.transitions['next_states'][idxs] if have_state else None
        obs = self.transitions['observations'][idxs]
        next_obs = self.transitions['next_observations'][idxs]
        actions = self.transitions['actions'][idxs]
        # --- Sample Value Goals ---
        value_goal_idxs = self._sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_geom_sample'],
            final_as_goal,
        )
        value_goals = self.transitions['observations'][value_goal_idxs]
        
        # Compute rewards and masks based on value goals
        successes = []
        if have_e_state:
            for i in range(len(idxs)):
                idx = idxs[i]
                value_goal_idx = value_goal_idxs[i]
                e_state_g = torch.tensor(self.transitions['e_states'][value_goal_idx]).unsqueeze(0)
                e_state = torch.tensor(self.transitions['e_states'][idx]).unsqueeze(0)
                #print("e_state_g.shape, e_states[idx].shape:", e_state_g.shape, e_state.shape)
                res = self.env.eval_state(e_state_g, e_state)
                '''
                if idx != value_goal_idx and res['success']:
                    print(res['success'], res['state_dist'])
                    print(self.transitions['observations'][idx].shape, self.transitions['observations'][value_goal_idx].shape)
                    plot_obses([torch.tensor(self.transitions['observations'][idx]), torch.tensor(self.transitions['observations'][value_goal_idx])])
                '''
                successes.append(res['success'])
            successes = np.array([float(s.item()) for s in successes])  # .item() extracts scalar from array/tensor
        elif have_z_dict:
            for i in range(len(idxs)):
                idx = idxs[i]
                value_goal_idx = value_goal_idxs[i]
                #print("len(self.z_dicts), value_goal_idx:", len(self.z_dicts), value_goal_idx)
                z_dict_g = self.z_dicts[value_goal_idx]
                z_dict = self.z_dicts[idx]
                loss = self.objective_fn(z_dict, z_dict_g)
                if loss < 0.05:
                    successes.append(float(1))
                else:
                    successes.append(float(0))
            successes = np.array(successes)
        else:
            successes = (idxs == value_goal_idxs).astype(float)
        #print("sum(successes), successes:", sum(successes), successes)
        rewards = successes - 1.0 # Standard -1 reward for non-goal states
        masks = 1.0 - successes   # Mask is 0 if the goal is reached, 1 otherwise
        #masks = (idxs != final_state_idxs).astype(float)
        
        # --- Set Low-Level Actor Goals ---
        #final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        low_actor_goals = self.transitions['observations'][low_goal_idxs]

        # --- Sample High-Level Actor Goals and Targets ---
        high_goal_idxs = self._sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_geom_sample']
        )
        high_actor_goals = self.transitions['observations'][high_goal_idxs]
        
        # The high-level actor's target is `k` steps towards its goal
        high_actor_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_goal_idxs)
        high_actor_targets = self.transitions['observations'][high_actor_target_idxs]

        if have_state:
            batch = {
                'states': states,
                'next_states': next_states,
                'observations': obs,
                'next_observations': next_obs,
                'actions': actions,
                'rewards': rewards,
                'masks': masks,
                'value_goals': value_goals,
                'low_actor_goals': low_actor_goals,
                'high_actor_goals': high_actor_goals,
                'high_actor_targets': high_actor_targets,
            }
        else:
            batch = {
                'observations': obs,
                'next_observations': next_obs,
                'actions': actions,
                'rewards': rewards,
                'masks': masks,
                'value_goals': value_goals,
                'low_actor_goals': low_actor_goals,
                'high_actor_goals': high_actor_goals,
                'high_actor_targets': high_actor_targets,
            }

        if return_idx:
            batch['idxs'] = idxs
            batch['value_goal_idxs'] = value_goal_idxs
            batch['low_goal_idxs'] = low_goal_idxs
            batch['high_goal_idxs'] = high_goal_idxs
            batch['high_actor_target_idxs'] = high_actor_target_idxs
        
        
        # Convert batch of numpy arrays to a batch of tensors on the correct device
        return {k: torch.from_numpy(v).float() for k, v in batch.items()}

def get_hiql_dataset_config():
    """Returns the default configuration for the HierarchicalGoalDataset."""
    return {
        'discount': 0.99,
        'subgoal_steps': 3,
        # --- Value Function Goal Sampling ---
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3, # Must sum to 1.0 with the above
        'value_geom_sample': False,
        # --- High-Level Actor Goal Sampling ---
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': 1.0,
        'actor_p_randomgoal': 0.0, # Must sum to 1.0 with the above
        'actor_geom_sample': False,
    }

class ResnetStack(nn.Module):
    def __init__(self, in_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding='same')
        self.max_pooling = max_pooling
        
        blocks = []
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding='same')
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.conv_in(x)
        if self.max_pooling:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        for block in self.blocks:
            x = x + block(x) # Residual connection
        return x

class ImpalaEncoder(nn.Module):
    def __init__(self, in_channels, stack_sizes=(16, 32, 32), num_blocks=2, mlp_dims=(512,), image_size=(84, 84), layer_norm=False):
        super(ImpalaEncoder, self).__init__()
        
        stacks = []
        current_channels = in_channels
        for i, features in enumerate(stack_sizes):
            stacks.append(ResnetStack(current_channels, features, num_blocks))
            current_channels = features
        self.stacks = nn.ModuleList(stacks)
        
        # Calculate flattened size for the MLP layer dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            x = dummy_input
            for stack in self.stacks:
                x = stack(x)
            x = F.relu(x)
            flattened_size = x.view(1, -1).shape[1]
        
        self.mlp = MLP([flattened_size] + list(mlp_dims), activate_final=True, layer_norm=layer_norm)

    def forward(self, x):
        x = x.float() / 255.0
        for stack in self.stacks:
            x = stack(x)
        x = F.relu(x)
        #x = x.view(x.shape[0], -1) # Flatten
        x = x.reshape(x.shape[0], -1) # safe for non-contiguous tensor
        x = self.mlp(x)
        return x

class GCEncoder(nn.Module):
    def __init__(self, state_encoder=None, goal_encoder=None, concat_encoder=None):
        super(GCEncoder, self).__init__()
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.concat_encoder = concat_encoder

    def forward(self, observations, goals=None, goal_encoded=False):
        reps = []
        if self.state_encoder:
            reps.append(self.state_encoder(observations))
        
        if goals is not None:
            if goal_encoded:
                assert self.goal_encoder is None and self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder:
                    # Correctly concatenate along channel dim for images
                    if len(observations.shape) > 2:
                        concat_input = torch.cat([observations, goals], dim=1)
                    else:
                        concat_input = torch.cat([observations, goals], dim=-1)
                    reps.append(self.concat_encoder(concat_input))

        return torch.cat(reps, dim=-1)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dims, activate_final=False, layer_norm=False):
        super(MLP, self).__init__()
        layers = []
        # The first dimension in hidden_dims is input_dim
        for i in range(len(hidden_dims) - 1):
            in_dim, out_dim = hidden_dims[i], hidden_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            # No activation or layer norm on the last layer
            if i < len(hidden_dims) - 2:
                layers.append(nn.GELU())
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
        
        # Final activation if specified
        if activate_final:
            layers.append(nn.GELU())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LengthNormalize(nn.Module):
    def __init__(self):
        super(LengthNormalize, self).__init__()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + 1e-6) * (x.shape[-1] ** 0.5)

class GCActor(nn.Module):
    def __init__(self, mlp_dims, action_dim, state_dependent_std=False, const_std=True, gc_encoder=None, heuristic_std=1):
        super(GCActor, self).__init__()
        self.gc_encoder = gc_encoder
        self.actor_net = MLP(mlp_dims, activate_final=True)
        self.mean_net = nn.Linear(mlp_dims[-1], action_dim)
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.heuristic_std = heuristic_std
        if state_dependent_std:
            self.log_std_net = nn.Linear(mlp_dims[-1], action_dim)
        elif not const_std:
            self.log_stds = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded)
        else:
            inputs = torch.cat([observations, goals], dim=-1)

        #print("observations.shape, goals.shape:", observations.shape, goals.shape)
        #print("inputs.shape:", inputs.shape)
        
        x = self.actor_net(inputs)
        mean = self.mean_net(x)

        if self.state_dependent_std:
            log_std = self.log_std_net(x)
        else:
            log_std = torch.full_like(mean, fill_value=np.log(self.heuristic_std)) if self.const_std else self.log_stds
            #log_std = torch.zeros_like(mean) if self.const_std else self.log_stds
        
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std) * temperature
        return Normal(mean, std)

class GCValue(nn.Module):
    def __init__(self, mlp_dims, layer_norm=True, ensemble=True, gc_encoder=None):
        super(GCValue, self).__init__()
        self.ensemble = ensemble
        self.gc_encoder = gc_encoder

        if ensemble:
            self.value_nets = nn.ModuleList([
                MLP(mlp_dims + [1], layer_norm=layer_norm) for _ in range(2)
            ])
        else:
            self.value_net = MLP(mlp_dims + [1], layer_norm=layer_norm)

    def forward(self, observations, goals=None, actions=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
            
        inputs = torch.cat(inputs, dim=-1)

        if self.ensemble:
            return [net(inputs).squeeze(-1) for net in self.value_nets]
        else:
            return self.value_net(inputs).squeeze(-1)

class HIQLAgent(nn.Module):
    def __init__(self, ex_observations, ex_actions, config):
        super(HIQLAgent, self).__init__()
        self.config = config
        self.discrete = config['discrete']

        action_dim = ex_actions.shape[-1]
            
        # Infer dimensions and architecture from example observation
        assert len(ex_observations.shape) > 2, "This agent is for pixel-based observations."
        in_channels = ex_observations.shape[1]
        image_size = (ex_observations.shape[2], ex_observations.shape[3])
        encoder_output_dim = config.get('encoder_output_dim', 512)
        
        # --- Define Goal Representation Network (phi) ---
        # This network processes concatenated (s, g) images
        goal_rep_encoder = ImpalaEncoder(
            in_channels=in_channels * 2, 
            num_blocks=1,
            mlp_dims=[encoder_output_dim], 
            image_size=image_size,
            layer_norm=config['layer_norm']
        )
        goal_rep_mlp = MLP(
            [encoder_output_dim] + list(config['value_hidden_dims']) + [config['rep_dim']],
            layer_norm = True
        )
        self.goal_rep_net = nn.Sequential(goal_rep_encoder, goal_rep_mlp, LengthNormalize())
        #self.goal_rep_net = nn.Sequential(goal_rep_encoder, goal_rep_mlp)

        # debug 1
        # --- Define Value Function Networks ---
        state_encoder = ImpalaEncoder(in_channels=in_channels, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        #goal_state_encoder = ImpalaEncoder(in_channels=in_channels*2, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        value_encoder = GCEncoder(state_encoder=state_encoder, concat_encoder=self.goal_rep_net)
        #value_encoder = GCEncoder(concat_encoder=goal_state_encoder)
        
        # value net V(s,g) = V(encode(s), goal_rep=ɸ(s, g))
        value_mlp_input_dim = encoder_output_dim + config['rep_dim']
        #value_mlp_input_dim = encoder_output_dim
        self.value_net = GCValue(mlp_dims=[value_mlp_input_dim] + list(config['value_hidden_dims']), gc_encoder=value_encoder)

        # Target Value Network (with separate encoder instances)
        target_state_encoder = ImpalaEncoder(in_channels=in_channels, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        #target_goal_state_encoder = ImpalaEncoder(in_channels=in_channels*2, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        target_value_encoder = GCEncoder(state_encoder=target_state_encoder, concat_encoder=self.goal_rep_net)
        #target_value_encoder = GCEncoder(concat_encoder=target_goal_state_encoder)
        self.target_value_net = GCValue(mlp_dims=[value_mlp_input_dim] + list(config['value_hidden_dims']), gc_encoder=target_value_encoder)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        
        # --- Define Actor Networks ---
        # Low-level Actor: π_low(s, g) = π_low(a | encode(s), goal_rep=ɸ(s,g))
        low_actor_state_encoder = ImpalaEncoder(in_channels=in_channels, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        low_actor_encoder = GCEncoder(state_encoder=low_actor_state_encoder)
        low_actor_input_dim = encoder_output_dim + config['rep_dim']
        self.low_actor = GCActor(
            mlp_dims=[low_actor_input_dim] + list(config['actor_hidden_dims']),
            action_dim=action_dim, 
            state_dependent_std=False, 
            const_std=False,
            gc_encoder=low_actor_encoder
        )
        
        # High-level Actor: π_high(goal_rep | encode(s, g))
        high_actor_concat_encoder = ImpalaEncoder(in_channels=in_channels * 2, num_blocks=1, mlp_dims=[encoder_output_dim], image_size=image_size, layer_norm=config['layer_norm'])
        high_actor_encoder = GCEncoder(concat_encoder=high_actor_concat_encoder)
        high_actor_input_dim = encoder_output_dim
        self.high_actor = GCActor(
            mlp_dims=[high_actor_input_dim] + list(config['actor_hidden_dims']),
            action_dim=config['rep_dim'],
            state_dependent_std=False, 
            const_std=False,
            gc_encoder=high_actor_encoder
        )

        # --- Optimizer ---
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

        #print("self.goal_rep_net: \n", self.goal_rep_net, "self.value_net: \n", self.value_net, "self.high_actor: \n", self.high_actor, "self.low_actor: \n", self.low_actor)

    # training phase
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return (weight * (diff**2)).mean()

    def value_loss(self, batch):
        #print("batch['next_observations'].shape, batch['value_goals'].shape:", batch['next_observations'].shape, batch['value_goals'].shape)
        with torch.no_grad():
            next_v1_t, next_v2_t = self.target_value_net(batch['next_observations'], batch['value_goals'])
            next_v_t = torch.min(next_v1_t, next_v2_t)
            q_t = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t
            
            v1_t, v2_t = self.target_value_net(batch['observations'], batch['value_goals'])
            adv = q_t - (v1_t + v2_t) / 2

        #print(batch['observations'].shape, batch['value_goals'].shape)
        v1, v2 = self.value_net(batch['observations'], batch['value_goals'])
        #print("v1", v1[:10])
        #print("v2", v2[:10])
        v = (v1 + v2) / 2
        #print("batch['rewards'].shape, batch['masks'].shape, next_v1_t.shape:", batch['rewards'].shape, batch['masks'].shape, next_v1_t.shape)
        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        #print("q1[:15]:", q1.detach().cpu().numpy().shape, q1[:15].detach().cpu().numpy().round(3))
        #print("v1[:15]:", v1.detach().cpu().numpy().shape, v1[:15].detach().cpu().numpy().round(3))
        #print("q2[:15]:", q2.detach().cpu().numpy().shape, q2[:15].detach().cpu().numpy().round(3))
        #print("v2[:15]:", v2.detach().cpu().numpy().shape, v2[:15].detach().cpu().numpy().round(3))
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile'])
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile'])
        value_loss = value_loss1 + value_loss2 - self.config['rep_coef'] * (v1.std(dim=0).mean() + v2.std(dim=0).mean())
        
        return value_loss, {
            'value_loss': value_loss,
            'v.mean': v.mean(),
            'v.max': v.max(),
            'v.min': v.min(),
            'v.std': v.std(),
            'value_loss1.mean': value_loss1.mean(),
            'value_loss2.mean': value_loss2.mean(),
        }

    def low_actor_loss(self, batch):
        with torch.no_grad():
            v1, v2 = self.value_net(batch['observations'], batch['low_actor_goals'])
            nv1, nv2 = self.value_net(batch['next_observations'], batch['low_actor_goals'])
            adv = ((nv1 + nv2) / 2) - ((v1 + v2) / 2)
            exp_a = torch.exp(adv * self.config['low_alpha']).clamp(max=100.0)
            
        goal_reps = self.goal_rep_net(torch.cat([batch['observations'], batch['low_actor_goals']], dim=1))
        #goal_reps = self.goal_rep_net(batch['low_actor_goals'])
        #print("goal_reps.mean(), goal_reps.max(), goal_reps.min(), goal_reps.std():", goal_reps.mean(), goal_reps.max(), goal_reps.min(), goal_reps.std())
        #print("batch['low_actor_goals'].shape:", batch['low_actor_goals'].shape)
        if self.config['low_actor_rep_grad'] == False:
            goal_reps = goal_reps.detach()
            
        dist = self.low_actor(batch['observations'], goal_reps, goal_encoded=True)
        log_prob = dist.log_prob(batch['actions']).sum(-1)
        actor_loss = -(exp_a * log_prob).mean()

        #print("dist.mean.shape, batch['actions'].shape:", dist.mean.shape, batch['actions'].shape)
        x = batch['actions']
        if self.config['over_all_element']:
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'std': dist.scale.mean(),
                'mse': torch.mean((dist.mean - x) ** 2),
                'mae': torch.mean(torch.abs(dist.mean - x)),
                'rep_std_mean': goal_reps.std(dim=0).mean(),
                'x.mean': x.mean(),
                'x.max': x.max(),
                'x.min': x.min(),
                'x.std': x.std(dim=0).mean(),
            }
        else:
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'std': dist.scale.mean(),
                'mse': torch.mean((dist.mean - x) ** 2),
                'mae': torch.mean(torch.abs(dist.mean - x)),
                'rep_std_mean': goal_reps.std(dim=0).mean(),
                'x.mean': x.mean(),
                'x.max': x.max(),
                'x.min': x.min(),
                'x.std': x.std(dim=0).mean(),
            }
        return actor_loss, actor_info

    def high_actor_loss(self, batch):
        with torch.no_grad():
            v1, v2 = self.value_net(batch['observations'], batch['high_actor_goals'])
            nv1, nv2 = self.value_net(batch['high_actor_targets'], batch['high_actor_goals'])
            adv = ((nv1 + nv2) / 2) - ((v1 + v2) / 2)
            exp_a = torch.exp(adv * self.config['high_alpha']).clamp(max=100.0)
            target_reps = self.goal_rep_net(torch.cat([batch['observations'], batch['high_actor_targets']], dim=1))
            #target_reps = self.goal_rep_net(batch['high_actor_targets'])
            
        dist = self.high_actor(batch['observations'], batch['high_actor_goals'])
        log_prob = dist.log_prob(target_reps).sum(-1)
        actor_loss = -(exp_a * log_prob).mean()

        #print("dist.mean.shape, target_reps.shape:", dist.mean.shape, target_reps.shape)
        x = target_reps
        if self.config['over_all_element']:
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'std': dist.scale.mean(),
                'mse': torch.mean((dist.mean - x) ** 2),
                'mae': torch.mean(torch.abs(dist.mean - x)),
                'rep_std_mean': x.std(dim=0).mean(),
                'x.mean': x.mean(),
                'x.max': x.max(),
                'x.min': x.min(),
                'x.std': x.std(dim=0).mean(),
            }
        else:
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'std': dist.scale.mean(),
                'mse': torch.mean((dist.mean - x) ** 2),
                'mae': torch.mean(torch.abs(dist.mean - x)),
                'rep_std_mean': x.std(dim=0).mean(),
                'x.mean': x.mean(),
                'x.max': x.max(),
                'x.min': x.min(),
                'x.std': x.std(dim=0).mean(),
            }
        return actor_loss, actor_info

    def update(self, batch):
        value_loss, value_info = self.value_loss(batch)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch)
        total_loss = value_loss + low_actor_loss + high_actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Target network soft update
        for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)

        info = {}
        info['value'] = value_info
        info['low_actor'] = low_actor_info
        info['high_actor'] = high_actor_info
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'low_actor_loss': low_actor_loss.item(),
            'high_actor_loss': high_actor_loss.item(),
        }, info

    @torch.no_grad()
    def sample_actions(self, observations, goals, temperature=1.0):
        high_dist = self.high_actor(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample()
        
        low_dist = self.low_actor(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample()
        return actions

def get_config():
    return {
        'lr': 3e-4,
        'actor_hidden_dims': (512, 512, 512),
        'value_hidden_dims': (512, 512, 512),
        'encoder_output_dim': 512,
        'layer_norm': True,
        'discount': 0.99,
        'tau': 0.05,
        'expectile': 0.7,
        'low_alpha': 3.0,
        'high_alpha': 3.0,
        'rep_dim': 10,
        'rep_coef': 0,
        'low_actor_rep_grad': True,
        'discrete': False,
    }

def plot_obses(obses, normalize=False):
    n = len(obses)
    plt.figure(figsize=(3 * n, 3))

    # obsese list of tensor (3, 64, 64)
    for i in range(n):
        img = obses[i].permute(1, 2, 0)
        if normalize:
            img = img / 255.0  # Normalize to [0, 1]
        
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"observation {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def get_cached_model(model_ckpt, model_cfg, num_action_repeat, device):
    key = str(model_ckpt.resolve())
    if key not in GLOBAL_MODEL_CACHE:
        print(f"Loading model from checkpoint: {model_ckpt}")
        GLOBAL_MODEL_CACHE[key] = load_model(model_ckpt, model_cfg, num_action_repeat, device)
    else:
        print(f"Using cached model for: {model_ckpt}")
    return GLOBAL_MODEL_CACHE[key]

def get_cached_env(env_name, n_evals, args, kwargs, use_serial):
    key = (env_name, n_evals, str(args), str(kwargs), use_serial)
    if key not in GLOBAL_ENV_CACHE:
        print(f"Creating new env: {env_name} x{n_evals}")
        if use_serial:
            from env.serial_vector_env import SerialVectorEnv
            #env = SerialVectorEnv([gym.make(env_name, *args, **kwargs) for _ in range(n_evals)])
            env = SerialVectorEnv([gym.make(env_name, disable_env_checker=True, *args, **kwargs) for _ in range(n_evals)])
        else:
            #env = SubprocVectorEnv([lambda: gym.make(env_name, *args, **kwargs) for _ in range(n_evals)])
            env = SubprocVectorEnv([lambda: ObsGoalDictWrapper(gym.make(env_name, disable_env_checker=True, *args, **kwargs)) for _ in range(n_evals)])
        GLOBAL_ENV_CACHE[key] = env
    else:
        print(f"Using cached env: {env_name} x{n_evals}")
    return GLOBAL_ENV_CACHE[key]

def get_cached_dataset(dataset_cfg, num_hist, num_pred, frameskip):
    # Create a hashable key from relevant config
    key = (
        str(dataset_cfg),
        num_hist,
        num_pred,
        frameskip,
    )
    if key not in GLOBAL_DATASET_CACHE:
        print("Loading dataset...")
        _, dset = hydra.utils.call(
            dataset_cfg,
            num_hist=num_hist,
            num_pred=num_pred,
            frameskip=frameskip,
        )
        dset = dset["valid"]
        GLOBAL_DATASET_CACHE[key] = dset
    else:
        print("Using cached dataset")
    return GLOBAL_DATASET_CACHE[key]

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result

class SimulationWorkspace:
    def __init__(self, cfg_dict, wm, dset, env, env_name, frameskip, wandb_run, agent, dtype=None):
        self.cfg_dict = cfg_dict
        self.cfg = cfg_dict
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device
        self.dtype = dtype
        
        self.seed = self.cfg_dict["seed"]
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        self.objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        self.preprocessor = self.data_preprocessor

        self.wm = wm
        self.agent = agent
        self.wm.to(self.device)

        # Create separate optimizers for each component
        '''
        self.agent.to(self.device)
        self.theta_optimizer = optim.Adam(self.agent.theta.parameters(), lr=self.cfg_dict["lr"])
        self.optimizer = torch.optim.Adam([
			{'params': self.agent._encoder.parameters(), 'lr': self.cfg_dict["lr"]*self.cfg_dict["enc_lr_scale"]},
			{'params': self.agent._reward.parameters()},
			{'params': self.agent._Qs.parameters()},
			{'params': self.agent._Vs.parameters()}
		], lr=self.cfg_dict["lr"], capturable=True)
        '''

        self.curriculum_loss = 3

    def refresh_targets(self):
        """Refresh obs_0, obs_g, state_0, state_g, gt_actions."""
        # have different seeds for each planning instances
        if not self.cfg.fix_seed:
            self.seed = random.randint(0, 10**6)
        self.eval_seed = [self.seed * (n + 1) for n in range(self.n_evals)]
        #print("seed: ", self.eval_seed)
        if self.goal_source == "file":
            self.prepare_targets_from_file(self.cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    # marker
    def interact_mpc(self, obs_0, obs_g, state_0, state_g, return_success_only=False, compare_wm_rollouts=False):
        self.evaluator = PlanEvaluator(
            obs_0=obs_0,
            obs_g=obs_g,
            state_0=state_0,
            state_g=state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )
        
        self.log_filename = "logs.json" 
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )
        
        actions_init = None
        actions, action_len, success_agent = self.planner.plan(
            obs_0=obs_0,
            obs_g=obs_g,
            actions=actions_init,
        )

        logs, success, e_obses_dict, e_states, i_z_obses_dict = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=False, filename="output_final", return_i_z_obses=True
        )
        
        # Public logging adapter: retain already-computed values, no extra rollout.
        self.last_evaluation = (logs, success, e_obses_dict, e_states, i_z_obses_dict)

        # frameskip parameters = (env rollout length - 1)/(imaginary rollout length - 1)
        frameskip = 5
        e_obses = e_obses_dict['visual'] 
        env_rollout_length = e_obses.shape[1]
        # Select every 5th step starting at 0
        indices = torch.arange(0, env_rollout_length-1, frameskip)  # last index excluded here (25 not included)
        # Append the last index, the final obs
        indices = torch.cat([indices, torch.tensor([env_rollout_length-1])])
        obs = torch.tensor(e_obses[:, indices, :, :, :])
        #print("e_obses['visual'].shape:", e_obses.shape) # (1, T*frameskip+1, 224, 224, 3)
        #print("obses.shape:", obs.shape)  # (1, T+1, 224, 224, 3)
        obs = obs.squeeze(0)                          
        obs = obs.permute(0, 3, 1, 2).unsqueeze(0)
        #print("obses.shape:", obs.shape)
        T, B, C = obs.shape[0], obs.shape[1], obs.shape[2]
        obs_reshaped = obs.view(-1, *obs.shape[2:]) # (T, B, C, H, W) → (T×B, C, H, W)
        obs_resized = F.interpolate(obs_reshaped, size=(64, 64), mode='bilinear', align_corners=False) # (T*B, C, 64, 64)
        obses = obs_resized.view(T, B, C, 64, 64) # (T, B, C, 64, 64)
        #print("resized obses shape:", obses.shape)

        if return_success_only or compare_wm_rollouts:
            info = TensorDict({
                "is_success": success,
            })
        else:
            info = TensorDict({
                "obs_0": obs_0,
                "obs_g": obs_g,
                "state_0": state_0,
                "state_g": state_g,
                "seed": self.eval_seed,
                "e_obses": obses,
                "actions": actions,
                "action_len": action_len,
                "is_success": success,
            })
        if compare_wm_rollouts:
            return info, e_obses_dict, i_z_obses_dict
        else:
            return info

    def merge_obs_visual_propriception(self, z_dct):
        proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, 1)
        z = torch.cat([z_dct['visual'], proprio_repeated], dim=3) 
        return z

    def form_z_wm(self, z_dct, act_emb):
        if act_emb.ndim == 2:  # [b, action_dim]
            act_emb = act_emb.unsqueeze(1)  # [b, 1, a]
    
        proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, 1) # self.num_proprio_repeat = 1
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
        act_repeated = act_tiled.repeat(1, 1, 1, 1) # self.num_action_repeat = 1
        z = torch.cat(
            [z_dct['visual'], proprio_repeated, act_repeated], dim=3
        )  # (b, num_frames, num_patches, dim + action_dim)
        return z
        
    def wm_next_interact(self, z, action):
        z_pred = self.wm.predict(z[:, -self.cfg.num_hist :]) # shape: (b, n, num_patches, emb_dim), self.num_hist = 3
        z_new = z_pred[:, -1:, ...]
        z_new = self.wm.replace_actions_from_z(z_new, action)
        z_cur_dict, _ = self.wm.separate_emb(z_new)
        return z_cur_dict

    def wm_decode(self, z_cur_dict, return_original=False):
        obs, _ = self.wm.decode_obs(z_cur_dict)
        obs = obs['visual'] # (T, B, C, H, W)
        T, B, C = obs.shape[0], obs.shape[1], obs.shape[2]
        obs_reshaped = obs.view(-1, *obs.shape[2:]) # (T, B, C, H, W) → (T×B, C, H, W)
        obs_resized = F.interpolate(obs_reshaped, size=(64, 64), mode='bilinear', align_corners=False) # (T*B, C, 64, 64)
        obs_resized = obs_resized.view(T, B, C, 64, 64) # (T, B, C, 64, 64)
        if not return_original:
            return obs_resized
        else:
            return obs_resized, obs
        
    @torch.no_grad()
    def interact(self, obs_0, obs_g):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)

        #print(obs_0['visual'].shape, obs_0['proprio'].shape)
        obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_cur_dict = self.wm.encode_obs(obs_0) # z_cur_dict["visual"] shape: (b, 1, 196, 384), z_cur_dict["proprioception"] shape: (b, 1, 1, 10)
        z_g_dict = self.wm.encode_obs(obs_g)

        obs = self.wm_decode(z_cur_dict) # (T, B, C, 64, 64)
        obs_g, obs_g_ori = self.wm_decode(z_g_dict, return_original=True) # (T, B, C, 64, 64)
        z_g_agent = self.batch_encode(obs_g) # (T, B, D), D here is agent latent dimension, which is often 512
        
        z_obses = []
        states = []
        actions = []
        rewards = []
        next_states = []

        states = []
        actions = []
        rewards = []
        next_states = []

        z_next_dict_list = [] 
        # interact with simulated world for max_iter steps
        i = 0
        while not np.all(self.is_success) and i < self.cfg_dict["max_iter"]:
            z_cur_agent = self.batch_encode(obs) # (T, B, D)
            #print("z_cur_agent.shape:", z_cur_agent.shape)
            z_agent = torch.cat([z_cur_agent, z_g_agent], dim=-1)  # shape: (T, B, 2*D)
            action, _ = self.agent.pi(z_agent) # shape: (b, 1, action_dim)
            
            if i == 0:
                z = self.form_z_wm(z_cur_dict, action)
            else:
                z_new = self.form_z_wm(z_cur_dict, action)
                z = torch.cat([z, z_new], dim=1)
                z = z[:, -self.cfg.num_hist:, ...] # only past self.num_hist info

            z_next_dict = self.wm_next_interact(z)
            next_obs, next_obs_ori = self.wm_decode(z_next_dict, return_original=True)
            loss = self.objective_fn(z_next_dict, z_g_dict)
            reward = torch.where(loss > self.curriculum_loss, torch.tensor(-1.0, device=loss.device), torch.tensor(0.0, device=loss.device))

            j = 0
            if i == 10 and j==0: 
                print("next_obs_ori.shape, obs_g_ori.shape:", next_obs_ori.shape, obs_g_ori.shape)
                print("next_obs.shape, obs_g.shape:", next_obs.shape, obs_g.shape)
                plot_current_and_final(next_obs_ori.squeeze(0).squeeze(0).cpu(), obs_g_ori.squeeze(0).squeeze(0).cpu())
                plot_current_and_final(next_obs.squeeze(0).squeeze(0).cpu(), obs_g.squeeze(0).squeeze(0).cpu())
                j=1

            states.append(obs)
            actions.append(action.squeeze(1))
            rewards.append(reward.unsqueeze(-1))
            next_states.append(next_obs)

            z_next_dict_list.append(z_next_dict)
            
            z_cur_dict = z_next_dict
            obs = next_obs
            i += 1

        states = torch.stack(states, dim=0).squeeze(2)         # shape: (T, B, C, 64, 64)
        actions = torch.stack(actions, dim=0)                  # shape: (T, B, A)
        rewards = torch.stack(rewards, dim=0)    # shape: (T, B, 1)
        next_states = torch.stack(next_states, dim=0).squeeze(2)  # shape: (T, B, P, 64, 64)
        self.curriculum_loss = 0.9 * self.curriculum_loss + 0.1 * loss.mean().item()
        print(f"Curriculum loss: {self.curriculum_loss:.4f}, Last reward mean: {rewards[-1].mean().item():.4f}")
        #print(states.shape, actions.shape, rewards.shape, next_states.shape)

        z_final_dict = z_next_dict
        obs_final = obs
        return states, actions, rewards, next_states, obs_g.squeeze(1).unsqueeze(0), obs_final.squeeze(1).unsqueeze(0), z_g_dict, z_final_dict, z_next_dict_list

    def wm_rollout(self, obs_0, obs_g, actions):
        #print(obs_0['visual'].shape, obs_0['proprio'].shape)
        obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        act_0 = actions[:, 0, :].unsqueeze(0)
        z = self.wm.encode(obs_0, act_0)

        t = 0
        inc = 1
        while t < (actions.shape[1]):
            z_pred = self.wm.predict(z[:, -self.cfg.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.wm.replace_actions_from_z(z_new, actions[:, t : t + inc, :])
            print("actions[:, t : t + inc, :].shape:", actions[:, t : t + inc, :].shape)
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.wm.predict(z[:, -self.cfg.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.wm.separate_emb(z)
        return z_obses, z

    
    def resize_obs(self, obs):
        obs = obs['visual'] # (T, B, C, H, W)
        T, B, C = obs.shape[0], obs.shape[1], obs.shape[2]
        obs_reshaped = obs.view(-1, *obs.shape[2:]) # (T, B, C, H, W) → (T×B, C, H, W)
        obs_resized = F.interpolate(obs_reshaped, size=(64, 64), mode='bilinear', align_corners=False) # (T*B, C, 64, 64)
        obs_resized = obs_resized.view(T, B, C, 64, 64) # (T, B, C, 64, 64)
        return obs_resized.squeeze(0)

    def wm_next(self, wm, state, action):
        z = state[:, :-1, ...]
        z_new = state[:, -1, ...].unsqueeze(1)
        #print("state.shape, z.shape, z_new.shape:", state.shape, z.shape, z_new.shape)
        z_new = wm.replace_actions_from_z(z_new, action)
        z = torch.cat([z, z_new], dim=1)
        z_pred = wm.predict(z[:, -self.cfg.num_hist :])
        #print(z_pred.shape, self.cfg.num_hist )
        z_new = z_pred[:, -1:, ...]
        z_cur_dict, _ = wm.separate_emb(z_new)
        obs_dict, _ = wm.decode_obs(z_cur_dict)
        obs = self.resize_obs(obs_dict)

        state = torch.cat([z, z_new], dim=1)
        return obs, state, z_cur_dict

    @torch.no_grad()
    def interact_wm(self, obs_0, obs_g, actor, wm, state_0=None, planned_actions=None, use_agent_action=False, max_step=5, env=None, return_z_dicts=False, return_good_data=False):
        #print(obs_0['visual'].shape, obs_0['proprio'].shape)
        obs_0_dict = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        obs_g_dict = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )

        #print("obs_0['visual'].shape, obs_g['visual'].shape:", obs_0_dict['visual'].shape, obs_g_dict['visual'].shape)
        obs = self.resize_obs(obs_0_dict)
        obs_0 = obs
        obs_g = self.resize_obs(obs_g_dict)
        obs_g_raw = self.reverse_transform(obs_g)
        #print("obs_0_r.shape, obs_g_r.shape:", obs.shape, obs_g.shape)
        if use_agent_action:
            obs_raw = self.reverse_transform(obs)
            #print("obs_raw.squeeze(1), obs_g_raw.squeeze(1):", obs_raw.squeeze(1).shape, obs_g_raw.squeeze(1).shape)
            action = actor.sample_actions(obs_raw.squeeze(1), obs_g_raw.squeeze(1)).unsqueeze(1)
        else:
            action = planned_actions[:, 0].unsqueeze(1)
        #print("action_0.shape:", action.shape)

        #print("obs_0_dict['visual'].shape, action.shape:", obs_0_dict['visual'].shape, action.shape)
        #print("action.shape:", action.shape)
        if not return_good_data:
            if state_0 is None:
                z = wm.encode(obs_0_dict, action)
                print("z.shape, action.shape:", z.shape, action.shape)
                z_pred = wm.predict(z[:, -self.cfg.num_hist :])
                z_new = z_pred[:, -1:, ...]
                #print("z_pred.shape, z_new.shape:", z_pred.shape, z_new.shape)
                z_cur_dict, _ = wm.separate_emb(z_new)
                obs_dict, _ = wm.decode_obs(z_cur_dict)
                #print("obs_dict['visual'].shape, obs_dict['proprio'].shape:", obs_dict['visual'].shape, obs_dict['proprio'].shape)
                obs = self.resize_obs(obs_dict)
                z_g_dict = wm.encode_obs(obs_g_dict)
                state = torch.cat([z, z_new], dim=1)
                print("state.shape:", state.shape)
            else:
                state = state_0
        else:
            z = wm.encode(obs_0_dict, action)
            state = z
            
        #print("state.shape:", state.shape)
        
        obses_to_plot = [self.reverse_transform(obs_0.squeeze(0).cpu().detach())]

        t = 0
        inc = 1
        if not return_good_data:
            # max_step+1 to make sliding window fully filled, so all state data from interact_wm have the same shape, i.e. window size (3, ...)
            max_iter = (max_step+1) if use_agent_action else planned_actions.shape[1] 
        else:
            max_iter = max_step if use_agent_action else planned_actions.shape[1] 
        states, next_states, observations, next_observations, actions = [], [], [], [], []   
        next_states_0, next_observations_0 = [], []
        z_dicts = []
        while (t < (max_iter)):
            if use_agent_action:
                obs_raw = self.reverse_transform(obs)
                action = actor.sample_actions(obs_raw.squeeze(1), obs_g_raw.squeeze(1)).unsqueeze(1)
            else:
                obs_raw = self.reverse_transform(obs)
                action = planned_actions[:, t].unsqueeze(1)
            #print("state.shape, action.shape:", state.shape, action.shape)
            if not return_good_data:
                if state.shape[1] >= self.cfg.num_hist:
                    next_obs, next_state, z_next_dict = self.wm_next(wm, state, action) 
                else:
                    # filling the initial window by unperturbed states
                    next_obs, next_state, z_next_dict = self.wm_next(self.wm, state, action) 
            else:
                next_obs, next_state, z_next_dict = self.wm_next(wm, state, action) 

            #print("z.shape, z_new.shape:", z.shape, z_new.shape)
            #print("state.shape, next_state.shape:", state.shape, next_state.shape)

            # marker
            if not return_good_data:
                if state.shape[1] >= self.cfg.num_hist:
                    states.append(state[:, -self.cfg.num_hist :].unsqueeze(1))
                    next_states.append(next_state[:, -self.cfg.num_hist :].unsqueeze(1))
                    if self.cfg.n_evals == 1:
                        observations.append(obs_raw.unsqueeze(1))
                        next_observations.append(self.reverse_transform(next_obs).unsqueeze(1))
                    else:
                        observations.append(obs_raw)
                        next_observations.append(self.reverse_transform(next_obs))
                        #print("z_cur_dict['visual'].shape, z_cur_dict['proprio'].shape:", z_cur_dict['visual'].shape, z_cur_dict['proprio'].shape)
                    if return_z_dicts:
                        z_dicts.append(z_cur_dict)
                    actions.append(action)
            else:
                if self.cfg.n_evals == 1:
                    observations.append(obs_raw.unsqueeze(1))
                    next_observations.append(self.reverse_transform(next_obs).unsqueeze(1))
                else:
                    observations.append(obs_raw)
                    next_observations.append(self.reverse_transform(next_obs))
                actions.append(action)
            
            obs, state = next_obs, next_state
            z_cur_dict = z_next_dict
            
            obses_to_plot.append(self.reverse_transform(obs.squeeze(0).cpu().detach()))

            t += inc

        if return_good_data:
            transition_dict = {
                'states': state[:, 1:].cpu(),
                'next_states': next_state[:, 1:].cpu(),
                #'observations': torch.cat(observations, dim=1),
                #'next_observations': torch.cat(next_observations, dim=1),
                'actions': torch.cat(actions, dim=1).cpu(),
            }
            return (transition_dict, None, None)
            
        transition_dict = {
            'states': torch.cat(states, dim=1),
            'next_states': torch.cat(next_states, dim=1),
            'observations': torch.cat(observations, dim=1),
            'next_observations': torch.cat(next_observations, dim=1),
            'actions': torch.cat(actions, dim=1),
        }

        #print("shapes:", transition_dict['states'].shape, transition_dict['next_states'].shape, transition_dict['observations'].shape, transition_dict['next_observations'].shape, transition_dict['actions'].shape)
        obses_to_plot.append(self.reverse_transform(obs_g.squeeze(0).cpu().detach()))
        # convert batched z_dicts to sequenced z_dict
        if self.cfg.n_evals != 1 and return_z_dicts:
            total_length = self.cfg.n_evals * len(z_dicts)
            z_dicts_new = []
            for j in range(self.cfg.n_evals):
                for i in range(len(z_dicts)):
                    z_dict = {
                        'visual': z_dicts[i]['visual'][j].unsqueeze(0),
                        'proprio': z_dicts[i]['proprio'][j].unsqueeze(0),
                    }
                    z_dicts_new.append(z_dict)
                    #print("z_cur_dict['visual'].shape, z_cur_dict['proprio'].shape:", z_dict['visual'].shape, z_dict['proprio'].shape)
            z_dicts = z_dicts_new

        transition_dict = {k: v.cpu() for k, v in transition_dict.items()}
        if return_z_dicts:
            return (transition_dict, z_dicts, z_g_dict)
        else:
            return transition_dict, obses_to_plot

    def reverse_transform(self, img):
        # Assume: state is (N, 3, H, W) and in [-1, 1]
        # Step 1: Unnormalize from [-1, 1] → [0, 1]
        img_raw = (img * 0.5) + 0.5
        
        # Step 2: Convert to uint8 in [0, 255]
        img_raw = (img_raw * 255.0).clamp(0, 255)  # (N, 3, H, W)

        return img_raw

    def generate_data(self, obs_0, obses, obs_g, actions, is_success, subgoal_steps=1, use_env_obses=True, her=True):
        """
        Generate HIQL-compatible rollout transitions with general hindsight relabeling.
        
        Args:
            obs_0 (dict): initial obs of shape (b, n, 3, H, W)
            obs_g (dict): goal obs (single image)
            actions (Tensor): (b, t+n, action_dim)
            is_success (bool): whether final goal succeeded
            subgoal_steps (int): how far future subgoal is
    
        Returns:
            dict of tensors, each with shape (num_transitions, ...)
        """
        obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        #print("obs_0.shape:", obs_0['visual'].shape)
        #print("obses.shape:", obses.shape)

        if not use_env_obses:
            with torch.no_grad():
                z_obses, z = self.wm.rollout(obs_0, actions)
                obses = self.wm_decode(z_obses, return_original=False)  # (1, T+1, 3, H, W)
                #print("wm obses.shape:", obses.shape)
                
        # Accumulate transitions
        states, actions_list, rewards, next_states, subgoals, goals = [], [], [], [], [], []

        # True transitions
        obs_g = obs_g['visual'].to(self.device)
        obs_g = obs_g.squeeze(0).squeeze(0).permute(2, 0, 1).unsqueeze(0).float()
        obs_g_resized = F.interpolate(obs_g, size=(64, 64), mode='bilinear', align_corners=False)
        goal_state = obs_g_resized.squeeze(0)
        #print("actions.shape, obses.shape:", actions.shape, obses.shape)
        length = obses.shape[1] - 1
        for i in range(length):
            obs = obses[:, i]
            next_obs = obses[:, i+1]
            #plot_current_and_final(obs.detach().squeeze(0).cpu(), permute=True)
            state = obs.squeeze(0)
            action = actions[:, i]
            reward = torch.tensor([0] if (i == (length - 1) and is_success) else [-1], dtype=torch.float32)
            next_state = next_obs.squeeze(0)
            subgoal_idx = min(i+subgoal_steps, length)
            subgoal_state = obses[:, subgoal_idx].squeeze(0)
            #print(state.shape, action.shape, reward.shape, next_state.shape, goal_state.shape)
            #print(length, i, reward)
            states.append(state)
            actions_list.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            subgoals.append(subgoal_state)
            goals.append(goal_state)

        # Convert to tensors
        transition_dict = {
            'states': torch.stack([x.cpu() for x in states]),
            'actions': torch.stack([x.cpu() for x in actions_list]),
            'rewards': torch.stack([x.cpu() for x in rewards]),
            'next_states': torch.stack([x.cpu() for x in next_states]),
            'subgoals': torch.stack([x.cpu() for x in subgoals]),
            'goals': torch.stack([x.cpu() for x in goals]),
        }
    
        return transition_dict, len(states)

    
    
    

        

        

def resize_obs(obs):
    if isinstance(obs, dict) and 'visual' in obs:
        obs_tensor = torch.tensor(obs['visual'])
    else:
        obs_tensor = torch.tensor(obs)
    while obs_tensor.ndim > 3:
        obs_tensor = obs_tensor.squeeze(0)
        
    if isinstance(obs, dict) and 'visual' in obs:
        obs_resized = F.interpolate(obs_tensor.permute(2, 0, 1).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
    if isinstance(obs, torch.Tensor): # obs: torch.size(224, 224, 3)
        obs_resized = F.interpolate(obs_tensor.permute(2, 0, 1).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
    if isinstance(obs, np.ndarray): # obs: torch.size(224, 224, 3)
        obs_tensor = torch.tensor(obs)
        obs_resized = F.interpolate(obs_tensor.permute(2, 0, 1).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
    return obs_resized

def print_cuda_stats(ignore=False):
    if ignore == False:
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3  # in GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
        
        print(f"CUDA memory - Free: {free_mem:.2f} GB / Total: {total_mem:.2f} GB")

def batched_forward(net, x, y, batch_size):
    outs1 = []
    outs2 = []
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        out1, out2 = net(x_batch, y_batch)
        #print(i)
        #print("out1.shape, out2.shape:", out1.shape, out2.shape)
        outs1.append(out1)
        outs2.append(out2)
    print_cuda_stats()
    torch.cuda.empty_cache()
    return torch.cat(outs1), torch.cat(outs2)

def apply_flat_grads_to_model(parameters, flat_grads):
    """
    Assigns a flattened gradient vector to a model's parameters.
    
    Args:
        parameters (iterable): The model parameters (e.g., model.parameters()).
        flat_grads (Tensor): A single flat tensor of gradients.
    """
    offset = 0
    for param in parameters:
        if param.requires_grad:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
    
            numel = param.numel()
            # Get the slice for the current parameter
            grad_slice = flat_grads[offset : offset + numel]
            
            # Reshape and assign to the .grad attribute
            param.grad.copy_(grad_slice.view_as(param))
            offset += numel

class ThetaPsi(torch.nn.Module):
    """
    algorithm for dino_wm
    world model part built upon framework by gaoyuezhou https://github.com/gaoyuezhou/dino_wm
    agent part adapted from framework by seohongpark https://github.com/seohongpark/ogbench
    """
    
    def __init__(self, cfg, env, agent, workspace):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.workspace = workspace
        self.objective_fn = hydra.utils.call(cfg["objective"])

        # instantiate models
        self.psi_0 = deepcopy(workspace.wm).to(self.device)
        self.psi = deepcopy(workspace.wm).to(self.device)
        self.theta_0 = deepcopy(agent).to(self.device)
        self.theta = deepcopy(agent).to(self.device)
        self.theta_star = deepcopy(agent).to(self.device)

        # Freeze all parameters in the trainable world model by default
        for param in self.psi.parameters():
            param.requires_grad = False
        self.psi.eval()
        # Then, unfreeze only the parameters of the predictor head
        for param in self.psi.predictor.parameters():
            param.requires_grad = True
            
        for param in self.psi_0.parameters():
            param.requires_grad = False
        for param in self.theta_0.parameters():
            param.requires_grad = False
        self.psi_0.eval()
        self.theta_0.eval()

        # optimizers
        self.psi_optimizer = torch.optim.AdamW(self.psi.predictor.parameters(), lr=5e-4) # from dino_wm 

        print_cuda_stats()
        
    def create_hiql_dataset(self, batched_data, subgoal_steps=3, objective_fn=None, z_dicts=None):
        data_dict = {}
        for k in batched_data:
            v = batched_data[k]  
            data_dict[k] = v.reshape(-1, *v.shape[2:])
        
        B, T = batched_data['states'].shape[:2]  
        terminals = torch.zeros(B * T)
        terminals[torch.arange(T - 1, B * T, T)] = 1
        terminals = terminals.cpu().numpy()
    
        dataset_config = get_hiql_dataset_config()
        dataset_config['subgoal_steps'] = subgoal_steps
        
        hiql_dataset = HierarchicalGoalDataset(
            transitions=data_dict,
            terminals=terminals,
            config=dataset_config,
            env=self.env,
            objective_fn=objective_fn,
            z_dicts=z_dicts,
        )
        return hiql_dataset
    
    # def refresh_targets
    # def interact_wm
    @torch.no_grad()
    def collect(self, wm=None, agent=None, max_step=8, num=1, sample_batch_size=64, return_dataset=False, return_batch_dataset=False):
        if not wm:
            wm = self.psi_0
        if not agent:
            agent = self.theta_0
            
        # set appropriate agent std for collecting experience
        const_std = True
        heuristic_std = 0.3
        
        agent.low_actor.const_std = const_std
        agent.high_actor.const_std = const_std
        
        agent.low_actor.heuristic_std = heuristic_std
        agent.high_actor.heuristic_std = heuristic_std
        
        # for num many times, each time collect n_evals(=16) trajectories each with max_iter(=10), and sample sample_batch_size(=16) times
        # sample_batch_size=16 as my pc only support this, the wm_next computation is storage intensive
        all_batched_data = None
        for i in range(num):
            #print(f"collect {i}th batch")
            self.workspace.refresh_targets()
            obs_0_dict, obs_g_dict, state_0, state_g, eval_seed = self.workspace.obs_0, self.workspace.obs_g, self.workspace.state_0, self.workspace.state_g, self.workspace.eval_seed
            env = self.workspace.env
            frameskip = self.workspace.frameskip
        
            (batched_data, z_dicts, _) = self.workspace.interact_wm(obs_0_dict, obs_g_dict, actor=agent, wm=wm, use_agent_action=True, max_step=max_step, env=env, return_z_dicts=True)
            if not all_batched_data:
                all_batched_data = batched_data
                all_z_dicts = z_dicts
            else:
                all_batched_data = {k: torch.cat([all_batched_data[k], batched_data[k]], dim=0) for k in batched_data}
                all_z_dicts += z_dicts

            #print(i, all_batched_data['observations'].shape)

        hiql_dataset = self.create_hiql_dataset(all_batched_data, objective_fn=self.objective_fn, z_dicts=all_z_dicts)
        #batch = hiql_dataset.sample(sample_batch_size, have_z_dict=True, return_idx=True)
        if return_dataset == False or return_batch_dataset == True:
            batch = hiql_dataset.sample(sample_batch_size, have_z_dict=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}

        # set appropriate agent std for update
        const_std = True
        heuristic_std = 1
        
        agent.low_actor.const_std = const_std
        agent.high_actor.const_std = const_std
        
        agent.low_actor.heuristic_std = heuristic_std
        agent.high_actor.heuristic_std = heuristic_std

        if return_batch_dataset == True:
            return batch, hiql_dataset
        if return_dataset:
            return hiql_dataset
        else:
            return batch

    # def update(
    def loss_theta(self, batch, minus_grads=False):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        value_loss, value_info = self.theta.value_loss(batch)
        low_actor_loss, low_actor_info = self.theta.low_actor_loss(batch)
        high_actor_loss, high_actor_info = self.theta.high_actor_loss(batch)
        #total_loss = value_loss + low_actor_loss + high_actor_loss
        actor_loss = low_actor_loss + high_actor_loss
        total_loss = - actor_loss

        print("value_loss, low_actor_loss, high_actor_loss:", value_loss.item(), low_actor_loss.item(), high_actor_loss.item())

        params_with_grad = [p for p in self.theta.parameters() if p.requires_grad and p.requires_grad is not None]
        grads = torch.autograd.grad(total_loss, params_with_grad, retain_graph=False, create_graph=False, allow_unused=True)
        # Disable gradients for unused params
        for param, grad in zip(self.theta.parameters(), grads):
            if grad is None:
                param.requires_grad = False
        
        if minus_grads == True:
            flat_grads = torch.cat([-g.view(-1) for g in grads if g is not None])
        else:
            flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])
        
        #print("value_flat_grads.shape, actor_flat_grads.shape:", value_flat_grads.shape, actor_flat_grads.shape)
        #print("value_loss, low_actor_loss, high_actor_loss:", value_loss, low_actor_loss, high_actor_loss)
        
        return total_loss, flat_grads.cpu()

    def loss_psi(self, dataset, wm=None):
        state = torch.tensor(dataset.transitions['states']).to(self.device)
        action = torch.tensor(dataset.transitions['actions']).unsqueeze(1).to(self.device)

        # form goals
        time_steps = state.shape[0] // self.workspace.n_evals
        value_goals = torch.tensor(dataset.transitions['observations'][dataset.terminal_locs]).to(self.device)
        value_goals = value_goals.repeat_interleave(time_steps, dim=0)

        # dino wm step function is computationally heavy, so carry the computation over time horizon instead of all at once
        #print("states.shape, actions.shape in loss_psi:", state.shape, action.shape)
        next_obs, next_state = batched_forward(self.wm_next, state, action, self.workspace.n_evals)
        #print("next_obs.shape, next_state.shape:",  next_obs.shape, next_state.shape)
        next_obs_raw = self.workspace.reverse_transform(next_obs).squeeze(1)
        #print_stats(next_obs_raw)
        #print("next_obs_raw.shape, value_goals.shape:", next_obs_raw.shape, value_goals.shape)
        #v1, v2 = self.theta.value_net(next_obs_raw, value_goals)
        # the same goes here, forward value net by mini-batch to avoid cuda out of memory
        v1, v2 = batched_forward(self.theta.value_net, next_obs_raw, value_goals, self.workspace.n_evals)
        #print("v1.shape, v2.shape:", v1.shape, v2.shape)
        v = (v1 + v2) / 2

        #print("v.mean(), v.max(), v.min(), v.std():", v.mean(), v.max(), v.min(), v.std())
        loss = v.mean()
        
        grads = torch.autograd.grad(loss, self.psi.predictor.parameters(), retain_graph=False, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])

        return loss, flat_grads.cpu()

    def consistency_loss(self, dataset, wm=None, obs_consistent=True):
        if wm is None:
            wm = self.psi
            
        state = torch.tensor(dataset.transitions['states']).to(self.device)
        action = torch.tensor(dataset.transitions['actions']).unsqueeze(1).to(self.device)
        next_state = torch.tensor(dataset.transitions['next_states']).to(self.device)
        next_obs = torch.tensor(dataset.transitions['next_observations']).to(self.device)

        #print("state.shape, action.shape, next_state.shape, next_obses.shape in consistency_loss:", state.shape, action.shape, next_state.shape, next_obs.shape )
        next_obs_pred, next_state_pred = batched_forward(self.wm_next, state, action, self.workspace.n_evals)

        # compute constraint loss
        z_new_pred = next_state_pred[:, -1, ...].unsqueeze(1)
        z_new = next_state[:, -1, ...].unsqueeze(1)
        #print("z_new_pred.shape, action.shape:", z_new_pred.shape, action.shape)
        z_new_pred = wm.replace_actions_from_z(z_new_pred, action)
        z_new = wm.replace_actions_from_z(z_new, action)
        #print("z_new_pred.shape, z_new.shape:", z_new_pred.shape, z_new.shape)
        latent_consistent_loss = F.mse_loss(z_new_pred, z_new)
        
        next_obs_pred = self.workspace.reverse_transform(next_obs_pred).squeeze(1)
        #print("next_obs_pred.shape, next_obs.shape:", next_obs_pred.shape, next_obs.shape)
        obs_consistent_loss = F.mse_loss(next_obs_pred, next_obs)
        
        #loss = latent_consistent_loss + obs_consistent_loss
        loss = latent_consistent_loss

        # debug
        debug = False
        if debug == True:
            next_obs_list = list(torch.unbind(next_obs.to('cpu'), dim=0))  # List of 32 tensors of shape (3, 64, 64)
            next_obs_pred_list = list(torch.unbind(next_obs_pred.detach().to('cpu'), dim=0))  # List of 32 tensors of shape (3, 64, 64)
            print("True obs")
            print_stats(next_obs)
            plot_obses(next_obs_list, normalize=True)
            print("Perturbed obs")
            print_stats(next_obs_pred)
            plot_obses(next_obs_pred_list, normalize=True)
            print("consistency loss, obs_consistent:", loss.item(), obs_consistent)
        
        grads = torch.autograd.grad(loss, self.psi.predictor.parameters(), retain_graph=False, create_graph=False)
        flat_grads = torch.cat([g.view(-1) for g in grads if g is not None])

        del state, action, next_state, next_obs
        del dataset
        return loss, flat_grads.cpu()

    def wm_next(self, state, action, wm=None):
        if wm is None:
            wm = self.psi
        z = state[:, :-1, ...]
        z_new = state[:, -1, ...].unsqueeze(1)
        #print("z_new.shape, action.shape in wm_next:", z_new.shape, action.shape)
        z_new = wm.replace_actions_from_z(z_new, action)
        z = torch.cat([z, z_new], dim=1)
        z_pred = wm.predict(z[:, -self.cfg.num_hist :])
        #print("z_pred[:, -1, ...].shape, z_pred[:, -1:, ...].shape:", z_pred[:, -1, ...].shape, z_pred[:, -1:, ...].shape)
        z_new = z_pred[:, -1, ...].unsqueeze(1)
        z_cur_dict, _ = wm.separate_emb(z_new)
        obs_dict, _ = wm.decode_obs(z_cur_dict)
        obs = self.resize_obs(obs_dict)

        state = torch.cat([z, z_new], dim=1)
        torch.cuda.empty_cache()
        return obs, state

    def resize_obs(self, obs):
        obs = obs['visual'] # (T, B, C, H, W)
        T, B, C = obs.shape[0], obs.shape[1], obs.shape[2]
        obs_reshaped = obs.view(-1, *obs.shape[2:]) # (T, B, C, H, W) → (T×B, C, H, W)
        obs_resized = F.interpolate(obs_reshaped, size=(64, 64), mode='bilinear', align_corners=False) # (T*B, C, 64, 64)
        obs_resized = obs_resized.view(T, B, C, 64, 64) # (T, B, C, 64, 64)
        return obs_resized.squeeze(0)
        
    # perform one iteration update of theta-psi
    def update(self):
        sample_num = 1
        sample_batch_size = 64
        # approximate theta * by updating theta T steps ahead
        self.theta_star.load_state_dict(self.theta.state_dict())
        T = 5
        for i in range(T):
            print_cuda_stats()
            if i == 0:
                batch_kk, dataset_kk = self.collect(self.psi, self.theta_star, num=sample_num, sample_batch_size=sample_batch_size, return_batch_dataset=True)
                batch = batch_kk
            else:
                batch = self.collect(self.psi, self.theta_star, num=sample_num, sample_batch_size=sample_batch_size)
            self.theta_star.update(batch)

        # collect trajectories from psi_k, theta^T
        #print("***collect trajectories from psi_k, theta^T***")
        dataset = self.collect(self.psi, self.theta_star, num=sample_num, sample_batch_size=sample_batch_size, return_dataset=True)
        loss_psi_kT, grads_psi_kT = self.loss_psi(dataset)
        del dataset
        
        # collect trajectories from psi_0, theta_k
        #print("***collect trajectories from psi_0, theta_k***")
        batch = self.collect(self.psi_0, self.theta, num=sample_num, sample_batch_size=sample_batch_size)
        loss_theta_0k, grads_theta_0k = self.loss_theta(batch)
        del batch

        # apply constraint
        # collect trajectories from psi_0, theta_0
        dataset = self.collect(self.psi_0, self.theta_0, num=sample_num, sample_batch_size=sample_batch_size, return_dataset=True)
        #loss_psi_constraint, grads_psi_constraint = torch.tensor(0).to(self.device), torch.zeros_like(grads_psi_kT).to(self.device)
        loss_psi_constraint, grads_psi_constraint = self.consistency_loss(dataset)
        del dataset
        
        # collect trajectories from psi_k, theta_k
        #print("***collect trajectories from psi_k, theta_k***")
        #batch, dataset = self.collect(self.psi, self.theta, num=sample_num, sample_batch_size=sample_batch_size, return_batch_dataset=True)
        loss_psi_kk, grads_psi_kk = self.loss_psi(dataset_kk)
        loss_theta_kk, minus_grads_theta_kk = self.loss_theta(batch_kk, minus_grads=True)
        del dataset_kk, batch_kk

        # compose grads[f], grads[q], and lamda_k from the four gradients about psi and theta
        # and perform co-update (psi_k+1, theta_k+1) <- (psi_k+1, theta_k+1) + Xi*{grads[f] + lamda_k*grads[q]}


        print(f"loss_psi_kT: {loss_psi_kT}, loss_psi_kk: {loss_psi_kk}, loss_theta_0k: {loss_theta_0k}, loss_theta_kk: {loss_theta_kk}")
        print(f"grad_psi_constraint norm: {grads_psi_constraint.norm()}")
        print(f"grad_psi_kT norm: {grads_psi_kT.norm()}, grad_psi_kk norm: {grads_psi_kk.norm()}, grad_theta_0k norm: {grads_theta_0k.norm()}, grad_theta_kk norm: {minus_grads_theta_kk.norm()}")
        # compute lambda_k
        eta = 0.5
        grads_f = (self.cfg.lambda_coef * grads_psi_constraint, grads_theta_0k)
        grads_psi_kT_minus_grads_psi_kk = grads_psi_kT - grads_psi_kk
        grads_q = (grads_psi_kT_minus_grads_psi_kk, minus_grads_theta_kk)
        grads_q_norm = grads_q[0].norm() + grads_q[1].norm()
        #print("shapes:", grads_f[0].shape, grads_q[0].shape, grads_f[1].shape, grads_q[1].shape)
        dot_product = torch.dot(grads_f[0], grads_q[0]) + torch.dot(grads_f[1], grads_q[1])
        print("dot_product[0], dot_product[1]:", torch.dot(grads_f[0], grads_q[0]), torch.dot(grads_f[1], grads_q[1]))
        print("grads_q_norm, dot_product:", grads_q_norm, dot_product)
        lambda_k = ((eta * grads_q_norm - dot_product) / (grads_q_norm + 1e-8)).detach() # detach lambda_k to avoid gradient propagation
        # when λ_k = 0, psi fits to psi_0, theta fits to theta_0, thus a upper bound of λ_k limits how much algorithm change we want to apply
        #lambda_k = lambda_k.clamp(min=0.0, max=100)  # Enforce λ_k ≥ 0
        lambda_k = lambda_k.clamp(min=0.0)  # Enforce λ_k ≥ 0
        
        # compute loss_psi and loss_theta
        loss_psi = self.cfg.lambda_coef * loss_psi_constraint + lambda_k * (loss_psi_kT - loss_psi_kk) # the psi loss term from f and q function
        loss_theta = loss_theta_0k - lambda_k * loss_theta_kk # the theta loss term from f and q function
        
        # update psi_k and theta_k
        self.psi_optimizer.zero_grad()
        #loss_psi.backward() 
        final_psi_grads = ((self.cfg.lambda_coef * grads_psi_constraint) + (lambda_k * (grads_psi_kT - grads_psi_kk))).to(self.device)
        print("len(final_psi_grads), final_psi_grads.norm():", len(final_psi_grads), final_psi_grads.norm())
        apply_flat_grads_to_model(self.psi.predictor.parameters(), final_psi_grads)
        self.psi_optimizer.step()
    
        self.theta.optimizer.zero_grad()
        #loss_theta.backward()
        final_theta_grads = (grads_theta_0k + (lambda_k * minus_grads_theta_kk)).to(self.device)
        print("len(final_theta_grads), final_theta_grads.norm():", len(final_theta_grads), final_theta_grads.norm())
        apply_flat_grads_to_model(self.theta.parameters(), final_theta_grads)
        #torch.nn.utils.clip_grad_norm_(self.theta.parameters(), max_norm=100.0)
        self.theta.optimizer.step()

        # Target network soft update
        for param, target_param in zip(self.theta.value_net.parameters(), self.theta.target_value_net.parameters()):
            target_param.data.copy_(self.theta.config['tau'] * param.data + (1 - self.theta.config['tau']) * target_param.data)

        print("loss_psi_constraint:", loss_psi_constraint.detach().cpu().item())
        print("lambda_k, loss_psi, loss_theta:", lambda_k.detach().cpu().item(), loss_psi.detach().cpu().item(), loss_theta.detach().cpu().item())
        info = TensorDict({
            "iteration": step,
            "lambda_k": lambda_k.detach().cpu().item(),
            "loss_psi": loss_psi.detach().cpu().item(),
            "loss_theta": loss_theta.detach().cpu().item(),
            "loss_psi_constraint": loss_psi_constraint.detach().cpu().item(),
        })

        #torch.cuda.empty_cache()
        return info

def build_dino_transition_dataset(dataset, device='cpu', num_hist=1):
    """
    dataset: dict with keys 'states', 'next_states', 'actions'
        states      [E, T, 196, 404]
        next_states [E, T, 196, 404]
        actions     [E, T, A]

    Returns (zs, actions, reward, next_z, terminated):

    - if num_hist == 1:
        zs       [E*T, 196, 404]
        next_z   [E*T, 196, 404]
        actions  [E*T, A]

    - if num_hist > 1:
        zs       [E*(T-num_hist+1), num_hist, 196, 404]
                 each entry is a window [t-num_hist+1 .. t]
        actions  [E*(T-num_hist+1), A]          (action at time t)
        next_z   [E*(T-num_hist+1), 196, 404]   (next_states at time t)
    """
    states      = dataset["states"].to(device)      # [E, T, 196, 404]
    next_states = dataset["next_states"].to(device) # [E, T, 196, 404]
    actions     = dataset["actions"].to(device)     # [E, T, A]
    print("states.shape, actions.shape, next_states.shape:", states.shape, actions.shape, next_states.shape)

    E, T, Z1, Z2 = states.shape
    A = actions.shape[-1]

    if num_hist == 1:
        # old behavior
        zs          = states.reshape(E * T, Z1, Z2)       # [N, 196, 404]
        next_z      = next_states.reshape(E * T, Z1, Z2)  # [N, 196, 404]
        actions_flat= actions.reshape(E * T, A)           # [N, A]
    else:
        # build sliding windows over time axis
        num_time = T - num_hist + 1             # valid t positions
        N = E * num_time

        zs          = torch.empty(N, num_hist, Z1, Z2, device=device)
        next_z      = torch.empty(N, Z1, Z2, device=device)
        actions_flat= torch.empty(N, A, device=device)

        idx = 0
        for e in range(E):
            # t is the *last* index in the window, so t ∈ [num_hist-1, T-1]
            for t in range(num_hist - 1, T):
                zs[idx]       = states[e, t - num_hist + 1 : t + 1]  # [num_hist, 196, 404]
                actions_flat[idx] = actions[e, t]                    # action at time t
                next_z[idx]   = next_states[e, t]                    # next state aligned to t
                idx += 1

        assert idx == N

    N = zs.shape[0]
    print(f"N transitions (num_hist={num_hist}):", N)

    reward     = torch.zeros(N, 1, device=device)
    terminated = torch.zeros(N, 1, dtype=torch.bool, device=device)

    return (zs, actions_flat, reward, next_z, terminated)

class DinoWMSystem(torch.nn.Module):
    """
    Wraps your dino_wm so that we can call:
        next_z = model.next(z, action, psi)

    - if z: [N, 196, 404]  (num_hist == 1 case)
      → internally reshaped to [N, 1, 196, 404]
    - if z: [N, num_hist, 196, 404]
      → passed through as-is

    action: [N, A]  → reshaped to [N, 1, A]
    psi:    world model parameters (e.g., target_psi / poison_psi)
    """
    def __init__(self, dino_wm, num_hist=1):
        super().__init__()
        self.model = dino_wm
        self.device = dino_wm.device
        self.num_hist = num_hist

    def next(self, state, action, psi):
        """
        state:  [N, 196, 404] or [N, T, 196, 404]
        action: [N, A] or [N, 1, A]
        """
        # ensure time dimension exists on state
        if state.dim() == 3:
            # [N, 196, 404] → [N, 1, 196, 404]
            state = state.unsqueeze(1)

        # ensure time dimension on action
        if action.dim() == 2:
            # [N, A] → [N, 1, A]
            action = action.unsqueeze(1)

        # wm_next signature: wm_next(self, state, action, wm)
        next_obs, next_state = self.model.wm_next(state, action, psi)

        # take the *last* latent in the returned sequence
        # next_state: [N, T', 196, 404] → [N, 196, 404]
        return next_state[:, -1, ...]
