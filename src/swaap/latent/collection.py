"""Retained per-trajectory ManiSkill collection seed/reset adapter."""
import math
from .io import require as need
class CollectionBoundary:
    """Observe shared collect's one reset/steps; seed planner after the real reset."""

    def __init__(self, environment, native, torch, np, device, task):
        self.env, self.native, self.torch, self.np = (environment, native, torch, np)
        self.device, self.task = (device, task)
        self.scope = None
        self.active = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def arm(self, root, role, offset):
        need(self.active is None, 'Previous shared trajectory was not closed')
        self.active = dict(role=role, **root, row_start=offset, reset_calls=0, outer_decisions=0, native_steps=0, return_observed=0.0, success_by_outer_step=[], native_terminated_by_outer_step=[], native_truncated_by_outer_step=[])

    def reset(self, *args, **kwargs):
        a = self.active
        need(a is not None and a['reset_calls'] == 0 and (not args) and (not kwargs), 'Shared collect must issue exactly one ordinary reset per trajectory')
        observation = self.env.reset(seed=a['environment_seed'])
        a['reset_calls'] = 1
        a['initial_observation_hash'] = self.native.observation_hash(observation, self.torch, self.np)
        a['observation_shape'] = list(observation.shape)
        native_env = self.env.env.unwrapped
        a['actual_episode_seed'] = int(native_env._episode_seed)
        a['reward_mode'] = native_env.reward_mode
        need(a['reward_mode'] == 'normalized_dense', 'Qualified normalized-dense runtime required')
        self.scope = self.native.isolated_planner_rng(a['planner_seed'], self.device, self.torch, self.np)
        self.scope.__enter__()
        return observation

    def step(self, action):
        a = self.active
        need(a is not None and a['role'] == 'clean', 'Model-only reference cannot step the environment')
        value = self.env.step(action)
        observation, reward, done, info = value
        a['outer_decisions'] += 1
        need(info['native_steps_this_decision'] == 2 and info['native_steps'] == 2 * a['outer_decisions'] and (bool(done) == (a['outer_decisions'] == 100)), 'Real Clean trajectory must retain all100 outer/200 native steps')
        a['native_steps'] = int(info['native_steps'])
        scalar_reward = float(reward)
        need(math.isfinite(scalar_reward), 'Nonfinite real reward; preserve partial collection')
        a['return_observed'] += scalar_reward
        a['success_by_outer_step'].append(bool(info.get('success', False)))
        a['native_terminated_by_outer_step'].append(bool(info.get('native_terminated', False)))
        a['native_truncated_by_outer_step'].append(bool(info.get('native_truncated', False)))
        a['last_observation_hash'] = self.native.observation_hash(observation, self.torch, self.np)
        return value

    def close_scope(self):
        if self.scope is not None:
            self.scope.__exit__(None, None, None)
            self.scope = None

    def finish(self):
        self.close_scope()
        a = self.active
        need(a['reset_calls'] == 1, 'Exactly one actual reset required')
        need(a['outer_decisions'] == (100 if a['role'] == 'clean' else 0), 'Wrong real-step count')
        a.update(row_stop=a['row_start'] + 100, rows=100, imagined_steps=100 if a['role'] == 'reference' else 0, trajectory_complete=True, stored_done_column='ALL_ZERO_SHARED_COLLECT')
        self.active = None
        return a


class ExplicitRootCollectionBoundary:
    """Fresh per-trajectory RNG binding around the unchanged native collector.

    This opt-in mode changes stochastic batching relative to historical batch
    collection. It does not claim byte-identical regeneration of an old bank.
    """
    def __init__(self,agent,native,torch,np):
        self.agent,self.native,self.torch,self.np=agent,native,torch,np
        self.scope=self.environment=self.active=None

    def arm(self,root,role,offset,*,use_environment):
        need(self.active is None,'Previous collection trajectory was not closed')
        self.active=dict(root,role=role,row_start=offset,reset_calls=0,outer_decisions=0,
            environment_steps=0,uses_environment=use_environment)

    def reset(self,*args,**kwargs):
        need(self.active is not None and not args and not kwargs and self.active['reset_calls']==0,
            'Native collect must make exactly one reset per trajectory')
        from .families import make_seeded_environment
        row=self.active
        self.environment,observation,methods=make_seeded_environment(self.agent.cfg,self.agent.cfg.task,
            row['environment_seed'],self.torch,self.np)
        row.update(reset_calls=1,seed_methods=methods,
            initial_observation_hash=self.native.observation_hash(observation,self.torch,self.np))
        self.scope=self.native.isolated_planner_rng(row['planner_seed'],self.agent.device,self.torch,self.np)
        self.scope.__enter__()
        return observation

    def step(self,action):
        need(self.active is not None and self.active['uses_environment'],'Imagined references must not step the environment')
        value=self.environment.step(action)
        self.active['outer_decisions']+=1
        self.active['environment_steps']+=1
        return value

    def close_scope(self):
        if self.scope is not None:self.scope.__exit__(None,None,None);self.scope=None
        if self.environment is not None:self.environment.close();self.environment=None

    def finish(self,expected_steps):
        row=self.active
        need(row is not None and row['reset_calls']==1,'Collection reset was not observed')
        need(row['environment_steps']==(expected_steps if row['uses_environment']else 0),'Incomplete declared collection trajectory')
        self.close_scope();row.update(row_stop=row['row_start']+expected_steps,rows=expected_steps,
            imagined_steps=0 if row['uses_environment']else expected_steps,trajectory_complete=True,
            stochastic_batching='one explicitly seeded trajectory per unchanged native collect call')
        self.active=None
        return row


def validate_collection_panels(panels,rows,reference_rows,horizon):
    """Reject duplicates/overlap before collection or model execution."""
    need(isinstance(panels,dict)and set(panels)>={'clean_roots','reference_roots'},'Explicit clean/reference collection root panels are required')
    all_env=set();all_planner=set()
    for role,count in [('clean',rows//horizon),('reference',reference_rows//horizon)]:
        roots=panels[role+'_roots'];need(isinstance(roots,list)and len(roots)==count,'Collection root count differs from complete trajectory count')
        need([r['episode_index']for r in roots]==list(range(count)),'Collection indices must be contiguous in declared order')
        es=[];ps=[]
        for r in roots:
            need(all(type(r[k])is int and 0<=r[k]<2**32 for k in ('environment_seed','planner_seed')),'Collection roots require unsigned32bit seeds')
            es.append(r['environment_seed']);ps.append(r['planner_seed'])
        need(len(set(es))==count and len(set(ps))==count and not all_env.intersection(es)and not all_planner.intersection(ps),
            'Clean/reference collection roots must be disjoint')
        all_env.update(es);all_planner.update(ps)
    return panels
