"""State-only subclasses of the source-verified corrected ManiSkill2 runtime.

No upstream module or installed source is modified. Physical stepping, robot
control, task initialization, observations, rewards and native flags are inherited
unchanged. Only renderer/camera/visual-material construction is omitted, as in
the retained state-only adapter. This Gymnasium-source adapter is separately
validated; an older Gym-based qualification is not claimed for these classes.

Derived BaseEnv initialization is from ManiSkill2 (see licenses/). The retained
rendererless adapter origin and exact upstream identities are in SOURCE_PROVENANCE.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import importlib
from pathlib import Path

SOURCE_SHA256={
 'mani_skill2.envs.sapien_env':'2fdb7b683c900e23c763690c2d40da01d31de3e32d7ea8274133a34761f8fc35',
 'mani_skill2.envs.pick_and_place.base_env':'346ff671c084644322a6662cf9b0a917adc644ebd1eed80c46038876bdb165d8',
 'mani_skill2.envs.pick_and_place.pick_cube':'d6581d2557d934ae078dfafca25c039cd20fb237582b7eefafcce5e57967241d',
 'mani_skill2.envs.pick_and_place.stack_cube':'95fadf8e12cc511245e5423bf1bc538b971a200e3b9372d68c806f610afaa520',
 'mani_skill2.agents.base_agent':'f7d35fae5bfcaebc50af7b399e0b54b10f8ff0a2c355603308cdd7f59a1ccea2',
 'mani_skill2.agents.robots.panda':'b847951c2ac59c7a11e90c134acb2d6fbadcc0f94f9223fbac82231d9d8f017a',
 'mani_skill2.utils.sapien_utils':'96537e001ba1bcfbe830f206817148f08f094ad035d8b987b15c586e9fcc9807',
}


def source_modules():
    result={}
    for name,expected in SOURCE_SHA256.items():
        module=importlib.import_module(name);path=Path(module.__file__).resolve()
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError('Corrected ManiSkill source differs: '+name)
        result[name]=module
    return result


def task_classes():
    """Create ordinary subclasses; do not monkeypatch upstream classes."""
    m=source_modules();base=m['mani_skill2.envs.sapien_env'];stationary=m['mani_skill2.envs.pick_and_place.base_env']
    pick=m['mani_skill2.envs.pick_and_place.pick_cube'];stack=m['mani_skill2.envs.pick_and_place.stack_cube']
    class StateBase(base.BaseEnv):
        def __init__(self,obs_mode=None,reward_mode=None,control_mode=None,render_mode=None,
                     sim_freq=500,control_freq=20,renderer='none',renderer_kwargs=None,shader_dir='ibl',
                     render_config=None,enable_shadow=False,camera_cfgs=None,render_camera_cfgs=None,bg_name=None):
            if not(obs_mode=='state' and self.robot_uid=='panda' and renderer=='none' and renderer_kwargs in(None,{})
                and shader_dir=='ibl' and render_config is None and enable_shadow is False and bg_name is None
                and camera_cfgs is None and render_camera_cfgs is None and render_mode is None):
                raise ValueError('Rendererless adapter requires the exact state/Panda/no-visual contract')
            self._engine=base.sapien.Engine()
            self._engine.set_log_level(base.os.getenv('MS2_SIM_LOG_LEVEL','error'))
            self._renderer_type='none';self._renderer=None;self._viewer=None
            self._sim_freq=sim_freq;self._control_freq=control_freq
            if sim_freq%control_freq!=0:base.logger.warning(f'sim_freq({sim_freq}) is not divisible by control_freq({control_freq}).')
            self._sim_steps_per_control=sim_freq//control_freq
            if obs_mode not in self.SUPPORTED_OBS_MODES:raise ValueError('Unsupported observation mode')
            self._obs_mode=obs_mode
            if reward_mode is None:reward_mode=self.SUPPORTED_REWARD_MODES[0]
            if reward_mode not in self.SUPPORTED_REWARD_MODES:raise ValueError('Unsupported reward mode')
            self._reward_mode=reward_mode;self._control_mode=control_mode
            if control_mode=='*':raise NotImplementedError('Multiple controllers are not supported yet.')
            self.render_mode=render_mode
            self._configure_agent();self._configure_cameras();self._configure_render_cameras()
            self.enable_shadow=enable_shadow;self.bg_name=bg_name
            self._main_seed=None;self.set_main_rng(2022)
            observation,_=self.reset(seed=2022,options=dict(reconfigure=True))
            self.observation_space=base.convert_observation_to_space(observation)
            self.action_space=self.agent.action_space

        def _add_ground(self,altitude=0.,render=True):
            return self._scene.add_ground(altitude=altitude,render=False,render_material=None)

        def _setup_cameras(self):
            self._cameras=OrderedDict();self._render_cameras=OrderedDict()

        def _setup_lighting(self):return None
        def _load_background(self):
            if self.bg_name is not None:raise ValueError('Visual background is prohibited')

    class StateStationary(stationary.StationaryManipulationEnv,StateBase):
        def _build_cube(self,half_size,color=(1,0,0),name='cube',static=False,render_material=None):
            if render_material is not None:raise ValueError('Visual material is prohibited')
            builder=self._scene.create_actor_builder();builder.add_box_collision(half_size=half_size)
            return builder.build_static(name)if static else builder.build(name)

        def _build_sphere_site(self,radius,color=(0,1,0),name='goal_site'):
            # Preserve static actor name/order, omitting only visual sphere.
            return self._scene.create_actor_builder().build_static(name)

        def _load_agent(self):
            agent_cls=self.SUPPORTED_ROBOTS[self.robot_uid]
            self.agent=agent_cls(self._scene,self._control_freq,self._control_mode,config=self._agent_cfg)
            self.tcp=stationary.get_entity_by_name(self.agent.robot.get_links(),self.agent.config.ee_link_name)
            if self.tcp is None:raise ValueError('Exact Panda TCP is required')

    class VisualGuards:
        def _forbidden(self,*args,**kwargs):raise RuntimeError('Renderer/camera entry is prohibited for state-only ManiSkill')
        render=render_human=render_rgb_array=update_render=take_picture=get_images=get_camera_params=gen_scene_pcd=_forbidden

    class StatePick(VisualGuards,pick.PickCubeEnv,StateStationary):pass
    class StateLift(pick.LiftCubeEnv,StatePick):pass
    class StateStack(VisualGuards,stack.StackCubeEnv,StateStationary):pass
    return {'lift-cube':StateLift,'pick-cube':StatePick,'stack-cube':StateStack}


def make_environment(task,*,obs_mode,control_mode,reward_mode):
    import gymnasium as gym
    cls=task_classes()[task]
    env=cls(obs_mode=obs_mode,control_mode=control_mode,reward_mode=reward_mode,renderer='none')
    if env._renderer is not None or env._renderer_type!='none' or env._cameras or env._render_cameras:
        raise RuntimeError('Rendererless surface differs')
    return gym.wrappers.TimeLimit(env,max_episode_steps=200)
