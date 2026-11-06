"""Only the retained PushT environment is registered by this optional runtime."""
from gym.envs.registration import register
register(id="pusht",entry_point="env.pusht.pusht_wrapper:PushTWrapper",max_episode_steps=300,reward_threshold=1.0)
