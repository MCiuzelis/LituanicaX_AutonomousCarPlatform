"""Cone-track task package.

Importing this package is all a script needs to do to make the task usable:

  * it registers the ``ConeTrack`` gym id, pointing at :mod:`cone_track.env` for
    the environment and :mod:`cone_track.ppo_cfg` for the RSL-RL runner config;
  * it registers the custom actor-critic (see :mod:`cone_track.actor_critic`)
    into rsl_rl's namespace, which must happen before any runner is built.

Module map
----------
  env.py           ConeTrackEnv + ConeTrackEnvCfg — physics, rewards, observations.
                   Every tunable number lives in ConeTrackEnvCfg.
  ppo_cfg.py       ConeTrackPPORunnerCfg — PPO hyper-parameters and network sizes.
  actor_critic.py  Two-branch actor-critic (depth trunk + proprio encoder).
"""

import gymnasium as gym

from .actor_critic import register_actor_critic

register_actor_critic()

gym.register(
    id="ConeTrack",
    entry_point=f"{__name__}.env:ConeTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:ConeTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:ConeTrackPPORunnerCfg",
    },
)
