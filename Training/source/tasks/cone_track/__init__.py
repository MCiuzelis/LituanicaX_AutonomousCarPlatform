"""Cone-track navigation task — gym registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="ConeTrack",
    entry_point=f"{__name__}.env:ConeTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:ConeTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:ConeTrackPPORunnerCfg",
    },
)
