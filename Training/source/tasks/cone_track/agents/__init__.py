"""PPO agent configuration for the cone-track task."""

from .actor_critic import ConeTrackActorCritic, SplitEncoderMLP, register

# rsl_rl resolves the policy class by name via eval() in its own module namespace,
# so the custom class has to be injected there before any runner is constructed.
# This package is imported by the task's gym registration, which happens well
# before train.py / play.py build the runner.
register()

__all__ = ["ConeTrackActorCritic", "SplitEncoderMLP"]
