"""PPO hyper-parameters for the cone-track task (RSL-RL ``OnPolicyRunner``).

This is the file to edit when tuning LEARNING; ``env.py`` is the file to edit when
tuning the ROBOT, the track, or the reward.  Values here are consumed by train.py
and play.py through the ``rsl_rl_cfg_entry_point`` registered in ``__init__.py``.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ConeTrackActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic with a dedicated encoder branch for the proprioceptive tail.

    See actor_critic.py for why the scalars need their own branch: as 11 columns
    out of 4331 they carry ~0.25 % of the first layer's activation variance, so a
    plain MLP has to spend a very long time learning to hear them over 4320 depth
    pixels.
    """

    class_name: str = "ConeTrackActorCritic"

    # Trailing observation entries treated as proprioception, i.e. routed through
    # the encoder branch instead of the depth trunk.  MUST match what
    # ConeTrackEnv._get_observations appends after the two depth images:
    #   include_wheel_vel (1) + 2 × prev_action_history_len (5) = 11.
    # A mismatch is silent — the split just lands in the wrong place — so update
    # this whenever either env option changes.
    num_proprio_obs: int = 11
    proprio_hidden_dims: list[int] = [64, 64]


@configclass
class ConeTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner + algorithm settings for a ConeTrack PPO run."""

    num_steps_per_env: int = 1200
    max_iterations: int = 1000
    save_interval: int = 10
    experiment_name: str = "cone_track"  # → logs/rsl_rl/cone_track/<timestamp>/

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: ConeTrackActorCriticCfg = ConeTrackActorCriticCfg(
        init_noise_std=0.65,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # First two entries size the depth trunk, the last sizes the head that
        # runs on [trunk_out | proprio_out].
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="mish",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=1e-4,
        # FIXED, not adaptive.  rsl_rl's adaptive controller (ppo.py:281-284) lives
        # INSIDE the minibatch loop, so it fires num_learning_epochs × num_mini_batches
        # = 24 times per iteration, each time multiplying or dividing by 1.5 between
        # hard bounds of [1e-5, 1e-2].  That is a bang-bang controller, not a
        # regulator, and on a wide actor fed raw depth pixels it just slams between
        # the rails: 12-28 sat on the 1e-5 floor for 70 iterations; 14-43 rode the
        # 1e-2 ceiling for iterations 4-9 (value loss 19.2) then collapsed into a
        # do-nothing policy; 18-00 blew up at iteration 30 at 6.7e-3; and 22-29 spent
        # 39 % of its iterations pinned to the floor at a median LR of 3.4e-5, which
        # is why lap time sat at 12.8 s for 90 iterations without improving.
        # A fixed 1e-4 removes every one of those failure modes.  Revisit only after
        # the depth observation goes through a CNN encoder rather than a dense layer.
        # NOTE: with schedule="fixed", rsl_rl never writes the LR into the optimizer
        # after construction — ppo.py:294 sits inside the `schedule == "adaptive"`
        # branch — while on_policy_runner.py:317 restores optimizer_state_dict on
        # resume.  So RESUMING a checkpoint that was trained on the adaptive
        # schedule silently inherits whatever LR it was saved at (the 22-29-47 run
        # ended at 1e-5, i.e. 10x too slow).  Start fresh, or overwrite the LR
        # manually after load.
        schedule="fixed",
        num_learning_epochs=3,
        num_mini_batches=8,
        clip_param=0.2,
        # 0.0045 held mean_noise_std at 0.49 after 148 iterations.  At 30 Hz that
        # much noise on throttle/steering IS the observed erratic driving, and it
        # also biases the mean action toward whatever survives its own jitter.
        entropy_coef=0.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        # 0.995 is a 200-step horizon; a lap is ~385 steps at 30 Hz, so the lap
        # bonus was discounted to 0.14 by the time it was visible from the lap
        # start.  0.998 gives ~500 steps, which actually covers a lap.
        gamma=0.998,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.75,
    )
