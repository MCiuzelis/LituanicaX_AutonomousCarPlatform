"""RSL-RL PPO configuration for MuSHR RC car maze navigation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MushrMazePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 1500#2000
    max_iterations:    int = 3000
    save_interval:     int = 40
    experiment_name:   str = "mushr_maze"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 512, 256],  # larger critic → better advantage estimates
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,
        schedule="adaptive",
        num_learning_epochs=8,
        num_mini_batches=8,
        clip_param=0.2,
        entropy_coef=0.01,        # was 0.005 — more exploration, prevents premature convergence
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.99,               # was 0.97 — longer horizon so lap completion reward propagates back
        lam=0.95,
        desired_kl=0.02,          # was 0.01 — allows faster policy updates when signal is clear
        max_grad_norm=1.0,
    )
