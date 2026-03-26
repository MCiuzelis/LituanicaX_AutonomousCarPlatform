"""RSL-RL PPO configuration for MuSHR RC car maze navigation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MushrMazePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 512
    max_iterations:    int = 6000
    save_interval:     int = 40
    experiment_name:   str = "mushr_maze"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # obs_size: 80×33 depth (2640) + 450 LiDAR + 4 scalars = 3094 → 512 → 256 → 128
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,
        schedule="adaptive",
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        # Higher entropy encourages more diverse exploration — important when
        # agents currently collapse to a single "drive straight and crash" behaviour.
        entropy_coef=0.005,  # was 0.05 — 8.67 action noise std; 0.005 → ~0.5 (converged)
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.99,   # longer horizon — lap completion is sparse; discount less aggressively
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
