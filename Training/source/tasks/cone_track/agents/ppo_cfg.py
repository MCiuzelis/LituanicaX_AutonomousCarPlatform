"""RSL-RL PPO configuration for the cone-track task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ConeTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
<<<<<<< HEAD
    num_steps_per_env: int = 1800
    max_iterations:    int = 3000
    save_interval:     int = 10
=======
    num_steps_per_env: int = 3000
    max_iterations:    int = 3000
    save_interval:     int = 20
>>>>>>> origin/main
    experiment_name:   str = "cone_track"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.75,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
<<<<<<< HEAD
        actor_hidden_dims=[512, 256, 128],                                                                                          
        critic_hidden_dims=[1024, 512, 256],
        # critic_hidden_dims=[512, 256, 128],
=======
        actor_hidden_dims=[1024, 512, 128],   # was [512, 256, 128]                                                                                           
        critic_hidden_dims=[2048, 512, 256],  # was [1024, 512, 256] 
>>>>>>> origin/main
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
<<<<<<< HEAD
        learning_rate=1e-4,   
        schedule="adaptive",     
        num_learning_epochs=6, 
        num_mini_batches=5,   
        clip_param=0.2,
        entropy_coef=0.002,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.997,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.75,
=======
        learning_rate=3e-4,
        schedule="adaptive",
        num_learning_epochs=8,
        num_mini_batches=8,
        clip_param=0.2,
        entropy_coef=0.003,   # was 0.01 — high entropy was preventing noise std from decaying below ~0.95
        value_loss_coef=0.5,  # was 1.0 — critic loss was climbing unbounded; halving stabilises advantage estimates
        use_clipped_value_loss=True,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.013,     # 0.01 caused near-zero LR (adaptive scheduler over-throttled); 0.013 lets it breathe without the large collapses seen at 0.02
        max_grad_norm=1.0,
>>>>>>> origin/main
    )
