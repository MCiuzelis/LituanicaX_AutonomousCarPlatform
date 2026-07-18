"""RSL-RL PPO configuration for the cone-track task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ConeTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 1000
    max_iterations: int = 1000
    save_interval: int = 10
    experiment_name: str = "cone_track"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # Small MLP — input is (4 × max_bboxes + optional wheel_speed) bbox coords
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="mish",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,
        schedule="adaptive",
        num_learning_epochs=4,
        num_mini_batches=8,
        clip_param=0.2,
        entropy_coef=0.006,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.993,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.75,
    )

    # policy: RslRlPpoActorCriticRecurrentCfg = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std=0.5,
    #     actor_obs_normalization=True,
    #     critic_obs_normalization=True,
    #     actor_hidden_dims=[256, 128],
    #     critic_hidden_dims=[512, 256],
    #     activation="elu",
    #     # Recurrent core: gives the policy temporal memory so it can integrate
    #     # wheel-speed odometry, anticipate corners beyond the 2 m sensing horizon,
    #     # and temporally filter the noisy bbox observations.
    #     rnn_type="lstm",
    #     rnn_hidden_dim=256,
    #     rnn_num_layers=1,
    # )

    # algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
    #     learning_rate=1e-4,
    #     schedule="adaptive",
    #     num_learning_epochs=4,
    #     # Few minibatches so RSL-RL keeps long trajectory sequences intact for
    #     # BPTT through the LSTM (24 would fragment them and hurt recurrence).
    #     num_mini_batches=2,
    #     clip_param=0.2,
    #     entropy_coef=0.004,
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     # Longer credit-assignment horizon (~330 steps ≈ 11 s) so the "brake now
    #     # to avoid a corner 1-2 s ahead" signal propagates back to the decision.
    #     gamma=0.993,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=0.75,
    # )
