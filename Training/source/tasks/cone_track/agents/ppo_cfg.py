"""RSL-RL PPO configuration for the cone-track task."""

from __future__ import annotations

import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
import torch
import torch.nn as nn
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.modules import MLP, EmpiricalNormalization
from torch.distributions import Normal


class _ObsShapeMarker(nn.Module):
    """Identity layer that exposes in_features for IsaacLab's ONNX exporter."""

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class _DepthCnnBackbone(nn.Module):
    """Encode the two flattened TOF depth frames into a compact feature vector."""

    def __init__(
        self,
        obs_dim: int,
        image_width: int,
        image_feature_dim: int,
        extra_obs_dim: int | None,
        activation: str,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.image_width = image_width
        self.extra_obs_dim = self._resolve_extra_obs_dim(
            obs_dim, image_width, extra_obs_dim
        )
        image_values = obs_dim - self.extra_obs_dim
        if image_values <= 0 or image_values % 2 != 0:
            raise ValueError(
                f"ConeTrackActorCritic expected two equal camera images, got obs_dim={obs_dim}, "
                f"extra_obs_dim={self.extra_obs_dim}. Make sure wheel velocity is appended after both cameras."
            )

        self.camera_flat_dim = image_values // 2
        if self.camera_flat_dim % image_width != 0:
            raise ValueError(
                f"Camera slice of {self.camera_flat_dim} values is not divisible by image_width={image_width}."
            )
        self.image_height = self.camera_flat_dim // image_width

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.Mish(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(24 * 2 * 4, image_feature_dim),
            self._activation(activation),
        )

    @staticmethod
    def _activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "elu":
            return nn.ELU()
        if name == "relu":
            return nn.ReLU()
        if name == "selu":
            return nn.SELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "mish":
            return nn.Mish()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation for ConeTrack CNN: {name}")

    @staticmethod
    def _resolve_extra_obs_dim(
        obs_dim: int, image_width: int, extra_obs_dim: int | None
    ) -> int:
        if extra_obs_dim is not None:
            return extra_obs_dim
        # Current env appends one wheel-speed scalar. If that scalar is disabled,
        # the no-extra case still resolves cleanly.
        for candidate in (1, 0):
            image_values = obs_dim - candidate
            if (
                image_values > 0
                and image_values % 2 == 0
                and (image_values // 2) % image_width == 0
            ):
                return candidate
        return 0

    @property
    def output_dim(self) -> int:
        return self.cnn[-2].out_features + self.extra_obs_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        left = obs[:, : self.camera_flat_dim]
        right = obs[:, self.camera_flat_dim : 2 * self.camera_flat_dim]
        images = torch.stack((left, right), dim=1).reshape(
            obs.shape[0], 2, self.image_height, self.image_width
        )
        features = self.cnn(images)
        if self.extra_obs_dim == 0:
            return features
        return torch.cat((features, obs[:, -self.extra_obs_dim :]), dim=-1)


class _DepthCnnMlp(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
        image_width: int,
        image_feature_dim: int,
        extra_obs_dim: int | None,
    ):
        super().__init__()
        self.encoder = _DepthCnnBackbone(
            obs_dim=obs_dim,
            image_width=image_width,
            image_feature_dim=image_feature_dim,
            extra_obs_dim=extra_obs_dim,
            activation=activation,
        )
        self.mlp = MLP(self.encoder.output_dim, output_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))


class ConeTrackActorCritic(nn.Module):
    """Actor-critic with CNN depth encoders followed by small MLP heads."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[192, 96],
        activation="mish",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        image_width: int = 160,
        image_feature_dim: int = 64,
        extra_obs_dim: int | None = 1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ConeTrackActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])

        self.actor = nn.Sequential(
            _ObsShapeMarker(num_actor_obs),
            _DepthCnnMlp(
                obs_dim=num_actor_obs,
                output_dim=num_actions,
                hidden_dims=actor_hidden_dims,
                activation=activation,
                image_width=image_width,
                image_feature_dim=image_feature_dim,
                extra_obs_dim=extra_obs_dim,
            ),
        )
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs)
            if actor_obs_normalization
            else torch.nn.Identity()
        )
        print(f"Actor CNN+MLP: {self.actor}")

        self.critic = nn.Sequential(
            _ObsShapeMarker(num_critic_obs),
            _DepthCnnMlp(
                obs_dim=num_critic_obs,
                output_dim=1,
                hidden_dims=critic_hidden_dims,
                activation=activation,
                image_width=image_width,
                image_feature_dim=image_feature_dim,
                extra_obs_dim=extra_obs_dim,
            ),
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs)
            if critic_obs_normalization
            else torch.nn.Identity()
        )
        print(f"Critic CNN+MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _obs_dim(obs, obs_group_names) -> int:
        total = 0
        for obs_group in obs_group_names:
            assert len(obs[obs_group].shape) == 2, (
                "ConeTrackActorCritic only supports flat observation tensors."
            )
            total += obs[obs_group].shape[-1]
        return total

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(obs)

    def get_actor_obs(self, obs):
        return torch.cat(
            [obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1
        )

    def get_critic_obs(self, obs):
        return torch.cat(
            [obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1
        )

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


@configclass
class ConeTrackCnnActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ConeTrackActorCritic"
    image_width: int = 160
    image_feature_dim: int = 64
    extra_obs_dim: int | None = 1


rsl_on_policy_runner.ConeTrackActorCritic = ConeTrackActorCritic


@configclass
class ConeTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 1000
    max_iterations: int = 1000
    save_interval: int = 10
    experiment_name: str = "cone_track"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: ConeTrackCnnActorCriticCfg = ConeTrackCnnActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[96, 48],
        critic_hidden_dims=[128, 64],
        activation="mish",
        image_width=160,
        image_feature_dim=64,
        extra_obs_dim=1,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=3e-5,
        schedule="adaptive",
        num_learning_epochs=6,
        num_mini_batches=10,
        clip_param=0.2,
        entropy_coef=0.007,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.995,
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
    #     # and temporally filter the noisy depth frames.
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
