"""CNN-encoder ActorCritic for cone-track: conv stack + adaptive pool + MLP."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

from ..parameters import (
    CNN_CHANNELS,
    CNN_KERNEL_SIZES,
    CNN_POOL_SIZE,
    CNN_STRIDES,
    POLICY_DEPTH_H,
    POLICY_DEPTH_W,
)


class ConeTrackActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        cnn_img_h=POLICY_DEPTH_H,
        cnn_img_w=POLICY_DEPTH_W,
        cnn_channels=CNN_CHANNELS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        cnn_strides=CNN_STRIDES,
        cnn_pool_size=CNN_POOL_SIZE,
        **kwargs,
    ):
        super().__init__()

        # Observation dimensions
        num_actor_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "only 1D observations supported"
            num_actor_obs += obs[g].shape[-1]
        num_critic_obs = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "only 1D observations supported"
            num_critic_obs += obs[g].shape[-1]

        # Flat obs layout: [vision_mask, wheel_vel_scalar]
        self._vision_size = num_actor_obs - 1
        self._scalar_size = 1
        self._img_h = cnn_img_h
        self._img_w = cnn_img_w

        exp_vis = cnn_img_h * cnn_img_w
        assert self._vision_size == exp_vis, (
            f"vision size {self._vision_size} != {cnn_img_h}*{cnn_img_w} = {exp_vis}"
        )

        # ── CNN encoder ──────────────────────────────────────────────────
        convs = []
        in_ch = 1
        for out_ch, k, s in zip(cnn_channels, cnn_kernel_sizes, cnn_strides):
            convs.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2))
            convs.append(nn.Mish())
            in_ch = out_ch
        convs.append(nn.AdaptiveAvgPool2d(cnn_pool_size))
        self.cnn_encoder = nn.Sequential(*convs)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, cnn_img_h, cnn_img_w)
            cnn_out = self.cnn_encoder(dummy)
            cnn_flat = cnn_out.reshape(1, -1).size(-1)
        encoded_dim = cnn_flat + self._scalar_size

        # ── MLP heads ────────────────────────────────────────────────────
        actor_hidden = kwargs.get("actor_hidden_dims", [256, 256, 256])
        critic_hidden = kwargs.get("critic_hidden_dims", [256, 256, 256])
        activation = kwargs.get("activation", "elu")
        init_noise_std = kwargs.get("init_noise_std", 1.0)
        act_norm = kwargs.get("actor_obs_normalization", False)
        crit_norm = kwargs.get("critic_obs_normalization", False)
        noise_std_type = kwargs.get("noise_std_type", "scalar")

        self.actor = MLP(encoded_dim, num_actions, actor_hidden, activation)
        self.critic = MLP(encoded_dim, 1, critic_hidden, activation)

        self.actor_obs_normalization = act_norm
        self.actor_obs_normalizer = (
            EmpiricalNormalization(encoded_dim) if act_norm else nn.Identity()
        )
        self.critic_obs_normalization = crit_norm
        self.critic_obs_normalizer = (
            EmpiricalNormalization(encoded_dim) if crit_norm else nn.Identity()
        )

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown std type '{noise_std_type}'. Use 'scalar' or 'log'."
            )

        self.obs_groups = obs_groups
        self._encoded_dim = encoded_dim
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ConeTrackActorCritic] CNN: {self.cnn_encoder}")
        print(f"[ConeTrackActorCritic] Encoded dim: {encoded_dim}")
        print(f"[ConeTrackActorCritic] Actor MLP: {self.actor}")
        print(f"[ConeTrackActorCritic] Critic MLP: {self.critic}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _encode(self, x):
        vision = x[:, : self._vision_size]
        scalar = x[:, self._vision_size :]
        img = vision.reshape(-1, 1, self._img_h, self._img_w)
        feats = self.cnn_encoder(img)
        feats = feats.reshape(feats.size(0), -1)
        return torch.cat([feats, scalar], dim=-1)

    # ── overrides ────────────────────────────────────────────────────────

    def forward(self):
        raise NotImplementedError

    def reset(self, dones=None):
        pass

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
            raise ValueError(f"Unknown std type '{self.noise_std_type}'")
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs):
        parts = []
        for g in self.obs_groups["policy"]:
            parts.append(obs[g])
        flat = torch.cat(parts, dim=-1)
        return self._encode(flat)

    def get_critic_obs(self, obs):
        parts = []
        for g in self.obs_groups["critic"]:
            parts.append(obs[g])
        flat = torch.cat(parts, dim=-1)
        return self._encode(flat)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

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
