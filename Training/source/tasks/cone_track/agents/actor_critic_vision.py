"""Lane-feature ActorCritic: compact lane params + MLP (no CNN)."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ConeTrackActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
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

        encoded_dim = num_actor_obs  # no CNN encoder — pass through directly

        # ── MLP heads (small — input is compact lane params) ─────────────
        actor_hidden = kwargs.get("actor_hidden_dims", [64, 32])
        critic_hidden = kwargs.get("critic_hidden_dims", [64, 32])
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

        print(f"[ConeTrackActorCritic] Encoded dim: {encoded_dim} (lane params)")
        print(f"[ConeTrackActorCritic] Actor MLP: {self.actor}")
        print(f"[ConeTrackActorCritic] Critic MLP: {self.critic}")

    # ── helpers ──────────────────────────────────────────────────────────

    def _encode(self, x):
        return x  # No CNN — the observation IS the encoded representation

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

    @torch.inference_mode()
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
