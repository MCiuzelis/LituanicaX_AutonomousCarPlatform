"""Two-branch actor-critic: depth trunk + dedicated proprioceptive encoder.

WHY THIS EXISTS
---------------
The cone-track observation is 4320 depth pixels followed by a handful of
proprioceptive scalars (wheel speed, previous action).  Fed to a plain MLP those
scalars are 3 input columns out of 4323, and that is a real handicap rather than
a cosmetic one:

  * At the standard 1/sqrt(fan_in) initialisation every first-layer unit's
    pre-activation is a sum of 4323 terms of comparable size.  The proprio
    scalars contribute 3/4323 ≈ 0.07 % of its variance — they are effectively
    invisible at init.
  * To make wheel speed count as much as the whole image, its weight column has
    to grow by ~sqrt(4320) ≈ 66x.  At lr 1e-4 that takes a very long time, and
    every gradient step is also fighting 4320 correlated pixel columns.

The natural reading of "feed it deeper" is to concatenate the scalars into layer
2 or 3.  That alone does NOT fix it: a scalar at layer 2 is 1-in-4096 instead of
1-in-4323, which is the same problem with a different denominator.  What matters
is representational MASS, not depth.  So the scalars get their own small encoder
and enter the trunk as a 64-wide feature block:

    depth (4320) ──> 4096 ──> 1024 ─┐
                                    ├─> 1088 ──> 256 ──> actions
    proprio (3)  ──> 64 ──> 64 ─────┘

64-in-1088 is ~6 % of the head's input width, roughly 85x the share the scalars
had at the input layer, and the block bypasses the depth bottleneck entirely so
its gradient path is short.

EXPORT COMPATIBILITY
--------------------
isaaclab_rl's exporter (rsl_rl/exporter.py) deep-copies `policy.actor`, calls it
on the flat normalised observation, and sizes the ONNX input from
`actor[0].in_features`.  SplitEncoderMLP therefore takes the *whole* flat
observation and splits it internally, and exposes `__getitem__(0)` so that
in_features lookup still resolves.  Both the JIT and ONNX paths work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class _InputSpec:
    """Minimal stand-in for a first `nn.Linear`, for the exporter's shape probe.

    Deliberately NOT an nn.Module: it holds no parameters, must not appear in the
    state dict, and must not be visited by torch.jit.script.
    """

    def __init__(self, in_features: int):
        self.in_features = in_features


class SplitEncoderMLP(nn.Module):
    """MLP over ``[vision | proprio]`` with a separate encoder for the proprio tail.

    The proprioceptive scalars are expected to be the LAST ``num_proprio`` entries
    of the observation vector, which is how ConeTrackEnv._get_observations lays it
    out.  ``hidden_dims`` is interpreted as: all but the last entry size the vision
    trunk, the last entry sizes the post-concatenation head.
    """

    def __init__(
        self,
        num_obs: int,
        num_proprio: int,
        num_out: int,
        hidden_dims: list[int],
        proprio_hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()

        if len(hidden_dims) < 3:
            raise ValueError(
                "SplitEncoderMLP needs at least 3 hidden dims (>=2 for the vision "
                f"trunk, 1 for the head); got {list(hidden_dims)}."
            )
        if len(proprio_hidden_dims) < 2:
            raise ValueError(
                f"proprio_hidden_dims needs at least 2 entries; got {list(proprio_hidden_dims)}."
            )
        if num_proprio <= 0 or num_proprio >= num_obs:
            raise ValueError(
                f"num_proprio ({num_proprio}) must be in (0, num_obs={num_obs})."
            )

        self.num_vision = num_obs - num_proprio
        self.num_proprio = num_proprio

        # Vision trunk: keeps its output activated, since it feeds a concat, not a head.
        self.vision = MLP(
            self.num_vision,
            hidden_dims[-2],
            list(hidden_dims[:-2]),
            activation,
            last_activation=activation,
        )
        self.proprio = MLP(
            num_proprio,
            proprio_hidden_dims[-1],
            list(proprio_hidden_dims[:-1]),
            activation,
            last_activation=activation,
        )
        self.head = MLP(
            hidden_dims[-2] + proprio_hidden_dims[-1],
            num_out,
            list(hidden_dims[-1:]),
            activation,
        )

        # Not a submodule — see _InputSpec.
        self._input_spec = _InputSpec(num_obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vision_feat = self.vision(x[..., : self.num_vision])
        proprio_feat = self.proprio(x[..., self.num_vision :])
        return self.head(torch.cat((vision_feat, proprio_feat), dim=-1))

    @torch.jit.unused
    def __getitem__(self, idx: int) -> _InputSpec:
        """Only index 0 is meaningful: the exporter's `actor[0].in_features` probe."""
        if idx != 0:
            raise IndexError(
                "SplitEncoderMLP is not a Sequential; only index 0 is exposed "
                "(input-shape probe for the policy exporter)."
            )
        return self._input_spec


class ConeTrackActorCritic(ActorCritic):
    """rsl_rl ActorCritic with both networks replaced by SplitEncoderMLP.

    Everything else — observation normalisers, the action distribution, the
    learnable noise std — is inherited untouched, so this stays compatible with
    stock PPO and with checkpoints' optimizer/normaliser state.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_proprio_obs: int = 3,
        proprio_hidden_dims: list[int] | None = None,
        actor_hidden_dims: list[int] = [4096, 1024, 256],
        critic_hidden_dims: list[int] = [4096, 1024, 256],
        activation: str = "mish",
        **kwargs,
    ):
        # Builds the stock dense MLPs first; they are discarded below.  The
        # transient allocation is ~140 MB for a 4323-wide input and is freed
        # immediately, which is cheap next to the RTX sensor buffers.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        if proprio_hidden_dims is None:
            proprio_hidden_dims = [64, 64]

        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        self.actor = SplitEncoderMLP(
            num_actor_obs,
            num_proprio_obs,
            num_actions,
            actor_hidden_dims,
            proprio_hidden_dims,
            activation,
        )
        self.critic = SplitEncoderMLP(
            num_critic_obs,
            num_proprio_obs,
            1,
            critic_hidden_dims,
            proprio_hidden_dims,
            activation,
        )
        print(f"Actor (split encoder, {num_proprio_obs} proprio dims): {self.actor}")
        print(f"Critic (split encoder, {num_proprio_obs} proprio dims): {self.critic}")


def register() -> None:
    """Make ConeTrackActorCritic resolvable by OnPolicyRunner.

    rsl_rl instantiates the policy with a bare ``eval(class_name)`` evaluated in
    on_policy_runner's own module namespace (on_policy_runner.py:418), so a custom
    class is only reachable if it is injected there.
    """
    import rsl_rl.runners.on_policy_runner as _on_policy_runner

    _on_policy_runner.ConeTrackActorCritic = ConeTrackActorCritic
