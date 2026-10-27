#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from JEPA-WAM (github.com/SpriteWithoutIce/JEPA_WAM, MIT License):
# prismatic/models/flow_gr00t_action_head.py and prismatic/models/flow_matching_head/*, which derive from
# NVIDIA Isaac GR00T, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
# Module structure and parameter names are kept so the released weights load unchanged.

"""JEPA-WAM flow-matching action head: a GR00T-style DiT conditioned on the LLM's action-placeholder states."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import (
    FlowConvention,
    euler_integrate,
    make_flow_matching_inputs,
    sample_beta,
    sample_noise,
)
from lerobot.utils.import_utils import _diffusers_available

from .configuration_jepa_wam import JEPAWAMQwenBackboneConfig

if TYPE_CHECKING or _diffusers_available:
    from diffusers.models.attention import Attention, FeedForward
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
else:
    Attention = None
    FeedForward = None
    TimestepEmbedding = None
    Timesteps = None


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(torch.relu(self.layer1(x)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_size)  # noqa: N815
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # noqa: N815
        self.W3 = nn.Linear(hidden_size, hidden_size)  # noqa: N815
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        """``actions``: ``(B, T, action_dim)``; ``timesteps``: ``(B,)`` bucket indices."""
        timesteps = timesteps.unsqueeze(1).expand(-1, actions.shape[1])
        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = F.silu(self.W2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.W3(x)


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, elementwise_affine=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """AdaLN block with one attention (cross-attention when ``cross_attention_dim`` is set) and a GELU MLP."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        cross_attention_dim: int | None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=True,
            cross_attention_dim=cross_attention_dim,
            out_bias=True,
        )
        self.norm3 = nn.LayerNorm(dim, 1e-5, elementwise_affine=False)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="gelu-approximate", final_dropout=True)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, temb: Tensor, encoder_hidden_states: Tensor | None) -> Tensor:
        attn_output = self.attn1(self.norm1(hidden_states, temb), encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.final_dropout(attn_output) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states


class DiT(nn.Module):
    """Even blocks cross-attend to the conditioning sequence, odd blocks self-attend."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cross_attention_dim: int,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim if idx % 2 == 0 else None,
                )
                for idx in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor, timestep: Tensor) -> Tensor:
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        for idx, block in enumerate(self.transformer_blocks):
            context = encoder_hidden_states if idx % 2 == 0 else None
            hidden_states = block(hidden_states, temb, context)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class JEPAWAMActionHead(nn.Module):
    """Flow matching with noise at ``t=0`` and data at ``t=1`` (GR00T convention).

    The token sequence is ``[state, future tokens, noisy actions]``; the velocity is read off the action positions.
    """

    def __init__(
        self, config: JEPAWAMQwenBackboneConfig, state_dim: int, action_dim: int, cross_attention_dim: int
    ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        input_embedding_dim = config.action_num_heads * config.action_attention_head_dim
        hidden_size = config.action_hidden_size
        self.model = DiT(
            num_attention_heads=config.action_num_heads,
            attention_head_dim=config.action_attention_head_dim,
            output_dim=hidden_size,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout,
            cross_attention_dim=cross_attention_dim,
        )
        self.state_encoder = MLP(state_dim, hidden_size, input_embedding_dim)
        self.state_dropout = nn.Dropout(p=config.state_dropout)
        self.action_encoder = ActionEncoder(action_dim, input_embedding_dim)
        self.action_decoder = MLP(hidden_size, hidden_size, action_dim)
        self.future_tokens = nn.Embedding(config.num_future_tokens, input_embedding_dim)
        self.position_embedding = nn.Embedding(config.action_max_seq_len, input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def _predict_velocity(self, memory: Tensor, actions: Tensor, timesteps: Tensor, state: Tensor) -> Tensor:
        action_features = self.action_encoder(actions, timesteps)
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        state_features = self.state_dropout(self.state_encoder(state).unsqueeze(1))
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        tokens = torch.cat((state_features, future_tokens, action_features), dim=1)
        output = self.model(hidden_states=tokens, encoder_hidden_states=memory, timestep=timesteps)
        return self.action_decoder(output)[:, -actions.shape[1] :]

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        sample = sample_beta(
            self.config.noise_beta_alpha, self.config.noise_beta_beta, batch_size, device, dtype=dtype
        )
        return (self.config.noise_s - sample) / self.config.noise_s

    def forward(self, memory: Tensor, state: Tensor, actions: Tensor, reduction: str = "mean") -> Tensor:
        """Flow-matching loss averaged over every step and dim, as upstream (padded steps included)."""
        dtype = memory.dtype
        state, actions = state.to(dtype), actions.to(dtype)
        noise = sample_noise(actions.shape, actions.device, dtype=dtype)
        time = self.sample_time(actions.shape[0], actions.device, dtype)
        x_t, velocity, model_time = make_flow_matching_inputs(
            actions, noise, time, convention=FlowConvention.NOISE_AT_ZERO
        )
        buckets = (model_time * self.config.num_timestep_buckets).long()
        pred = self._predict_velocity(memory, x_t, buckets, state)
        loss = (pred - velocity) ** 2
        return loss.mean() if reduction == "mean" else loss.mean(dim=(1, 2))

    @torch.no_grad()
    def sample_actions(
        self, memory: Tensor, state: Tensor, chunk_size: int, num_steps: int, noise: Tensor | None = None
    ) -> Tensor:
        dtype = memory.dtype
        state = state.to(dtype)
        if noise is None:
            noise = sample_noise((memory.shape[0], chunk_size, self.action_dim), memory.device, dtype=dtype)
        buckets_per_step = [
            int(step / num_steps * self.config.num_timestep_buckets) for step in range(num_steps)
        ]

        def denoise(x_t: Tensor, _time: Tensor, step: int) -> Tensor:
            timesteps = torch.full(
                (x_t.shape[0],), buckets_per_step[step], device=x_t.device, dtype=torch.long
            )
            return self._predict_velocity(memory, x_t, timesteps, state)

        return euler_integrate(
            denoise, noise.to(dtype), num_steps, convention=FlowConvention.NOISE_AT_ZERO, step_aware=True
        )
