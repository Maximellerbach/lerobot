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
# V-JEPA 2.1 differences ported from Meta's facebookresearch/vjepa2 (app/vjepa_2_1/models), MIT License,
# Copyright (c) Meta Platforms, Inc. and affiliates. Parameter names follow huggingface/transformers#45497
# so this file can be deleted once transformers supports V-JEPA 2.1.

"""V-JEPA 2.1 encoder as a thin subclass of the transformers V-JEPA 2 encoder.

V-JEPA 2.1 changes five things relative to the V-JEPA 2 encoder in transformers:

1. single images go through their own tubelet-1 patch embedding instead of a duplicated frame,
2. learned image / video modality embeddings are added after patch embedding,
3. RoPE rotates interleaved pairs (``repeat_interleave``) instead of tiled halves,
4. RoPE height/width positions are rescaled to the pretraining grid (``interpolate_rope``),
5. the output norm is the last of the hierarchical norms, which maps onto ``layernorm``.
"""

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import (
        VJEPA2Embeddings,
        VJEPA2Encoder,
        VJEPA2PatchEmbeddings3D,
        VJEPA2RopeAttention,
    )
else:
    VJEPA2Config = None
    VJEPA2Embeddings = nn.Module
    VJEPA2Encoder = nn.Module
    VJEPA2PatchEmbeddings3D = nn.Module
    VJEPA2RopeAttention = nn.Module

# Meta's RoPE reference grid: positions are rescaled to a 256 px / 16 px pretraining grid.
PRETRAINED_ROPE_GRID_SIZE = 16


def rotate_interleaved(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """RoPE rotation on interleaved channel pairs ``(x0, x1), (x2, x3), ...`` (V-JEPA 2.1)."""
    head_dim = x.shape[-1]
    omega = torch.arange(head_dim // 2, dtype=x.dtype, device=x.device)
    omega /= head_dim / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().repeat_interleave(2, dim=-1)
    emb_cos = freq.cos().repeat_interleave(2, dim=-1)
    y1, y2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    rotated = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (rotated * emb_sin)


class VJEPA21RopeAttention(VJEPA2RopeAttention):
    def get_position_ids(self, x, masks=None):
        frame_ids, height_ids, width_ids = super().get_position_ids(x, masks=masks)
        scale = (PRETRAINED_ROPE_GRID_SIZE - 1.0) / max(self.grid_size - 1.0, 1.0)
        return frame_ids, height_ids.float() * scale, width_ids.float() * scale

    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        parts = []
        for dim, pos in ((self.d_dim, d_mask), (self.h_dim, h_mask), (self.w_dim, w_mask)):
            parts.append(rotate_interleaved(qk[..., s : s + dim], pos))
            s += dim
        if s < self.attention_head_size:
            parts.append(qk[..., s:])
        # The fp32 positions promote the rotation; attention runs in the input dtype.
        return torch.cat(parts, dim=-1).to(qk.dtype)


class VJEPA21Embeddings(VJEPA2Embeddings):
    def __init__(self, config: VJEPA2Config, hidden_size: int = 1024):
        super().__init__(config, hidden_size=hidden_size)
        image_config = copy.copy(config)
        image_config.tubelet_size = 1
        self.patch_embeddings_img = VJEPA2PatchEmbeddings3D(image_config, hidden_size=hidden_size)
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """``pixel_values_videos``: ``(batch, frames, channels, height, width)``; one frame is an image."""
        is_image = pixel_values_videos.shape[1] == 1
        patch_embeddings = self.patch_embeddings_img if is_image else self.patch_embeddings
        pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4).to(
            patch_embeddings.proj.weight.dtype
        )
        embeddings = patch_embeddings(pixel_values_videos)
        return embeddings + (self.img_mod_embed if is_image else self.video_mod_embed)


class VJEPA21Encoder(VJEPA2Encoder):
    """V-JEPA 2.1 encoder. Output is ``last_hidden_state``: ``(batch, frames // tubelet * grid**2, hidden)``."""

    def __init__(self, config: VJEPA2Config):
        super().__init__(config)
        self.embeddings = VJEPA21Embeddings(config, hidden_size=config.hidden_size)
        for layer in self.layer:
            layer.attention = VJEPA21RopeAttention(config, config.hidden_size, config.num_attention_heads)


def make_vjepa2_1_config(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    mlp_ratio: float,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
) -> VJEPA2Config:
    """Encoder-only V-JEPA 2.1 config (the predictor fields are never built)."""
    config = VJEPA2Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        mlp_ratio=mlp_ratio,
        crop_size=image_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=tubelet_size,
        layer_norm_eps=1e-6,
        hidden_act="gelu",
        qkv_bias=True,
        drop_path_rate=0.0,
        attention_probs_dropout_prob=0.0,
    )
    config._attn_implementation = "sdpa"
    return config
