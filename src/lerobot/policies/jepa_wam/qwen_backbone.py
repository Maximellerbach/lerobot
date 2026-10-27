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
# Ported from JEPA-WAM (github.com/SpriteWithoutIce/JEPA_WAM, MIT License): prismatic/models/vlms/prismatic.py,
# prismatic/models/action_heads.py and prismatic/models/backbones/vision/vjepa_vit.py.

"""The paper's native JEPA-WAM model: V-JEPA 2.1 -> projector -> Qwen2.5-0.5B -> flow-matching DiT."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from .action_head import JEPAWAMActionHead
from .configuration_jepa_wam import JEPAWAMConfig, JEPAWAMQwenBackboneConfig
from .vjepa2_1 import VJEPA21Encoder, make_vjepa2_1_config

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2Config, Qwen2Model
else:
    Qwen2Config = None
    Qwen2Model = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VisualTokenCosineHead(nn.Module):
    """Maps LLM visual-token states onto the V-JEPA target space (training only)."""

    def __init__(self, llm_dim: int, target_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(llm_dim, 2 * target_dim)
        self.act_fn1 = nn.GELU()
        self.fc2 = nn.Linear(2 * target_dim, target_dim)

    def forward(self, llm_states: Tensor, target: Tensor) -> Tensor:
        """Per-sample mean cosine distance, ``(B,)``."""
        prediction = F.normalize(self.fc2(self.act_fn1(self.fc1(llm_states))), dim=-1)
        target = F.normalize(target.detach(), dim=-1)
        return (1 - (prediction * target).sum(dim=-1)).mean(dim=-1)


class JEPAWAMQwenModel(nn.Module):
    def __init__(self, config: JEPAWAMConfig):
        super().__init__()
        if not isinstance(config.backbone, JEPAWAMQwenBackboneConfig):
            raise TypeError(f"Expected a qwen backbone config, got {type(config.backbone).__name__}.")
        self.config = config
        backbone = config.backbone
        encoder, llm = backbone.encoder, backbone.llm
        self.camera_keys = list(config.image_features)

        self.vision_encoder = VJEPA21Encoder(
            make_vjepa2_1_config(
                hidden_size=encoder.hidden_size,
                num_hidden_layers=encoder.num_layers,
                num_attention_heads=encoder.num_heads,
                mlp_ratio=encoder.mlp_ratio,
                image_size=encoder.image_size,
                patch_size=encoder.patch_size,
                tubelet_size=encoder.tubelet_size,
            )
        )
        self.projector = nn.Sequential(
            nn.Linear(encoder.hidden_size, llm.hidden_size),
            nn.GELU(),
            nn.Linear(llm.hidden_size, llm.hidden_size),
        )
        qwen_config = Qwen2Config(
            hidden_size=llm.hidden_size,
            intermediate_size=llm.intermediate_size,
            num_hidden_layers=llm.num_hidden_layers,
            num_attention_heads=llm.num_attention_heads,
            num_key_value_heads=llm.num_key_value_heads,
            vocab_size=llm.vocab_size,
            max_position_embeddings=llm.max_position_embeddings,
            rope_theta=llm.rope_theta,
            rms_norm_eps=llm.rms_norm_eps,
            hidden_act="silu",
            tie_word_embeddings=True,
            use_sliding_window=False,
        )
        qwen_config._attn_implementation = "sdpa"
        self.llm = Qwen2Model(qwen_config)
        state_feature, action_feature = config.robot_state_feature, config.action_feature
        if state_feature is None or action_feature is None:
            raise ValueError("JEPA-WAM needs 'observation.state' and 'action' features.")
        self.action_head = JEPAWAMActionHead(
            backbone,
            state_dim=state_feature.shape[0],
            action_dim=action_feature.shape[0],
            cross_attention_dim=llm.hidden_size,
        )
        self.visual_token_cosine_head = VisualTokenCosineHead(llm.hidden_size, encoder.hidden_size)

        if backbone.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        if backbone.freeze_projector:
            self.projector.requires_grad_(False)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.backbone.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    def _prepare_frames(self, frames: Tensor) -> Tensor:
        """``(N, T, C, H, W)`` in [0, 1] -> bicubic resize to the encoder resolution, ImageNet-normalized."""
        n, t = frames.shape[:2]
        size = self.config.backbone.encoder.image_size
        flat = frames.flatten(0, 1).float()
        if flat.shape[-2:] != (size, size):
            flat = F.interpolate(flat, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
        flat = flat.clamp(0.0, 1.0).unflatten(0, (n, t))
        return ((flat - self.image_mean) / self.image_std).to(self.projector[0].weight.dtype)

    def _encode(self, frames: Tensor) -> Tensor:
        """``(B, V, T, C, H, W)`` -> ``(B, V * tokens_per_frame, D)``. ``T=1`` is the image path."""
        b, v = frames.shape[:2]
        with torch.set_grad_enabled(
            not self.config.backbone.freeze_vision_encoder and torch.is_grad_enabled()
        ):
            out = self.vision_encoder(self._prepare_frames(frames.flatten(0, 1))).last_hidden_state
        return out.unflatten(0, (b, v)).flatten(1, 2)

    def _stack_cameras(self, batch: dict[str, Tensor]) -> Tensor:
        """Cameras in config order, ``(B, V, T, C, H, W)``; a 4-D camera tensor is a single frame."""
        frames = [batch[key] if batch[key].ndim == 5 else batch[key].unsqueeze(1) for key in self.camera_keys]
        return torch.stack(frames, dim=1)

    def _llm_states(self, batch: dict[str, Tensor], visual_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Run Qwen on ``[<|im_start|>][visual][prompt][placeholders]``; return visual and placeholder states.

        The prompt is left-padded, so its pads sit between the visual tokens and the prompt. They are masked
        and skipped by the position ids, which makes a padded sample see exactly its unpadded sequence and
        keeps the placeholders at a fixed ``[-num_action_tokens:]`` slice.
        """
        backbone = self.config.backbone
        tokens, text_mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        b, device = tokens.shape[0], tokens.device
        special = torch.tensor([backbone.im_start_token_id, backbone.placeholder_token_id], device=device)
        embed = self.llm.get_input_embeddings()
        im_start, placeholder = embed(special).unbind(0)
        projected = self.projector(visual_tokens)
        num_visual, num_actions = projected.shape[1], backbone.num_action_tokens
        inputs = torch.cat(
            [
                im_start.expand(b, 1, -1),
                projected,
                embed(tokens),
                placeholder.expand(b, num_actions, -1),
            ],
            dim=1,
        )
        ones = torch.ones(b, 1 + num_visual, dtype=torch.bool, device=device)
        mask = torch.cat(
            [ones, text_mask, torch.ones(b, num_actions, dtype=torch.bool, device=device)], dim=1
        )
        position_ids = (mask.long().cumsum(dim=1) - 1).clamp(min=0)
        hidden = self.llm(
            inputs_embeds=inputs, attention_mask=mask, position_ids=position_ids
        ).last_hidden_state
        return hidden[:, 1 : 1 + num_visual], hidden[:, -num_actions:]

    def compute_loss(
        self, batch: dict[str, Tensor], reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, float]]:
        cameras = self._stack_cameras(batch)
        visual_tokens = self._encode(cameras[:, :, :1])
        visual_states, action_memory = self._llm_states(batch, visual_tokens)
        action_loss = self.action_head(action_memory, batch[OBS_STATE], batch[ACTION], reduction="none")
        loss = action_loss
        logs = {"action_loss": action_loss.mean().item()}
        weight = self.config.backbone.jepa_loss_weight
        if weight > 0:
            if cameras.shape[2] < 2:
                raise ValueError(
                    "The JEPA objective needs the [t, t + jepa_target_frame_offset] frame pair per camera; "
                    "got a single frame. Train from a dataset so `image_observation_delta_indices` applies, "
                    "or set `--policy.backbone.jepa_loss_weight=0`."
                )
            with torch.no_grad():
                target = self._encode(cameras[:, :, :2])
            jepa_loss = self.visual_token_cosine_head(visual_states, target)
            loss = loss + weight * jepa_loss
            logs["jepa_loss"] = jepa_loss.mean().item()
        return (loss.mean() if reduction == "mean" else loss), logs

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        cameras = self._stack_cameras(batch)[:, :, :1]
        _, action_memory = self._llm_states(batch, self._encode(cameras))
        return self.action_head.sample_actions(
            action_memory,
            batch[OBS_STATE],
            chunk_size=self.config.chunk_size,
            num_steps=self.config.num_inference_steps,
            noise=noise,
        )
