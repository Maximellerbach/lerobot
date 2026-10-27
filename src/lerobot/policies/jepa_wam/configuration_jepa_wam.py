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

from dataclasses import dataclass, field

import draccus
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@dataclass
class VJEPA21EncoderConfig:
    """Frozen V-JEPA 2.1 encoder architecture. Defaults are ViT-L/16 at 384 px."""

    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    image_size: int = 384
    patch_size: int = 16
    tubelet_size: int = 2


@dataclass
class JEPAWAMBackboneConfig(draccus.ChoiceRegistry):
    """Backbone the JEPA objective is instantiated on, chosen with `--policy.backbone.type`."""

    chunk_size: int = 20
    # Weight of the JEPA objective (cosine distance to frozen V-JEPA features of the [t, t + offset] pair).
    jepa_loss_weight: float = 0.5

    optimizer_lr: float = 2e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_200
    scheduler_decay_steps: int = 40_000
    scheduler_decay_lr: float = 1e-5


@dataclass
class JEPAWAMQwen2Config:
    """Qwen2.5-0.5B architecture, so the language model builds from config without a download."""

    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    # Qwen2.5's 151665 tokens, padded to a multiple of 64 as in the released checkpoint.
    vocab_size: int = 151680
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6


@JEPAWAMBackboneConfig.register_subclass("qwen")
@dataclass
class JEPAWAMQwenBackboneConfig(JEPAWAMBackboneConfig):
    """The paper's native model: V-JEPA 2.1 ViT-L -> projector -> Qwen2.5-0.5B -> flow-matching DiT.

    Qwen reads `[<|im_start|>][visual tokens][prompt][placeholders]`. The DiT is conditioned on the
    placeholder states, and the JEPA objective regresses the visual-token states onto V-JEPA features
    of the `[t, t + offset]` frame pair.
    """

    encoder: VJEPA21EncoderConfig = field(default_factory=VJEPA21EncoderConfig)
    llm: JEPAWAMQwen2Config = field(default_factory=JEPAWAMQwen2Config)
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    tokenizer_max_length: int = 256
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    # `{task}` is replaced by the lowercased task string.
    prompt_template: str = "What action should the robot take to {task}?"
    im_start_token_id: int = 151644
    # An ordinary vocabulary token repeated as the action-memory slots (upstream ACTION_TOKEN_BEGIN_IDX).
    placeholder_token_id: int = 151386
    num_action_tokens: int = 64

    freeze_vision_encoder: bool = True
    freeze_projector: bool = True

    action_hidden_size: int = 1024
    action_num_layers: int = 16
    action_num_heads: int = 32
    action_attention_head_dim: int = 48
    action_dropout: float = 0.2
    state_dropout: float = 0.5
    num_future_tokens: int = 32
    action_max_seq_len: int = 1024
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000


@PreTrainedConfig.register_subclass("jepa_wam")
@dataclass
class JEPAWAMConfig(PreTrainedConfig):
    """JEPA-WAM: a policy trained with a training-only JEPA objective on future V-JEPA 2.1 features.

    Args:
        backbone: Which instantiation of the method to build (`--policy.backbone.type`).
        n_action_steps: Actions executed from each predicted chunk.
        num_inference_steps: Euler steps of the flow-matching sampler.
        jepa_target_frame_offset: Frames between the current observation and the future frame of the JEPA target.
    """

    backbone: JEPAWAMBackboneConfig = field(default_factory=JEPAWAMQwenBackboneConfig)
    n_action_steps: int = 8
    num_inference_steps: int = 4
    jepa_target_frame_offset: int = 31

    dtype: torch.dtype | None = torch.float32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.jepa_target_frame_offset < 1:
            raise ValueError("jepa_target_frame_offset must be at least 1.")

    @property
    def chunk_size(self) -> int:
        return self.backbone.chunk_size

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("JEPA-WAM requires at least one camera.")
        if self.action_feature is None:
            raise ValueError("JEPA-WAM requires 'action' in output_features.")
        if self.robot_state_feature is None:
            raise ValueError(
                "JEPA-WAM conditions its action head on the robot state; 'observation.state' is required."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        backbone = self.backbone
        return AdamWConfig(
            lr=backbone.optimizer_lr,
            betas=backbone.optimizer_betas,
            eps=backbone.optimizer_eps,
            weight_decay=backbone.optimizer_weight_decay,
            grad_clip_norm=backbone.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        backbone = self.backbone
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=backbone.optimizer_lr,
            decay_lr=backbone.scheduler_decay_lr,
            num_warmup_steps=backbone.scheduler_warmup_steps,
            num_decay_steps=backbone.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int]:
        # The future frame is only the JEPA target; inference consumes the current frame alone.
        return [0, self.jepa_target_frame_offset]

    @property
    def state_observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
