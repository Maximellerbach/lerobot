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

"""JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling (arXiv:2608.09381)."""

from collections import deque
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from ..chunk import ChunkPolicySpec
from ..pretrained import PreTrainedPolicy
from .configuration_jepa_wam import JEPAWAMConfig, JEPAWAMQwenBackboneConfig
from .qwen_backbone import JEPAWAMQwenModel


class JEPAWAMPolicy(PreTrainedPolicy):
    config_class = JEPAWAMConfig
    name = "jepa_wam"
    _fsdp_wrap_modules = ["Qwen2DecoderLayer", "VJEPA2Layer", "BasicTransformerBlock"]

    def __init__(self, config: JEPAWAMConfig, **kwargs: Any):
        require_package("transformers", extra="jepa_wam")
        require_package("diffusers", extra="jepa_wam")
        super().__init__(config)
        config.validate_features()
        self.config = config
        if isinstance(config.backbone, JEPAWAMQwenBackboneConfig):
            self.model = JEPAWAMQwenModel(config)
        else:
            raise ValueError(f"Unsupported JEPA-WAM backbone: {type(config.backbone).__name__}")
        if config.dtype is not None:
            self.to(config.dtype)
        self.reset()

    def reset(self) -> None:
        self._queues: dict[str, deque[Tensor]] = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        loss, logs = self.model.compute_loss(batch, reduction=reduction)
        return loss, logs

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        return self.model.sample_actions(batch, noise=noise).float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """The future camera frame is a training target only; inference needs the current observation."""
        return ChunkPolicySpec(
            prediction_steps=self.config.chunk_size, execution_steps=self.config.n_action_steps
        )

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA on every Qwen linear (the paper's recipe); the action and cosine heads train in full."""
        return {
            "target_modules": r"model\.llm\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
            "modules_to_save": ["model.action_head", "model.visual_token_cosine_head"],
        }
