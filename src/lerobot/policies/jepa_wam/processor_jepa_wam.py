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
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_jepa_wam import JEPAWAMConfig, JEPAWAMQwenBackboneConfig

# Normalization modes that map the data range onto [-1, 1], where clamping is a saturation.
_BOUNDED_MODES = {NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES, NormalizationMode.QUANTILE10}


@ProcessorStepRegistry.register(name="jepa_wam_prompt")
@dataclass
class JEPAWAMPromptProcessorStep(ProcessorStep):
    """Render the task into JEPA-WAM's Qwen chat prompt.

    The leading ``<|im_start|>`` is left out: the model inserts it as a token, then the visual tokens,
    then this text. ``<|im_start|>`` is a special token, so the text after it tokenizes identically.
    """

    system_prompt: str = ""
    prompt_template: str = "{task}"
    task_key: str = "task"

    def render(self, task: str) -> str:
        message = self.prompt_template.format(task=task.strip().lower())
        return (
            f"system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{message}<|im_end|>\n"
            "<|im_start|>assistant\n <|im_end|><|endoftext|>"
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or complementary_data.get(self.task_key) is None:
            raise ValueError(f"JEPA-WAM needs a task string in complementary data '{self.task_key}'.")
        tasks = complementary_data[self.task_key]
        rendered = self.render(tasks) if isinstance(tasks, str) else [self.render(t) for t in tasks]
        transition = transition.copy()
        transition[TransitionKey.COMPLEMENTARY_DATA] = {**complementary_data, self.task_key: rendered}
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "prompt_template": self.prompt_template,
            "task_key": self.task_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="jepa_wam_clamp_normalized")
@dataclass
class JEPAWAMClampNormalizedProcessorStep(ProcessorStep):
    """Clamp normalized state/action to [-1, 1], as upstream does after its q01/q99 normalization.

    Only valid right after a bounded normalization (MIN_MAX / QUANTILES / QUANTILE10); the factory only
    lists keys normalized that way.
    """

    observation_keys: list[str] = field(default_factory=list)
    clamp_action: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None and self.observation_keys:
            observation = dict(observation)
            for key in self.observation_keys:
                if isinstance(observation.get(key), torch.Tensor):
                    observation[key] = observation[key].clamp(-1.0, 1.0)
            transition[TransitionKey.OBSERVATION] = observation
        action = transition.get(TransitionKey.ACTION)
        if self.clamp_action and isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"observation_keys": list(self.observation_keys), "clamp_action": self.clamp_action}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_jepa_wam_pre_post_processors(
    config: JEPAWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    backbone = config.backbone
    if not isinstance(backbone, JEPAWAMQwenBackboneConfig):
        raise ValueError(f"Unsupported JEPA-WAM backbone: {type(backbone).__name__}")
    steps = make_default_policy_processor_steps(config, dataset_stats)

    def bounded(feature_type: FeatureType) -> bool:
        return config.normalization_mapping.get(feature_type.value) in _BOUNDED_MODES

    clamp = JEPAWAMClampNormalizedProcessorStep(
        observation_keys=[OBS_STATE]
        if bounded(FeatureType.STATE) and OBS_STATE in (config.input_features or {})
        else [],
        clamp_action=bounded(FeatureType.ACTION) and ACTION in (config.output_features or {}),
    )
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        clamp,
        JEPAWAMPromptProcessorStep(
            system_prompt=backbone.system_prompt, prompt_template=backbone.prompt_template
        ),
        TokenizerProcessorStep(
            tokenizer_name=backbone.tokenizer_name,
            max_length=backbone.tokenizer_max_length,
            padding_side="left",
            padding="longest",
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
