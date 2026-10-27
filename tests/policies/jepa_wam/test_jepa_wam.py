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

"""End-to-end tests for JEPA-WAM on a tiny offline config (no Hub access)."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

import draccus  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.jepa_wam import (  # noqa: E402
    JEPAWAMConfig,
    JEPAWAMPolicy,
    make_jepa_wam_pre_post_processors,
)
from lerobot.policies.jepa_wam.configuration_jepa_wam import (  # noqa: E402
    JEPAWAMQwen2Config,
    JEPAWAMQwenBackboneConfig,
    VJEPA21EncoderConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERAS = ("observation.images.image", "observation.images.image2")
STATE_DIM, ACTION_DIM, CHUNK = 8, 7, 6
TASKS = ["pick up the red cube", "put the cube in the bowl"]


@pytest.fixture
def tokenizer_path(tmp_path: Path) -> str:
    words = ["[UNK]", "[PAD]", "pick", "up", "the", "red", "cube", "put", "bowl", "in"]
    tokenizer = Tokenizer(WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        path
    )
    return str(path)


def make_config(tokenizer_path: str, **backbone_overrides) -> JEPAWAMConfig:
    backbone = {
        "chunk_size": CHUNK,
        "encoder": VJEPA21EncoderConfig(hidden_size=48, num_layers=2, num_heads=3, image_size=32),
        "llm": JEPAWAMQwen2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=64,
        ),
        "tokenizer_name": tokenizer_path,
        "im_start_token_id": 10,
        "placeholder_token_id": 11,
        "num_action_tokens": 4,
        "action_hidden_size": 32,
        "action_num_layers": 2,
        "action_num_heads": 2,
        "action_attention_head_dim": 16,
        "num_future_tokens": 3,
        "action_max_seq_len": 16,
        **backbone_overrides,
    }
    return JEPAWAMConfig(
        device="cpu",
        n_action_steps=4,
        input_features={
            **{cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 48)) for cam in CAMERAS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        backbone=JEPAWAMQwenBackboneConfig(**backbone),
    )


STATS = {
    OBS_STATE: {"q01": -torch.ones(STATE_DIM), "q99": torch.ones(STATE_DIM)},
    ACTION: {"q01": -torch.ones(ACTION_DIM), "q99": torch.ones(ACTION_DIM)},
}


def make_batch(batch_size: int = 2, num_frames: int = 2) -> dict:
    torch.manual_seed(0)
    return {
        **{cam: torch.rand(batch_size, num_frames, 3, 40, 48) for cam in CAMERAS},
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        ACTION: torch.randn(batch_size, CHUNK, ACTION_DIM).clamp(-1, 1),
        "task": TASKS[:batch_size],
    }


def observation(batch: dict, index: slice) -> dict:
    """The inference view of a training batch: current frame only, no action."""
    return {
        key: (value[index, 0] if key in CAMERAS else value[index])
        for key, value in batch.items()
        if key != ACTION
    }


def test_training_reduces_both_losses(tokenizer_path):
    config = make_config(tokenizer_path, action_dropout=0.0, state_dropout=0.0)
    policy = JEPAWAMPolicy(config).train()
    preprocessor, _ = make_jepa_wam_pre_post_processors(config, STATS)
    batch = preprocessor(make_batch())
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)

    history = []
    for _ in range(60):
        torch.manual_seed(1)  # same flow noise and time every step, so the objective is fixed
        loss, logs = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(logs)

    assert all(isinstance(v, float) for v in history[0].values())
    for key in ("action_loss", "jepa_loss"):
        assert history[-1][key] < 0.5 * history[0][key], key
    # The frozen V-JEPA encoder and projector never receive gradients.
    assert all(p.grad is None for p in policy.model.vision_encoder.parameters())
    assert all(p.grad is None for p in policy.model.projector.parameters())


def test_mixed_prompt_lengths_match_unbatched(tokenizer_path):
    """Upstream slices the placeholders off a right-padded batch; ours must be batch-invariant."""
    config = make_config(tokenizer_path)
    policy = JEPAWAMPolicy(config).eval()
    preprocessor, _ = make_jepa_wam_pre_post_processors(config, STATS)
    batch = make_batch()
    noise = torch.randn(2, CHUNK, ACTION_DIM)

    batched_inputs = preprocessor(observation(batch, slice(0, 2)))
    mask = batched_inputs["observation.language.attention_mask"]
    assert not mask.all(), "the two prompts should tokenize to different lengths"
    batched = policy.predict_action_chunk(batched_inputs, noise=noise)
    for i in range(2):
        single = policy.predict_action_chunk(
            preprocessor(observation(batch, slice(i, i + 1))), noise=noise[i : i + 1]
        )
        torch.testing.assert_close(batched[i : i + 1], single, atol=1e-5, rtol=1e-5)


def test_save_and_reload_from_relative_path(tokenizer_path, tmp_path, monkeypatch):
    config = make_config(tokenizer_path)
    policy = JEPAWAMPolicy(config).eval()
    preprocessor, postprocessor = make_jepa_wam_pre_post_processors(config, STATS)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)

    monkeypatch.chdir(tmp_path)
    reloaded = JEPAWAMPolicy.from_pretrained("checkpoint").eval()
    reloaded_pre, reloaded_post = make_pre_post_processors(reloaded.config, pretrained_path="checkpoint")
    assert isinstance(reloaded.config.backbone, JEPAWAMQwenBackboneConfig)
    assert reloaded.config.backbone.num_action_tokens == config.backbone.num_action_tokens

    obs = observation(make_batch(), slice(0, 2))
    noise = torch.randn(2, CHUNK, ACTION_DIM)
    expected = postprocessor(policy.select_action(preprocessor(obs), noise=noise))
    actual = reloaded_post(reloaded.select_action(reloaded_pre(obs), noise=noise))
    torch.testing.assert_close(actual, expected)


def test_backbone_is_selected_from_the_cli():
    @dataclass
    class Wrapper:
        policy: PreTrainedConfig = field(default_factory=JEPAWAMConfig)

    parsed = draccus.parse(
        Wrapper,
        args=["--policy.type=jepa_wam", "--policy.backbone.type=qwen", "--policy.backbone.chunk_size=10"],
    )
    assert isinstance(parsed.policy.backbone, JEPAWAMQwenBackboneConfig)
    assert parsed.policy.chunk_size == 10
    assert draccus.decode(Wrapper, draccus.encode(parsed)) == parsed
