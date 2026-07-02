# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for entropy-aware (ReSET) decoding temperature scaling.

Covers the standalone transform and its wiring into ``Sampler.sample``.
"""

import math

import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.entropy_temperature import (
    EntropyTemperatureConfig,
    entropy_scaled_temperature,
    normalized_entropy,
)
from vllm.v1.sample.sampler import Sampler

VOCAB_SIZE = 512


def _peaked_logits(batch_size: int) -> torch.Tensor:
    logits = torch.full((batch_size, VOCAB_SIZE), -20.0)
    logits[:, 0] = 20.0
    return logits


def _uniform_logits(batch_size: int) -> torch.Tensor:
    return torch.zeros(batch_size, VOCAB_SIZE)


def test_normalized_entropy_bounds():
    # A near-deterministic distribution has ~0 entropy; uniform has ~1.
    low = normalized_entropy(_peaked_logits(3))
    high = normalized_entropy(_uniform_logits(3))
    assert torch.all(low < 0.05)
    assert torch.allclose(high, torch.ones(3), atol=1e-4)


def test_normalized_entropy_ignores_masked_logits():
    # -inf (masked) entries must not produce NaNs in the entropy sum.
    logits = torch.zeros(2, VOCAB_SIZE)
    logits[:, VOCAB_SIZE // 2 :] = float("-inf")
    h = normalized_entropy(logits)
    assert torch.isfinite(h).all()
    # Half the vocab is live and uniform -> H = log(V/2)/log(V).
    expected = math.log(VOCAB_SIZE / 2) / math.log(VOCAB_SIZE)
    assert torch.allclose(h, torch.full((2,), expected), atol=1e-4)


def test_entropy_scaled_temperature_direction():
    cfg = EntropyTemperatureConfig(
        target_entropy=0.5, strength=1.0, min_scale=0.5, max_scale=2.0
    )
    base = torch.ones(2)
    logits = torch.cat([_peaked_logits(1), _uniform_logits(1)], dim=0)
    scaled = entropy_scaled_temperature(logits, base, cfg)
    # Low-entropy row sharpened (temp down), high-entropy row smoothed (up).
    assert scaled[0] < base[0]
    assert scaled[1] > base[1]
    # Multipliers stay within the configured clamp.
    assert torch.all(scaled >= base * cfg.min_scale - 1e-6)
    assert torch.all(scaled <= base * cfg.max_scale + 1e-6)


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_ENTROPY_TEMPERATURE_TARGET", "0.3")
    monkeypatch.setenv("VLLM_ENTROPY_TEMPERATURE_STRENGTH", "2.0")
    monkeypatch.setenv("VLLM_ENTROPY_TEMPERATURE_MIN_SCALE", "0.25")
    monkeypatch.setenv("VLLM_ENTROPY_TEMPERATURE_MAX_SCALE", "4.0")
    cfg = EntropyTemperatureConfig.from_env()
    assert cfg.target_entropy == 0.3
    assert cfg.strength == 2.0
    assert cfg.min_scale == 0.25
    assert cfg.max_scale == 4.0


def _sampling_metadata(temperature: torch.Tensor) -> SamplingMetadata:
    batch_size = temperature.shape[0]
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        prompt_token_ids=None,
        output_token_ids=[[] for _ in range(batch_size)],
        spec_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_sampler_wiring_is_argmax_invariant():
    # Enabling entropy scaling must not change the picked token for greedy
    # (temperature == 0) requests routed through the real Sampler.sample.
    sampler = Sampler()
    sampler.entropy_temp_config = EntropyTemperatureConfig()

    logits = torch.randn(4, VOCAB_SIZE)
    expected = logits.argmax(dim=-1)
    # All-greedy rows take the early return; use a mixed batch so the
    # entropy-scaling + apply_temperature path actually runs.
    temperature = torch.tensor([0.0, 0.0, 1.0, 0.0])
    md = _sampling_metadata(temperature)

    sampled, _ = sampler.sample(logits.clone(), md)
    greedy_rows = (temperature == 0.0).nonzero(as_tuple=True)[0]
    assert torch.equal(sampled[greedy_rows], expected[greedy_rows])


def test_sampler_wiring_disabled_by_default():
    # Without an explicit config the new path is inert.
    sampler = Sampler()
    assert sampler.entropy_temp_config is None
