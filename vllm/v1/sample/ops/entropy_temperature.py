# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entropy-aware decoding temperature scaling.

Adapts the per-request decoding temperature to the model's token-level
predictive uncertainty. The transform is argmax-invariant (it only rescales
logits by a positive per-row factor), so it never changes greedy outputs and
slots in next to ``Sampler.apply_temperature``.

Motivation comes from ReSET (Reasoning-step Entropy-based temperature
Scaling, https://arxiv.org/abs/2606.13233): low-precision (e.g. NVFP4)
inference makes confident, low-entropy "symbolic" tokens sample incorrectly,
while over-concentrating probability mass on a few tokens at high-uncertainty
reasoning steps. Adapting the temperature to the current step's entropy
counteracts both: sharpen where the model is confident, smooth where it is
uncertain. Here we implement the online token-level signal, which needs no
per-request history; the paper's step-level accumulation and its CUDA-core
NVFP4 kernel are out of scope for this Python-level transform.
"""

import os
from dataclasses import dataclass

import torch

_EPS = 1e-10


@dataclass(frozen=True)
class EntropyTemperatureConfig:
    """Tuning for :func:`entropy_scaled_temperature`.

    Attributes:
        target_entropy: Normalized-entropy pivot in ``[0, 1]``. Rows above it
            get their temperature raised, rows below it lowered.
        strength: How aggressively the scale departs from ``1.0`` per unit of
            normalized-entropy deviation from ``target_entropy``.
        min_scale: Lower clamp on the temperature multiplier.
        max_scale: Upper clamp on the temperature multiplier.
    """

    target_entropy: float = 0.5
    strength: float = 1.0
    min_scale: float = 0.5
    max_scale: float = 2.0

    @classmethod
    def from_env(cls) -> "EntropyTemperatureConfig":
        """Build a config from ``VLLM_ENTROPY_TEMPERATURE_*`` env overrides."""
        return cls(
            target_entropy=float(os.getenv("VLLM_ENTROPY_TEMPERATURE_TARGET", "0.5")),
            strength=float(os.getenv("VLLM_ENTROPY_TEMPERATURE_STRENGTH", "1.0")),
            min_scale=float(os.getenv("VLLM_ENTROPY_TEMPERATURE_MIN_SCALE", "0.5")),
            max_scale=float(os.getenv("VLLM_ENTROPY_TEMPERATURE_MAX_SCALE", "2.0")),
        )


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of ``softmax(logits)`` per row, normalized to ``[0, 1]``.

    Normalization is by ``log(vocab_size)`` (entropy of the uniform
    distribution), making the result comparable across vocabularies. Masked
    logits (``-inf``) contribute zero and do not poison the sum.

    Args:
        logits: ``[num_rows, vocab_size]`` logits (any float dtype).

    Returns:
        ``[num_rows]`` tensor of normalized entropies.
    """
    logp = logits.log_softmax(dim=-1)
    p = logp.exp()
    # p * logp is 0 wherever p == 0; guard the 0 * -inf -> nan case.
    contrib = torch.where(p > 0, p * logp, torch.zeros_like(p))
    entropy = -contrib.sum(dim=-1)
    vocab_size = logits.shape[-1]
    norm = torch.log(torch.tensor(float(vocab_size), device=logits.device))
    return (entropy / norm.clamp_min(_EPS)).clamp_(0.0, 1.0)


def entropy_scaled_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    config: EntropyTemperatureConfig,
) -> torch.Tensor:
    """Scale per-request temperature by token-level predictive entropy.

    The multiplier is a clamped linear map of the deviation of each row's
    normalized entropy from ``config.target_entropy``::

        scale = clamp(1 + strength * (H_norm - target_entropy), min_scale, max_scale)
        new_temperature = temperature * scale

    High-entropy rows (uncertain steps) get a higher temperature to avoid
    over-concentration; low-entropy rows (confident symbolic tokens) get a
    lower temperature to suppress quantization-induced mis-sampling. The map
    is positive and row-wise, so applying it before ``apply_temperature``
    leaves the argmax (and thus greedy requests) unchanged.

    Args:
        logits: ``[num_rows, vocab_size]`` logits, before temperature.
        temperature: ``[num_rows]`` base per-request temperature.
        config: Scaling parameters.

    Returns:
        A new ``[num_rows]`` temperature tensor (input is not mutated).
    """
    h_norm = normalized_entropy(logits).to(temperature.dtype)
    scale = 1.0 + config.strength * (h_norm - config.target_entropy)
    scale = scale.clamp_(config.min_scale, config.max_scale)
    return temperature * scale
