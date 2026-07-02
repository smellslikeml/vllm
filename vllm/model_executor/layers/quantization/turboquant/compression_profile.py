# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache compression / residency profiling for TurboQuant presets.

Quantifies the memory-vs-quality trade-off of each TurboQuant KV-cache
preset so that operators can reason about *cache residency* jointly with
task quality — i.e. how many more tokens fit in a fixed KV budget, and
what the quality cost is.

This framing (task quality, cache residency, and serving throughput must
be measured jointly for 4-bit KV caching on context-heavy / multi-round
agent workloads) is contribution #1 of UltraQuant: 4-bit KV Caching for
Context-Heavy Agents (arXiv:2606.20474). The byte accounting here is
derived from the live ``TurboQuantConfig`` packed-slot contract, not
hardcoded, so it stays correct as presets evolve. UltraQuant's FP4 /
UE8M0 / scaled-MFMA decode path is hardware-specific (CDNA4) and is out
of scope for this CPU-side planner.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)

# Documented perplexity-degradation cost of each preset relative to the
# uncompressed baseline (from the TurboQuantConfig docstring / preset
# characterization). Used to weigh quality against compression when
# recommending a preset. Keyed by preset name; presets absent here are
# treated as having unknown cost and excluded from quality-bounded picks.
PRESET_QUALITY_COST_PCT: dict[str, float] = {
    "turboquant_k8v4": 1.17,
    "turboquant_4bit_nc": 2.71,
    "turboquant_k3v4_nc": 10.63,
    "turboquant_3bit_nc": 20.59,
}

# Bytes per element for the full-precision and FP8 KV baselines that the
# compression ratios are quoted against.
_FP16_BYTES_PER_ELEM = 2
_FP8_BYTES_PER_ELEM = 1


@dataclass(frozen=True)
class CompressionProfile:
    """Per-preset KV-cache footprint and quality summary (one KV head).

    Attributes:
        preset: TurboQuant preset name (a ``--kv-cache-dtype`` value).
        head_dim: Attention head dimension the profile was computed for.
        bytes_per_token: Allocated KV bytes per token per KV head, i.e.
            the aligned combined key+value slot the cache manager reserves.
        ratio_vs_fp16: Compression factor over an fp16 K+V baseline.
        ratio_vs_fp8: Compression factor over an fp8 K+V baseline (the
            UltraQuant deployment anchor).
        quality_cost_pct: Documented perplexity degradation (percent), or
            ``None`` when uncharacterized.
    """

    preset: str
    head_dim: int
    bytes_per_token: int
    ratio_vs_fp16: float
    ratio_vs_fp8: float
    quality_cost_pct: float | None

    @property
    def residency_gain_vs_fp16(self) -> float:
        """How many more tokens fit in a fixed budget vs the fp16 baseline."""
        return self.ratio_vs_fp16


def profile_preset(cache_dtype: str, head_dim: int) -> CompressionProfile:
    """Build the compression/quality profile for a single preset.

    Args:
        cache_dtype: A TurboQuant preset name (e.g. ``turboquant_4bit_nc``).
        head_dim: Attention head dimension.

    Returns:
        The :class:`CompressionProfile` for the preset at ``head_dim``.

    Raises:
        ValueError: If ``cache_dtype`` is not a known TurboQuant preset.
    """
    cfg = TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim)
    # Aligned slot is what the paged-KV cache manager actually reserves.
    slot = cfg.slot_size_aligned
    fp16_baseline = _FP16_BYTES_PER_ELEM * head_dim * 2  # K + V
    fp8_baseline = _FP8_BYTES_PER_ELEM * head_dim * 2
    return CompressionProfile(
        preset=cache_dtype,
        head_dim=head_dim,
        bytes_per_token=slot,
        ratio_vs_fp16=fp16_baseline / slot,
        ratio_vs_fp8=fp8_baseline / slot,
        quality_cost_pct=PRESET_QUALITY_COST_PCT.get(cache_dtype),
    )


def all_profiles(head_dim: int) -> list[CompressionProfile]:
    """Profiles for every registered TurboQuant preset, densest last."""
    profiles = [profile_preset(name, head_dim) for name in TQ_PRESETS]
    profiles.sort(key=lambda p: p.ratio_vs_fp16)
    return profiles


def recommend_preset(
    head_dim: int,
    max_quality_cost_pct: float | None = None,
    min_compression: float | None = None,
) -> str | None:
    """Pick the densest preset meeting quality/compression constraints.

    Implements the joint quality-vs-residency decision: among presets that
    stay within ``max_quality_cost_pct`` and reach ``min_compression`` (vs
    fp16), return the one with the highest compression, breaking ties
    toward lower quality cost.

    Args:
        head_dim: Attention head dimension.
        max_quality_cost_pct: Upper bound on documented perplexity
            degradation. Presets with unknown cost are excluded when this
            bound is set. ``None`` disables the quality bound.
        min_compression: Minimum acceptable compression factor over fp16.
            ``None`` disables the compression floor.

    Returns:
        The recommended preset name, or ``None`` if none qualify.
    """
    candidates = all_profiles(head_dim)
    if max_quality_cost_pct is not None:
        candidates = [
            p
            for p in candidates
            if p.quality_cost_pct is not None
            and p.quality_cost_pct <= max_quality_cost_pct
        ]
    if min_compression is not None:
        candidates = [p for p in candidates if p.ratio_vs_fp16 >= min_compression]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda p: (p.ratio_vs_fp16, -(p.quality_cost_pct or 0.0)),
    )
    return best.preset


def build_residency_report(cache_dtype: str, head_dim: int) -> str:
    """One-line startup summary of the selected preset's footprint.

    Surfaces the memory-vs-quality trade-off at engine-config time and
    points at denser / higher-quality alternatives so the residency choice
    is explicit rather than implicit in the ``--kv-cache-dtype`` flag.
    """
    prof = profile_preset(cache_dtype, head_dim)
    cost = (
        f"{prof.quality_cost_pct:.2f}% quality cost"
        if prof.quality_cost_pct is not None
        else "uncharacterized quality cost"
    )
    msg = (
        f"TurboQuant KV cache preset {cache_dtype!r}: "
        f"{prof.bytes_per_token} B/token/head "
        f"({prof.ratio_vs_fp16:.2f}x vs fp16, {prof.ratio_vs_fp8:.2f}x vs fp8), "
        f"{cost}."
    )
    denser = recommend_preset(head_dim, min_compression=prof.ratio_vs_fp16 + 1e-6)
    if denser is not None:
        denser_prof = profile_preset(denser, head_dim)
        msg += (
            f" For more cache residency: {denser!r} "
            f"({denser_prof.ratio_vs_fp16:.2f}x vs fp16)."
        )
    return msg
