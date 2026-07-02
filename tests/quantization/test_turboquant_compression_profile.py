# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the TurboQuant KV-cache compression/residency profiler.

The profiler must stay anchored to the live ``TurboQuantConfig`` packed-slot
contract (a NON-new module) rather than hardcoding byte counts, so these
tests cross-check profiler output against the config the rest of the engine
uses.

Run: .venv/bin/python -m pytest \
    tests/quantization/test_turboquant_compression_profile.py -v
"""

import pytest

from vllm.model_executor.layers.quantization.turboquant.compression_profile import (
    PRESET_QUALITY_COST_PCT,
    all_profiles,
    build_residency_report,
    profile_preset,
    recommend_preset,
)

# NON-new modules: the live preset registry and config contract the profiler
# integrates with.
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)

ALL_PRESETS = list(TQ_PRESETS.keys())
HEAD_DIMS = [64, 128, 256]


@pytest.mark.parametrize("preset", ALL_PRESETS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_bytes_per_token_matches_config_contract(preset, head_dim):
    """Profiler footprint must equal the config's aligned reserved slot."""
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
    prof = profile_preset(preset, head_dim)
    assert prof.bytes_per_token == cfg.slot_size_aligned


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_ratios_are_consistent_and_documented(preset):
    """fp16/fp8 ratios follow the baseline definitions and beat 1x."""
    head_dim = 128
    prof = profile_preset(preset, head_dim)
    assert prof.ratio_vs_fp16 == pytest.approx(2 * prof.ratio_vs_fp8)
    assert prof.ratio_vs_fp16 > 1.0
    # All shipped presets are quality-characterized.
    assert prof.quality_cost_pct == PRESET_QUALITY_COST_PCT[preset]


def test_all_profiles_cover_registry_sorted_by_density():
    profiles = all_profiles(128)
    assert {p.preset for p in profiles} == set(TQ_PRESETS)
    ratios = [p.ratio_vs_fp16 for p in profiles]
    assert ratios == sorted(ratios)


def test_recommend_respects_quality_budget():
    """A tight quality budget must exclude the aggressive presets."""
    # 3.0% budget admits only k8v4 (1.17%) and 4bit_nc (2.71%); of those
    # 4bit_nc is denser, so it wins.
    assert recommend_preset(128, max_quality_cost_pct=3.0) == "turboquant_4bit_nc"
    # A very tight budget leaves only the FP8-key preset.
    assert recommend_preset(128, max_quality_cost_pct=2.0) == "turboquant_k8v4"
    # Impossible budget => no recommendation.
    assert recommend_preset(128, max_quality_cost_pct=0.1) is None


def test_recommend_respects_compression_floor():
    """A compression floor selects the highest-quality preset that clears it."""
    chosen = recommend_preset(128, min_compression=4.0)
    prof = profile_preset(chosen, 128)
    assert prof.ratio_vs_fp16 >= 4.0
    # Nothing reaches 100x.
    assert recommend_preset(128, min_compression=100.0) is None


def test_residency_report_mentions_preset_and_alternative():
    report = build_residency_report("turboquant_k8v4", 128)
    assert "turboquant_k8v4" in report
    assert "vs fp16" in report
    # k8v4 is the least aggressive preset, so a denser alternative exists.
    assert "more cache residency" in report


def test_residency_report_for_densest_preset_has_no_denser_alt():
    densest = all_profiles(128)[-1].preset
    report = build_residency_report(densest, 128)
    assert densest in report
    assert "more cache residency" not in report
