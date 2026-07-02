# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import CodingAgentTrace, get_samples


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("gpt2")


def _write_trace(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "agent_trace.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _args(trace_path: Path, num_prompts: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_name="coding_agent_trace",
        dataset_path=str(trace_path),
        disable_shuffle=True,
        num_prompts=num_prompts,
        seed=0,
        request_id_prefix="",
    )


_TRACE = [
    {
        "timestamp": 1.0,
        "messages": [{"role": "user", "content": "read the file"}],
        "output_tokens": 16,
        # OpenAI-format tool: must be passed through verbatim.
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    },
    {
        "ts": 2.0,  # alias for the timestamp field
        "messages": [{"role": "user", "content": "now edit it"}],
        "output_len": 8,  # alias for the output-length field
        # Bare function dict: must be wrapped into OpenAI tool format.
        "functions": [{"name": "edit_file", "parameters": {"type": "object"}}],
    },
    {
        "timestamp": 3.0,
        "prompt": "run the tests",  # single-string alias -> wrapped to a message
        "completion_tokens": 4,  # alias for the output-length field
        # No tools: request_overrides must be left as None.
    },
]


@pytest.mark.benchmark
def test_get_samples_emits_timed_tool_requests(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """get_samples dispatches to CodingAgentTrace and produces requests
    carrying chat messages, cumulative inter-arrival timestamps, and tool
    overrides -- the timestamp + tool combination that the trace-only
    (timed_trace) and tool-only (BFCL) datasets each capture only half of.
    """
    trace = _write_trace(tmp_path, _TRACE)
    samples = get_samples(_args(trace), hf_tokenizer)

    assert len(samples) == 3

    # Inter-arrival deltas [1, 2, 3] accumulate to cumulative [1, 3, 6]
    # and are emitted as a monotonically non-decreasing timestamp series.
    timestamps = [s.timestamp for s in samples]
    assert timestamps == pytest.approx([1.0, 3.0, 6.0])
    assert timestamps == sorted(timestamps)

    # Record 0: openai-format tool passed through; messages set directly.
    s0 = samples[0]
    assert s0.chat_messages == [{"role": "user", "content": "read the file"}]
    assert s0.expected_output_len == 16
    assert s0.request_overrides is not None
    # messages must live on their own typed field, not in overrides.
    assert "messages" not in s0.request_overrides
    assert s0.request_overrides["tool_choice"] == "auto"
    assert s0.request_overrides["tools"][0]["function"]["name"] == "read_file"

    # Record 1: bare function dict wrapped into OpenAI tool format.
    s1 = samples[1]
    tool1 = s1.request_overrides["tools"][0]
    assert tool1["type"] == "function"
    assert tool1["function"]["name"] == "edit_file"
    assert s1.expected_output_len == 8

    # Record 2: no tools -> no overrides; prompt alias wrapped to a message.
    s2 = samples[2]
    assert s2.request_overrides is None
    assert s2.chat_messages[0]["role"] == "user"
    assert s2.chat_messages[0]["content"] == "run the tests"
    assert s2.expected_output_len == 4

    for s in samples:
        assert s.prompt_len > 0


@pytest.mark.benchmark
def test_missing_timestamps_leave_pacing_to_scheduler(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """A trace with no timing information should leave timestamps as None so
    the benchmark falls back to its rate-based scheduler instead of forcing
    a degenerate all-at-once replay."""
    trace = _write_trace(
        tmp_path,
        [{"messages": [{"role": "user", "content": "hi"}], "output_tokens": 4}],
    )
    samples = get_samples(_args(trace, num_prompts=1), hf_tokenizer)

    assert len(samples) == 1
    assert samples[0].timestamp is None


@pytest.mark.benchmark
def test_absolute_timestamps_used_directly_when_not_inter_arrival(
    hf_tokenizer: PreTrainedTokenizerBase, tmp_path: Path
) -> None:
    """With inter_arrival disabled, already-cumulative timestamps are used
    as-is (never accumulated), preserving their wall-clock meaning."""
    records = [
        {"timestamp": 100.0, "messages": [{"role": "user", "content": "a"}]},
        {"timestamp": 50.0, "messages": [{"role": "user", "content": "b"}]},
        {"timestamp": 150.0, "messages": [{"role": "user", "content": "c"}]},
    ]
    trace = _write_trace(tmp_path, records)
    dataset = CodingAgentTrace(
        dataset_path=str(trace),
        disable_shuffle=True,
        coding_agent_trace_inter_arrival=False,
    )
    samples = dataset.sample(tokenizer=hf_tokenizer, num_requests=3)

    # Absolute cumulative mode clamps to non-decreasing but does not sum:
    # [100, max(100,50)=100, max(100,150)=150].
    timestamps = [s.timestamp for s in samples]
    assert timestamps == pytest.approx([100.0, 100.0, 150.0])


@pytest.mark.benchmark
def test_load_data_rejects_invalid_json(tmp_path: Path) -> None:
    """A malformed line must raise an actionable ValueError naming the line,
    not an opaque json error."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": true}\n{not valid json\n')
    dataset = CodingAgentTrace(dataset_path=str(bad), disable_shuffle=True)
    with pytest.raises(ValueError, match="line 2"):
        dataset.load_data()


@pytest.mark.benchmark
def test_prompt_len_includes_rendered_tool_schemas(tmp_path: Path) -> None:
    """prompt_len must reflect tool schemas (not just the user message), so
    latency-percentile buckets aren't biased low for tool-heavy coding-agent
    traffic."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=True
        ):
            base = " ".join(m.get("content", "") for m in messages)
            return base + " " + (json.dumps(tools) if tools else "")

        def __call__(self, text):
            # 1 "token" per whitespace-separated word.
            return type("Enc", (), {"input_ids": text.split()})()

    trace = _write_trace(tmp_path, _TRACE[:1])
    samples = get_samples(_args(trace, num_prompts=1), _FakeTokenizer())

    assert len(samples) == 1
    # User message "read the file" is 3 whitespace tokens; the rendered tool
    # schema adds many more, so the estimate must exceed the message-only count.
    assert samples[0].prompt_len > 3
