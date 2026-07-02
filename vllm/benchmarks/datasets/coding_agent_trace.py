# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coding-agent trace replay dataset for the online serving benchmark.

Coding agents (e.g. Claude Code, Codex) drive serving traffic that looks
nothing like ShareGPT or synthetic-random workloads: long multi-turn
contexts, short completions, a heavy-tailed mix of tool calls, and
arrival pacing driven by human-paced gaps between autonomous loops.
Replaying that traffic is the only way to evaluate serving optimizations
(tool-call overhead, append-length-aware prefill, KV-cache management
around idle gaps) against realistic request shapes.

:class:`CodingAgentTrace` loads a newline-delimited JSON trace -- one
record per LLM step -- and emits :class:`SampleRequest` objects that
carry both the per-request inter-arrival ``timestamp`` (replayed by the
benchmark's self-timed scheduler) and the tool definitions available at
that step (attached via ``request_overrides`` for an ``openai-chat``
backend). That combination -- timed replay *plus* tool calling -- is the
coding-agent workload profile that the trace-only (``timed_trace``) and
tool-only (``hf``/BFCL) datasets each capture only half of.

Trace record schema (one JSON object per line)::

    {
        "timestamp": 1.25,          # seconds; inter-arrival delta
        "messages": [               # openai chat messages (the prompt)
            {"role": "user", "content": "refactor foo()"}
        ],
        "output_tokens": 64,        # expected completion length
        "tools": [                  # optional; openai tool schemas
            {"type": "function", "function": {"name": "edit", ...}}
        ]
    }

Common field aliases are accepted (``ts``/``time``/``arrival_time`` for
the timestamp, ``output_len``/``completion_tokens`` for the output
length, ``functions`` for tools, ``prompt`` for a single-string prompt).

Adapted from the workload characterization released by the TraceLab
project (https://github.com/uw-syfi/TraceLab), Apache-2.0.
"""

import json
import logging
import random
from typing import Any

from vllm.benchmarks.datasets.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike

logger = logging.getLogger(__name__)

# Field-name aliases tolerated in trace records. The first key found (in
# list order) wins, so canonical names take precedence over shorthand.
_TIMESTAMP_KEYS = ("timestamp", "ts", "time", "arrival_time", "t")
_MESSAGES_KEYS = ("messages", "conversation")
_PROMPT_KEYS = ("prompt", "text")
_OUTPUT_KEYS = (
    "output_tokens",
    "output_len",
    "completion_tokens",
    "num_output_tokens",
    "max_tokens",
)
_TOOLS_KEYS = ("tools", "functions")


class CodingAgentTrace(BenchmarkDataset):
    """Replay a coding-agent serving trace as timed tool-calling requests.

    The trace is read once at construction (newline-delimited JSON, one
    LLM step per record) and sampled without network access; the dataset
    itself is supplied by the user via ``--dataset-path``, exactly like
    the other trace/HF datasets. See the module docstring for the record
    schema.
    """

    DEFAULT_OUTPUT_LEN = 256

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        random.seed(self.random_seed)
        # Optional tunables (no CLI args today; honored if ever added).
        self.sec_multiplier = float(
            kwargs.get("coding_agent_trace_sec_multiplier", 1.0)
        )
        # Treat ``timestamp`` as an inter-arrival delta (accumulated into a
        # monotonic cumulative series) unless told otherwise. Coding-agent
        # traces are naturally expressed as gaps between steps.
        self.inter_arrival = bool(kwargs.get("coding_agent_trace_inter_arrival", True))
        self.load_data()

    # -- loading ---------------------------------------------------------

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")
        records: list[dict] = []
        with open(self.dataset_path) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"coding_agent_trace: invalid JSON on line {line_no} "
                        f"of {self.dataset_path}: {e}"
                    ) from e
                if not isinstance(record, dict):
                    raise ValueError(
                        f"coding_agent_trace: line {line_no} of "
                        f"{self.dataset_path} is not a JSON object."
                    )
                records.append(record)
        if not records:
            raise ValueError(
                f"coding_agent_trace: no records found in {self.dataset_path}."
            )
        self.data = records

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _first(record: dict, keys: tuple[str, ...], default: Any = None) -> Any:
        for k in keys:
            if k in record:
                return record[k]
        return default

    @classmethod
    def _normalize_tools(cls, raw: Any) -> list[dict] | None:
        """Coerce a record's tool list to OpenAI ``tools`` format.

        Accepts either ready-made OpenAI tools (``{"type": "function",
        "function": {...}}``) or bare function dicts (``{"name": ...,
        "parameters": {...}}``) and wraps the latter. ``None``/empty maps
        to ``None`` (no tools attached to that request).
        """
        if not raw:
            return None
        if isinstance(raw, dict):
            raw = [raw]
        tools: list[dict] = []
        for fn in raw:
            if not isinstance(fn, dict):
                continue
            if fn.get("type") == "function" and "function" in fn:
                tools.append(fn)
            else:
                tools.append({"type": "function", "function": fn})
        return tools or None

    def _cumulative_timestamps(self, raw_ts: list[float | None]) -> list[float | None]:
        """Build the monotonic cumulative-seconds series the scheduler wants.

        ``raw_ts`` holds one value per record (``None`` where the record
        omitted a timestamp). With ``inter_arrival`` (default) each value
        is a delta since the previous step and is accumulated; otherwise
        values are treated as already-cumulative. The result is
        non-decreasing -- negative deltas (clock jitter) collapse onto the
        previous step.
        """
        if all(v is None for v in raw_ts):
            # No timing information at all: leave pacing to the
            # benchmark's rate-based scheduler.
            return [None] * len(raw_ts)
        out: list[float | None] = []
        cumulative = 0.0
        for v in raw_ts:
            delta = 0.0 if v is None else max(0.0, float(v))
            scaled = delta * self.sec_multiplier
            if self.inter_arrival:
                cumulative += scaled
            else:
                # Absolute cumulative: never go backwards.
                cumulative = max(cumulative, scaled)
            out.append(cumulative)
        return out

    @staticmethod
    def _estimate_prompt_len(
        tokenizer: TokenizerLike,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> int:
        """Token-count estimate that includes rendered tool schemas.

        Coding-agent prompts are dominated by long multi-turn context
        *and* tool definitions; estimating from the last user message
        alone badly understates input length and skews latency buckets.
        Render the full chat template with ``tools=`` when the tokenizer
        supports it, falling back to a plain-text concatenation for
        older/legacy tokenizers.
        """
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # Legacy tokenizer without a ``tools`` kwarg.
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(
                "coding_agent_trace: apply_chat_template failed for a "
                "sample, falling back to plain-text prompt length: %s",
                e,
                exc_info=True,
            )
            text = "\n".join(str(m.get("content", "")) for m in messages)
            return len(tokenizer(text).input_ids)
        if not isinstance(rendered, str):
            text = "\n".join(str(m.get("content", "")) for m in messages)
            return len(tokenizer(text).input_ids)
        return len(tokenizer(rendered).input_ids)

    # -- sampling --------------------------------------------------------

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        assert tokenizer is not None, "Tokenizer must be provided, now is Null"
        assert self.data is not None, "Data must be loaded before sampling"

        # Order matters for trace replay: inter-arrival timestamps are
        # only meaningful in trace order, so we do not shuffle (matching
        # the sibling ``timed_trace`` dataset) and take the first N steps.
        records = self.data[:num_requests]
        raw_ts = [self._first(r, _TIMESTAMP_KEYS) for r in records]
        timestamps = self._cumulative_timestamps(raw_ts)

        samples: list[SampleRequest] = []
        for ind, record in enumerate(records):
            messages = self._first(record, _MESSAGES_KEYS)
            if messages is None:
                prompt_text = self._first(record, _PROMPT_KEYS, "")
                messages = [{"role": "user", "content": str(prompt_text)}]
            if not isinstance(messages, list) or not messages:
                continue

            output_len = self._first(record, _OUTPUT_KEYS, self.DEFAULT_OUTPUT_LEN)
            try:
                output_len = int(output_len)
            except (TypeError, ValueError):
                output_len = self.DEFAULT_OUTPUT_LEN
            output_len = max(1, output_len)

            tools = self._normalize_tools(self._first(record, _TOOLS_KEYS))
            prompt_len = self._estimate_prompt_len(tokenizer, messages, tools)
            overrides = {"tools": tools, "tool_choice": "auto"} if tools else None

            samples.append(
                SampleRequest(
                    prompt=messages[-1].get("content", ""),
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(ind),
                    chat_messages=messages,
                    request_overrides=overrides,
                    timestamp=timestamps[ind],
                )
            )
        return samples
