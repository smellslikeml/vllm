# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic coding-agent workload for serving benchmarks.

The request shape here is modeled on the workload characterization in
TraceLab: Characterizing Coding Agent Workloads for LLM Serving
(https://arxiv.org/abs/2606.30560v1). That trace study of real Claude Code
and Codex sessions reports that coding-agent traffic is dominated by long
*autonomous loops*: a large context (system prompt, files, prior turns) that
**grows** step by step as tool results are appended, paired with short model
outputs. Consecutive steps therefore share a long -- but steadily growing --
prefix, which produces a high yet *imperfect* prefix-cache hit rate.

`RandomDataset` samples independent prompts and `PrefixRepetitionRandomDataset`
shares a *fixed* prefix; neither reproduces the append-and-re-prefill loop that
this study identifies as the defining pattern. This dataset fills that gap so
serving benchmarks can stress prefix-cache management and append-length-aware
prefill against a realistic coding-agent trace.
"""

import numpy as np

from vllm.benchmarks.datasets.datasets import (
    BenchmarkDataset,
    SampleRequest,
    gen_prompt_decode_to_target_len,
    logger,
)
from vllm.tokenizers import TokenizerLike


class CodingAgentDataset(BenchmarkDataset):
    """Synthetic autonomous coding-agent loop.

    Each session begins with a shared context of ``context_len`` tokens and
    runs for ``steps_per_session`` steps. Every step appends ``append_len``
    fresh tokens (a tool result / observation) to the context and requests a
    short ``output_len`` completion, so the prompt at step ``k`` is a strict
    prefix of the prompt at step ``k + 1``. Requests are emitted in step order
    so that a prefix-caching server sees the growing-prefix reuse pattern.
    """

    # Defaults chosen to reflect the trace study: long context, short tool
    # appends, short outputs, and a multi-step autonomous loop.
    DEFAULT_CONTEXT_LEN = 4096
    DEFAULT_APPEND_LEN = 256
    DEFAULT_OUTPUT_LEN = 128
    DEFAULT_STEPS_PER_SESSION = 8

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Isolate from global RNG state, matching RandomDataset.
        self._rng = np.random.default_rng(self.random_seed)

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        context_len: int = DEFAULT_CONTEXT_LEN,
        append_len: int = DEFAULT_APPEND_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        steps_per_session: int = DEFAULT_STEPS_PER_SESSION,
        **kwargs,
    ) -> list[SampleRequest]:
        if context_len < 1 or append_len < 1 or steps_per_session < 1:
            raise ValueError(
                "context_len, append_len, and steps_per_session must all be "
                f">= 1 (got context_len={context_len}, append_len={append_len}"
                f", steps_per_session={steps_per_session})."
            )

        vocab_size = tokenizer.vocab_size
        prohibited_tokens = set(tokenizer.all_special_ids)
        allowed_tokens = np.array(
            [tid for tid in range(vocab_size) if tid not in prohibited_tokens]
        )

        # Full length a session reaches on its final step.
        full_len = context_len + steps_per_session * append_len

        requests: list[SampleRequest] = []
        token_mismatch_total = 0
        while len(requests) < num_requests:
            session_seq, mismatch = self._generate_session_sequence(
                tokenizer=tokenizer,
                allowed_tokens=allowed_tokens,
                full_len=full_len,
            )
            token_mismatch_total += mismatch

            for step in range(steps_per_session):
                if len(requests) >= num_requests:
                    break
                step_len = context_len + (step + 1) * append_len
                prompt_tokens = session_seq[:step_len]
                prompt = tokenizer.decode(prompt_tokens)
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=len(prompt_tokens),
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(len(requests)),
                    )
                )

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )

        self.maybe_oversample_requests(
            requests, num_requests, request_id_prefix, no_oversample
        )
        return requests

    def _generate_session_sequence(
        self,
        *,
        tokenizer: TokenizerLike,
        allowed_tokens: np.ndarray,
        full_len: int,
    ) -> tuple[list[int], int]:
        """Build one session's full token sequence of length ``full_len``.

        A fresh per-session offset (drawn from the advancing RNG stream) makes
        sessions differ (distinct "codebases") while keeping generation
        reproducible for a fixed seed.
        """
        offset = int(self._rng.integers(0, len(allowed_tokens)))
        inner_seq = allowed_tokens[
            (offset + np.arange(full_len)) % len(allowed_tokens)
        ].tolist()
        _, adjusted_tokens, token_mismatch = gen_prompt_decode_to_target_len(
            tokenizer=tokenizer,
            token_sequence=inner_seq,
            target_token_len=full_len,
            add_special_tokens=False,
            rng=self._rng,
        )
        return adjusted_tokens, token_mismatch
