# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import (
    CodingAgentDataset,
    add_dataset_parser,
    get_samples,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

CONTEXT_LEN = 16
APPEND_LEN = 4
OUTPUT_LEN = 5
STEPS_PER_SESSION = 3


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("gpt2")


def _assert_growing_loop(samples, num_requests) -> None:
    """Every session is a growing-prefix autonomous loop with short outputs."""
    assert len(samples) == num_requests
    for i, req in enumerate(samples):
        step = i % STEPS_PER_SESSION
        # Prompt grows by exactly one appended block per step.
        assert req.prompt_len == CONTEXT_LEN + (step + 1) * APPEND_LEN
        # Long context, short output -- the defining coding-agent shape.
        assert req.expected_output_len == OUTPUT_LEN
        # Within a session each step re-prefills the previous prompt plus the
        # newly appended block, so it must extend the prior prompt verbatim.
        if step > 0:
            assert req.prompt.startswith(samples[i - 1].prompt)
            assert len(req.prompt) > len(samples[i - 1].prompt)


@pytest.mark.benchmark
def test_coding_agent_growing_prefix_loop(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    num_requests = 2 * STEPS_PER_SESSION
    samples = CodingAgentDataset(random_seed=0).sample(
        tokenizer=hf_tokenizer,
        num_requests=num_requests,
        context_len=CONTEXT_LEN,
        append_len=APPEND_LEN,
        output_len=OUTPUT_LEN,
        steps_per_session=STEPS_PER_SESSION,
    )
    _assert_growing_loop(samples, num_requests)


@pytest.mark.benchmark
def test_coding_agent_same_seed_is_reproducible(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    def run():
        return CodingAgentDataset(random_seed=7).sample(
            tokenizer=hf_tokenizer,
            num_requests=STEPS_PER_SESSION,
            context_len=CONTEXT_LEN,
            append_len=APPEND_LEN,
            output_len=OUTPUT_LEN,
            steps_per_session=STEPS_PER_SESSION,
        )

    a = [(s.prompt, s.prompt_len, s.expected_output_len) for s in run()]
    b = [(s.prompt, s.prompt_len, s.expected_output_len) for s in run()]
    assert a == b


@pytest.mark.benchmark
def test_coding_agent_rejects_invalid_shape(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    with pytest.raises(ValueError):
        CodingAgentDataset(random_seed=0).sample(
            tokenizer=hf_tokenizer,
            num_requests=1,
            context_len=0,
        )


@pytest.mark.benchmark
def test_coding_agent_registered_in_get_samples(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Exercise the get_samples/add_dataset_parser wiring in datasets.py."""
    parser = FlexibleArgumentParser()
    add_dataset_parser(parser)
    num_requests = 2 * STEPS_PER_SESSION
    args = parser.parse_args(
        [
            "--dataset-name",
            "coding_agent",
            "--num-prompts",
            str(num_requests),
            "--coding-agent-context-len",
            str(CONTEXT_LEN),
            "--coding-agent-append-len",
            str(APPEND_LEN),
            "--coding-agent-output-len",
            str(OUTPUT_LEN),
            "--coding-agent-steps-per-session",
            str(STEPS_PER_SESSION),
        ]
    )
    samples = get_samples(args, hf_tokenizer)
    _assert_growing_loop(samples, num_requests)
