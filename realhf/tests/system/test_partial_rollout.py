import asyncio
import random
import string
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from transformers import PreTrainedTokenizerFast

from realhf.api.cli_args import NameResolveConfig
from realhf.api.core.model_api import (
    APIGenerateInput,
    APIGenerateOutput,
    GenerationHyperparameters,
    LLMAPIClient,
)
from realhf.base import constants, name_resolve, names, testing
from realhf.base.names import gen_server_manager
from realhf.system.partial_rollout import PartialRolloutManager


class MockLLMAPIClient(LLMAPIClient):
    def setup(self):
        pass

    async def close(self):
        await asyncio.sleep(0.5)

    async def async_add_generate_request(
        self, req: APIGenerateInput, stream: bool = True
    ) -> APIGenerateOutput:
        await asyncio.sleep(1)
        output_len = req.gconfig.max_new_tokens
        out = APIGenerateOutput.from_input(req)
        out.output_ids = [
            [
                random.randint(0, testing.TESTING_MODEL_VOCAB_SIZE - 1)
                for _ in range(output_len)
            ]
            for _ in range(req.gconfig.n)
        ]
        if req.return_logprob:
            out.output_logprobs = [
                [random.random() for _ in range(output_len)]
                for _ in range(req.gconfig.n)
            ]
        out.no_eos = [True for _ in range(req.gconfig.n)]
        return out

    async def async_update_weights_from_disk(self, path):
        await asyncio.sleep(10)

    async def get_cur_version(self):
        await asyncio.sleep(0)
        return 1


new_tokens_per_chunk = 16


@pytest.fixture
def partial_rollout_manager():
    # Set up mocked tokenizer
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    # Set up mocked request and reply queues
    request_queue = asyncio.Queue()
    reply_queue = asyncio.Queue()

    name_resolve.reconfigure(
        NameResolveConfig("nfs", "/tmp/areal/test-partial-rollout")
    )

    testing.clear_name_resolve()
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    name = gen_server_manager(testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME)
    name_resolve.add(name, "http://fake.com")

    global new_tokens_per_chunk

    # Initialize PartialRolloutManager
    manager = PartialRolloutManager(
        worker_index=0,
        request_queue=request_queue,
        reply_queue=reply_queue,
        new_tokens_per_chunk=new_tokens_per_chunk,
        tokenizer=mock_tokenizer,
        timeout=300,
    )
    yield manager
    # Cleanup if needed


@pytest.mark.asyncio
async def test_run_step_calls_issue_generation(partial_rollout_manager):
    """
    Test `run_step` calls `_issue_generation` the correct number of times.
    """
    with (
        patch(
            "realhf.system.partial_rollout.PartialRolloutManager._schedule_request",
            new_callable=AsyncMock,
        ),
        patch(
            "realhf.impl.model.backend.sglang.SGLangAPIClient",
            new_callable=lambda: MockLLMAPIClient,
        ),
    ):

        # Define inputs
        gen_random_qid = lambda length=8: "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(length)
        )
        group_size = 10
        test_prompt_ids = list(range(100))
        test_max_new_tokens = 128
        assert test_max_new_tokens % new_tokens_per_chunk == 0
        test_raw_gconfig = GenerationHyperparameters(
            n=group_size, max_new_tokens=test_max_new_tokens
        )
        q_len = 8  # query number
        test_qids = [f"{gen_random_qid(10)}-{i}" for i in range(q_len)]
        for test_qid in test_qids:
            partial_rollout_manager.request_queue.put_nowait(
                (test_qid, test_prompt_ids.copy(), test_raw_gconfig.new())
            )

        with patch.object(
            partial_rollout_manager,
            "_issue_generation",
            side_effect=partial_rollout_manager._issue_generation,
        ) as mock_issue_generation:
            while partial_rollout_manager.reply_queue.qsize() != q_len:
                await partial_rollout_manager.run_step()

            assert (
                mock_issue_generation.call_count
                == q_len
                * group_size
                * test_max_new_tokens
                // partial_rollout_manager.new_tokens_per_chunk
            )
