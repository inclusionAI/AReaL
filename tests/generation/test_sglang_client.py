import asyncio
import random
import uuid

import pytest
import torch

from realhf.api.cli_args import GenerationHyperparameters
from realhf.api.core.model_api import (
    APIGenerateInput,
    APIGenerateOutput,
    BundledGenerationOutputs,
)


@pytest.fixture
def sglang_client(request):
    from sglang.test.test_utils import is_in_ci

    if is_in_ci():
        from patch import launch_server_cmd
    else:
        from sglang.utils import launch_server_cmd

    from sglang.utils import terminate_process, wait_for_server

    server_process, port = launch_server_cmd(
        f"python -m sglang.launch_server --model-path {request.param} --host 0.0.0.0 --skip-tokenizer-init "
    )

    wait_for_server(f"http://localhost:{port}")
    from realhf.impl.model.backend.sglang import SGLangAPIClient

    client = SGLangAPIClient(
        generate_url=f"http://localhost:{port}/generate",
        update_weights_url=f"http://localhost:{port}/update_weights_from_disk",
    )
    yield client
    terminate_process(server_process)


@pytest.mark.parametrize(
    "sglang_client",
    ["/storage/openpsi/models/Qwen__Qwen2.5-7B-Instruct/"],
    indirect=True,
)
@pytest.mark.parametrize("group_size", [16])
@pytest.mark.asyncio
async def test_batch_generate(sglang_client, group_size):
    bs = 8
    # genlen = 16384
    genlen = 10
    prompt_len = 100
    async with sglang_client:
        tasks = []
        qids = []
        for i in range(bs):
            qid = str(uuid.uuid4())
            prompt_ids = [random.randint(10, 100) for _ in range(prompt_len)]
            gconfig = GenerationHyperparameters(
                n=group_size,
                max_new_tokens=genlen,
            )
            req = APIGenerateInput(
                qid=qid,
                prompt_ids=prompt_ids,
                input_ids=prompt_ids,
                gconfig=gconfig,
                return_logprob=True,
            )
            tasks.append(sglang_client.async_add_generate_request(req, stream=False))
            qids.append(qid)

        outputs = {}
        for r in asyncio.as_completed(tasks):
            out = await r
            outputs[out.qid] = out

        results = [outputs[key] for key in qids]
        assert all([isinstance(r, APIGenerateOutput) for r in results])
    batch_token_ids = []
    batch_logprobs = []
    max_seqlen = -1
    for x in results:
        max_seqlen = max(max_seqlen, max(x.output_lens))
        batch_token_ids += x.output_ids
        batch_logprobs += x.output_logprobs

    pad_token_id = 0
    # To be consistent with our internal implementation,
    # we should pad generated tokens and logprobs
    batch_token_ids = [
        t + [pad_token_id] * (max_seqlen - len(t)) for t in batch_token_ids
    ]
    batch_logprobs = [p + [0.0] * (max_seqlen - len(p)) for p in batch_logprobs]

    tokens = torch.tensor(batch_token_ids, dtype=torch.long, device="cpu")
    assert tokens.shape == (bs * group_size, genlen)
    logprobs = torch.tensor(batch_logprobs, dtype=torch.float32, device="cpu")
    assert logprobs.shape == (bs * group_size, genlen)
