# Copyright 2025 Ant Group Inc.

import functools
import multiprocessing as mp
import os
import uuid

import numpy as np
import torch
import torch.distributed as dist
import transformers

from realhf.api.core import data_api, model_api
from realhf.api.core.config import ModelName
from realhf.api.core.data_api import MicroBatchSpec
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging
from realhf.base.testing import init_global_constants

logger = logging.getLogger("test sglang backend")


def check_sequences_consistency(
    batched_seq1: torch.LongTensor, batched_seq2: torch.LongTensor
):
    matched_tokens = 0
    matched_seqs = 0
    total_tokens = 0
    assert len(batched_seq1) == len(batched_seq2)
    for i in range(len(batched_seq1)):
        a = batched_seq1[i]
        b = batched_seq2[i]
        assert torch.is_tensor(a) and torch.is_tensor(b)
        assert a.dim() == 1 and b.dim() == 1, (a.shape, b.shape)
        gen_len = a.shape[0] if a.shape[0] < b.shape[0] else b.shape[0]
        b = b[:gen_len]
        a = a[:gen_len]
        for j in range(gen_len):
            if a[j] != b[j]:
                logger.info(f"Mismatch at sequence {i} position {j}")
                break
            matched_tokens += 1
        else:
            matched_seqs += 1
        total_tokens += gen_len
    logger.info(
        f"Matched {matched_seqs}/{len(batched_seq1)} "
        f"sequences and {matched_tokens}/{total_tokens} tokens"
    )
    return (
        matched_seqs,
        matched_tokens,
        float(matched_tokens) / total_tokens,
        float(matched_seqs) / len(batched_seq1),
    )


def _fn(
    rank: int,
    world_size: int,
    path: str,
    model_family_name: str,
    dp: int,
    pp: int,
    tp: int,
):
    assert not torch.cuda.is_initialized()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)
    assert world_size == (
        dp * pp * tp
    ), f"dp={dp}, pp={pp}, tp={tp}, world_size={world_size}"
    # Initialize distributed environment.
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://localhost:7777",
    )
    torch.cuda.set_device(0)
    model_name = ModelName("default", 0)
    constants.set_experiment_trial_names("slang-test", str(uuid.uuid4()))
    init_global_constants(
        num_dp=dp,
        num_tp=tp,
        num_pp=pp,
        sequence_parallel=False,
        model_name=model_name,
        max_prompt_len=128,
    )

    from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

    mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(
        transformers.AutoConfig.from_pretrained(
            path,
            trust_remote_code=True,
            force_download=True,
        )
    )
    with constants.model_scope(model_name):
        module = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        module._instantiation_hooks.append(
            lambda: getattr(module, f"from_{model_family_name}")(
                load_dir=path, init_critic_from_actor=False
            )
        )
        add_helper_functions(module)
        module.instantiate()
        module.eval()
        tokenizer = data_api.load_hf_tokenizer(path)

        from realhf.impl.model.backend.sglang import SGLangGenerationBackend

        backend = SGLangGenerationBackend(
            model_path=path,
            dtype="bfloat16" if module.dtype == torch.bfloat16 else torch.float16,
        )
        model = model_api.Model(
            name=model_name,
            module=module,
            tokenizer=tokenizer,
            device=module.device,
            dtype=module.dtype,
        )
        ft_spec = model_api.FinetuneSpec(
            total_train_epochs=1,
            dataset_size=100,
            train_batch_size=1,
        )
        model = backend.initialize(model, ft_spec)

        gconfig = model_api.GenerationHyperparameters(
            n=1,
            max_new_tokens=32,
            min_new_tokens=0,
            greedy=True,
            top_p=1.0,
            top_k=int(1e8),
            temperature=1.0,
            use_cuda_graph=False,
        )

        bs = 8
        for i in range(1):
            seqlens = [torch.randint(5, 10, (1,)).cuda() for _ in range(bs)]

            for s in seqlens:
                dist.broadcast(s, src=0)
            seqlens = [int(s) for s in seqlens]

            token_ids = (
                torch.randint(0, mconfig.vocab_size, (sum(seqlens),)).long().cuda()
            )
            dist.broadcast(token_ids, src=0)

            max_seqlen = max(seqlens)
            cu_seqlens = torch.nn.functional.pad(
                torch.tensor(seqlens, device="cuda").cumsum(0),
                (1, 0),
            ).int()

            res = module.generate(
                tokenizer=tokenizer,
                packed_input_ids=token_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gconfig=gconfig,
            )
            gen_tokens1 = res.sequences
            logprobs1 = res.scores

            x = data_api.SequenceSample.from_default(
                seqlens=seqlens,
                ids=list(range(bs)),
                data=dict(packed_input_ids=token_ids),
            )
            gen_tokens2, logprobs2, _ = model.module.generate(
                input_=x,
                mb_spec=MicroBatchSpec(),
                tokenizer=tokenizer,
                gconfig=gconfig,
            )
            if constants.tensor_parallel_rank() == 0:
                # The outputs are Nones for tp_rank > 1 in SGLang
                _, _, token_match_percent, seq_match_percent = (
                    check_sequences_consistency(gen_tokens1, gen_tokens2)
                )
                assert token_match_percent > 0.8, token_match_percent
                assert seq_match_percent > 0.8, seq_match_percent

        print("success")

    dist.destroy_process_group()


def check_sglang_consistency(tp: int, dp: int, path: str, model_family_name: str):
    mp.set_start_method("spawn", force=True)
    world_size = dp * tp
    procs = [
        mp.Process(
            target=_fn,
            args=(
                i,
                world_size,
            ),
            kwargs=dict(
                path=path,
                model_family_name=model_family_name,
                dp=dp,
                pp=1,
                tp=tp,
            ),
        )
        for i in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        [p.terminate() for p in procs]
        [p.join() for p in procs]


if __name__ == "__main__":
    path = "/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct/"
    model_family_name = "qwen2"
    # test_fn(
    #     rank=0,
    #     world_size=1,
    #     path=path,
    #     model_family_name=model_family_name,
    #     dp=1,
    #     pp=1,
    #     tp=1,
    # )
    check_sglang_consistency(
        tp=2,
        dp=2,
        path=path,
        model_family_name=model_family_name,
    )
