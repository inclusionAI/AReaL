# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import gc
import itertools
import queue
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.utils.checkpoint
import transformers

import realhf.impl.model.utils.cuda_graph as cuda_graph
from realhf.api.core.model_api import GenerationHyperparameters, ReaLModelConfig
from realhf.base import constants, logging
from realhf.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData
from realhf.impl.model.utils.functional import mask_eos_token
from realhf.impl.model.utils.logits_warper import top_k_top_p_logits
from realhf.impl.model.utils.padding import index_first_axis, unpad_input

if TYPE_CHECKING:
    from .real_llm_api import ReaLModel

logger = logging.getLogger("ReaLModel Generation")


def genstep(
    next_token_logits: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
    unfinished_sequences: torch.Tensor,
    generated_idx: Union[torch.IntTensor, int],
    gconfig: GenerationHyperparameters,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool, torch.Tensor]:
    """Advance generation by one step given logits.

    Args:
        next_token_logits (torch.Tensor): Shape [bs, vocab_size].
        tokenizer (transformers.PreTrainedTokenizerFast): .
        unfinished_sequences (torch.Tensor): Bool tensor indicator of whether a sequence is finished.
            Shape [bs].
        generated_idx (int): The token index to be generated.
        gconfig (GenerationHyperparameters): .

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        A tuple of
            next_tokens: Shape [bs].
            logprob: The log probability of selected tokens. May be re-normalized
                according to the mask machanism. Shape [bs].
            logits_mask: The mask of logits. Shape [bs, vocab_size].
            terminate: Whether the generation should be terminated.
            unfinished_sequences: Bool tensor indicator of whether a sequence is finished.
                Shape [bs].
    """
    if constants.tensor_parallel_world_size() > 1:
        from realhf.impl.model.parallelism.tensor_parallel.mappings import (
            gather_from_tensor_model_parallel_region,
        )

        next_token_logits = gather_from_tensor_model_parallel_region(next_token_logits)

    unfinished_sequences = unfinished_sequences.bool()
    next_token_logits = next_token_logits.float()
    if isinstance(generated_idx, int):
        if generated_idx < gconfig.min_new_tokens:
            next_token_logits = mask_eos_token(
                next_token_logits, eos_token_id=tokenizer.eos_token_id
            )
    else:
        assert isinstance(generated_idx, torch.Tensor)
        if (generated_idx < gconfig.min_new_tokens).any():
            _batch_indices = (generated_idx < gconfig.min_new_tokens).unsqueeze(1)
            _vocab_indices = _batch_indices.new_zeros((1, next_token_logits.shape[1]))
            if tokenizer.eos_token_id is not None:
                _vocab_indices[:, tokenizer.eos_token_id] = 1
            next_token_logits.masked_fill_(
                _batch_indices * _vocab_indices,
                torch.finfo(next_token_logits.dtype).min,
            )

    if not gconfig.greedy:
        next_token_logits /= gconfig.temperature
        next_token_logits = top_k_top_p_logits(
            next_token_logits,
            top_k=gconfig.top_k,
            top_p=gconfig.top_p,
            inplace=True,
            ordered=False,
        )

    distrb = torch.distributions.Categorical(logits=next_token_logits)
    next_tokens = distrb.mode if gconfig.greedy else distrb.sample()
    logprob = distrb.log_prob(next_tokens)

    if constants.tensor_parallel_world_size() > 1:
        if constants.tensor_parallel_rank() > 0:
            logprob[:] = 0
            next_tokens[:] = 0
        handle = torch.distributed.all_reduce(
            logprob,
            torch.distributed.ReduceOp.SUM,
            async_op=True,
            group=constants.tensor_parallel_group(),
        )
        torch.distributed.all_reduce(
            next_tokens,
            torch.distributed.ReduceOp.SUM,
            group=constants.tensor_parallel_group(),
        )

    if tokenizer.eos_token_id is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            )
        next_tokens.masked_fill_(
            unfinished_sequences.logical_not(), tokenizer.pad_token_id
        )
        # next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
    # unfinished_sequences = next_tokens.ne(tokenizer.eos_token_id).long() * unfinished_sequences
    unfinished_sequences.logical_and_(next_tokens.ne(tokenizer.eos_token_id))

    # terminate check
    if isinstance(generated_idx, int):
        terminate = (generated_idx >= gconfig.max_new_tokens - 1) or (
            unfinished_sequences.max() == 0
        )
    else:
        unfinished_sequences.logical_and_(generated_idx < gconfig.max_new_tokens - 1)
        terminate = unfinished_sequences.max() == 0

    if gconfig.force_no_logits_mask:
        logits_mask = None
    else:
        logits_mask = next_token_logits == torch.finfo(next_token_logits.dtype).min
        if not logits_mask.any():
            logits_mask = None

    if constants.tensor_parallel_world_size() > 1:
        handle.wait()

    return next_tokens, logprob, logits_mask, terminate, unfinished_sequences


def prepare_generate_inputs(
    module: "ReaLModel",
    gconfig: GenerationHyperparameters,
    x: PipeTransferData,
    ys: List[PipeCacheData],
    cuda_graph_name: str,
):
    if gconfig.top_p >= 1 and gconfig.top_k >= module.config.vocab_size:
        gconfig.force_no_logits_mask = True

    cu_seqlens = x.cu_seqlens
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    # assert constants.pipe_parallel_world_size() >= 2
    layer_indices = range(module.layer_idx_start, module.layer_idx_end)
    min_layer_index = module.layer_idx_start
    bs = input_lens.shape[0]
    cache_seqlens = input_lens.clone().to(dtype=torch.int32)

    block_ys = ys
    assert len(layer_indices) == len(block_ys), (len(block_ys), layer_indices)
    if constants.is_first_pipe_stage():
        layer_indices = layer_indices[1:]
        ys[0].cache_seqlens = cache_seqlens
        block_ys = ys[1:]
    if constants.is_last_pipe_stage():
        layer_indices = layer_indices[:-1]
        ys[-1].cache_seqlens = cache_seqlens
        block_ys = block_ys[:-1]

    for y, layer_idx in zip(block_ys, layer_indices):
        assert (
            y.k_cache is not None
            and y.v_cache is not None
            and y.cache_seqlens is not None
        ), (y.k_cache is None, y.v_cache is None, y.cache_seqlens is None)
        kvcache_seqlen = max(
            constants.max_prompt_len() + gconfig.max_new_tokens,
            module.config.hidden_dim // module.config.head_dim + 10,
        )
        k_cache_handle = cuda_graph.input_buffer_handle(cuda_graph_name, "k_caches")
        v_cache_handle = cuda_graph.input_buffer_handle(cuda_graph_name, "v_caches")
        if k_cache_handle is not None and v_cache_handle is not None:
            k_cache = k_cache_handle[layer_idx - min_layer_index][:bs, :kvcache_seqlen]
            v_cache = v_cache_handle[layer_idx - min_layer_index][:bs, :kvcache_seqlen]
        else:
            k_cache = torch.zeros(
                (bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                dtype=y.k_cache.dtype,
                device=y.k_cache.device,
            )
            v_cache = torch.zeros(
                (bs, kvcache_seqlen, *y.v_cache.shape[1:]),
                dtype=y.v_cache.dtype,
                device=y.v_cache.device,
            )
        indices = (
            torch.arange(kvcache_seqlen, device=module.device, dtype=torch.long)[
                None, :
            ]
            < input_lens[:, None]
        )
        k_cache[indices] = y.k_cache
        v_cache[indices] = y.v_cache
        y.k_cache = k_cache
        y.v_cache = v_cache
        y.cache_seqlens = cache_seqlens

    return x, ys


def maybe_capture_cudagraph(
    module: "ReaLModel",
    x: PipeTransferData,
    ys: List[PipeCacheData],
    cuda_graph_name: str,
    force_recapture: bool,
):
    bs = x.cu_seqlens.shape[0] - 1
    cache_seqlens = ys[0].cache_seqlens
    input_buffers = dict(
        input_ids=torch.ones((bs,), dtype=torch.long, device="cuda"),
        position_ids=cache_seqlens.clone(),
        k_caches=[y.k_cache for y in ys],
        v_caches=[y.v_cache for y in ys],
        cache_seqlens=cache_seqlens.clone(),
        max_seqlen=x.max_seqlen,
        cu_seqlens=x.cu_seqlens.clone(),
        hidden_states=(
            torch.zeros(
                (bs, x.pp_input.shape[1]), dtype=x.pp_input.dtype, device="cuda"
            )
            if x.pp_input is not None
            else None
        ),
    )

    graph, input_buffers, output_buffers = cuda_graph.capture_func(
        cuda_graph_name,
        module._forward,
        input_buffers,
        force_recapture=force_recapture,
        no_grad=True,
    )

    return graph, input_buffers, output_buffers


@torch.no_grad()
def generate(
    model: "ReaLModel",
    tokenizer: transformers.PreTrainedTokenizerFast,
    packed_input_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlen: Optional[int] = None,
    gconfig: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    ),
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[PipeCacheData],
    Optional[torch.Tensor],
]:
    """Generete a sequence with a ReaLModel."""
    bs = cu_seqlens.shape[0] - 1
    device = model.device
    mconfig: ReaLModelConfig = model.config

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(bs, dtype=torch.long, device=device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    prompt_logits = None
    # Prepare inputs for generation iterations

    # Input_ids may have different lengths, we should first pack them into a large batch
    # to use varlen flash attention, then record kv caches for the following inferences.
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    if constants.max_prompt_len() < max_seqlen:
        raise RuntimeError(
            f"Input sequence length {max_seqlen} is larger than the maximum sequence length "
            f"supported by the model {constants.max_prompt_len()}."
        )
    x = PipeTransferData(
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, store_kv_cache=True
    )
    # one embedding layer, n_layers transformer block, one output layer
    ys = [PipeCacheData(packed_input_ids=packed_input_ids)] + [
        PipeCacheData() for _ in range(mconfig.n_layers + 1)
    ]

    # Model forward will set k/v cache in PipeCacheData.
    prompt_logits = model(x, ys)[0].pp_output
    logits = prompt_logits[cu_seqlens[1:] - 1]

    # Next, we will generate the next token after prompts.
    # cache_seqlens is exactly the lengths of prompts.
    # We perform a genstep outside the loop due to a historical reason.
    next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
        logits, tokenizer, unfinished_sequences, generated_idx, gconfig
    )
    del logits

    gen_token_ph.append(next_tokens)
    gen_logprob_ph.append(logprob)
    gen_logits_mask_ph.append(logits_mask)
    generated_idx += 1

    cuda_graph_name = "decoding"
    x, ys = prepare_generate_inputs(model, gconfig, x, ys, cuda_graph_name)
    graph = None
    if gconfig.use_cuda_graph:
        graph, input_buffers, output_buffers = maybe_capture_cudagraph(
            model,
            x,
            ys,
            cuda_graph_name,
            force_recapture=gconfig.force_cudagraph_recapture,
        )

    # The main loop.
    while not terminate:
        # the next round of inference
        if graph is not None:
            input_buffers["input_ids"][:bs].copy_(next_tokens, non_blocking=True)
            input_buffers["position_ids"][:bs].copy_(
                ys[0].cache_seqlens, non_blocking=True
            )
            input_buffers["cache_seqlens"][:bs].copy_(
                ys[0].cache_seqlens, non_blocking=True
            )
            # K/v cache will be changed in-place with flash attention.
            graph.replay()
            logits = output_buffers["output"][:bs].squeeze(1)
        else:
            ys[0].packed_input_ids = next_tokens
            ys[0].packed_position_ids = None
            x.cu_seqlens = torch.arange(bs + 1, dtype=torch.int32, device=device)
            x.max_seqlen = 1
            # K/v cache will be changed in-place with flash attention.
            logits = model(x, ys)[0].pp_output.squeeze(dim=1)

        ys[
            0
        ].cache_seqlens += (
            1  # The global handle. This will increase all handles in ys by 1.
        )
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

    gen_tokens, log_probs, logits_mask = _gather_gen_output_from_list(
        gen_token_ph, gen_logprob_ph, gen_logits_mask_ph
    )
    if gconfig.use_cuda_graph and gconfig.force_cudagraph_recapture:
        cuda_graph.destroy(cuda_graph_name)

    return gen_tokens, log_probs, logits_mask, ys[1:-1], prompt_logits


def _gather_gen_output_from_list(
    gen_token_ph: List[torch.LongTensor],
    gen_logprob_ph: List[torch.FloatTensor],
    gen_logits_mask_ph: List[torch.BoolTensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack over the sequence dimension given a list of single-token
    tensors."""
    gen_tokens = torch.stack(gen_token_ph, -1)  # [bs, seqlen]
    log_probs = torch.stack(gen_logprob_ph, -1)  # [bs, seqlen]
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [
            torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph
        ]
        logits_mask = torch.stack(gen_logits_mask_ph, 1)  # [bs, seqlen, vocab_size]
    return gen_tokens, log_probs, logits_mask


def _gather_minibatch_gen_outputs(
    all_gen_tokens: List[torch.LongTensor],
    all_log_probs: List[torch.FloatTensor],
    all_logits_mask: List[torch.BoolTensor],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concate over the batch dimension given multiple [bs, seqlen] mini-batch
    tensors.

    Since different minibatches may have different generated lengths, we
    should pad them to the same length.
    """
    gen_tokens_lengths = [t.shape[-1] for t in all_gen_tokens]
    max_gen_tokens_length = max(gen_tokens_lengths)

    padded_gen_tokens = []
    padded_log_probs = []
    padded_logits_mask = []

    n_mbs = len(all_gen_tokens)
    for i in range(n_mbs):
        assert all_gen_tokens[i].shape == all_log_probs[i].shape
        gen_len = all_gen_tokens[i].shape[-1]

        gen_token = all_gen_tokens[i]
        log_probs = all_log_probs[i]
        logits_mask = all_logits_mask[i]

        if gen_len < max_gen_tokens_length:
            pad_size = max_gen_tokens_length - gen_len
            gen_token = torch.nn.functional.pad(
                gen_token, (0, pad_size), value=pad_token_id
            )
            log_probs = torch.nn.functional.pad(log_probs, (0, pad_size))
            if logits_mask is not None:
                logits_mask = torch.nn.functional.pad(
                    logits_mask,
                    (0, 0, 0, pad_size),
                    mode="constant",
                    value=1,
                )

        padded_gen_tokens.append(gen_token)
        padded_log_probs.append(log_probs)
        padded_logits_mask.append(logits_mask)

    gen_tokens = torch.cat(padded_gen_tokens, 0)
    log_probs = torch.cat(padded_log_probs, 0)
    if all([m is None for m in padded_logits_mask]):
        logits_mask = None
    else:
        mm = next(m for m in padded_logits_mask if m is not None)
        padded_logits_mask = [
            torch.ones_like(mm) if m is None else m for m in padded_logits_mask
        ]
        logits_mask = torch.cat(padded_logits_mask, 0)  # [bs, seqlen, vocab_size]

    return (gen_tokens, log_probs, logits_mask)


def concat_prompt_to_generation_output(
    packed_prompts: torch.LongTensor,
    prompt_lengths: torch.IntTensor,
    gen_tokens: torch.LongTensor,
    logprobs: torch.FloatTensor,
    logits_mask: torch.BoolTensor | None,
    gen_lengths: torch.IntTensor,
) -> Tuple[
    torch.LongTensor,
    torch.FloatTensor,
    torch.BoolTensor,
    torch.IntTensor,
    torch.BoolTensor,
]:
    device = packed_prompts.device

    prompts_list, prompt_log_probs_list, prompt_logits_mask_list = [], [], []
    gen_tokens_list, gen_log_probs_list, gen_logits_mask_list = [], [], []

    bs = prompt_lengths.shape[0]
    prompt_cu_seqlens = torch.nn.functional.pad(prompt_lengths.cumsum(0), (1, 0))
    for i in range(bs):
        prompt_len, gen_len = prompt_lengths[i].item(), gen_lengths[i].item()

        # log_probs is one-step shorter than token sequences.
        prompts_list.append(
            packed_prompts[prompt_cu_seqlens[i] : prompt_cu_seqlens[i + 1]]
        )
        prompt_log_probs_list.append(logprobs.new_zeros(prompt_len - 1))
        if logits_mask is not None:
            prompt_logits_mask_list.append(
                logits_mask.new_ones((prompt_len - 1, logits_mask.shape[-1]))
            )

        # Generated tokens are right-padded.
        gen_tokens_list.append(gen_tokens[i, :gen_len])
        gen_log_probs_list.append(logprobs[i, :gen_len])
        if logits_mask is not None:
            gen_logits_mask_list.append(
                torch.cat(
                    [
                        logits_mask[i, :gen_len],
                        logits_mask.new_ones(1, logits_mask.shape[-1]),
                    ]
                )
            )

    seq = torch.cat(
        list(itertools.chain.from_iterable(zip(prompts_list, gen_tokens_list)))
    )
    seq_lengths = prompt_lengths + gen_lengths
    packed_logprobs = torch.cat(
        list(
            itertools.chain.from_iterable(
                zip(prompt_log_probs_list, gen_log_probs_list)
            )
        )
    )
    assert seq.shape[0] == packed_logprobs.shape[0] + bs, (
        seq.shape,
        packed_logprobs.shape,
        bs,
    )
    packed_logits_mask = None
    if gen_logits_mask_list:
        packed_logits_mask = torch.cat(
            list(
                itertools.chain.from_iterable(
                    zip(prompt_logits_mask_list, gen_logits_mask_list)
                )
            )
        )

    prompt_mask = zip(
        [torch.ones(plen, dtype=torch.bool, device=device) for plen in prompt_lengths],
        [torch.zeros(glen, dtype=torch.bool, device=device) for glen in gen_lengths],
    )
    prompt_mask = torch.cat(list(itertools.chain.from_iterable(prompt_mask)))

    return (seq, packed_logprobs, packed_logits_mask, seq_lengths, prompt_mask)


@torch.no_grad()
def vanilla_packed_generate(
    model: "ReaLModel",
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    ),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: ReaLModelConfig = model.config

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            input_ids, attention_mask
        )
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(packed_input_ids=packed_input_ids)] + [
            PipeCacheData() for _ in range(mconfig.n_layers + 1)
        ]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output
        logits = logits[cu_seqlens[1:] - 1]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [
            torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph
        ]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


@torch.no_grad()
def vanilla_cpu_generate(
    model: "ReaLModel",
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    ),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: ReaLModelConfig = model.config
    assert str(input_ids.device) == "cpu"

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        x = PipeTransferData(attention_mask=attention_mask)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(packed_input_ids=input_ids)] + [
            PipeCacheData() for _ in range(mconfig.n_layers + 1)
        ]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output[:, -1, :]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [
            torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph
        ]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


class InflightBatchingGenerator:

    def __init__(
        self,
        inqueue: queue.Queue,
        outqueue: queue.Queue,
        model: "ReaLModel",
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters,
        batch_size: int,
        max_prompt_len: int,
    ):
        self.inqueue = inqueue
        self.outqueue = outqueue

        self.model = model
        self.mconfig = mconfig = model.config
        self.tokenizer = tokenizer

        self.gconfig = gconfig
        self.batch_size = batch_size
        self.max_prompt_len = max_prompt_len

        kvcache_seqlen = max(
            max_prompt_len + gconfig.max_new_tokens,
            mconfig.hidden_dim // mconfig.head_dim + 10,
        )
        _p = next(self.model.parameters())
        dtype, device = _p.dtype, _p.device

        # Cache
        self.k_caches = [
            torch.zeros(
                (
                    batch_size,
                    kvcache_seqlen,
                    mconfig.n_kv_heads,
                    mconfig.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.mconfig.n_layers)
        ]
        self.v_caches = [
            torch.zeros(
                (
                    batch_size,
                    kvcache_seqlen,
                    mconfig.n_kv_heads,
                    mconfig.head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.mconfig.n_layers)
        ]
        self.cache_seqlens = torch.zeros(
            (batch_size,), dtype=torch.int32, device=device
        )

        # Input buffers
        self.input_buf = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=device
        )
        self.input_buf_lens = torch.zeros(
            (batch_size,), dtype=torch.int32, device=device
        )

        # Save prompts for output
        self.prompt_tokens = [None for _ in range(batch_size)]

        # Generation state
        self.generate_idx = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        self.unfinished_sequences = torch.zeros(
            (batch_size,), dtype=torch.float32, device=device
        )

        self.ys = (
            [
                PipeCacheData(
                    cache_seqlens=self.cache_seqlens,
                )
            ]
            + [
                PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=self.cache_seqlens)
                for k, v in zip(self.k_caches, self.v_caches)
            ]
            + [PipeCacheData()]
        )

        # output buffers
        self.output_tokens_buf = [[] for _ in range(batch_size)]
        self.output_logprob_buf = [[] for _ in range(batch_size)]
        self.output_logits_mask = [[] for _ in range(batch_size)]

    def _get_non_eos_logits(self) -> torch.FloatTensor:
        self.ys[0].packed_position_ids = None
        self.ys[0].packed_input_ids = self.input_buf[:, :1]
        logits = self.model(PipeTransferData(), self.ys).pp_output.squeeze(dim=1)

        self.cache_seqlens += 1
        return logits.float()

    def _get_inflight_logits(self) -> torch.FloatTensor:
        finished_sequences = self.unfinished_sequences.logical_not()
        assert finished_sequences.any()

        finish_indices = finished_sequences.nonzero().squeeze(-1).tolist()

        # pop out finished sequences and clear corresponding buffers
        for i in finish_indices:
            prompt_tokens = self.prompt_tokens[i]

            # Used to skip the first call.
            if prompt_tokens is not None:
                gen_tokens = torch.stack(self.output_tokens_buf[i])
                gen_logp = torch.stack(self.output_logprob_buf[i])
                if all([m is None for m in self.output_logits_mask[i]]):
                    gen_logits_mask = None
                else:
                    mm = next(m for m in self.output_logits_mask[i] if m is not None)
                    gen_logits_mask = [
                        torch.ones_like(mm) if m is None else m
                        for m in self.output_logits_mask[i]
                    ]
                    gen_logits_mask = torch.stack(gen_logits_mask, -2)

                res = dict(
                    prompt=prompt_tokens,
                    gen=gen_tokens,
                    logp=gen_logp,
                    logits_mask=gen_logits_mask,
                )
                try:
                    self.outqueue.put_nowait(res)
                except queue.Full as e:
                    raise RuntimeError(
                        "Output queue is full. Please set a larger queue size."
                    ) from e

            # clear cache
            self.cache_seqlens[i] = 0

            # clear input buffers and prompts
            self.input_buf[i] = 0
            self.input_buf_lens[i] = 0
            self.prompt_tokens[i] = None

            # clear generation state
            self.generate_idx[i] = 0
            self.unfinished_sequences[i] = 1

            self.output_logits_mask[i] = []
            self.output_tokens_buf[i] = []
            self.output_logprob_buf[i] = []

        # build packed input ids with variable lengths for the next-step inference
        for i in range(self.batch_size):
            if i in finish_indices:
                try:
                    prompt = self.inqueue.get_nowait()
                    self.prompt_tokens[i] = prompt
                    self.input_buf[i, : prompt.shape[0]] = prompt
                    self.input_buf_lens[i] = prompt.shape[0]
                except queue.Empty as e:
                    raise RuntimeError(
                        "Input queue is empty. This should not happen."
                    ) from e

        input_lens = self.input_buf_lens
        valid_input_mask = torch.arange(
            self.max_prompt_len, device=self.input_buf.device, dtype=torch.int32
        ).unsqueeze(0) < input_lens.unsqueeze(-1)
        indices = torch.nonzero(valid_input_mask.flatten(), as_tuple=False).flatten()
        packed_input_ids = self.input_buf.flatten()[indices]
        max_seqlen = int(max(input_lens))
        cu_seqlens = torch.nn.functional.pad(
            input_lens.cumsum(0), (1, 0), value=0
        ).int()

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        self.ys[0].packed_position_ids = None
        self.ys[0].packed_input_ids = packed_input_ids
        logits = self.model(x, self.ys).pp_output
        logits = index_first_axis(logits, (cu_seqlens[1:] - 1).long())

        self.cache_seqlens += input_lens

        return logits.float()

    def advance_one_genstep(self):
        if self.unfinished_sequences.logical_not().any():
            logits = self._get_inflight_logits()
        else:
            logits = self._get_non_eos_logits()

        next_tokens, logprob, logits_mask, _, self.unfinished_sequences = genstep(
            logits,
            self.tokenizer,
            self.unfinished_sequences,
            self.generate_idx,
            self.gconfig,
        )

        for i in range(self.batch_size):
            self.output_tokens_buf[i].append(next_tokens[i].long())
            self.output_logprob_buf[i].append(logprob[i].float())
            if logits_mask is not None:
                self.output_logits_mask[i].append(logits_mask[i].bool())
            else:
                self.output_logits_mask[i].append(None)

        self.generate_idx += 1
        self.input_buf[:, 0] = next_tokens

    def step_for(self, n: int):
        for _ in range(n):
            self.advance_one_genstep()
