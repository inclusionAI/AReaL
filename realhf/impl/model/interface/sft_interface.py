# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
from typing import Dict, List, Literal

import torch
import torch.distributed as dist
import torch.utils.data
import tqdm

import realhf.api.core.model_api as model_api
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.base import constants, stats_tracker
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils.functional import (
    build_shift_one_indices,
    gather_packed_shifted_log_probs,
)


def compute_packed_sft_loss(
    logits: torch.Tensor,
    input_: SequenceSample,
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_.data["packed_input_ids"]
    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    prompt_mask = input_.data["prompt_mask"]

    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    prompt_mask = prompt_mask[shift_one_indices]
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss = -logprobs.sum() / prompt_mask.logical_not().count_nonzero()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp.detach()).sum() / (
                m.numel() - m.count_nonzero()
            )

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=prompt_mask.logical_not(),
        prompt_tokens=prompt_mask,
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    dist.all_reduce(
        vocab_min_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MIN
    )
    dist.all_reduce(
        vocab_max_logits, group=constants.tensor_parallel_group(), op=dist.ReduceOp.MAX
    )
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss


@dataclasses.dataclass
class SFTInterface(model_api.ModelInterface):
    token_normalize_scope: Literal["global", "dp"] = "global"

    def train_step(
        self, model: model_api.Model, data: SequenceSample, mb_spec: MicroBatchSpec
    ) -> Dict | List[Dict]:
        module = model.module

        module.train()

        with stats_tracker.scope("sft"):
            stats = module.train_batch(
                input_=data,
                loss_fn=compute_packed_sft_loss,
                loss_weight_fn=lambda x: x.data["prompt_mask"]
                .logical_not()
                .count_nonzero(),
                token_normalize_scope=self.token_normalize_scope,
                mb_spec=mb_spec,
                version_steps=model.version.global_step,
            )
            stats_tracker.scalar(**stats)

        model.inc_version()

        return stats_tracker.export()

    def save(self, model: model_api.Model, save_dir: str):
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model_: model_api.Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()

        for step, x in enumerate(tqdm.tqdm(eval_dataloader)):
            x: SequenceSample

            with stats_tracker.scope("sft-eval"):
                module.eval_batch(
                    input_=x.to_device(device),
                    loss_fn=compute_packed_sft_loss,
                    mb_spec=MicroBatchSpec(),
                )

        return stats_tracker.export()


model_api.register_interface("sft", SFTInterface)
