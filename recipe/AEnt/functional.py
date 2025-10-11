from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed.nn.functional as dist_F
from areal.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


@torch.compile
def clamped_softmax_entropy(logits: torch.Tensor, entropy_clamp: float):
    """Assuming softmax policy, calculate entropy with token space clamping."""
    logits_cpu = logits.cpu().detach()
    # compute token space clamping mask
    with torch.no_grad():   
        k = int(logits_cpu.size(-1)*entropy_clamp)
        _, rm_indices = torch.topk(logits_cpu,k=k,dim=-1,largest=False)
        row_indices = torch.arange(logits_cpu.size(0)).unsqueeze(1)
        rm_mask = torch.zeros_like(logits_cpu,dtype=torch.bool)
        rm_mask[row_indices,rm_indices]=True
        del logits_cpu, row_indices, rm_indices
    rm_mask = rm_mask.to(logits.device)
    clamped_logits = logits.masked_fill(rm_mask, -torch.inf)
    clamped_logprobs = F.log_softmax(clamped_logits, dim=-1)
    clamped_logprobs = torch.where(rm_mask, 0., clamped_logprobs)
    del rm_mask
    torch.cuda.empty_cache()
    clamped_probs = F.softmax(clamped_logits, dim=-1)
    clamped_entropy = -torch.sum(clamped_probs * clamped_logprobs, dim=-1)
    # clamped_entropy = torch.logsumexp(clamped_logits, dim=-1) - torch.sum(clamped_probs * logits, dim=-1)
    return clamped_entropy


@torch.compile
def _gather_logprobs_clamped_entropy(
    logits: torch.Tensor, labels: torch.Tensor, entropy_clamp: float, temperature: float = 1.0,
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    clamped_entropy = clamped_softmax_entropy(logits.float() / temperature, entropy_clamp)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, clamped_entropy


def gather_logprobs_clamped_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    entropy_clamp: float,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]
    log_probs_labels_list = []
    entropy_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs, chunk_entropy = _gather_logprobs_clamped_entropy(
            chunk_logits, chunk_labels, entropy_clamp, temperature
        )

        log_probs_labels_list.append(chunk_log_probs)
        entropy_list.append(chunk_entropy)

    logprobs = torch.cat(log_probs_labels_list)
    entropy = torch.cat(entropy_list)

    if get_ulysses_sequence_parallel_world_size() > 1:
        sp_group = get_ulysses_sequence_parallel_group()
        logprobs = dist_F.all_gather(logprobs, group=sp_group)
        logprobs = torch.cat(logprobs, dim=-1)
        entropy = dist_F.all_gather(entropy, group=sp_group)
        entropy = torch.cat(entropy, dim=-1)

    return logprobs, entropy