import functools

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel

from areal.utils.functional import (
    chunked_apply,
    chunked_gather_logprobs,
    chunked_gather_logprobs_entropy,
)


# Copied from verl:
# https://github.com/volcengine/verl/blob/11a43b6cad8d6f1f52738af49ca5307cd5b1b1be/verl/utils/megatron/tensor_parallel.py#L109
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(
            logits_max,
            op=dist.ReduceOp.MAX,
            group=mpu.get_tensor_model_parallel_group(),
        )
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(
            normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group()
        )
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(
            sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group()
        )
        entropy = (
            logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        )
        ctx.save_for_backward(
            vocab_parallel_logits, softmax_logits, sum_softmax_times_logits
        )
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = (
            ctx.saved_tensors
        )
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def _vocab_parallel_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    _logits = logits.float() / temperature
    return -tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits=_logits, target=labels
    )  # type: ignore


def _vocab_parallel_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    _logits = logits.float() / temperature
    entropy = _VocabParallelEntropy.apply(_logits)
    logprobs = -tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits=_logits, target=labels
    )  # type: ignore
    return logprobs, entropy


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute log probabilities with optional vocab parallelism for Megatron.

    When tensor parallelism is enabled, uses Megatron's vocab_parallel_cross_entropy
    to avoid gathering the full vocab dimension across TP ranks, which would
    significantly increase GPU memory usage.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        Log probabilities at the label positions with shape [...].
    """
    if mpu.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
        # NOTE: When tensor parallelism is enabled, logits are parallelized across TP ranks.
        # If we explicitly gather logits, it will significantly increase GPU memory usage.
        fn = functools.partial(_vocab_parallel_logprobs, temperature=temperature)
        return chunked_apply(fn, logits, labels, chunk_size)

    return chunked_gather_logprobs(logits, labels, temperature, chunk_size)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy with optional vocab parallelism for Megatron.

    When tensor parallelism is enabled, uses Megatron's vocab_parallel_cross_entropy
    and a custom _VocabParallelEntropy implementation to avoid gathering the full
    vocab dimension across TP ranks.

    Args:
        logits: Model logits with shape [..., vocab_size] or [..., vocab_size/tp]
            when tensor parallelism is enabled.
        labels: Token indices with shape [...] for which to compute log probabilities.
        temperature: Softmax temperature scaling. Default is 1.0.
        chunk_size: Chunk size for memory-efficient processing along the sequence
            dimension. Default is 1024.

    Returns:
        A tuple of (logprobs, entropy):
            - logprobs: Log probabilities at the label positions with shape [...].
            - entropy: Entropy of the probability distribution with shape [...].
    """
    if mpu.is_initialized() and mpu.get_tensor_model_parallel_world_size() > 1:
        fn = functools.partial(
            _vocab_parallel_logprobs_entropy, temperature=temperature
        )
        return chunked_apply(fn, logits, labels, chunk_size)

    return chunked_gather_logprobs_entropy(logits, labels, temperature, chunk_size)
