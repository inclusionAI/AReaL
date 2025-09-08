import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu


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
