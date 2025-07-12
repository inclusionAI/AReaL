import torch


@torch.compile
def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels
