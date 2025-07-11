# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import functools
from typing import List, Optional, Tuple

import torch


class LogitsWarper:
    """Abstract base class for all logit processors that can be applied during
    generation.

    Cloned from huggingface transformers/src/transformers/generation/logits_process.py,
    except that we can optionally change the logits inplace.
    """

    def __call__(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
        inplace: bool = False,
    ) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


@dataclasses.dataclass
class TemperatureLogitsWarper(LogitsWarper):
    temperature: float

    def __post_init__(self):
        if self.temperature > 1.0 or self.temperature < 0.0:
            raise ValueError("temperature has to be between 0 and 1")

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        if inplace:
            logits.div_(self.temperature)
        else:
            logits = logits / self.temperature
        return logits / self.temperature


@dataclasses.dataclass
class TopPLogitsWarper(LogitsWarper):
    top_p: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        self.top_p = top_p = float(self.top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(self.min_tokens_to_keep, int) or (
            self.min_tokens_to_keep < 1
        ):
            raise ValueError(
                f"`min_tokens_to_keep` has to be a positive integer, "
                f"but is {self.min_tokens_to_keep}"
            )

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        self.filter_value = torch.finfo(logits.dtype).min
        if inplace:
            logits.masked_fill_(indices_to_remove, self.filter_value)
        else:
            logits = logits.masked_fill(indices_to_remove, self.filter_value)
        return logits


@dataclasses.dataclass
class TopKLogitsWarper(LogitsWarper):
    top_k: int
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        top_k = self.top_k
        min_tokens_to_keep = self.min_tokens_to_keep
        self.top_k = max(top_k, min_tokens_to_keep)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        top_k = min(self.top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        self.filter_value = torch.finfo(logits.dtype).min
        if inplace:
            logits.masked_fill_(indices_to_remove, self.filter_value)
        else:
            logits = logits.masked_fill(indices_to_remove, self.filter_value)
        return logits


@dataclasses.dataclass
class EpsilonLogitsWarper(LogitsWarper):
    epsilon: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __post_init__(self):
        self.epsilon = epsilon = float(self.epsilon)
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(
                f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}"
            )

        self.min_tokens_to_keep = min_tokens_to_keep = int(self.min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

    def __call__(
        self,
        _,
        logits: torch.FloatTensor,
        inplace: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # Calculate the adaptive cutoff
        probabilities = logits.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[
            ..., None
        ]
        indices_to_remove = probabilities < eta

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, logits.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (
            logits < torch.topk(logits, top_k)[0][..., -1, None]
        )

        self.filter_value = torch.finfo(logits.dtype).min
        if inplace:
            logits.masked_fill_(indices_to_remove, self.filter_value)
        else:
            logits = logits.masked_fill(indices_to_remove, self.filter_value)
        return logits


def chained_logits_wraper(xs: List[LogitsWarper], inplace: bool = False):

    def foo(
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        for x in xs:
            logits = x(input_ids, logits, inplace)
        return logits

    return foo


def unioned_logits_wraper(xs: List[LogitsWarper], inplace: bool = False):

    def foo(
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
    ) -> torch.FloatTensor:
        processed_logits = [x(input_ids, logits, inplace=False) for x in xs]
        masks = [logits != pl for pl in processed_logits]
        mask = functools.reduce(torch.logical_or, masks)
        if inplace:
            logits.masked_fill_(mask, torch.finfo(logits.dtype).min)
        else:
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    return foo


def top_k_top_p_logits(
    logits: torch.Tensor,
    top_k=0,
    top_p=1.0,
    inplace: bool = False,
    ordered: bool = False,
) -> torch.FloatTensor:
    if top_p == 1.0 and top_k >= logits.shape[-1]:
        return logits
    if top_p == 1.0:
        return TopKLogitsWarper(top_k=top_k)(None, logits, inplace=inplace)
    if top_k >= logits.shape[-1]:
        return TopPLogitsWarper(top_p=top_p)(None, logits, inplace=inplace)
    warper_fn = unioned_logits_wraper if not ordered else chained_logits_wraper
    p = warper_fn(
        [TopKLogitsWarper(top_k=top_k), TopPLogitsWarper(top_p=top_p)],
        inplace=inplace,
    )
    return p(None, logits)
