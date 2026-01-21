# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import functools
import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from realhf.base import constants, logging

logger = logging.getLogger("Modules")


class ExponentialRunningMeanStd(nn.Module):

    def __init__(self, beta=0.999, epsilon=1e-5, high_precision=True):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon

        self.__dtype = torch.float64 if high_precision else torch.float32

        self.__mean = nn.Parameter(
            torch.zeros((1,), dtype=self.__dtype, device=constants.current_device()),
            requires_grad=False,
        )
        self.__mean_sq = nn.Parameter(
            torch.zeros((1,), dtype=self.__dtype, device=constants.current_device()),
            requires_grad=False,
        )
        self.__debiasing_term = nn.Parameter(
            torch.zeros((1,), dtype=self.__dtype, device=constants.current_device()),
            requires_grad=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError()

    @torch.no_grad()
    def update(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x.to(self.__dtype)
        if mask is not None:
            mask = mask.to(self.__dtype)
        if mask is None:
            factor = torch.tensor(np.prod(x.shape), dtype=self.__dtype, device=x.device)
        else:
            x = x * mask
            factor = mask.sum()

        x_sum = x.sum()
        x_sum_sq = x.square().sum()
        if dist.is_initialized():
            dist.all_reduce(
                factor, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
            dist.all_reduce(
                x_sum, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
            dist.all_reduce(
                x_sum_sq, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
        batch_mean = x_sum / factor
        batch_sq_mean = x_sum_sq / factor

        self.__mean.data[:] = self.__beta * self.__mean.data[:] + batch_mean * (
            1.0 - self.__beta
        )
        self.__mean_sq.data[:] = self.__beta * self.__mean_sq.data[
            :
        ] + batch_sq_mean * (1.0 - self.__beta)
        self.__debiasing_term.data[:] = (
            self.__beta * self.__debiasing_term.data[:] + 1.0 - self.__beta
        )

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (
            ((x - mean) / std).clip(-5, 5).float()
        )  # clipping is a trick from hide and seek

    @torch.no_grad()
    def denormalize(self, x):
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (x * std + mean).float()


class MovingAverageRunningMeanStd(nn.Module):

    def __init__(self, high_precision=True):
        super().__init__()

        self.__dtype = torch.float64 if high_precision else torch.float32

        self.__mean = nn.Parameter(
            torch.zeros((1,), dtype=self.__dtype, device=constants.current_device()),
            requires_grad=False,
        )
        self.__mean_sq = nn.Parameter(
            torch.zeros((1,), dtype=self.__dtype, device=constants.current_device()),
            requires_grad=False,
        )
        self.__accum_denominator = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__accum_denominator = 0

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError()

    @torch.no_grad()
    def update(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x.to(self.__dtype)
        if mask is not None:
            mask = mask.to(self.__dtype)
        if mask is None:
            factor = torch.tensor(np.prod(x.shape), dtype=self.__dtype, device=x.device)
        else:
            x = x * mask
            factor = mask.sum()

        x_sum = x.sum()
        x_sum_sq = x.square().sum()
        if dist.is_initialized():
            dist.all_reduce(
                factor, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
            dist.all_reduce(
                x_sum, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )
            dist.all_reduce(
                x_sum_sq, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
            )

        self.__mean.data[:] = (
            self.__accum_denominator * self.__mean.data[:] + x_sum
        ) / (self.__accum_denominator + factor)
        self.__mean_sq.data[:] = (
            self.__accum_denominator * self.__mean_sq.data[:] + x_sum_sq
        ) / (self.__accum_denominator + factor)
        self.__accum_denominator += factor

    @torch.no_grad()
    def mean_std(self):
        return (
            self.__mean.clone(),
            (self.__mean_sq - self.__mean**2).clamp(min=1e-2).sqrt(),
        )

    @torch.no_grad()
    def normalize(self, x):
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (
            ((x - mean) / std).clip(-5, 5).float()
        )  # clipping is a trick from hide and seek

    @torch.no_grad()
    def denormalize(self, x):
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (x * std + mean).float()


class PopArtValueHead(nn.Module):

    def __init__(
        self,
        input_dim,
        critic_dim,
        beta=0.99999,
        epsilon=1e-5,
        burn_in_updates=torch.inf,
        rms_type: str = "exp",
        high_precision=True,
    ):
        super().__init__()
        if rms_type == "exp":
            rms_cls = functools.partial(ExponentialRunningMeanStd, beta=beta)
        elif rms_type == "ma":
            rms_cls = MovingAverageRunningMeanStd
        else:
            raise NotImplementedError(f"Unknown rms type {rms_type}")
        self.__rms = rms_cls(
            input_shape=(critic_dim,),
            epsilon=epsilon,
            high_precision=high_precision,
        )

        self.__weight = nn.Parameter(torch.zeros(critic_dim, input_dim))
        self.__bias = nn.Parameter(torch.zeros(critic_dim))
        # The same initialization as `nn.Linear`.
        torch.nn.init.kaiming_uniform_(self.__weight, a=math.sqrt(5))
        torch.nn.init.uniform_(
            self.__bias, -1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim)
        )

        self.__burn_in_updates = burn_in_updates
        self.__update_cnt = 0

    @property
    def weight(self):
        return self.__weight

    @property
    def bias(self):
        return self.__bias

    def forward(self, feature):
        return torch.nn.functional.linear(feature, self.__weight, self.__bias)

    @torch.no_grad()
    def update(self, x, mask):
        old_mean, old_std = self.__rms.mean_std()
        self.__rms.update(x, mask)
        new_mean, new_std = self.__rms.mean_std()
        self.__update_cnt += 1

        if self.__update_cnt > self.__burn_in_updates:
            self.__weight.data[:] = self.__weight * (old_std / new_std).unsqueeze(-1)
            self.__bias.data[:] = (
                old_std * self.__bias + old_mean - new_mean
            ) / new_std

    @torch.no_grad()
    def normalize(self, x):
        return self.__rms.normalize(x)

    @torch.no_grad()
    def denormalize(self, x):
        return self.__rms.denormalize(x)
