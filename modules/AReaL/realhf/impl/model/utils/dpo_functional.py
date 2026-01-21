# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import Tuple

import torch
import torch.nn.functional as F


def dpo_loss(
    pi_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float,
):
    assert len(pi_logps.shape) == 1 and pi_logps.shape[0] % 2 == 0, (
        pi_logps.shape,
        ref_logps.shape,
    )
    assert len(ref_logps.shape) == 1 and ref_logps.shape[0] % 2 == 0, (
        pi_logps.shape,
        ref_logps.shape,
    )
    pi_logps = pi_logps.view(-1, 2)
    ref_logps = ref_logps.view(-1, 2)
    pi_yw_logps, pi_yl_logps = pi_logps[:, 0], pi_logps[:, 1]
    ref_yw_logps, ref_yl_logps = ref_logps[:, 0], ref_logps[:, 1]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
    pos_score = beta * (pi_yw_logps - ref_yw_logps).detach().sum()
    neg_score = beta * (pi_yl_logps - ref_yl_logps).detach().sum()
    kl = -(pi_logps - ref_logps).detach().sum()
    return losses, pos_score, neg_score, kl
