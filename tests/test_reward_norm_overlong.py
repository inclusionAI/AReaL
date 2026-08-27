# SPDX-License-Identifier: Apache-2.0

"""Row-length penalties survive logical trajectory reward normalization."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization


@pytest.mark.parametrize("supplied_reference", [False, True])
def test_split_row_rewards_require_a_shared_original_reference(supplied_reference):
    """Fail explicitly rather than select an arbitrary branch as the outcome."""
    norm = Normalization(NormConfig(mean_level="group", std_level=None))
    with pytest.raises(RuntimeError, match="row rewards differ"):
        norm(
            torch.tensor([1.0, 0.0]),
            traj_starts=torch.tensor([1, 0]),
            unpenalized_rewards=torch.tensor([1.0, 2.0])
            if supplied_reference
            else None,
        )


def test_singleton_tiny_variance_keeps_existing_scale():
    """The split-reference scale fallback does not change singleton normalization."""
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", group_size=2)
    )
    scores = torch.tensor([1.0, 1.0 + 1e-7])

    expected = norm(scores)
    actual = norm(
        scores,
        traj_starts=torch.tensor([1, 1]),
        unpenalized_rewards=torch.tensor([2.0, 2.0]),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _distributed_split_reference_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        scores = torch.tensor([0.0, 1.0] if rank == 0 else [0.0])
        original = torch.tensor([1.0, 1.0] if rank == 0 else [0.0])
        starts = torch.tensor([1, 0] if rank == 0 else [1])
        for mean_level, std_level in [
            ("batch", "batch"),
            ("batch", "group"),
            ("group", "batch"),
        ]:
            norm = Normalization(
                NormConfig(
                    mean_level=mean_level, std_level=std_level, std_unbiased=False
                )
            )
            actual = norm(
                scores,
                traj_starts=starts,
                group_sizes=[scores.numel()],
                unpenalized_rewards=original,
                reduce_group=dist.group.WORLD,
            )
            if mean_level == "batch":
                expected = torch.tensor([-0.5, 0.5] if rank == 0 else [-0.5]) / (
                    0.5 + norm.eps
                )
            else:
                expected = torch.tensor([-1.0, 0.0] if rank == 0 else [0.0])
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        # One malformed rank must not leave its peer in the mean/std collectives.
        norm = Normalization(NormConfig(mean_level="batch", std_level="batch"))
        invalid_original = torch.tensor([1.0, 2.0] if rank == 0 else [0.0])
        with pytest.raises(RuntimeError, match="row rewards differ"):
            norm(
                scores,
                traj_starts=starts,
                unpenalized_rewards=invalid_original,
                reduce_group=dist.group.WORLD,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo"
)
def test_batch_normalization_split_references_match_real_dp_collectives(tmp_path):
    """Ragged logical units and invalid-reference checks agree across two ranks."""
    mp.spawn(
        _distributed_split_reference_worker,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
