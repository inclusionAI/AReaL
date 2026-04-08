"""
tests/torchrun/run_megatron_pp_nccl_distributed.py

Distributed torchrun test script for per-PP-rank NCCL weight synchronisation.

Launch with 8 GPUs to simulate a DP=2, PP=2, TP=2 training topology that
broadcasts weights to a mock inference fleet (also PP=2, TP=2).

Usage:
    torchrun --nproc_per_node=8 tests/torchrun/run_megatron_pp_nccl_distributed.py

What this script does
---------------------
1. Initialises torch.distributed with 8 ranks.
2. Assigns a (dp_rank, pp_rank, tp_rank) coordinate to each rank using a
   fixed DP=2, PP=2, TP=2 mapping.
3. Each PP source rank (dp=0, tp=0) creates a per-PP-rank NCCL sub-group
   with the appropriate world_size.
4. Source ranks broadcast a known tensor through their sub-group.
5. Remaining ranks (acting as mock inference workers) receive the tensor
   and verify its value.
6. Exits 0 on success.

NOTE: This script is a *structural* validation.  It does NOT start real
SGLang servers; instead it uses extra torch.distributed ranks to emulate
the inference-side participants.  For a full integration test with real
SGLang, see the CI pipeline configuration.

GPU requirement: 8 GPUs (or use ``--nproc_per_node=8`` on a multi-GPU node).
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import sys
from typing import Dict, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

DP_SIZE = 2
PP_SIZE = 2
TP_SIZE = 2
WORLD_SIZE = DP_SIZE * PP_SIZE * TP_SIZE  # 8

# Rank mapping: rank = dp_rank * (PP_SIZE * TP_SIZE) + pp_rank * TP_SIZE + tp_rank
#
# rank | dp | pp | tp
# -----+----+----+---
#   0  |  0 |  0 |  0   <- PP source for stage 0
#   1  |  0 |  0 |  1
#   2  |  0 |  1 |  0   <- PP source for stage 1
#   3  |  0 |  1 |  1
#   4  |  1 |  0 |  0
#   5  |  1 |  0 |  1
#   6  |  1 |  1 |  0
#   7  |  1 |  1 |  1


def rank_to_coords(rank):
    # type: (int) -> Tuple[int, int, int]
    """Return (dp_rank, pp_rank, tp_rank) for the given global rank."""
    dp = rank // (PP_SIZE * TP_SIZE)
    remainder = rank % (PP_SIZE * TP_SIZE)
    pp = remainder // TP_SIZE
    tp = remainder % TP_SIZE
    return dp, pp, tp


def is_pp_source(dp, tp):
    # type: (int, int) -> bool
    return dp == 0 and tp == 0


# ---------------------------------------------------------------------------
# Per-PP-rank NCCL group simulation
# ---------------------------------------------------------------------------

def find_free_port():
    # type: () -> int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_per_pp_rank_broadcast(rank, world_size):
    # type: (int, int) -> None
    """Core test logic executed by every rank."""

    dp, pp, tp = rank_to_coords(rank)
    print("[Rank {}] dp={}, pp={}, tp={}".format(rank, dp, pp, tp), flush=True)

    # -- Step 1: Build per-PP-rank sub-groups --------------------------
    # We create two sub-groups (one per PP stage).
    #
    # For PP stage p:
    #   Members = all ranks with pp_rank == p
    #   In DP=2 PP=2 TP=2:
    #     PP stage 0: ranks {0, 1, 4, 5}
    #     PP stage 1: ranks {2, 3, 6, 7}

    pp_groups = {}  # type: Dict[int, object]
    for p in range(PP_SIZE):
        members = []
        for r in range(world_size):
            d, pp_r, t = rank_to_coords(r)
            if pp_r == p:
                members.append(r)
        members.sort()
        group = dist.new_group(ranks=members)
        pp_groups[p] = group
        if rank == 0:
            print("  PP group {}: members={}".format(p, members), flush=True)

    # -- Step 2: PP source ranks broadcast a known tensor --------------
    # Source rank for PP stage p is the one with dp=0, tp=0, pp=p.

    device = torch.device("cuda:{}".format(rank % torch.cuda.device_count()))
    torch.cuda.set_device(device)

    for p in range(PP_SIZE):
        d, pp_r, t = rank_to_coords(rank)
        if pp_r != p:
            # This rank is not in PP stage p.
            continue

        group = pp_groups[p]

        # Determine the source (rank within the sub-group).
        # The source is the rank with dp=0, tp=0 inside this sub-group.
        # In our sorted member list, the source is always index 0
        # (rank 0 for PP0, rank 2 for PP1).
        members_sorted = []
        for r in range(world_size):
            rd, rp, rt = rank_to_coords(r)
            if rp == p:
                members_sorted.append(r)
        members_sorted.sort()
        source_global_rank = members_sorted[0]
        # dist.broadcast uses global rank as src when given a sub-group.

        # Prepare tensor.
        expected_value = float(100 + p)  # PP0 -> 100.0, PP1 -> 101.0
        if rank == source_global_rank:
            tensor = torch.full((64,), expected_value, device=device)
            print(
                "  [Rank {}] Broadcasting {} in PP group {}".format(
                    rank, expected_value, p
                ),
                flush=True,
            )
        else:
            tensor = torch.zeros(64, device=device)

        dist.broadcast(tensor, src=source_global_rank, group=group)

        # -- Step 3: Verify --------------------------------------------
        if not torch.allclose(tensor, torch.full_like(tensor, expected_value)):
            print(
                "  [Rank {}] FAIL: expected {}, got {}".format(
                    rank, expected_value, tensor[:4].tolist()
                ),
                flush=True,
            )
            sys.exit(1)
        else:
            print(
                "  [Rank {}] OK: received {} in PP group {}".format(
                    rank, expected_value, p
                ),
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        print(
            "\n=== All per-PP-rank broadcasts verified successfully ===",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", help="Distributed backend")
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != WORLD_SIZE:
        if rank == 0:
            print(
                "ERROR: expected world_size={} (DP={}, PP={}, TP={}), got {}. "
                "Launch with: torchrun --nproc_per_node={} ...".format(
                    WORLD_SIZE, DP_SIZE, PP_SIZE, TP_SIZE, world_size, WORLD_SIZE
                ),
                file=sys.stderr,
            )
        dist.destroy_process_group()
        sys.exit(1)

    try:
        run_per_pp_rank_broadcast(rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
