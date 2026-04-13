# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import torch  # noqa: F401 (used by callers)


def patch_dist_group_timeout(timeout: timedelta):
    """
    Patch the default timeout for process groups in torch.distributed.

    Args:
        timeout (timedelta): Default timeout to set for all process group backends.
    """
    from torch.distributed import distributed_c10d

    if hasattr(distributed_c10d, "default_pg_timeout"):
        distributed_c10d.default_pg_timeout = timeout

    if hasattr(distributed_c10d, "default_pg_nccl_timeout"):
        distributed_c10d.default_pg_nccl_timeout = timeout


# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# This is needed because torch.distributed.init_process_group() only creates
# the default global group, and torch.distributed.new_group() only creates
# subgroups of the default group. AReaL needs independent process groups
# for weight synchronization between training and inference engines that
# run in separate launcher contexts (separate init_process_group calls).
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    backend_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    if store is not None and init_method is not None:
        raise RuntimeError("Cannot specify both init_method and store.")

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=backend_options,
        timeout=timeout,
        group_desc=group_name,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
