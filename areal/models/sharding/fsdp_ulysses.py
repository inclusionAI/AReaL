# Adapted from verl

from torch.distributed.device_mesh import DeviceMesh

from areal.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)


class FSDPUlyssesShardingManager:
    """A context manager to handle Ulysses sequence parallelism group changes.

    This manager is designed to be used with Fully Sharded Data Parallelism (FSDP)
    when different parts of a model might be sharded across different device meshes,
    each with its own sequence parallelism (SP) group.

    When entering the context, it saves the current global Ulysses SP group and
    sets a new one based on the provided `device_mesh`. Upon exiting the context,
    it restores the original SP group, ensuring that the correct parallelism
    settings are used for different model components.

    Args:
        device_mesh (DeviceMesh | None): The device mesh for a specific model
            component. The manager will use the `sp` dimension of this mesh to
            set the active sequence parallel group. If `None`, the context
            manager has no effect.

    Example:
        >>> # Assuming device_mesh is a valid DeviceMesh object
        >>> with FSDPUlyssesShardingManager(device_mesh):
        ...     # Operations within this block will use the SP group
        ...     # from device_mesh.
        ...     run_model_part()
        >>> # Outside the block, the original SP group is restored.
    """

    def __init__(self, device_mesh: DeviceMesh | None):
        self.device_mesh = device_mesh
        self.prev_sp_group = None

    def __enter__(self):
        if self.device_mesh is not None:
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_mesh is not None:
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
