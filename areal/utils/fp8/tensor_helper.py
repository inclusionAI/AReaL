import math

import torch


class FP8BlockwiseTensorHelper(torch.Tensor):
    """A helper wrapper tensor that maps operations on data to operations on both data and scale_inv.

    This is a customized helper class for blockwise FP8 quantization, not a general-purpose tensor.
    It allows conversion functions to work with FP8 tensors as if they were regular tensors,
    while automatically handling the corresponding scale_inv transformations.

    - scale_inv shape mirrors data shape, but each dimension is divided by block_size
    - When data is [M, K], scale_inv is [ceil(M/block), ceil(K/block)]
    - When data is reshaped to [A, B, C], scale_inv is reshaped to [A, ceil(B/block), ceil(C/block)]
      (assuming the reshape is compatible with block boundaries)
    - Operations like chunk, split, cat on data apply to scale_inv on the SAME dimension
    """

    @staticmethod
    def __new__(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        block_size: int = 128,
    ):
        # Create a tensor subclass that wraps rowwise_data
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            rowwise_data.shape,
            dtype=rowwise_data.dtype,
            device=rowwise_data.device,
            requires_grad=False,
        )
        obj._rowwise_data = rowwise_data
        obj._rowwise_scale_inv = rowwise_scale_inv
        obj._block_size = block_size
        return obj

    def __repr__(self) -> str:
        return f"FP8BlockwiseTensorHelper(data={self._rowwise_data}\nscale_inv={self._rowwise_scale_inv}\ndata_shape={self.shape}, scale_shape={self._rowwise_scale_inv.shape}, block_size={self._block_size})"

    def _ceil_div(self, a: int, b: int) -> int:
        return (a + b - 1) // b

    def _map_data_dim_to_scale_dim(self, data_dim_size: int) -> int:
        """Map a data dimension size to the corresponding scale dimension size."""
        return self._ceil_div(data_dim_size, self._block_size)

    def _compute_scale_shape(
        self, old_data_shape: tuple, new_data_shape: tuple, old_scale_shape: tuple
    ) -> tuple:
        """Compute scale_inv shape after a view/reshape operation.

        - When data dimension is divided by block_size to get scale dimension
        - Reshapes should preserve this relationship

        For example:
        - data: [4096, 4096] -> scale: [32, 32]
        - After view to [8, 512, 4096]:
          - scale should be [8, 4, 32]
          - 8 is a factor that doesn't need scaling (it's a "grouping" dimension)
        """
        assert old_data_shape[-1] == new_data_shape[-1], (
            "last dimension of old_data_shape and new_data_shape must be the same"
        )
        # Same number of dimensions
        if len(old_data_shape) == len(new_data_shape):
            assert old_data_shape == new_data_shape, (
                "old_data_shape and new_data_shape must be the same"
            )
            return old_scale_shape

        # For view to dimensions like view(A, B, C, K)
        # Try to infer scale shape
        new_scale_shape = []

        for i, data_dim in enumerate(new_data_shape):
            if i == len(new_data_shape) - 1 or i == len(new_data_shape) - 2:
                # Last two dimensions
                scale_dim = self._map_data_dim_to_scale_dim(data_dim)
                new_scale_shape.append(scale_dim)
            else:
                new_scale_shape.append(data_dim)

        assert math.prod(new_scale_shape) == math.prod(old_scale_shape), (
            f"product of new_scale_shape {new_scale_shape} and old_scale_shape {old_scale_shape} must be the same"
        )
        return tuple(new_scale_shape)

    def chunk(self, chunks: int, dim: int = 0):
        """Chunk operation: split both data and scale_inv along the same dimension."""
        assert self._rowwise_data.ndim == self._rowwise_scale_inv.ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        data_chunks = self._rowwise_data.chunk(chunks, dim=dim)
        scale_inv_chunks = self._rowwise_scale_inv.chunk(chunks, dim=dim)

        return tuple(
            FP8BlockwiseTensorHelper(data, scale_inv, self._block_size)
            for data, scale_inv in zip(data_chunks, scale_inv_chunks)
        )

    def split(self, split_size_or_sections, dim: int = 0):
        """Split operation: split both data and scale_inv along the same dimension."""
        assert self._rowwise_data.ndim == self._rowwise_scale_inv.ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        # Do not split on last two dims
        assert dim < self._rowwise_data.ndim - 2 or dim < -2, (
            "do not split on last two dims"
        )

        data_splits = list(self._rowwise_data.split(split_size_or_sections, dim=dim))
        scale_inv_splits = list(
            self._rowwise_scale_inv.split(split_size_or_sections, dim=dim)
        )

        return tuple(
            FP8BlockwiseTensorHelper(data, scale_inv, self._block_size)
            for data, scale_inv in zip(data_splits, scale_inv_splits)
        )

    def view(self, *shape):
        """View operation: reshape both data and scale_inv.

        When data is reshaped, scale_inv needs to be reshaped correspondingly.
        From hf_load.py pattern:
        - data: view(num_groups, -1, head_dim, hidden) with shape like [8, 16, 128, 4096]
        - scale: view(num_groups, -1, head_dim/block, hidden/block) with shape like [8, 16, 1, 32]
        """
        old_data_shape = self._rowwise_data.shape
        new_data = self._rowwise_data.view(*shape)
        new_data_shape = new_data.shape

        new_scale_shape = self._compute_scale_shape(
            old_data_shape, new_data_shape, self._rowwise_scale_inv.shape
        )
        new_scale = self._rowwise_scale_inv.view(*new_scale_shape)

        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def reshape(self, *shape):
        """Reshape operation: same as view but allows non-contiguous tensors."""
        old_data_shape = self._rowwise_data.shape
        new_data = self._rowwise_data.reshape(*shape)
        new_data_shape = new_data.shape

        new_scale_shape = self._compute_scale_shape(
            old_data_shape, new_data_shape, self._rowwise_scale_inv.shape
        )
        new_scale = self._rowwise_scale_inv.reshape(*new_scale_shape)
        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def __getitem__(self, indices):
        """Indexing operation: slice data and scale_inv accordingly."""
        new_data = self._rowwise_data[indices]

        if isinstance(indices, slice):
            # slicing on first dimension
            start = indices.start if indices.start is not None else 0
            stop = indices.stop if indices.stop is not None else self.shape[0]
            scale_start = start // self._block_size
            scale_stop = self._ceil_div(stop, self._block_size)
            new_scale = self._rowwise_scale_inv[scale_start:scale_stop]
        elif isinstance(indices, int):
            # slicing on first dimension
            scale_idx = indices // self._block_size
            new_scale = self._rowwise_scale_inv[scale_idx : scale_idx + 1]
        elif isinstance(indices, tuple):
            # indexing on multiple dimensions
            scale_indices = []
            for index in indices:
                if isinstance(index, slice):
                    start = index.start if index.start is not None else 0
                    stop = index.stop if index.stop is not None else self.shape[0]
                    scale_start = start // self._block_size
                    scale_stop = self._ceil_div(stop, self._block_size)
                    scale_indices.append(slice(scale_start, scale_stop))
                elif isinstance(index, int):
                    scale_idx = index // self._block_size
                    scale_indices.append(scale_idx)
                else:
                    raise NotImplementedError(
                        f"indexing with {type(index)} is not supported"
                    )
            new_scale = self._rowwise_scale_inv[scale_indices]
        else:
            raise NotImplementedError(f"indexing with {type(indices)} is not supported")

        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    @staticmethod
    def cat(tensors, dim: int = 0):
        """Concatenate FP8BlockwiseTensorHelper instances along specified dimension.

        Both data and scale_inv are concatenated along the same dimension.
        """
        if not tensors:
            raise ValueError("cat expects at least one tensor")
        if not all(isinstance(t, FP8BlockwiseTensorHelper) for t in tensors):
            raise ValueError("All tensors must be FP8BlockwiseTensorHelper instances")

        # Check that all tensors have matching dimensions
        first_ndim = tensors[0]._rowwise_data.ndim
        first_scale_ndim = tensors[0]._rowwise_scale_inv.ndim
        assert first_ndim == first_scale_ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        assert all(t._rowwise_data.ndim == first_ndim for t in tensors), (
            "All tensors must have the same number of dimensions"
        )
        assert all(t._rowwise_scale_inv.ndim == first_scale_ndim for t in tensors), (
            "All scale_inv tensors must have the same number of dimensions"
        )

        block_size = tensors[0]._block_size
        data_tensors = [t._rowwise_data for t in tensors]
        scale_tensors = [t._rowwise_scale_inv for t in tensors]

        new_data = torch.cat(data_tensors, dim=dim)
        new_scale = torch.cat(scale_tensors, dim=dim)

        return FP8BlockwiseTensorHelper(new_data, new_scale, block_size)

    def contiguous(self):
        """Make both data and scale_inv contiguous."""
        return FP8BlockwiseTensorHelper(
            self._rowwise_data.contiguous(),
            self._rowwise_scale_inv.contiguous(),
            self._block_size,
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        """Flatten operation."""
        new_data = self._rowwise_data.flatten(start_dim, end_dim)
        new_scale = self._rowwise_scale_inv.flatten(start_dim, end_dim)
        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def to_pytorch_fp8(
        self, scale_inv_dtype: torch.dtype = torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert FP8BlockwiseTensorHelper to PyTorch float8 tensor.

        This function extracts the data and scale_inv from this FP8BlockwiseTensorHelper
        and converts them to PyTorch FP8 format

        Returns:
            Tuple of (torch_fp8_tensor, scale_inv) where:
            - torch_fp8_tensor: PyTorch float8 tensor (torch.float8_e4m3fn)
            - scale_inv: Inverse scale tensor (1/scale) with compact blockwise shape
                         [M/block_size, K/block_size] without padding
        """
        # rowwise_data is stored in uint8 format, convert to PyTorch FP8
        rowwise_data_uint8 = self._rowwise_data
        torch_fp8_tensor = rowwise_data_uint8.view(torch.float8_e4m3fn)

        # Extract rowwise_scale_inv and remove padding if needed
        # FP8BlockwiseTensorHelper's scale_inv should already be compact, but we verify and trim padding
        rowwise_scale_inv = self._rowwise_scale_inv.to(scale_inv_dtype)

        # Calculate the actual (unpadded) scale_inv shape from the data shape
        data_shape = torch_fp8_tensor.shape
        M = data_shape[0] if len(data_shape) >= 1 else 1
        K = data_shape[-1] if len(data_shape) >= 1 else 1

        # For 2D tensor (M, K), scale_inv should be (M // block_size, K // block_size)
        actual_scale_rows = (M + self._block_size - 1) // self._block_size
        actual_scale_cols = (K + self._block_size - 1) // self._block_size

        # Extract only the valid (non-padded) portion
        scale_inv = rowwise_scale_inv[:actual_scale_rows, :actual_scale_cols].clone()

        return torch_fp8_tensor, scale_inv

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Intercept torch operations and handle them appropriately."""
        if kwargs is None:
            kwargs = {}

        # Handle torch.cat
        if func is torch.ops.aten.cat.default:
            tensors = args[0] if args else kwargs.get("tensors", [])
            if tensors and all(
                isinstance(t, FP8BlockwiseTensorHelper) for t in tensors
            ):
                dim = kwargs.get("dim", 0) if not args or len(args) < 2 else args[1]
                return FP8BlockwiseTensorHelper.cat(tensors, dim=dim)
            else:
                raise RuntimeError(f"cat delegate failed with tensors {tensors}")

        # Handle torch.split
        if func is torch.ops.aten.split.default:
            tensor = args[0]
            if isinstance(tensor, FP8BlockwiseTensorHelper):
                split_size = (
                    args[1] if len(args) > 1 else kwargs.get("split_size_or_sections")
                )
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                return tensor.split(split_size, dim=dim)

        # Handle torch.chunk
        if func is torch.ops.aten.chunk.default:
            tensor = args[0]
            if isinstance(tensor, FP8BlockwiseTensorHelper):
                chunks = args[1] if len(args) > 1 else kwargs.get("chunks")
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                return tensor.chunk(chunks, dim=dim)

        # Default: operate on underlying data and return regular tensor
        # This handles operations that don't preserve FP8 semantics
        def unwrap(x):
            if isinstance(x, FP8BlockwiseTensorHelper):
                return x._rowwise_data
            return x

        new_args = tuple(unwrap(a) for a in args)
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return func(*new_args, **new_kwargs)
