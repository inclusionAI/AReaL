import math
import re

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from torch import Tensor
from torch.nn.parameter import Parameter

from areal.utils.fp8 import quantize_params


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
            assert old_scale_shape == new_data_shape, (
                "old_scale_shape and new_data_shape must be the same"
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


def _all_gather_and_concat(
    tensor: torch.Tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    name: str,
) -> torch.Tensor:
    """All-gather tensor partitions and concatenate along partition dimension."""
    partitions = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(partitions, tensor, group=tp_group)

    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name:
        partitions = [p.chunk(2, dim=0) for p in partitions]
        partitions = [p[0] for p in partitions] + [p[1] for p in partitions]

    # this is bug in megatron's grouped moe.
    partition_dim = (
        1 if "linear_fc2.weight" in name and partition_dim == 0 else partition_dim
    )

    return torch.cat(partitions, dim=partition_dim)


def _all_gather_fp8_blockwise_tensor(
    fp8_tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    name: str,
    block_size: int = 128,
) -> FP8BlockwiseTensorHelper:
    """All-gather a Float8BlockwiseQTensor along the partition dimension.

    Returns FP8BlockwiseTensorHelper that wraps rowwise_data and rowwise_scale_inv.
    This allows conversion functions to work with FP8 tensors as regular tensors.
    """
    gathered_rowwise_data = _all_gather_and_concat(
        fp8_tensor._rowwise_data, tp_size, tp_group, partition_dim, name
    )
    gathered_rowwise_scale_inv = _all_gather_and_concat(
        fp8_tensor._rowwise_scale_inv, tp_size, tp_group, partition_dim, name
    )

    return FP8BlockwiseTensorHelper(
        gathered_rowwise_data, gathered_rowwise_scale_inv, block_size
    )


# Adapted from slime
def all_gather_param(
    name: str, param: Parameter | Tensor, fp8_direct_convert: bool = False
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    if "expert_bias" in name:
        return param

    if not hasattr(param, "tensor_model_parallel"):
        raise ValueError(f"{name} does not have tensor_model_parallel attribute")

    param_is_fp8 = is_float8tensor(param)

    if (
        not param.tensor_model_parallel
        or getattr(param, "parallel_mode", None) == "duplicated"
    ):
        # For FP8 tensors, return the tensor directly without accessing .data
        # because accessing .data on QuantizedTensor triggers __torch_dispatch__
        # which dequantizes the tensor to bfloat16
        if param_is_fp8 and fp8_direct_convert:
            return param
        # If param is TE FP8, .data will implicitly convert TE FP8 to bf16,
        # and then be converted to PyTorch FP8 later in convert_to_hf
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    partition_dim = param.partition_dim
    assert param.partition_stride == 1, "partition_stride != 1 is not supported"

    # Handle FP8 tensors specially
    if param_is_fp8 and fp8_direct_convert:
        # Get block_size from quantization config if available
        # Default to 128 if not specified
        block_size = 128  # TODO: get from quantization_config if available
        return _all_gather_fp8_blockwise_tensor(
            param, tp_size, tp_group, partition_dim, name, block_size
        )

    # bf16/fp32
    param = _all_gather_and_concat(param.data, tp_size, tp_group, partition_dim, name)
    return param


# Adapted from slime
def remove_padding(name: str, param: Parameter | Tensor, vocab_size: int):
    if (
        name == "module.module.embedding.word_embeddings.weight"
        or name == "module.module.output_layer.weight"
    ):
        return param[:vocab_size]
    return param


# Adapted from slime
def convert_qwen3moe_to_hf(
    tf_config: TransformerConfig, name: str, param: Parameter | Tensor
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen3-MoE models should have num_query_groups")

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [
                (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
def convert_qwen2_to_hf(
    tf_config: TransformerConfig, name: str, param: Parameter | Tensor
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen2 models should have num_query_groups")

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
def convert_deepseekv3_to_hf(
    tf_config: TransformerConfig, name: str, param: Parameter | Tensor
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif (
            rest == "self_attention.linear_qkv.layer_norm_weight"
            or rest == "input_layernorm.weight"
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [
                (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)
            ]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
# A registry for conversion functions is more extensible.
_CONVERSION_FN_REGISTRY = {
    "qwen3_moe": convert_qwen3moe_to_hf,
    "qwen2": convert_qwen2_to_hf,
    "qwen3": convert_qwen2_to_hf,
    "deepseekv3": convert_deepseekv3_to_hf,
}


def convert_to_hf(
    tf_config: TransformerConfig,
    model_name: str,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    quantization_config: dict[str, int | str | list[str]] | None = None,
    fp8_direct_convert: bool = False,
):
    """Convert Megatron parameter to HuggingFace format, optionally with FP8 quantization.

    Args:
        tf_config: Transformer configuration
        model_name: Model name (e.g., "qwen2", "qwen3_moe")
        name: Parameter name in Megatron format
        param: Parameter tensor or FP8BlockwiseTensorHelper
        quantization_config: Optional quantization config dict with keys:
            - quant_method: "fp8"
            - fmt: "e4m3"
            - activation_scheme: "dynamic"
            - weight_block_size: Optional tuple/list of [block_m, block_n] for blockwise quantization
        fp8_direct_convert: If True, directly convert TE FP8 tensors to PyTorch FP8 format.
            If False, dequantize TE FP8 to bf16 first, then quantize to PyTorch FP8.

    Returns:
        List of (name, tensor) tuples in HuggingFace format. For FP8 quantization,
        returns both quantized weight and scale tensors.
    """
    for key, conversion_fn in _CONVERSION_FN_REGISTRY.items():
        if key in model_name:
            converted_named_tensors = conversion_fn(tf_config, name, param)
            if quantization_config:
                if fp8_direct_convert:
                    converted_fp8_named_tensors = []
                    for hf_name, hf_tensor in converted_named_tensors:
                        if isinstance(hf_tensor, FP8BlockwiseTensorHelper):
                            # FP8BlockwiseTensorHelper from all_gather
                            weight, scale_inv = hf_tensor.to_pytorch_fp8()
                            converted_fp8_named_tensors.append((hf_name, weight))
                            scale_inv_name = f"{hf_name}_scale_inv"
                            converted_fp8_named_tensors.append(
                                (scale_inv_name, scale_inv)
                            )
                        else:
                            # Keep non-FP8 or non-weight tensors as is
                            converted_fp8_named_tensors.append((hf_name, hf_tensor))
                    return converted_fp8_named_tensors
                else:
                    # Quantize from bf16 to PyTorch FP8
                    return quantize_params(
                        name, converted_named_tensors, quantization_config
                    )
            return converted_named_tensors

    raise ValueError(f"Unsupported model for HF conversion: {model_name}")


def get_named_parameters(model_module, num_experts):
    def _iter_single(single_module):
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        if num_experts:
            expert_offset = ep_rank * num_experts // ep_size
        else:
            expert_offset = 0

        config = getattr(single_module, "config", None)
        if config is None and hasattr(single_module, "module"):
            config = getattr(single_module.module, "config", None)
        if config is None:
            raise AttributeError("Megatron module does not expose transformer config")

        vp_stage = getattr(single_module, "virtual_pipeline_model_parallel_rank", None)
        if vp_stage is None and hasattr(single_module, "module"):
            vp_stage = getattr(
                single_module.module, "virtual_pipeline_model_parallel_rank", None
            )
        if vp_stage is None:
            try:
                vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()
            except AssertionError:
                vp_stage = None

        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for name, param in single_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # mtp layer starts from layer 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in single_module.named_buffers():
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer

    if isinstance(model_module, (list, tuple)):
        try:
            vp_world = mpu.get_virtual_pipeline_model_parallel_world_size()
            original_vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        except AssertionError:
            original_vp_rank = None
            vp_world = None

        for vpp_rank, single_module in enumerate(model_module):
            if vp_world and vp_world > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            yield from _iter_single(single_module)

        if (
            vp_world
            and vp_world > 1
            and original_vp_rank is not None
            and original_vp_rank >= 0
        ):
            mpu.set_virtual_pipeline_model_parallel_rank(original_vp_rank)
        return

    yield from _iter_single(model_module)
