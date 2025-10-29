"""Tensor serialization utilities for RPC communication.

This module provides utilities to serialize and deserialize PyTorch tensors
for transmission over HTTP/JSON. Tensors are encoded as base64 strings with
metadata stored in Pydantic models.

Assumptions:
- All tensors are on CPU
- Gradient tracking (requires_grad) is not preserved
"""

import base64
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field


class SerializedTensor(BaseModel):
    """Pydantic model for serialized tensor with metadata.

    Attributes
    ----------
    type : str
        Type marker, always "tensor"
    data : str
        Base64-encoded tensor data
    shape : List[int]
        Tensor shape
    dtype : str
        String representation of dtype (e.g., "torch.float32")
    """

    type: Literal["tensor"] = Field(default="tensor")
    data: str
    shape: list[int]
    dtype: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "SerializedTensor":
        """Create SerializedTensor from a PyTorch tensor.

        Assumes tensor is on CPU or will be moved to CPU for serialization.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to serialize

        Returns
        -------
        SerializedTensor
            Serialized tensor with metadata
        """
        # Move to CPU for serialization (detach to avoid gradient tracking)
        cpu_tensor = tensor.detach().cpu()

        # Convert to bytes and encode as base64
        buffer = cpu_tensor.numpy().tobytes()
        data_b64 = base64.b64encode(buffer).decode("utf-8")

        return cls(
            data=data_b64,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct PyTorch tensor from serialized data.

        Returns CPU tensor without gradient tracking.

        Returns
        -------
        torch.Tensor
            Reconstructed CPU tensor
        """
        # Decode base64 to bytes
        buffer = base64.b64decode(self.data.encode("utf-8"))

        # Parse dtype string (e.g., "torch.float32" -> torch.float32)
        dtype_str = self.dtype.replace("torch.", "")
        dtype = getattr(torch, dtype_str)

        # Reconstruct tensor from bytes
        import numpy as np

        np_array = np.frombuffer(buffer, dtype=self._torch_dtype_to_numpy(dtype))
        # Copy the array to make it writable before converting to tensor
        np_array = np_array.copy()
        tensor = torch.from_numpy(np_array).reshape(self.shape)

        # Cast to correct dtype (numpy might have different dtype)
        tensor = tensor.to(dtype)

        return tensor

    @staticmethod
    def _torch_dtype_to_numpy(torch_dtype: torch.dtype):
        """Convert torch dtype to numpy dtype for buffer reading.

        Parameters
        ----------
        torch_dtype : torch.dtype
            PyTorch data type

        Returns
        -------
        numpy.dtype
            Corresponding NumPy data type
        """
        import numpy as np

        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return dtype_map.get(torch_dtype, np.float32)


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value, converting tensors to SerializedTensor dicts.

    This function transparently handles:
    - torch.Tensor -> SerializedTensor dict (CPU only, no gradient tracking)
    - dict -> recursively serialize values
    - list/tuple -> recursively serialize elements
    - primitives (int, float, str, bool, None) -> unchanged

    Parameters
    ----------
    value : Any
        Value to serialize (can be nested structure)

    Returns
    -------
    Any
        Serialized value (JSON-compatible with SerializedTensor dicts)
    """
    # Handle None
    if value is None:
        return None

    # Handle torch.Tensor
    if isinstance(value, torch.Tensor):
        return SerializedTensor.from_tensor(value).model_dump()

    # Handle dict - recursively serialize values
    if isinstance(value, dict):
        return {key: serialize_value(val) for key, val in value.items()}

    # Handle list - recursively serialize elements
    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    # Handle tuple - convert to list and recursively serialize
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]

    # Primitives (int, float, str, bool) pass through unchanged
    return value


def deserialize_value(value: Any) -> Any:
    """Recursively deserialize a value, converting SerializedTensor dicts back to tensors.

    This function transparently handles:
    - SerializedTensor dict -> torch.Tensor (CPU, no gradient tracking)
    - dict -> recursively deserialize values
    - list -> recursively deserialize elements
    - primitives -> unchanged

    Parameters
    ----------
    value : Any
        Value to deserialize (potentially containing SerializedTensor dicts)

    Returns
    -------
    Any
        Deserialized value with torch.Tensor objects restored
    """
    # Handle None
    if value is None:
        return None

    # Handle dict - check if it's a SerializedTensor
    if isinstance(value, dict):
        # Check for SerializedTensor marker
        if value.get("type") == "tensor":
            try:
                serialized_tensor = SerializedTensor.model_validate(value)
                return serialized_tensor.to_tensor()
            except Exception:
                # If parsing fails, treat as regular dict
                pass

        # Regular dict - recursively deserialize values
        return {key: deserialize_value(val) for key, val in value.items()}

    # Handle list - recursively deserialize elements
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]

    # Primitives pass through unchanged
    return value
