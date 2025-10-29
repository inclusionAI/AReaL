"""Tensor and dataclass serialization utilities for RPC communication.

This module provides utilities to serialize and deserialize PyTorch tensors
and dataclass instances for transmission over HTTP/JSON. Tensors are encoded
as base64 strings and dataclasses preserve their type information with metadata
stored in Pydantic models.

Assumptions:
- All tensors are on CPU
- Gradient tracking (requires_grad) is not preserved
- Dataclasses are reconstructed with their original types
"""

import base64
import importlib
from dataclasses import is_dataclass
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


class SerializedDataclass(BaseModel):
    """Pydantic model for serialized dataclass with metadata.

    Attributes
    ----------
    type : str
        Type marker, always "dataclass"
    class_path : str
        Full import path to the dataclass (e.g., "areal.api.cli_args.InferenceEngineConfig")
    data : dict
        Dataclass fields as dictionary (recursively serialized)
    """

    type: Literal["dataclass"] = Field(default="dataclass")
    class_path: str
    data: dict[str, Any]

    @classmethod
    def from_dataclass(cls, dataclass_instance: Any) -> "SerializedDataclass":
        """Create SerializedDataclass from a dataclass instance.

        Parameters
        ----------
        dataclass_instance : Any
            Dataclass instance to serialize

        Returns
        -------
        SerializedDataclass
            Serialized dataclass with metadata
        """
        class_path = (
            f"{dataclass_instance.__class__.__module__}."
            f"{dataclass_instance.__class__.__name__}"
        )
        # Get fields without recursive conversion to preserve nested dataclass instances
        # We'll handle recursive serialization in serialize_value()
        from dataclasses import fields

        data = {}
        for field in fields(dataclass_instance):
            data[field.name] = getattr(dataclass_instance, field.name)

        return cls(class_path=class_path, data=data)

    def to_dataclass(self) -> Any:
        """Reconstruct dataclass instance from serialized data.

        Returns
        -------
        Any
            Reconstructed dataclass instance

        Raises
        ------
        ImportError
            If the dataclass module cannot be imported
        AttributeError
            If the dataclass class is not found in the module
        """
        # Dynamically import the dataclass type
        module_path, class_name = self.class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        dataclass_type = getattr(module, class_name)

        # Return the dataclass type and data for caller to deserialize fields
        return dataclass_type, self.data


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value, converting tensors and dataclasses to serialized dicts.

    This function transparently handles:
    - torch.Tensor -> SerializedTensor dict (CPU only, no gradient tracking)
    - dataclass instances -> SerializedDataclass dict (preserves type information)
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
        Serialized value (JSON-compatible with SerializedTensor and SerializedDataclass dicts)
    """
    # Handle None
    if value is None:
        return None

    # Handle torch.Tensor
    if isinstance(value, torch.Tensor):
        return SerializedTensor.from_tensor(value).model_dump()

    # Handle dataclass instances (check before dict, as dataclasses can be dict-like)
    # Note: is_dataclass returns True for both classes and instances, so check it's not a type
    if is_dataclass(value) and not isinstance(value, type):
        serialized_dc = SerializedDataclass.from_dataclass(value)
        # Recursively serialize the data fields
        serialized_data = {
            key: serialize_value(val) for key, val in serialized_dc.data.items()
        }
        return {
            "type": "dataclass",
            "class_path": serialized_dc.class_path,
            "data": serialized_data,
        }

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
    """Recursively deserialize a value, converting SerializedTensor and SerializedDataclass dicts back.

    This function transparently handles:
    - SerializedTensor dict -> torch.Tensor (CPU, no gradient tracking)
    - SerializedDataclass dict -> dataclass instance (reconstructed with original type)
    - dict -> recursively deserialize values
    - list -> recursively deserialize elements
    - primitives -> unchanged

    Parameters
    ----------
    value : Any
        Value to deserialize (potentially containing SerializedTensor and SerializedDataclass dicts)

    Returns
    -------
    Any
        Deserialized value with torch.Tensor and dataclass objects restored
    """
    # Handle None
    if value is None:
        return None

    # Handle dict - check if it's a SerializedDataclass or SerializedTensor
    if isinstance(value, dict):
        # Check for SerializedDataclass marker (check before tensor)
        if value.get("type") == "dataclass":
            try:
                serialized_dc = SerializedDataclass.model_validate(value)
                dataclass_type, data = serialized_dc.to_dataclass()
                # Recursively deserialize the fields
                deserialized_data = {
                    key: deserialize_value(val) for key, val in data.items()
                }
                # Reconstruct the dataclass instance
                return dataclass_type(**deserialized_data)
            except Exception:
                # If parsing fails, treat as regular dict
                pass

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
