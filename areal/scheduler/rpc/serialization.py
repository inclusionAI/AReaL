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
import importlib.util
import io
import os
import tempfile
import zipfile
from dataclasses import fields, is_dataclass
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from areal.utils import logging

TOKENIZER_ARCHIVE_INLINE_THRESHOLD = 512 * 1024
TOKENIZER_ZSTD_THRESHOLD = 20 * 1024 * 1024
TokenizerCompression = Literal["zip", "zstd"]

logger = logging.getLogger("SyncRPCServer")


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

        # For dtypes that NumPy cannot represent directly (e.g., bfloat16),
        # upcast to a compatible storage dtype for the raw buffer. We keep
        # the original torch dtype in metadata so that deserialization can
        # restore it exactly.
        storage_tensor = cpu_tensor
        if cpu_tensor.dtype is torch.bfloat16:
            storage_tensor = cpu_tensor.to(torch.float32)

        # Convert to bytes and encode as base64
        buffer = storage_tensor.numpy().tobytes()
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

        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            # NumPy does not have a native bfloat16 scalar type in all
            # environments. We store bfloat16 tensors as float32 buffers and
            # map them back via a float32 NumPy view and a final cast in
            # to_tensor().
            torch.bfloat16: np.float32,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return dtype_map.get(torch_dtype, np.float32)


class SerializedNDArray(BaseModel):
    """Pydantic model for serialized NumPy ndarrays.

    Attributes
    ----------
    type : str
        Type marker, always "ndarray"
    data : str
        Base64-encoded contiguous bytes of the array
    shape : List[int]
        Array shape
    dtype : str
        NumPy dtype string representation (e.g., "<f4")
    """

    type: Literal["ndarray"] = Field(default="ndarray")
    data: str
    shape: list[int]
    dtype: str

    @classmethod
    def from_array(cls, array: np.ndarray) -> "SerializedNDArray":
        """Serialize a NumPy array into base64-encoded payload."""

        if array.dtype.kind in {"O", "V"}:
            msg = "Object or void dtype arrays are not supported for serialization"
            raise ValueError(msg)

        contiguous = np.ascontiguousarray(array)
        buffer = contiguous.tobytes()
        data_b64 = base64.b64encode(buffer).decode("utf-8")
        return cls(data=data_b64, shape=list(array.shape), dtype=array.dtype.str)

    def to_array(self) -> np.ndarray:
        """Reconstruct a NumPy array from serialized payload."""

        buffer = base64.b64decode(self.data.encode("utf-8"))
        dtype = np.dtype(self.dtype)
        array = np.frombuffer(buffer, dtype=dtype)
        # Copy to detach from the underlying immutable buffer and ensure writability
        array = array.copy()
        return array.reshape(self.shape)


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


class SerializedTokenizer(BaseModel):
    """Pydantic model for serialized Hugging Face tokenizers.

    Attributes
    ----------
    type : str
        Type marker, always "tokenizer"
    name_or_path : str
        Original ``name_or_path`` attribute captured from the tokenizer
    data : str
        Base64-encoded ZIP (optionally Zstandard-compressed) archive of the tokenizer files
    compression : {"zip", "zstd"}
        Compression algorithm applied to the archive payload
    """

    type: Literal["tokenizer"] = Field(default="tokenizer")
    name_or_path: str
    data: str
    compression: TokenizerCompression = Field(default="zip")

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "SerializedTokenizer":
        """Create a serialized representation from a Hugging Face tokenizer."""

        name_or_path = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        blob = cls._archive_tokenizer(tokenizer)
        blob, compression = cls._maybe_compress(blob)
        data_b64 = base64.b64encode(blob).decode("utf-8")
        return cls(name_or_path=name_or_path, data=data_b64, compression=compression)

    def to_tokenizer(self) -> Any:
        """Reconstruct a Hugging Face tokenizer from serialized data."""

        blob = base64.b64decode(self.data.encode("utf-8"))
        blob = self._maybe_decompress(blob)
        from transformers import AutoTokenizer

        zip_buffer = io.BytesIO(blob)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_buffer) as zf:
                zf.extractall(tmpdir)
            tokenizer = AutoTokenizer.from_pretrained(tmpdir)

        if hasattr(tokenizer, "name_or_path"):
            tokenizer.name_or_path = self.name_or_path
        return tokenizer

    @staticmethod
    def _is_tokenizer(obj: Any) -> bool:
        try:
            from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        except ImportError:  # pragma: no cover - optional dependency
            return False

        return isinstance(obj, (PreTrainedTokenizer, PreTrainedTokenizerFast))

    @staticmethod
    def _archive_tokenizer(tokenizer: Any) -> bytes:
        zip_buffer = io.BytesIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            total_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(tmpdir)
                for file in files
            )
            compression = (
                zipfile.ZIP_STORED
                if total_size < TOKENIZER_ARCHIVE_INLINE_THRESHOLD
                else zipfile.ZIP_DEFLATED
            )
            compress_kwargs = (
                {"compresslevel": 6} if compression == zipfile.ZIP_DEFLATED else {}
            )
            with zipfile.ZipFile(
                zip_buffer, "w", compression=compression, **compress_kwargs
            ) as zf:
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, tmpdir)
                        zf.write(full_path, arcname=arcname)
        return zip_buffer.getvalue()

    @staticmethod
    def _maybe_compress(blob: bytes) -> tuple[bytes, TokenizerCompression]:
        if (
            len(blob) > TOKENIZER_ZSTD_THRESHOLD
            and importlib.util.find_spec("zstandard") is not None
        ):
            import zstandard as zstd

            return zstd.ZstdCompressor(level=3).compress(blob), "zstd"
        return blob, "zip"

    def _maybe_decompress(self, blob: bytes) -> bytes:
        if self.compression == "zip":
            return blob
        if self.compression == "zstd":
            import zstandard as zstd

            return zstd.ZstdDecompressor().decompress(blob)
        msg = f"Unsupported tokenizer compression: {self.compression}"
        raise ValueError(msg)


def serialize_value(value: Any) -> Any:
    """Recursively serialize a value, converting tensors and dataclasses to serialized dicts.

    This function transparently handles:
    - torch.Tensor -> SerializedTensor dict (CPU only, no gradient tracking)
    - numpy.ndarray -> SerializedNDArray dict
    - dataclass instances -> SerializedDataclass dict (preserves type information)
    - Hugging Face tokenizers -> SerializedTokenizer dict
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

    # Handle numpy.ndarray
    if isinstance(value, np.ndarray):
        return SerializedNDArray.from_array(value).model_dump()

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

    if SerializedTokenizer._is_tokenizer(value):
        tokenizer_payload = SerializedTokenizer.from_tokenizer(value)
        return tokenizer_payload.model_dump()

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
    - SerializedNDArray dict -> numpy.ndarray
    - SerializedDataclass dict -> dataclass instance (reconstructed with original type)
    - SerializedTokenizer dict -> Hugging Face tokenizer
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
            except Exception as e:
                # If parsing fails, treat as regular dict
                logger.warning(
                    f"Failed to deserialize dataclass, treating as regular dict: {e}"
                )

        # Check for SerializedTokenizer marker
        if value.get("type") == "tokenizer":
            try:
                serialized_tokenizer = SerializedTokenizer.model_validate(value)
                return serialized_tokenizer.to_tokenizer()
            except Exception as e:
                logger.warning(
                    f"Failed to deserialize tokenizer, treating as regular dict: {e}"
                )

        # Check for SerializedNDArray marker
        if value.get("type") == "ndarray":
            try:
                serialized_array = SerializedNDArray.model_validate(value)
                return serialized_array.to_array()
            except Exception as e:
                logger.warning(
                    f"Failed to deserialize ndarray, treating as regular dict: {e}"
                )

        # Check for SerializedTensor marker
        if value.get("type") == "tensor":
            try:
                serialized_tensor = SerializedTensor.model_validate(value)
                return serialized_tensor.to_tensor()
            except Exception as e:
                logger.warning(
                    f"Failed to deserialize tensor, treating as regular dict: {e}"
                )

        # Regular dict - recursively deserialize values
        return {key: deserialize_value(val) for key, val in value.items()}

    # Handle list - recursively deserialize elements
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]

    # Primitives pass through unchanged
    return value
