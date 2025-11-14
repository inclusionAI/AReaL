import importlib.util
import io
import os
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from enum import IntEnum
from inspect import isclass
from typing import Any, TypeAlias

import numpy as np
import torch
from msgspec import msgpack
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class ExtensionTypeCode(IntEnum):
    RAW_VIEW = 1
    TOKENIZER = 2


bytestr: TypeAlias = bytes | bytearray | memoryview


def tensor_data(obj: torch.Tensor) -> memoryview:
    """Extract the raw bytes from a tensor."""
    return memoryview(obj.detach().numpy().tobytes())


class Serializer:
    """A flexible serialization/deserialization handler for RPC communication.

    This class provides a serialization protocol that supports:
    - PyTorch tensors
    - NumPy arrays
    - Hugging Face tokenizers
    - Dataclasses
    """

    magic_symbol = b"\x7e\x5c\x2e\x5e"

    def __init__(self, size_threshold: int = 1024):
        self.size_threshold = size_threshold
        self.encode_buffer: list[bytestr] | None = None
        self.decode_buffer: Sequence[bytestr] = ()
        self._encoder = msgpack.Encoder(enc_hook=self._enc_hook)

    def serialize(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.encode_buffer = bufs = [b""]
            bufs[0] = self._encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.encode_buffer = None

    def deserialize(
        self, bufs: bytestr | Sequence[bytestr], decoded_type: Any | None = None
    ) -> Any:
        args = () if decoded_type is None else (decoded_type,)
        decoder = msgpack.Decoder(
            *args, dec_hook=self._dec_hook, ext_hook=self._ext_hook
        )
        if isinstance(bufs, bytestr):  # type: ignore
            return decoder.decode(bufs)

        self.decode_buffer = bufs
        try:
            return decoder.decode(bufs[0])
        finally:
            self.decode_buffer = ()

    def _dec_hook(self, decoded_type: type, obj: Any) -> Any:
        """
        Given native types in `obj`, convert to type `t`.
        """
        if isclass(decoded_type):
            if issubclass(decoded_type, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(decoded_type, torch.Tensor):
                return self._decode_tensor(obj)
            if decoded_type is slice:
                return slice(*obj)
        return obj

    def _ext_hook(self, code: int, data: memoryview) -> Any:
        if code == int(ExtensionTypeCode.RAW_VIEW):
            return data
        if code == int(ExtensionTypeCode.TOKENIZER):
            return self._decode_tokenizer(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")

    def _enc_hook(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)

        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        if isinstance(obj, PreTrainedTokenizer | PreTrainedTokenizerFast):
            return self._encode_tokenizer(obj)

        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        raise NotImplementedError(f"Type {type(obj)} is not supported")

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.decode_buffer[data] if isinstance(data, int) else data
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.decode_buffer[data] if isinstance(data, int) else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_tokenizer(
        self, blob: memoryview
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        blob, name_or_path = self._pop_magic_info(blob.tobytes())
        if blob[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic header
            import zstandard as zstd

            blob = zstd.ZstdDecompressor().decompress(blob)

        from transformers import AutoTokenizer

        zip_buffer = io.BytesIO(blob)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_buffer) as zf:
                zf.extractall(tmpdir)
            cls = AutoTokenizer
            tokenizer = cls.from_pretrained(tmpdir)
            if isinstance(tokenizer, PreTrainedTokenizerFast):
                tokenizer.name_or_path = name_or_path.decode("utf-8")
            elif isinstance(tokenizer, PreTrainedTokenizer):
                tokenizer.name_or_path = name_or_path.decode("utf-8")
            return tokenizer

    def _encode_tokenizer(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> bytestr:
        zip_buffer = io.BytesIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            total_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(tmpdir)
                for f in files
            )

            compression = (
                zipfile.ZIP_STORED if total_size < 512 * 1024 else zipfile.ZIP_DEFLATED
            )
            with zipfile.ZipFile(
                zip_buffer, "w", compression=compression, compresslevel=6
            ) as zf:
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        zf.write(
                            os.path.join(root, f),
                            arcname=os.path.relpath(os.path.join(root, f), tmpdir),
                        )

        blob = zip_buffer.getvalue()

        if len(blob) > 20 * 1024 * 1024 and importlib.util.find_spec("zstandard"):
            import zstandard as zstd

            blob = zstd.ZstdCompressor(level=3).compress(blob)
        blob = self._append_magic_info(blob, tokenizer.name_or_path.encode("utf-8"))

        return msgpack.Ext(int(ExtensionTypeCode.TOKENIZER), blob)

    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.encode_buffer is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(int(ExtensionTypeCode.RAW_VIEW), arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.encode_buffer)
            self.encode_buffer.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `encode_buffer`.
        return obj.dtype.str, obj.shape, data

    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.encode_buffer is not None
        # view the tensor as a contiguous 1D array of bytes
        arr_data = tensor_data(obj)
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(int(ExtensionTypeCode.RAW_VIEW), arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.encode_buffer)
            self.encode_buffer.append(arr_data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    def _append_magic_info(self, blob: bytes, info: bytes) -> bytes:
        """
        Append magic symbol and info to the end of the blob with specified length.
        [raw blob] + [info] + [magic symbol] + [length of info] + [magic symbol]
        """
        return (
            blob
            + info
            + self.magic_symbol
            + len(info).to_bytes(4, "big")
            + self.magic_symbol
        )

    def _pop_magic_info(self, blob: bytes) -> tuple[bytes, bytes]:
        if blob[-len(self.magic_symbol) :] != self.magic_symbol:
            return blob, b""

        info_len = int.from_bytes(
            blob[-len(self.magic_symbol) - 4 : -len(self.magic_symbol)], "big"
        )

        info_end = -len(self.magic_symbol) - 4 - len(self.magic_symbol)
        info_start = info_end - info_len
        info = blob[info_start:info_end]
        raw_blob = blob[:info_start]
        return raw_blob, info


serializer = Serializer()

serialize = serializer.serialize
deserialize = serializer.deserialize
