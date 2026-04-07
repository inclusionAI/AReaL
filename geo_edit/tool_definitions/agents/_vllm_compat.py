"""vLLM compatibility layer for different versions."""

from importlib.metadata import version as _pkg_version
from packaging.version import Version as _V

_vllm_version = _V(_pkg_version("vllm"))

if _vllm_version >= _V("0.11.0") and _vllm_version < _V("0.12.0"):
    from vllm.entrypoints.llm import LLM
else:
    from vllm import LLM

from vllm import SamplingParams

__all__ = ["LLM", "SamplingParams"]
