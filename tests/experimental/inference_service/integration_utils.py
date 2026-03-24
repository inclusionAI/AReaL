"""Integration test utilities for the experimental gateway.

This module provides helper functions and constants for integration testing
of the gateway module. All imports from areal.* are lazy (inside functions)
for Python 3.10 compatibility.
"""

from __future__ import annotations

import os

import requests
import torch
from huggingface_hub import snapshot_download

# =============================================================================
# Constants
# =============================================================================

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
SERVER_STARTUP_TIMEOUT = 180  # seconds
EXPR_NAME = "test_gateway_controller_integration"
TRIAL_NAME = "trial_0"


# =============================================================================
# Helper Functions
# =============================================================================


def _get_model_path(local_path: str, hf_id: str) -> str:
    """Get model path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the model from HuggingFace Hub using snapshot_download.

    Args:
        local_path: Local path to check first.
        hf_id: HuggingFace model ID to download if local not found.

    Returns:
        Path to the model (either local or downloaded).
    """
    if os.path.exists(local_path):
        return local_path

    try:
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
        )
        return downloaded_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {hf_id}: {e}") from e


def has_gpu() -> bool:
    """Check if GPU is available.

    Returns:
        bool: True if CUDA is available and at least one GPU device is present.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_test_model_path() -> str:
    """Get the model path for tests (lazy evaluation).

    Returns:
        str: Path to the test model, falling back to HuggingFace if not available locally.
    """
    return _get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def check_server_health(base_url: str) -> bool:
    """Check if the inference server is healthy.

    Args:
        base_url: The base URL of the server (e.g., "http://localhost:8000").

    Returns:
        bool: True if the server responds with a 200 status to the /health endpoint.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
