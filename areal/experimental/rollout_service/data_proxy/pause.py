"""Pause/resume state management for generation coordination.

The controller calls POST /pause on the data proxy to:
  1. Set the PauseState flag to True
  2. Call SGLang POST /pause_generation (aborting in-flight requests)

When ready to resume, the controller calls POST /resume:
  1. Call SGLang POST /continue_generation
  2. Set the PauseState flag to False

SGLangBackend (backend.py) polls PauseState and transparently
resubmits aborted requests once resumed.
"""

from __future__ import annotations

import asyncio

import httpx

from areal.utils import logging

logger = logging.getLogger("RolloutDataProxy")


class PauseState:
    """Async-safe pause flag for weight-update coordination."""

    def __init__(self) -> None:
        self._paused: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def set_paused(self, paused: bool) -> None:
        async with self._lock:
            self._paused = paused

    async def is_paused(self) -> bool:
        # Single bool read is atomic under CPython GIL — no lock needed.
        return self._paused


async def pause_backend(backend_addr: str) -> None:
    """Call SGLang POST /pause_generation to abort in-flight requests."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{backend_addr}/pause_generation", json={})
        resp.raise_for_status()
    logger.info("SGLang pause_generation called on %s", backend_addr)


async def resume_backend(backend_addr: str) -> None:
    """Call SGLang POST /continue_generation to resume inference."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{backend_addr}/continue_generation", json={})
        resp.raise_for_status()
    logger.info("SGLang continue_generation called on %s", backend_addr)
