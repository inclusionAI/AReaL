"""
SGLang-side patches for per-PP-rank NCCL weight update.

This module provides TWO critical patches:

1. patch_sglang_backend():
   Monkey-patches SGLangBackend.build_init_weights_group_request to support PP > 1.
   The original raises NotImplementedError for PP > 1. The patched version computes
   world_size = tp_size + 1 and rank_offset = 1 for per-PP-rank groups.

2. patch_sglang_scheduler_source():
   Modifies the SGLang scheduler source file on disk to add pp_rank-based filtering.
   This is necessary because SGLang runs as a separate subprocess and its PP forwarding
   chain broadcasts every request to ALL PP*TP schedulers. Without filtering, non-target
   PP workers would try to join NCCL groups they don't belong to, causing rank collisions
   and deadlocks.

   The scheduler patch adds checks to init_weights_update_group and
   update_weights_from_distributed that skip requests where the group_name
   is "areal-pp_{N}" and the scheduler's pp_rank != N.

MUST be called BEFORE SGLang server is launched.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("SGLangPPPatches")

# ---------------------------------------------------------------------------
#  Patch 1: Monkey-patch SGLangBackend in the AReaL process
# ---------------------------------------------------------------------------

def patch_sglang_backend() -> None:
    """Monkey-patch SGLangBackend.build_init_weights_group_request to support PP > 1.

    The original method raises NotImplementedError when gen_parallel.pp_size != 1.
    The patched version:
    - Uses world_size = tp_size + 1 (not pp*tp + 1)
    - Uses rank_offset = 1 (Megatron is always rank 0 in every per-PP-rank group)
    - Passes meta.nccl_group_name to the SGLang HTTP request
    """
    from areal.engine.sglang_remote import SGLangBackend
    from areal.infra.platforms import current_platform
    from areal.utils.network import format_host_for_url

    _original_build = SGLangBackend.build_init_weights_group_request

    def _pp_build_init_weights_group_request(
        self, addr: str, server_idx: int, meta
    ):
        """Build SGLang init weights group request with PP > 1 support."""
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel

        if gen_parallel.pp_size == 1:
            # Fall back to original for PP=1
            return _original_build(self, addr, server_idx, meta)

        tp_size = gen_parallel.tp_size

        # Per-PP-rank group: world_size = tp_size + 1 (Megatron + TP workers)
        # rank_offset = 1 because Megatron is rank 0, first TP worker is rank 1
        # server_idx is the dp-level server index (0 for dp=1)
        rank_offset = 1 + server_idx * tp_size
        world_size = tp_size + 1

        logger.info(
            "[PP SGLangBackend] build_init_weights_group_request: "
            "addr=%s server_idx=%d pp_size=%d tp_size=%d "
            "group_name=%s rank_offset=%d world_size=%d",
            addr, server_idx, gen_parallel.pp_size, tp_size,
            meta.nccl_group_name, rank_offset, world_size,
        )

        from areal.api.io_struct import HttpRequest
        payload = {
            "master_address": format_host_for_url(meta.nccl_master_address),
            "master_port": str(meta.nccl_master_port),
            "rank_offset": rank_offset,
            "world_size": world_size,
            "backend": current_platform.communication_backend,
            "group_name": meta.nccl_group_name,
        }
        return HttpRequest(endpoint="/init_weights_update_group", payload=payload)

    SGLangBackend.build_init_weights_group_request = _pp_build_init_weights_group_request
    logger.info("[patch_sglang_backend] SGLangBackend patched for PP > 1 support.")


# ---------------------------------------------------------------------------
#  Patch 2: Modify SGLang scheduler source file for pp_rank filtering
# ---------------------------------------------------------------------------

# The code to inject into the scheduler mixin
_SCHEDULER_INIT_FILTER = '''
    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group.
        [PATCHED by AReaL per-PP-rank weight update]
        Filters by pp_rank for per-PP-rank groups (group_name = "areal-pp_{N}").
        """
        import re as _re
        _match = _re.match(r"areal-pp_(\\d+)", recv_req.group_name)
        if _match:
            _target_pp = int(_match.group(1))
            if hasattr(self, "pp_rank") and self.pp_rank != _target_pp:
                logger.info(
                    f"[PP Filter] Skipping init_weights_update_group: "
                    f"my pp_rank={self.pp_rank}, target={_target_pp}, "
                    f"group={recv_req.group_name}"
                )
                return InitWeightsUpdateGroupReqOutput(
                    True, f"Skipped: pp_rank {self.pp_rank} != target {_target_pp}"
                )
            logger.info(
                f"[PP Filter] Processing init_weights_update_group: "
                f"pp_rank={self.pp_rank}, group={recv_req.group_name}"
            )
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)
'''

_SCHEDULER_UPDATE_FILTER = '''
    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter.
        [PATCHED by AReaL per-PP-rank weight update]
        Filters by pp_rank for per-PP-rank groups (group_name = "areal-pp_{N}").
        """
        import re as _re
        _match = _re.match(r"areal-pp_(\\d+)", recv_req.group_name)
        if _match:
            _target_pp = int(_match.group(1))
            if hasattr(self, "pp_rank") and self.pp_rank != _target_pp:
                logger.info(
                    f"[PP Filter] Skipping update_weights_from_distributed: "
                    f"my pp_rank={self.pp_rank}, target={_target_pp}, "
                    f"group={recv_req.group_name}"
                )
                return UpdateWeightsFromDistributedReqOutput(
                    True, f"Skipped: pp_rank {self.pp_rank} != target {_target_pp}"
                )
            logger.info(
                f"[PP Filter] Processing update_weights_from_distributed: "
                f"pp_rank={self.pp_rank}, group={recv_req.group_name}, "
                f"n_params={len(recv_req.names)}"
            )
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)
'''

_PATCH_MARKER = "# [PATCHED by AReaL per-PP-rank weight update]"


def _find_sglang_scheduler_mixin_path() -> Path | None:
    """Find the installed SGLang scheduler_update_weights_mixin.py file."""
    # Try to find via importlib
    try:
        spec = importlib.util.find_spec("sglang.srt.managers.scheduler_update_weights_mixin")
        if spec and spec.origin:
            return Path(spec.origin)
    except (ModuleNotFoundError, ValueError):
        pass

    # Search common paths
    for base in sys.path:
        candidate = Path(base) / "sglang" / "srt" / "managers" / "scheduler_update_weights_mixin.py"
        if candidate.exists():
            return candidate

    # Search in site-packages
    import site
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(site_dir) / "sglang" / "srt" / "managers" / "scheduler_update_weights_mixin.py"
        if candidate.exists():
            return candidate

    return None


def patch_sglang_scheduler_source(sglang_path: str | None = None) -> bool:
    """Modify SGLang scheduler source file to add pp_rank filtering.

    This patches two methods in SchedulerUpdateWeightsMixin:
    - init_weights_update_group: skip if group_name="areal-pp_{N}" and pp_rank != N
    - update_weights_from_distributed: skip if group_name="areal-pp_{N}" and pp_rank != N

    The patch is idempotent: running it multiple times is safe.

    Args:
        sglang_path: Optional explicit path to scheduler_update_weights_mixin.py.
                     If None, auto-detects the installed location.

    Returns:
        True if patch was applied (or already applied), False if file not found.
    """
    if sglang_path:
        fpath = Path(sglang_path)
    else:
        fpath = _find_sglang_scheduler_mixin_path()

    if fpath is None or not fpath.exists():
        logger.error(
            "[patch_sglang_scheduler_source] Could not find "
            "scheduler_update_weights_mixin.py. Searched sys.path and site-packages. "
            "Please provide the path explicitly via sglang_path argument."
        )
        return False

    logger.info("[patch_sglang_scheduler_source] Found SGLang scheduler mixin at: %s", fpath)

    content = fpath.read_text()

    # Check if already patched
    if _PATCH_MARKER in content:
        logger.info("[patch_sglang_scheduler_source] Already patched, skipping.")
        return True

    # Create backup
    backup_path = fpath.with_suffix(".py.bak_areal_pp")
    if not backup_path.exists():
        shutil.copy2(fpath, backup_path)
        logger.info("[patch_sglang_scheduler_source] Backup created: %s", backup_path)

    # Replace init_weights_update_group method
    init_pattern = (
        r'    def init_weights_update_group\(\s*\n'
        r'        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput\s*\n'
        r'    \):\s*\n'
        r'        """Initialize the online model parameter update group\."""\s*\n'
        r'        success, message = self\.tp_worker\.init_weights_update_group\(recv_req\)\s*\n'
        r'        return InitWeightsUpdateGroupReqOutput\(success, message\)'
    )

    init_replacement = _PATCH_MARKER + "\n" + _SCHEDULER_INIT_FILTER.strip()

    new_content, init_count = re.subn(init_pattern, init_replacement, content)
    if init_count == 0:
        logger.warning(
            "[patch_sglang_scheduler_source] Could not find init_weights_update_group "
            "method to patch. The SGLang version may have changed. "
            "Attempting line-based replacement..."
        )
        new_content = _line_based_patch_init(content)
        if new_content is None:
            logger.error("[patch_sglang_scheduler_source] Line-based init patch also failed.")
            return False

    # Replace update_weights_from_distributed method
    update_pattern = (
        r'    def update_weights_from_distributed\(\s*\n'
        r'        self,\s*\n'
        r'        recv_req: UpdateWeightsFromDistributedReqInput,\s*\n'
        r'    \) -> Tuple\[bool, str\]:\s*\n'
        r'        """Update the online model parameter\."""\s*\n'
        r'        success, message = self\.tp_worker\.update_weights_from_distributed\(recv_req\)\s*\n'
        r'        if success:\s*\n'
        r'            if recv_req\.flush_cache:\s*\n'
        r'                flush_cache_success = self\.flush_cache\(\)\s*\n'
        r'                assert flush_cache_success, "Cache flush failed after updating weights"\s*\n'
        r'        else:\s*\n'
        r'            logger\.error\(message\)\s*\n'
        r'        return UpdateWeightsFromDistributedReqOutput\(success, message\)'
    )

    update_replacement = _PATCH_MARKER + "\n" + _SCHEDULER_UPDATE_FILTER.strip()

    final_content, update_count = re.subn(update_pattern, update_replacement, new_content)
    if update_count == 0:
        logger.warning(
            "[patch_sglang_scheduler_source] Could not find update_weights_from_distributed "
            "method to patch. Attempting line-based replacement..."
        )
        final_content = _line_based_patch_update(new_content)
        if final_content is None:
            logger.error("[patch_sglang_scheduler_source] Line-based update patch also failed.")
            return False

    # Write patched file
    fpath.write_text(final_content)
    logger.info(
        "[patch_sglang_scheduler_source] Successfully patched %s "
        "(init_weights: %s, update_weights: %s)",
        fpath,
        "regex" if init_count > 0 else "line-based",
        "regex" if update_count > 0 else "line-based",
    )
    return True


def _line_based_patch_init(content: str) -> str | None:
    """Fallback: replace init_weights_update_group using line-by-line matching."""
    lines = content.split("\n")
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        if (
            "def init_weights_update_group(" in line
            and "self: Scheduler" in (lines[i + 1] if i + 1 < len(lines) else "")
        ):
            # Find the end of this method (next method or end of class)
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith("def ") and j > i + 1:
                    break
                if lines[j].strip().startswith("class ") and j > i + 1:
                    break
                j += 1
            # Replace with our patched version
            new_lines.append(_PATCH_MARKER)
            new_lines.append(_SCHEDULER_INIT_FILTER.rstrip())
            i = j
            patched = True
        else:
            new_lines.append(line)
            i += 1

    return "\n".join(new_lines) if patched else None


def _line_based_patch_update(content: str) -> str | None:
    """Fallback: replace update_weights_from_distributed using line-by-line matching."""
    lines = content.split("\n")
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        if (
            "def update_weights_from_distributed(" in line
            and "UpdateWeightsFromDistributedReqInput" in (lines[i + 2] if i + 2 < len(lines) else "")
        ):
            # Find the end of this method
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith("def ") and j > i + 1:
                    break
                if lines[j].strip().startswith("class ") and j > i + 1:
                    break
                j += 1
            new_lines.append(_PATCH_MARKER)
            new_lines.append(_SCHEDULER_UPDATE_FILTER.rstrip())
            i = j
            patched = True
        else:
            new_lines.append(line)
            i += 1

    return "\n".join(new_lines) if patched else None


def restore_sglang_scheduler_source(sglang_path: str | None = None) -> bool:
    """Restore the original SGLang scheduler source from backup.

    Returns True if restored, False if backup not found.
    """
    if sglang_path:
        fpath = Path(sglang_path)
    else:
        fpath = _find_sglang_scheduler_mixin_path()

    if fpath is None:
        logger.error("[restore] Could not find scheduler mixin path.")
        return False

    backup_path = fpath.with_suffix(".py.bak_areal_pp")
    if not backup_path.exists():
        logger.warning("[restore] No backup found at %s", backup_path)
        return False

    shutil.copy2(backup_path, fpath)
    logger.info("[restore] Restored original scheduler from %s", backup_path)
    return True


# ---------------------------------------------------------------------------
#  Combined application
# ---------------------------------------------------------------------------

def apply_all_sglang_pp_patches(sglang_scheduler_path: str | None = None) -> None:
    """Apply all SGLang-side patches for per-PP-rank weight update.

    MUST be called BEFORE SGLang server subprocess is launched.

    1. Patches SGLangBackend (in AReaL process) for PP > 1 support
    2. Patches SGLang scheduler source file (on disk) for pp_rank filtering
    """
    logger.info("[apply_all_sglang_pp_patches] Starting...")

    # Patch 1: SGLangBackend monkey-patch (in-process)
    patch_sglang_backend()

    # Patch 2: SGLang scheduler source file (on disk, for subprocess)
    success = patch_sglang_scheduler_source(sglang_scheduler_path)
    if not success:
        logger.error(
            "[apply_all_sglang_pp_patches] CRITICAL: Failed to patch SGLang scheduler. "
            "Per-PP-rank weight update will NOT work correctly. "
            "All PP*TP workers will try to join every NCCL group, causing deadlock."
        )
        raise RuntimeError(
            "Failed to patch SGLang scheduler source. Cannot proceed with PP > 1 weight update."
        )

    logger.info("[apply_all_sglang_pp_patches] All SGLang patches applied successfully.")
