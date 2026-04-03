"""Startup hook for SGLang PP fix — injected via PYTHONPATH.

Only activates when AREAL_SGLANG_PP_FIX=1 is set.
Registers a post-import hook that patches GroupCoordinator.send/recv
when sglang.srt.distributed.parallel_state is imported.
"""

import os as _os

if _os.environ.get("AREAL_SGLANG_PP_FIX") == "1":
    import builtins as _builtins
    import sys as _sys

    _real_import = _builtins.__import__

    def _import_hook(name, *args, **kwargs):
        result = _real_import(name, *args, **kwargs)
        # Check if the target module has been loaded (handles all import styles)
        ps = _sys.modules.get("sglang.srt.distributed.parallel_state")
        if ps is not None and hasattr(ps, "GroupCoordinator") and not getattr(
            ps, "_areal_pp_fixed", False
        ):
            ps._areal_pp_fixed = True
            _builtins.__import__ = _real_import  # Unhook immediately
            _apply_pp_fix(ps.GroupCoordinator)
        return result

    def _apply_pp_fix(GroupCoordinator):
        _orig_send = GroupCoordinator.send
        _orig_recv = GroupCoordinator.recv

        def _patched_send(self, tensor, dst=None):
            if dst is not None and dst >= self.world_size:
                if dst in self.ranks:
                    dst = self.ranks.index(dst)
            return _orig_send(self, tensor, dst)

        def _patched_recv(self, size, dtype, src=None):
            if src is not None and src >= self.world_size:
                if src in self.ranks:
                    src = self.ranks.index(src)
            return _orig_recv(self, size, dtype, src)

        GroupCoordinator.send = _patched_send
        GroupCoordinator.recv = _patched_recv

    _builtins.__import__ = _import_hook

# Chain to original sitecustomize.py (if any exists elsewhere)
_hook_dir = _os.path.dirname(_os.path.abspath(__file__))
try:
    import sys as _sys2
    _filtered = [p for p in _sys2.path if _os.path.abspath(p) != _hook_dir]
    _saved = _sys2.path[:]
    _sys2.path[:] = _filtered
    _self_mod = _sys2.modules.pop("sitecustomize", None)
    try:
        import sitecustomize  # noqa: F811
    except ImportError:
        pass
    finally:
        if _self_mod is not None:
            _sys2.modules["sitecustomize"] = _self_mod
        _sys2.path[:] = _saved
except Exception:
    pass
