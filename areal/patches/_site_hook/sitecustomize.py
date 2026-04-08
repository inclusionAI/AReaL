"""
sitecustomize.py — Auto-injected into every Python subprocess via PYTHONPATH.

When the environment variable ``AREAL_SGLANG_PP_FIX`` is set to ``"1"``,
this module installs a lightweight ``builtins.__import__`` hook that
monkey-patches SGLang internals **the moment they are imported** by
the subprocess.

Currently patched targets
-------------------------
1. ``GroupCoordinator.send / .recv``  (PP weight-tying rank fix)
2. ``Qwen2ForCausalLM / Qwen3ForCausalLM``  (PP vocab-size fix)
3. ``ModelRunner.init_weights_update_group``  (PP NCCL rank collision fix)

The hook is intentionally minimal: it chains to any pre-existing
``sitecustomize.py`` via ``importlib`` so that third-party hooks are
preserved.
"""

import os as _os

if _os.environ.get("AREAL_SGLANG_PP_FIX") == "1":
    import builtins as _builtins
    import sys as _sys

    _real_import = _builtins.__import__

    def _import_hook(name, *args, **kwargs):
        result = _real_import(name, *args, **kwargs)

        # --- 1. Patch GroupCoordinator (PP weight-tying send/recv fix) ---
        ps = _sys.modules.get("sglang.srt.distributed.parallel_state")
        if (
            ps is not None
            and hasattr(ps, "GroupCoordinator")
            and not getattr(ps, "_areal_pp_fixed", False)
        ):
            ps._areal_pp_fixed = True
            _apply_pp_fix(ps.GroupCoordinator)

        # --- 2. Patch Qwen models (PP vocab-size fix) ---
        qwen2 = _sys.modules.get("sglang.srt.models.qwen2")
        if (
            qwen2 is not None
            and hasattr(qwen2, "Qwen2ForCausalLM")
            and not getattr(qwen2, "_areal_vocab_fixed", False)
        ):
            qwen2._areal_vocab_fixed = True
            _apply_vocab_fix(qwen2.Qwen2ForCausalLM)

        qwen3 = _sys.modules.get("sglang.srt.models.qwen3")
        if (
            qwen3 is not None
            and hasattr(qwen3, "Qwen3ForCausalLM")
            and not getattr(qwen3, "_areal_vocab_fixed", False)
        ):
            qwen3._areal_vocab_fixed = True
            _apply_vocab_fix(qwen3.Qwen3ForCausalLM)

        # --- 3. Patch ModelRunner (PP NCCL rank collision fix) ---
        mr = _sys.modules.get("sglang.srt.model_executor.model_runner")
        if (
            mr is not None
            and hasattr(mr, "ModelRunner")
            and not getattr(mr.ModelRunner, "_areal_pp_rank_fixed", False)
        ):
            _apply_pp_rank_fix(mr.ModelRunner)

        return result

    # ---- Patch helpers ----

    def _apply_pp_fix(GroupCoordinator):
        """Fix GroupCoordinator.send/recv: use local group index, not global rank."""
        import logging

        import torch
        import torch.distributed as dist

        _log = logging.getLogger("areal.patches.sitecustomize.pp_fix")

        _orig_send = GroupCoordinator.send
        _orig_recv = GroupCoordinator.recv

        def _patched_send(self, tensor: torch.Tensor, dst: int = None):
            if dst is None:
                dst_idx = self.rank_in_group + 1
            elif isinstance(dst, int) and dst >= self.world_size:
                dst_idx = self.rank_in_group + 1
            else:
                group_ranks = dist.get_process_group_ranks(self.device_group)
                if dst in group_ranks:
                    dst_idx = group_ranks.index(dst)
                else:
                    dst_idx = self.rank_in_group + 1
            _log.debug(
                "Patched send: global_dst=%s → local_idx=%s (group_size=%s)",
                dst,
                dst_idx,
                self.world_size,
            )
            return _orig_send(self, tensor, dst_idx)

        def _patched_recv(self, tensor: torch.Tensor, src: int = None):
            if src is None:
                src_idx = self.rank_in_group - 1
            elif isinstance(src, int) and src >= self.world_size:
                src_idx = self.rank_in_group - 1
            else:
                group_ranks = dist.get_process_group_ranks(self.device_group)
                if src in group_ranks:
                    src_idx = group_ranks.index(src)
                else:
                    src_idx = self.rank_in_group - 1
            _log.debug(
                "Patched recv: global_src=%s → local_idx=%s (group_size=%s)",
                src,
                src_idx,
                self.world_size,
            )
            return _orig_recv(self, tensor, src_idx)

        GroupCoordinator.send = _patched_send
        GroupCoordinator.recv = _patched_recv
        _log.info("Patched GroupCoordinator.send/recv for PP fix")

    def _apply_vocab_fix(ModelCls):
        """Fix tie_word_embeddings handling for PP."""
        import logging

        _log = logging.getLogger("areal.patches.sitecustomize.vocab_fix")
        _orig_init = ModelCls.__init__
        _orig_load = ModelCls.load_weights

        def _patched_init(self, *a, **kw):
            config = a[0] if a else kw.get("config")
            saved_tie = None
            if config is not None and getattr(config, "tie_word_embeddings", False):
                saved_tie = config.tie_word_embeddings
                config.tie_word_embeddings = False
            _orig_init(self, *a, **kw)
            if saved_tie is not None:
                config.tie_word_embeddings = saved_tie

        def _patched_load(self, weights):
            _orig_load(self, weights)
            try:
                pp_group = None
                try:
                    from sglang.srt.distributed.parallel_state import get_pp_group

                    pp_group = get_pp_group()
                except Exception:
                    pass

                if pp_group is not None and pp_group.is_last_rank:
                    embed = dict(self.named_parameters()).get(
                        "model.embed_tokens.weight"
                    )
                    lm_head = dict(self.named_parameters()).get("lm_head.weight")
                    if embed is not None and lm_head is not None:
                        lm_head.data.copy_(embed.data)
                        _log.info(
                            "Copied embed_tokens.weight → lm_head.weight on last PP rank"
                        )
            except Exception as e:
                _log.warning("vocab fix load_weights post-hook failed: %s", e)

        ModelCls.__init__ = _patched_init
        ModelCls.load_weights = _patched_load
        _log.info("Patched %s for PP vocab fix", ModelCls.__name__)

    def _apply_pp_rank_fix(ModelRunner):
        """Fix ModelRunner.init_weights_update_group: include pp_rank in the
        NCCL rank computation to avoid rank collisions when PP > 1."""
        import logging

        _log = logging.getLogger("areal.patches.sitecustomize.pp_rank_fix")

        _orig_init_group = ModelRunner.init_weights_update_group

        def _patched_init_weights_update_group(
            self,
            master_address,
            master_port,
            rank_offset,
            world_size,
            group_name,
            backend="nccl",
        ):
            pp_rank = getattr(self, "pp_rank", 0)
            tp_rank = getattr(self, "tp_rank", 0)
            tp_size = getattr(self, "tp_size", 1)

            correct_local_index = pp_rank * tp_size + tp_rank

            _log.info(
                "[PP Rank Fix] init_weights_update_group: "
                "pp_rank=%d, tp_rank=%d, tp_size=%d, rank_offset=%d, "
                "original_rank=%d, corrected_rank=%d, world_size=%d, "
                "group=%s",
                pp_rank,
                tp_rank,
                tp_size,
                rank_offset,
                rank_offset + tp_rank,
                rank_offset + correct_local_index,
                world_size,
                group_name,
            )

            saved_tp_rank = self.tp_rank
            self.tp_rank = correct_local_index
            try:
                return _orig_init_group(
                    self,
                    master_address,
                    master_port,
                    rank_offset,
                    world_size,
                    group_name,
                    backend,
                )
            finally:
                self.tp_rank = saved_tp_rank

        ModelRunner.init_weights_update_group = _patched_init_weights_update_group
        ModelRunner._areal_pp_rank_fixed = True
        _log.info("Patched ModelRunner.init_weights_update_group for PP rank fix")

    # Install the hook
    _builtins.__import__ = _import_hook

# Chain to any pre-existing sitecustomize.py
import importlib.util as _importlib_util

_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_prev_path = (
    [p for p in _sys.path if _os.path.abspath(p) != _this_dir]
    if "_sys" in dir()
    else []
)
if not _prev_path:
    import sys as _sys

    _prev_path = [p for p in _sys.path if _os.path.abspath(p) != _this_dir]

for _p in _prev_path:
    _candidate = _os.path.join(_p, "sitecustomize.py")
    if _os.path.isfile(_candidate) and _os.path.abspath(_candidate) != _os.path.abspath(
        __file__
    ):
        _spec = _importlib_util.spec_from_file_location(
            "_prev_sitecustomize", _candidate
        )
        if _spec and _spec.loader:
            _mod = _importlib_util.module_from_spec(_spec)
            try:
                _spec.loader.exec_module(_mod)
            except Exception:
                pass
        break
