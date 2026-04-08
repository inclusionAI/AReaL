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
4. ``TpModelWorker._init_model_runner``  (safety net for ModelRunner patch)
5. ``Scheduler.process_input_requests`` + ``SchedulerPPMixin._pp_send_pyobj_to_next_stage``
   (PP event-loop deadlock fix for NCCL weight update)

The hook is intentionally minimal: it chains to any pre-existing
``sitecustomize.py`` via ``importlib`` so that third-party hooks are
preserved.

IMPORTANT: All stdlib imports used by patch helpers (logging, torch, etc.)
MUST be imported at module level BEFORE the __import__ hook is installed,
to prevent infinite recursion when the hook intercepts those imports.
"""

import os as _os

if _os.environ.get("AREAL_SGLANG_PP_FIX") == "1":
    import builtins as _builtins
    import sys as _sys

    # ================================================================
    # Pre-import all stdlib/third-party modules used by patch helpers
    # BEFORE installing the __import__ hook.
    # ================================================================
    import logging as _logging

    _real_import = _builtins.__import__

    # Re-entrancy guard
    _hook_applying_patches = False

    def _import_hook(name, *args, **kwargs):
        global _hook_applying_patches

        result = _real_import(name, *args, **kwargs)

        if _hook_applying_patches:
            return result

        # --- 1. Patch GroupCoordinator (PP weight-tying send/recv fix) ---
        ps = _sys.modules.get("sglang.srt.distributed.parallel_state")
        if (
            ps is not None
            and hasattr(ps, "GroupCoordinator")
            and not getattr(ps, "_areal_pp_fixed", False)
        ):
            ps._areal_pp_fixed = True
            _hook_applying_patches = True
            try:
                _apply_pp_fix(ps.GroupCoordinator)
            finally:
                _hook_applying_patches = False

        # --- 2. Patch Qwen models (PP vocab-size fix) ---
        qwen2 = _sys.modules.get("sglang.srt.models.qwen2")
        if (
            qwen2 is not None
            and hasattr(qwen2, "Qwen2ForCausalLM")
            and not getattr(qwen2, "_areal_vocab_fixed", False)
        ):
            qwen2._areal_vocab_fixed = True
            _hook_applying_patches = True
            try:
                _apply_vocab_fix(qwen2.Qwen2ForCausalLM)
            finally:
                _hook_applying_patches = False

        qwen3 = _sys.modules.get("sglang.srt.models.qwen3")
        if (
            qwen3 is not None
            and hasattr(qwen3, "Qwen3ForCausalLM")
            and not getattr(qwen3, "_areal_vocab_fixed", False)
        ):
            qwen3._areal_vocab_fixed = True
            _hook_applying_patches = True
            try:
                _apply_vocab_fix(qwen3.Qwen3ForCausalLM)
            finally:
                _hook_applying_patches = False

        # --- 3. Patch ModelRunner (PP NCCL rank collision fix) ---
        _try_apply_model_runner_patch()

        # --- 4. Patch TpModelWorker (safety net for ModelRunner) ---
        _try_apply_tp_worker_patch()

        # --- 5. Patch Scheduler PP event-loop deadlock ---
        _try_apply_scheduler_pp_deadlock_fix()

        return result

    # ================================================================
    # Targeted patch attempts (called on every import hook invocation)
    # ================================================================

    def _try_apply_model_runner_patch():
        global _hook_applying_patches
        mr = _sys.modules.get("sglang.srt.model_executor.model_runner")
        if (
            mr is not None
            and hasattr(mr, "ModelRunner")
            and not getattr(mr.ModelRunner, "_areal_pp_rank_fixed", False)
        ):
            mr.ModelRunner._areal_pp_rank_fixed = True
            _hook_applying_patches = True
            try:
                _apply_pp_rank_fix(mr.ModelRunner)
            except Exception as _e:
                _logging.getLogger("areal.patches.sitecustomize").error(
                    "Failed to apply PP rank fix: %s", _e, exc_info=True
                )
                mr.ModelRunner._areal_pp_rank_fixed = False
            finally:
                _hook_applying_patches = False

    def _try_apply_tp_worker_patch():
        global _hook_applying_patches
        for mod_name in (
            "sglang.srt.managers.tp_model_worker",
            "sglang.srt.managers.tp_worker",
        ):
            tp_mod = _sys.modules.get(mod_name)
            if tp_mod is not None and not getattr(
                tp_mod, "_areal_tp_worker_patched", False
            ):
                TpCls = getattr(tp_mod, "TpModelWorker", None)
                if TpCls is not None and hasattr(TpCls, "_init_model_runner"):
                    tp_mod._areal_tp_worker_patched = True
                    _hook_applying_patches = True
                    try:
                        _apply_tp_worker_patch(TpCls)
                    finally:
                        _hook_applying_patches = False

    def _try_apply_scheduler_pp_deadlock_fix():
        global _hook_applying_patches
        # We need BOTH the scheduler module (for process_input_requests)
        # and the PP mixin (for _pp_send_pyobj_to_next_stage).
        # Patch when the scheduler module is loaded (it inherits from PP mixin).
        for sched_mod_name in (
            "sglang.srt.managers.scheduler",
            "sglang.srt.entrypoints.engine.scheduler",
        ):
            sched_mod = _sys.modules.get(sched_mod_name)
            if sched_mod is None:
                continue
            Scheduler = getattr(sched_mod, "Scheduler", None)
            if Scheduler is None:
                continue
            if getattr(Scheduler, "_areal_pp_deadlock_fixed", False):
                continue
            # Also need the PP mixin for _pp_send_pyobj_to_next_stage
            if not hasattr(Scheduler, "_pp_send_pyobj_to_next_stage"):
                continue
            if not hasattr(Scheduler, "process_input_requests"):
                continue
            Scheduler._areal_pp_deadlock_fixed = True
            _hook_applying_patches = True
            try:
                _apply_scheduler_pp_deadlock_fix(Scheduler)
            finally:
                _hook_applying_patches = False

    # ================================================================
    # Patch implementation helpers
    # ================================================================

    def _apply_pp_fix(GroupCoordinator):
        """Fix GroupCoordinator.send/recv: use local group index, not global rank."""
        import torch
        import torch.distributed as dist

        _log = _logging.getLogger("areal.patches.sitecustomize.pp_fix")

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
                "Patched send: global_dst=%s -> local_idx=%s (group_size=%s)",
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
                "Patched recv: global_src=%s -> local_idx=%s (group_size=%s)",
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
        _log = _logging.getLogger("areal.patches.sitecustomize.vocab_fix")
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
                            "Copied embed_tokens.weight -> lm_head.weight on last PP rank"
                        )
            except Exception as e:
                _log.warning("vocab fix load_weights post-hook failed: %s", e)

        ModelCls.__init__ = _patched_init
        ModelCls.load_weights = _patched_load
        _log.info("Patched %s for PP vocab fix", ModelCls.__name__)

    def _apply_pp_rank_fix(ModelRunner):
        """Fix ModelRunner.init_weights_update_group: include pp_rank in the
        NCCL rank computation to avoid rank collisions when PP > 1.

        Without this fix, rank = rank_offset + tp_rank, ignoring pp_rank.
        With PP=2, TP=2: PP0-TP0=1, PP0-TP1=2, PP1-TP0=1(dup!), PP1-TP1=2(dup!).
        Fixed: rank = rank_offset + pp_rank * tp_size + tp_rank.
        """
        _log = _logging.getLogger("areal.patches.sitecustomize.pp_rank_fix")

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
        _log.info("Patched ModelRunner.init_weights_update_group for PP rank fix")

    def _apply_tp_worker_patch(TpModelWorker):
        """Safety net: ensure ModelRunner PP rank fix is applied after
        ModelRunner is fully imported and instantiated."""
        _log = _logging.getLogger("areal.patches.sitecustomize.tp_worker_patch")

        _orig_init_mr = TpModelWorker._init_model_runner

        def _patched_init_model_runner(self, *a, **kw):
            result = _orig_init_mr(self, *a, **kw)
            try:
                mr_mod = _sys.modules.get("sglang.srt.model_executor.model_runner")
                if mr_mod is not None and hasattr(mr_mod, "ModelRunner"):
                    MR = mr_mod.ModelRunner
                    if not getattr(MR, "_areal_pp_rank_fixed", False):
                        MR._areal_pp_rank_fixed = True
                        _apply_pp_rank_fix(MR)
                        _log.info(
                            "[Safety Net] Applied PP rank fix via TpModelWorker hook"
                        )
            except Exception as e:
                _log.error("TpModelWorker safety-net failed: %s", e, exc_info=True)
            return result

        TpModelWorker._init_model_runner = _patched_init_model_runner
        _log.info(
            "Patched TpModelWorker._init_model_runner (safety net for PP rank fix)"
        )

    def _apply_scheduler_pp_deadlock_fix(Scheduler):
        """Fix the PP event-loop deadlock for NCCL weight update operations.

        ROOT CAUSE:
        In event_loop_pp(), the execution order is:
          1. recv_reqs = recv_requests()
          2. process_input_requests(recv_reqs)   ← BLOCKS on NCCL broadcast
          3. _pp_send_pyobj_to_next_stage(recv_reqs)  ← never reached!

        When a weight update request arrives at PP rank 0, step 2 calls
        update_weights_from_distributed() → ModelRunner.update_weights_from_distributed()
        → torch.distributed.broadcast() with handle.wait(). This NCCL collective
        needs ALL PP stages to participate. But PP rank 1 hasn't received the
        request yet (step 3 hasn't executed). Classic NCCL deadlock.

        FIX:
        Wrap process_input_requests to detect NCCL-blocking requests
        (UpdateWeightsFromDistributedReqInput, InitWeightsUpdateGroupReqInput).
        When found on a non-last PP rank, pre-forward recv_reqs to the next
        PP stage BEFORE processing. Set a flag so the normal forwarding step
        in event_loop_pp skips the duplicate send.
        """
        _log = _logging.getLogger("areal.patches.sitecustomize.pp_deadlock_fix")

        # Lazily resolve NCCL-blocking request types
        _nccl_req_types = None

        def _get_nccl_req_types():
            nonlocal _nccl_req_types
            if _nccl_req_types is not None:
                return _nccl_req_types
            try:
                io_mod = _sys.modules.get("sglang.srt.managers.io_struct")
                if io_mod is None:
                    io_mod = _real_import(
                        "sglang.srt.managers.io_struct", fromlist=["*"]
                    )
                types = []
                for attr in (
                    "UpdateWeightsFromDistributedReqInput",
                    "InitWeightsUpdateGroupReqInput",
                ):
                    t = getattr(io_mod, attr, None)
                    if t is not None:
                        types.append(t)
                _nccl_req_types = tuple(types) if types else ()
            except Exception:
                _nccl_req_types = ()
            return _nccl_req_types

        # ------ Patch process_input_requests ------
        _orig_process = Scheduler.process_input_requests

        def _patched_process_input_requests(self, recv_reqs):
            nccl_types = _get_nccl_req_types()
            has_nccl_req = False
            if nccl_types and recv_reqs:
                has_nccl_req = any(isinstance(r, nccl_types) for r in recv_reqs)

            # Pre-forward on non-last PP ranks when NCCL-blocking ops are present
            pp_group = getattr(self, "pp_group", None)
            if (
                has_nccl_req
                and pp_group is not None
                and hasattr(pp_group, "is_last_rank")
                and not pp_group.is_last_rank
            ):
                _log.info(
                    "[PP Deadlock Fix] Pre-forwarding %d reqs to next PP stage "
                    "before NCCL-blocking processing",
                    len(recv_reqs),
                )
                # Commit any previously pending async send
                send_work = getattr(self, "send_req_work", [])
                if send_work:
                    for p2p in send_work:
                        p2p.work.wait()
                    send_work.clear()

                # Forward ALL recv_reqs to next PP stage NOW
                self.send_req_work = self._pp_send_pyobj_to_next_stage(
                    recv_reqs, async_send=True
                )

                # Set flag so event_loop_pp skips its duplicate forwarding
                self._areal_pp_already_forwarded = True

            # Process all requests (may block on NCCL for weight updates,
            # but that's OK because next PP stage already has the requests)
            return _orig_process(self, recv_reqs)

        Scheduler.process_input_requests = _patched_process_input_requests

        # ------ Patch _pp_send_pyobj_to_next_stage ------
        _orig_pp_send = Scheduler._pp_send_pyobj_to_next_stage

        def _patched_pp_send(self, data, async_send=False):
            if getattr(self, "_areal_pp_already_forwarded", False):
                self._areal_pp_already_forwarded = False
                _log.debug(
                    "[PP Deadlock Fix] Skipping duplicate PP send "
                    "(already pre-forwarded)"
                )
                # Return existing send_req_work — it holds the pre-forwarded
                # work handle. The event loop will store it back and commit
                # it in the next iteration, which is correct.
                return getattr(self, "send_req_work", [])
            return _orig_pp_send(self, data, async_send=async_send)

        Scheduler._pp_send_pyobj_to_next_stage = _patched_pp_send

        _log.info(
            "Patched Scheduler.process_input_requests + "
            "_pp_send_pyobj_to_next_stage for PP event-loop deadlock fix"
        )

    # ================================================================
    # Install the hook
    # ================================================================
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
    if _os.path.isfile(_candidate) and _os.path.abspath(
        _candidate
    ) != _os.path.abspath(__file__):
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