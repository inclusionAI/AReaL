"""
sitecustomize.py — Auto-injected into every Python subprocess via PYTHONPATH.
"""
import os as _os

if _os.environ.get("AREAL_SGLANG_PP_FIX") == "1":
    import builtins as _builtins
    import sys as _sys

    _real_import = _builtins.__import__

    def _import_hook(name, *args, **kwargs):
        result = _real_import(name, *args, **kwargs)
        
        # 1. Patch GroupCoordinator
        ps = _sys.modules.get("sglang.srt.distributed.parallel_state")
        if (
            ps is not None
            and hasattr(ps, "GroupCoordinator")
            and not getattr(ps, "_areal_pp_fixed", False)
        ):
            ps._areal_pp_fixed = True
            _apply_pp_fix(ps.GroupCoordinator)

        # 2. Patch Qwen models
        qwen2 = _sys.modules.get("sglang.srt.models.qwen2")
        if (
            qwen2 is not None 
            and hasattr(qwen2, "Qwen2ForCausalLM")
            and not getattr(qwen2, "_areal_vocab_fixed", False)
        ):
            qwen2._areal_vocab_fixed = True
            _apply_vocab_fix(qwen2.Qwen2ForCausalLM, "Qwen2ForCausalLM")

        qwen3 = _sys.modules.get("sglang.srt.models.qwen3")
        if (
            qwen3 is not None 
            and hasattr(qwen3, "Qwen3ForCausalLM")
            and not getattr(qwen3, "_areal_vocab_fixed", False)
        ):
            qwen3._areal_vocab_fixed = True
            _apply_vocab_fix(qwen3.Qwen3ForCausalLM, "Qwen3ForCausalLM")
            
        # Unhook if all possible target modules have been patched
        # (This is a simplified approach, leaving the hook active is generally fine since it's fast)
        return result

    def _resolve_group_local_rank(coordinator, rank_value, param_name):
        ranks = coordinator.ranks
        world_size = coordinator.world_size
        rank_in_group = coordinator.rank_in_group

        if rank_value >= world_size:
            if rank_value in ranks:
                return ranks.index(rank_value)
            return rank_value

        if rank_value in ranks and ranks.index(rank_value) != rank_value:
            if param_name == "src" and rank_in_group == world_size - 1 and rank_value == ranks[0]:
                return 0
            if param_name == "dst" and rank_in_group == 0 and rank_value == ranks[-1]:
                return world_size - 1

        return rank_value

    def _apply_pp_fix(GroupCoordinator):
        _orig_send = GroupCoordinator.send
        _orig_recv = GroupCoordinator.recv

        def _patched_send(self, tensor, dst=None):
            if dst is not None:
                dst = _resolve_group_local_rank(self, dst, "dst")
            return _orig_send(self, tensor, dst)

        def _patched_recv(self, size, dtype, src=None):
            if src is not None:
                src = _resolve_group_local_rank(self, src, "src")
            return _orig_recv(self, size, dtype, src)

        GroupCoordinator.send = _patched_send
        GroupCoordinator.recv = _patched_recv
        
    def _apply_vocab_fix(model_cls, name):
        if hasattr(model_cls, "__init__"):
            orig_init = model_cls.__init__
            if not getattr(orig_init, "_areal_pp_patched", False):
                def patched_init(self, config, *args, **kwargs):
                    original_tie = config.tie_word_embeddings
                    if original_tie:
                        config.tie_word_embeddings = False
                        try:
                            orig_init(self, config, *args, **kwargs)
                        finally:
                            config.tie_word_embeddings = original_tie
                    else:
                        orig_init(self, config, *args, **kwargs)
                patched_init._areal_pp_patched = True
                model_cls.__init__ = patched_init
                
        if hasattr(model_cls, "load_weights"):
            orig_lw = model_cls.load_weights
            if not getattr(orig_lw, "_areal_pp_patched", False):
                def patched_load_weights(self, weights):
                    need_pp_tying = (
                        hasattr(self, "pp_group")
                        and self.pp_group.world_size > 1
                        and hasattr(self, "config")
                        and getattr(self.config, "tie_word_embeddings", False)
                    )
                    if not need_pp_tying:
                        return orig_lw(self, weights)

                    def _intercepted_weights():
                        for weight_name, loaded_weight in weights:
                            yield weight_name, loaded_weight
                            if (
                                weight_name == "model.embed_tokens.weight"
                                and self.pp_group.is_last_rank
                            ):
                                params_dict = dict(self.named_parameters())
                                if "lm_head.weight" in params_dict:
                                    lm_head_param = params_dict["lm_head.weight"]
                                    weight_loader = getattr(lm_head_param, "weight_loader", None)
                                    if weight_loader is not None:
                                        weight_loader(lm_head_param, loaded_weight)
                                    else:
                                        lm_head_param.data.copy_(
                                            loaded_weight[: lm_head_param.shape[0]]
                                        )

                    return orig_lw(self, _intercepted_weights())
                patched_load_weights._areal_pp_patched = True
                model_cls.load_weights = patched_load_weights

    _builtins.__import__ = _import_hook

# Chain to original sitecustomize.py if one exists
import importlib as _importlib
import importlib.util as _importlib_util

_this_dir = _os.path.dirname(_os.path.abspath(__file__))
_orig_path = [p for p in _sys.path if _os.path.abspath(p) != _this_dir]
for _p in _orig_path:
    _candidate = _os.path.join(_p, "sitecustomize.py")
    if _os.path.isfile(_candidate) and _os.path.abspath(_candidate) != _os.path.abspath(__file__):
        _spec = _importlib_util.spec_from_file_location("_orig_sitecustomize", _candidate)
        if _spec and _spec.loader:
            _mod = _importlib_util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
        break
