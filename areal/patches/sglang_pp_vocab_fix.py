"""
Monkey-patch for SGLang PP + TP vocab_size mismatch bug.

FIX STRATEGY:
1. Suppress the buggy send/recv in __init__ by temporarily setting
   config.tie_word_embeddings = False
2. Patch load_weights() to handle PP weight tying correctly: when the last PP rank
   encounters "model.embed_tokens.weight" in the checkpoint, copy it into lm_head.weight.
"""

_PATCHED_MODELS = set()


def _make_patched_init(original_init, model_name):
    def patched_init(self, config, *args, **kwargs):
        original_tie = config.tie_word_embeddings
        if original_tie:
            config.tie_word_embeddings = False
            try:
                original_init(self, config, *args, **kwargs)
            finally:
                config.tie_word_embeddings = original_tie
        else:
            original_init(self, config, *args, **kwargs)
    return patched_init


def _make_patched_load_weights(original_load_weights, model_name):
    def patched_load_weights(self, weights):
        need_pp_tying = (
            hasattr(self, "pp_group")
            and self.pp_group.world_size > 1
            and hasattr(self, "config")
            and getattr(self.config, "tie_word_embeddings", False)
        )
        if not need_pp_tying:
            return original_load_weights(self, weights)

        def _intercepted_weights():
            for name, loaded_weight in weights:
                yield name, loaded_weight
                if (
                    name == "model.embed_tokens.weight"
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

        return original_load_weights(self, _intercepted_weights())
    return patched_load_weights


def apply_sglang_pp_vocab_fix():
    global _PATCHED_MODELS
    model_classes = []
    try:
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM
        if "Qwen2ForCausalLM" not in _PATCHED_MODELS:
            model_classes.append(("Qwen2ForCausalLM", Qwen2ForCausalLM))
    except ImportError:
        pass
    try:
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM
        if "Qwen3ForCausalLM" not in _PATCHED_MODELS:
            model_classes.append(("Qwen3ForCausalLM", Qwen3ForCausalLM))
    except ImportError:
        pass

    for name, cls in model_classes:
        if hasattr(cls, "__init__"):
            orig_init = cls.__init__
            if not getattr(orig_init, "_areal_pp_patched", False):
                patched_init = _make_patched_init(orig_init, name)
                patched_init._areal_pp_patched = True
                cls.__init__ = patched_init
        if hasattr(cls, "load_weights"):
            orig_lw = cls.load_weights
            if not getattr(orig_lw, "_areal_pp_patched", False):
                patched_lw = _make_patched_load_weights(orig_lw, name)
                patched_lw._areal_pp_patched = True
                cls.load_weights = patched_lw
        _PATCHED_MODELS.add(name)
