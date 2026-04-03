"""
Monkey-patch for SGLang PP + TP vocab_size mismatch bug.
"""

_PATCHED_MODELS = set()


def _do_pp_weight_tying(model_instance, config):
    """Perform corrected PP weight tying after model init."""
    pp_group = model_instance.pp_group
    if pp_group.world_size <= 1 or not config.tie_word_embeddings:
        return

    import torch

    if pp_group.is_first_rank:
        pp_group.send(
            model_instance.model.embed_tokens.weight,
            dst=pp_group.last_rank,
        )
    elif pp_group.is_last_rank:
        num_embeddings = model_instance.lm_head.num_embeddings_per_partition
        emb_token_weight = pp_group.recv(
            size=(num_embeddings, config.hidden_size),
            dtype=next(model_instance.model.parameters()).dtype,
            src=pp_group.first_rank,
        )
        model_instance.lm_head.weight.copy_(emb_token_weight)


def _make_patched_init(original_init, model_name):
    def patched_init(self, config, *args, **kwargs):
        original_tie = config.tie_word_embeddings
        if original_tie:
            config.tie_word_embeddings = False
            try:
                original_init(self, config, *args, **kwargs)
            finally:
                config.tie_word_embeddings = original_tie
            _do_pp_weight_tying(self, config)
        else:
            original_init(self, config, *args, **kwargs)
    return patched_init


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
        if not hasattr(cls, "__init__"):
            continue
        original_init = cls.__init__
        if getattr(original_init, "_areal_vocab_patched", False):
            continue
        patched = _make_patched_init(original_init, name)
        patched._areal_vocab_patched = True
        cls.__init__ = patched
        _PATCHED_MODELS.add(name)
