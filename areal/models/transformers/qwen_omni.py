"""Omni-model-specific patches for distributed training.

Provides:
- Generator module freezing (Talker, Code2Wav, Token2Wav)
- Thinker language model resolution
- DeepStack TP patch for the Thinker's language model

Note on Ulysses SP: The Thinker's language model is a standard Qwen2 (for
Qwen2.5-Omni) or Qwen3Moe (for Qwen3-Omni) LLM.  Ulysses sequence
parallelism is handled by the generic ``_ulysses_flash_attention_forward``
patch in ``ulyssess_patch.py``, which patches
``transformers.integrations.flash_attention._flash_attention_forward``
globally.
"""

import types

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from areal.utils import logging

logger = logging.getLogger("QwenOmniPatch")

# Generator module name prefixes that should be frozen during RL training.
# Only the Thinker is trained; Talker and audio synthesis modules are excluded.
_GENERATOR_PREFIXES = ("talker", "code2wav", "token2wav")


def freeze_and_exclude_generator(model: nn.Module) -> list[str]:
    """Freeze generator modules and return the list of frozen top-level names.

    Iterates over the model's immediate children and freezes any whose name
    matches a known generator prefix.  Returns the names of frozen modules
    so callers can log or verify.
    """
    frozen: list[str] = []
    for name, child in model.named_children():
        if any(name.startswith(prefix) for prefix in _GENERATOR_PREFIXES):
            for param in child.parameters():
                param.requires_grad = False
            frozen.append(name)
    return frozen


def get_thinker_language_model(model: nn.Module) -> nn.Module:
    """Resolve the Thinker's language model from an Omni model.

    Qwen2.5-Omni / Qwen3-Omni wrap the Thinker under ``model.thinker``
    (when loaded via ``AutoModelForImageTextToText``).  The language model
    is then at ``model.thinker.language_model`` or ``model.thinker.model``.
    """
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        # Fallback: some HF revisions nest under model.model.thinker
        inner = getattr(model, "model", None)
        if inner is not None:
            thinker = getattr(inner, "thinker", None)
    if thinker is None:
        raise AttributeError(
            "Cannot locate the Thinker module on the Omni model. "
            "Expected model.thinker or model.model.thinker."
        )
    lang = getattr(thinker, "language_model", None)
    if lang is not None:
        return lang
    lang = getattr(thinker, "model", None)
    if lang is not None:
        return lang
    raise AttributeError(
        "Cannot locate the language model inside the Thinker. "
        "Expected thinker.language_model or thinker.model."
    )


def patch_qwen_omni_deepstack_process_for_tp(language_model: nn.Module) -> None:
    """Patch ``_deepstack_process`` on the Omni Thinker's language model for TP.

    Follows the same pattern as
    :func:`areal.models.transformers.qwen3_vl.patch_qwen3_vl_deepstack_process_for_tp`.
    """
    if not hasattr(language_model, "_deepstack_process"):
        logger.info("Omni language model has no _deepstack_process; skipping TP patch.")
        return

    original_fn = language_model._deepstack_process

    def patched_deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            device_mesh = hidden_states.device_mesh
            placements = hidden_states.placements
            replicated = [Replicate()] * device_mesh.ndim

            hidden_states = hidden_states.redistribute(placements=replicated)
            hidden_states = hidden_states.to_local().clone()

            hidden_states = original_fn(hidden_states, visual_pos_masks, visual_embeds)

            hidden_states = DTensor.from_local(
                hidden_states, device_mesh, replicated, run_check=False
            )
            return hidden_states.redistribute(placements=placements)
        return original_fn(hidden_states, visual_pos_masks, visual_embeds)

    language_model._deepstack_process = types.MethodType(
        patched_deepstack_process, language_model
    )
    logger.info("Patched Omni Thinker _deepstack_process for TP.")
