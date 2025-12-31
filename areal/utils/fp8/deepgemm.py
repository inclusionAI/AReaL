# DeepGEMM detection and configuration
# Adapted from SGLang sglang/srt/layers/deep_gemm_wrapper/configurer.py

from areal.utils.cuda import get_device_sm, is_blackwell


def _compute_enable_deep_gemm() -> bool:
    """Check if DeepGEMM JIT is enabled.

    DeepGEMM requires:
    1. SM90+ GPU (Hopper or newer)
    2. deep_gemm package installed
    """
    try:
        sm_version = get_device_sm()
        if sm_version < 90:
            return False
    except Exception:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return True


# Cache the values at module load time
ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()
DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL


def should_deepgemm_weight_requant_ue8m0(
    weight_block_size: list[int] | tuple[int, int] | None,
) -> bool:
    """Should we requant fp8 weights into UE8M0 format when loading the model.

    UE8M0 format is used by DeepGEMM on Blackwell GPUs for optimal performance.

    From SGLang sglang/srt/model_loader/utils.py.

    Args:
        weight_block_size: Block size for weight quantization, e.g. [128, 128]

    Returns:
        True if UE8M0 requantization should be applied
    """
    return (
        ENABLE_JIT_DEEPGEMM and DEEPGEMM_SCALE_UE8M0 and weight_block_size is not None
    )
