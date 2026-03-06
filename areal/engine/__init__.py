__all__ = [
    "FSDPEngine",
    "FSDPPPOActor",
    "FSDPPPOCritic",
    "FSDPLMEngine",
    "FSDPRWEngine",
    "MegatronEngine",
    "MegatronPPOActor",
    "MegatronPPOCritic",
    "MegatronLMEngine",
    "RemoteSGLangEngine",
    "RemotevLLMEngine",
]

_LAZY_IMPORTS = {
    "FSDPEngine": "areal.engine.fsdp_engine",
    "FSDPPPOActor": "areal.engine.fsdp_engine",
    "FSDPPPOCritic": "areal.engine.fsdp_engine",
    "FSDPLMEngine": "areal.engine.fsdp_engine",
    "FSDPRWEngine": "areal.engine.fsdp_engine",
    "MegatronEngine": "areal.engine.megatron_engine",
    "MegatronPPOActor": "areal.engine.megatron_engine",
    "MegatronPPOCritic": "areal.engine.megatron_engine",
    "MegatronLMEngine": "areal.engine.megatron_engine",
    "RemoteSGLangEngine": "areal.engine.sglang_remote",
    "RemotevLLMEngine": "areal.engine.vllm_remote",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
