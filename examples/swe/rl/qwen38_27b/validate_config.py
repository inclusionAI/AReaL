import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from examples.swe.utils import SWEPPOConfig

from areal.api.cli_args import to_structured_cfg
from areal.utils import checkpoint_pointer

root = Path(os.environ["EXPERIMENT_ROOT"])
cfg = OmegaConf.to_object(
    to_structured_cfg(
        OmegaConf.load(os.environ.get("QWEN_CONFIG", str(root / "train.yaml"))),
        SWEPPOConfig,
    )
)
assert cfg.trial_name == os.environ["QWEN_ARENA_TRIAL"]
assert (
    cfg.stats_logger.trial_name
    == cfg.recover.trial_name
    == cfg.saver.trial_name
    == cfg.trial_name
)
assert cfg.gconfig.top_p == cfg.eval_gconfig.top_p == 1.0
assert cfg.gconfig.top_k == cfg.eval_gconfig.top_k == 100000
assert cfg.gconfig.reward_normalization_use_std is True
assert cfg.sglang.disable_radix_cache is False
assert cfg.sglang.enable_memory_saver is True
assert cfg.sglang.disable_cuda_graph is False
assert cfg.actor.scheduling_spec[0].mount == os.environ["QWEN_MOUNTS"]
assert cfg.rollout.scheduling_spec[0].mount == os.environ["QWEN_MOUNTS"]
assert cfg.actor.temperature == cfg.gconfig.temperature == 1.0
assert cfg.actor.reward_norm is None
assert cfg.rollout.return_routed_experts is False
assert cfg.rollout.max_concurrent_rollouts == 16
assert cfg.gconfig.n_samples == 12
assert cfg.train_dataset.batch_size == cfg.rollout.consumer_batch_size == 8
assert cfg.recover.mode == "auto"
assert cfg.recover.freq_steps is None
assert cfg.recover.freq_secs == 7200
assert cfg.recover.no_save_optim is False
assert cfg.recover.no_load_optim is False
assert cfg.actor.megatron.use_checkpoint_opt_param_scheduler is True
assert (
    "python3 -m examples.swe.qwen38_flash_next.patch_sglang_mm_executor"
    not in cfg.rollout.scheduling_spec[0].additional_bash_cmds
)
checkpoint = (
    Path(cfg.recover.fileroot)
    / "checkpoints"
    / os.environ["USER"]
    / cfg.experiment_name
    / cfg.trial_name
)

source = checkpoint_pointer.resolve_checkpoint(str(checkpoint), None)
require_recovery = os.environ.get("QWEN_REQUIRE_RECOVER", "0") == "1"
if require_recovery and source is None:
    raise RuntimeError("Recovery is required, but this trial has no valid checkpoint")
if checkpoint.exists() and any(checkpoint.iterdir()) and source is None:
    raise RuntimeError(
        "Checkpoint directory is nonempty but no valid resume pointer exists"
    )
for key in (
    "QWEN_ISOLATE_REQUEST_CACHE",
    "QWEN_QSA_STABLE_TOPK",
    "QWEN_BATCH_REPLAY_PATH",
    "QWEN_BATCH_REPLAY_PATHS",
    "QWEN_BATCH_SNAPSHOT_DIR",
):
    assert not os.environ.get(key), f"Unexpected diagnostic environment {key}"
assert cfg.actor.megatron.bridge_type == "megatron-bridge"
# Qwen3.5 MODEL_THD VLM is unsupported by the AReaL chunked projection path.
# Use the native LM Head settings verified by the ZJW successful experiment.
assert cfg.actor.megatron.enable_chunked_logits is False
assert cfg.actor.megatron.lm_head_loss_chunk_size == 0
assert cfg.actor.scheduling_spec[0].cmd == "python3 -m qwen27_actor_worker"
assert cfg.actor.path == cfg.tokenizer_path == os.environ["QWEN_MODEL"]
assert cfg.actor.backend == "megatron:d1p4t4c4e1"
assert cfg.sglang.ep_size == 1
assert cfg.sglang.context_length == cfg.gconfig.max_tokens == 262144
assert cfg.actor.scheduling_spec[0].env_vars["QWEN_PLE_CHUNK_TOKENS"] == "0"
assert cfg.actor.scheduling_spec[0].env_vars["QWEN_GDN_CP_COMPAT"] == "0"
result = {
    "schema_valid": True,
    "structured_config_validated": True,
    "trial": cfg.trial_name,
    "sampling": {
        "top_p": cfg.gconfig.top_p,
        "top_k": cfg.gconfig.top_k,
        "temperature": cfg.gconfig.temperature,
    },
    "reward_normalization_use_std": True,
    "cache_enabled": True,
    "r3": False,
    "lm_head": {
        "enable_chunked_logits": cfg.actor.megatron.enable_chunked_logits,
        "lm_head_loss_chunk_size": cfg.actor.megatron.lm_head_loss_chunk_size,
        "reference": "ZJW native VLM",
    },
    "enable_memory_saver": True,
    "disable_cuda_graph": False,
    "max_concurrent_rollouts": 16,
    "n_samples": 12,
    "checkpoint_root": str(checkpoint),
    "resume_checkpoint_available": source is not None,
    "resume_required": require_recovery,
    "recover_mode": cfg.recover.mode,
    "recover_freq_steps": cfg.recover.freq_steps,
    "recover_freq_secs": cfg.recover.freq_secs,
    "checkpoint_saving_retained": True,
    "images": {
        "actor": os.environ["QWEN_ACTOR_IMAGE"],
        "rollout": os.environ["QWEN_ROLLOUT_IMAGE"],
    },
    "custom_flashinfer_kernel_mounted": False,
    "no_trainer_instantiated": True,
}
(root / "structured-config-validation.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("STRUCTURED_CONFIG_REVIEW_PASS " + json.dumps(result), flush=True)
