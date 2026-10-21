# SPDX-License-Identifier: Apache-2.0

"""PPO and fixed-batch LM groups for infrastructure lifecycle validation."""

import getpass
import math
import secrets
from pathlib import Path

from examples.autopilot.elastic.swe.trajectory import trajectory_payload

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    SchedulingSpec,
)
from areal.autopilot.contracts import BatchWorkload
from areal.autopilot.elastic.runtime.state.checkpoint import (
    WEIGHT_TYING,
    artifact_manifest,
    atomic_json,
    verify_tied_export,
)
from areal.autopilot.elastic.runtime.training.workers import TrainGroup


class PPOGroup(TrainGroup):
    """Four ranks may replicate one small batch; queue progress counts it once."""

    @staticmethod
    def workload(payload: dict) -> BatchWorkload:
        trajectory = payload["trajectory"]
        return BatchWorkload(
            1,
            payload["generated_tokens"],
            sum(sum(row) for row in trajectory["attention_mask"]),
        )

    def validate_trajectory(self, payload: dict) -> None:
        """Validate on CPU before deciding to allocate training GPUs."""
        validated = trajectory_payload(
            payload["trajectory"],
            task_id=payload["task_id"],
            rollout_id=payload["rollout_id"],
            model_revision=payload["model_revision"],
            policy_version=payload["policy_version"],
        )
        if (
            validated != payload
            or payload["model_revision"] != self.compatibility["model_revision"]
        ):
            raise ValueError("Trajectory provenance or schema mismatch")
        if payload["policy_version"] != self.step:
            raise ValueError(
                "Strict on-policy update requires the current policy version"
            )

    def train_trajectory(self, payload: dict) -> dict:
        self.validate_trajectory(payload)
        from areal.infra.rpc.serialization import serialize_value

        self.actor._gateway_post_result(
            "/ppo/actor/update",
            {
                "args": serialize_value([[payload]]),
                "kwargs": serialize_value({}),
            },
        )
        records = self.actor.rank_records("ppo", self.step + 1)
        for record in records:
            stats = record["optimizer"]
            if (
                stats["update_successful"] != 1
                or not math.isfinite(stats["grad_norm"])
                or stats["grad_norm"] <= 0
            ):
                raise RuntimeError(
                    "PPO optimizer did not complete a finite nonzero-gradient update"
                )
            if record["optimizer_updates"] != 1 or record["parameter_max_delta"] <= 0:
                raise RuntimeError(
                    "Expected one optimizer update and changed model parameters"
                )
        self.actor.step_lr_scheduler()
        self.step += 1
        self.actor.set_version(self.step)
        return {
            "step": self.step,
            "rank_stats": records,
            "real_interactions": len(payload["trajectory"]["input_ids"]),
        }

    def _checkpoint_extra(self, path: Path) -> None:
        serving = path / "hf"
        self.actor.save(
            SaveLoadMeta(path=str(serving), weight_format="hf", with_optim=False)
        )
        verify_tied_export(serving)
        manifest = artifact_manifest(serving)
        if (
            not any(name.endswith(".safetensors") for name in manifest)
            or "tokenizer_config.json" not in manifest
        ):
            raise RuntimeError("Serving export is incomplete")
        atomic_json(
            path / "serving.json",
            {
                "version": self.step,
                "files": manifest,
                "weight_tying": self.compatibility["weight_tying"],
            },
        )


def create_ppo_group(
    *, root: Path, model: Path, revision: str, devices: list[int], elastic: bool = False
) -> PPOGroup:
    """Fixed-baseline critic-free clipped PG, with untouched binary task rewards."""
    config = PPOActorConfig(
        experiment_name="elastic-ppo",
        trial_name="actor",
        path=str(model.resolve()),
        backend=f"fsdp:d{len(devices)}",
        dtype="bfloat16",
        optimizer_dtype="float32",
        _version="v2",
        admin_api_key=secrets.token_urlsafe(32),
        setup_timeout=180,
        workers_ready_timeout=120,
        request_timeout=300,
        disable_dropout=True,
        gradient_checkpointing=True,
        attn_impl="flash_attention_2",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=8192),
        optimizer=OptimizerConfig(
            lr=1e-5,
            weight_decay=0.0,
            lr_scheduler_type="constant",
            warmup_steps_proportion=0.0,
        ),
        reward_bias=-0.5,
        reward_scaling=1.0,
        reward_norm=None,
        adv_norm=None,
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=1.0,
        mask_no_eos_with_zero=False,
        ppo_n_minibatches=1,
        recompute_logprob=False,
        use_decoupled_loss=False,
        temperature=0.6,
        scheduling_spec=(
            SchedulingSpec(
                cpu=4,
                gpu=1,
                env_vars={
                    "LOGNAME": getpass.getuser(),
                    "CUDA_VISIBLE_DEVICES": "",
                    "AREAL_ELASTIC_GPU_POOL": ":".join(map(str, devices)),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NCCL_CUMEM_ENABLE": "0",
                    "AREAL_ELASTIC_DIAGNOSTICS_DIR": str(root / "diagnostics"),
                },
            ),
        ),
    )
    compatibility = {
        "model": str(model.resolve()),
        "model_revision": revision,
        "backend": config.backend,
        "algorithm": "fixed-baseline-clipped-pg-v1",
        "baseline": 0.5,
        "lr": 1e-5,
        "temperature": 0.6,
        "seed": 17,
        "replicated_rank_batch": True,
        "weight_tying": WEIGHT_TYING,
    }
    if elastic:
        compatibility.pop("backend")
        compatibility.update(engine="fsdp", elastic_dp="fsdp-pure-dp-v1")
    return PPOGroup(
        devices=devices,
        config=config,
        engine_class="examples.autopilot.elastic.fixtures.ppo_engine.ElasticPPOEngine",
        ft_spec=FinetuneSpec(1, 32 * 3 * len(devices), 3 * len(devices)),
        root=root,
        compatibility=compatibility,
        keep_frontend=True,
    )


class LMGroup(TrainGroup):
    """Fixed LM-loss updates used only by the continuous/restarted fixture."""

    def train_fixed_batch(self, batch):
        if self.is_v2:
            from areal.infra.rpc.serialization import serialize_value

            self.actor._gateway_post_result(
                "/sft/train",
                {"args": serialize_value([batch]), "kwargs": serialize_value({})},
            )
            records = self.actor.rank_records("update", self.step + 1)
            probe = [record["rng_probe"] for record in records]
            stats = [record["stats"] for record in records]
        else:
            probe = self.call_all("rng_probe")
            stats = self.call_all("fixed_batch_update", batch)
        if any(result.get("update_successful", 0) != 1 for result in stats):
            raise RuntimeError(f"Optimizer update failed: {stats}")
        self.actor.step_lr_scheduler()
        self.step += 1
        self.actor.set_version(self.step)
        if self.actor.get_version() != self.step:
            raise RuntimeError("Controller version disagrees with completed update")
        return {"step": self.step, "rng_probe": probe, "rank_stats": stats}
