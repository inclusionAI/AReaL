"""Unit tests for colocated orchestration and scheduler-driven trainer behavior."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType
from areal.api.io_struct import WeightUpdateMeta
from areal.infra.colocated import ColocatedOrchestrator
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.controller.train_controller import TrainController
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.scheduler.local import LocalScheduler, _apply_env_patch
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils import names


@pytest.fixture
def mock_train_engine():
    engine = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    return engine


@pytest.fixture
def mock_inf_engine():
    engine = MagicMock()
    engine.pause = MagicMock()
    engine.resume = MagicMock()
    engine.pause_generation = MagicMock()
    engine.continue_generation = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    engine.sync_weights_from_disk = MagicMock()
    return engine


@pytest.fixture
def orchestrator(mock_train_engine, mock_inf_engine):
    return ColocatedOrchestrator(
        train_engine=mock_train_engine,
        inf_engine=mock_inf_engine,
    )


class TestColocatedOrchestrator:
    def test_initial_state(self, orchestrator):
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is True

    def test_initial_offload_training(self, orchestrator, mock_train_engine):
        orchestrator.initial_offload_training()

        mock_train_engine.offload.assert_called_once()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_for_training_switches_gpu_owner(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_called_once()
        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is False

    def test_prepare_for_training_orders_local_and_remote_rollout_shutdown(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        events: list[str] = []
        mock_inf_engine.pause.side_effect = lambda: events.append("pause")
        mock_inf_engine.pause_generation.side_effect = lambda: events.append(
            "pause_generation"
        )
        mock_inf_engine.offload.side_effect = lambda: events.append("inf_offload")
        mock_train_engine.onload.side_effect = lambda: events.append("train_onload")

        orchestrator.initial_offload_training()
        events.clear()

        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()

        assert events == [
            "pause",
            "pause_generation",
            "inf_offload",
            "train_onload",
        ]

    def test_prepare_for_training_only_coordinator_controls_shared_rollout_server(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        with (
            patch("areal.infra.colocated.dist.is_initialized", return_value=True),
            patch("areal.infra.colocated.dist.get_rank", return_value=3),
            patch("areal.infra.colocated.dist.barrier") as mock_barrier,
        ):
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_not_called()
        mock_inf_engine.offload.assert_not_called()
        mock_train_engine.onload.assert_called_once()
        mock_barrier.assert_called()
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is False

    def test_prepare_for_inference_switches_gpu_owner_and_syncs_weights(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
        mock_inf_engine.pause.reset_mock()
        mock_inf_engine.pause_generation.reset_mock()
        mock_inf_engine.offload.reset_mock()
        mock_inf_engine.onload.reset_mock()
        mock_inf_engine.sync_weights_from_disk.reset_mock()
        mock_inf_engine.continue_generation.reset_mock()
        mock_inf_engine.resume.reset_mock()
        mock_train_engine.onload.reset_mock()
        mock_train_engine.offload.reset_mock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v1")
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_inference(meta)

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_called_once()
        mock_inf_engine.sync_weights_from_disk.assert_called_once_with(meta)
        mock_inf_engine.continue_generation.assert_called_once()
        mock_inf_engine.resume.assert_called_once()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_for_inference_only_coordinator_controls_shared_rollout_server(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
        mock_train_engine.offload.reset_mock()
        mock_inf_engine.onload.reset_mock()
        mock_inf_engine.sync_weights_from_disk.reset_mock()
        mock_inf_engine.continue_generation.reset_mock()
        mock_inf_engine.resume.reset_mock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v3")
        with patch(
            "areal.infra.colocated.dist.is_initialized", return_value=True
        ):
            with patch("areal.infra.colocated.dist.get_rank", return_value=5):
                with patch("areal.infra.colocated.dist.barrier") as mock_barrier:
                    orchestrator.prepare_for_inference(meta)

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_not_called()
        mock_inf_engine.sync_weights_from_disk.assert_not_called()
        mock_inf_engine.continue_generation.assert_not_called()
        mock_inf_engine.resume.assert_called_once()
        mock_barrier.assert_called()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_calls_are_idempotent(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_called_once()
        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()


class TestTrainControllerColocatedInterfaces:
    def test_offload_updates_state_and_dispatches(self):
        controller = TrainController.__new__(TrainController)
        controller._custom_function_call = MagicMock()
        controller.is_offload = False

        controller.offload()

        controller._custom_function_call.assert_called_once_with("offload")
        assert controller.is_offload is True

    def test_onload_updates_state_and_dispatches(self):
        controller = TrainController.__new__(TrainController)
        controller._custom_function_call = MagicMock()
        controller.is_offload = True

        controller.onload()

        controller._custom_function_call.assert_called_once_with("onload")
        assert controller.is_offload is False

    def test_prepare_batch_context_is_noop(self):
        controller = TrainController.__new__(TrainController)

        with controller.prepare_batch_context():
            pass


class TestRolloutControllerColocatedInterfaces:
    def test_sync_weights_from_disk_uses_run_async_task(self):
        controller = RolloutController.__new__(RolloutController)
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v2")

        with patch(
            "areal.infra.controller.rollout_controller.run_async_task"
        ) as mock_run_async_task:
            controller.sync_weights_from_disk(meta)

        mock_run_async_task.assert_called_once_with(
            controller.update_weights_from_disk, meta
        )

    def test_pause_generation_and_continue_generation_use_run_async_task(self):
        controller = RolloutController.__new__(RolloutController)

        with patch(
            "areal.infra.controller.rollout_controller.run_async_task"
        ) as mock_run_async_task:
            controller.pause_generation()
            controller.continue_generation()

        assert mock_run_async_task.call_count == 2
        assert mock_run_async_task.call_args_list[0].args[0].__name__ == "_pause_generation_async"
        assert (
            mock_run_async_task.call_args_list[1].args[0].__name__
            == "_continue_generation_async"
        )

    def test_offload_and_onload_delegate_to_collective_rpc(self):
        controller = RolloutController.__new__(RolloutController)
        controller._collective_rpc = MagicMock()

        controller.offload()
        controller.onload(tags=["lora"])

        assert controller._collective_rpc.call_args_list == [
            (("offload",), {"http_timeout": 60.0}),
            (("onload",), {"tags": ["lora"], "http_timeout": 60.0}),
        ]

    def test_update_weights_from_disk_does_not_mutate_original_meta(self):
        controller = RolloutController.__new__(RolloutController)
        controller._collective_rpc_async = AsyncMock()

        temp_dir = Path(tempfile.mkdtemp(prefix="areal-colocated-test-"))
        try:
            meta = WeightUpdateMeta(
                type="disk",
                path=str(temp_dir),
                clear_checkpoint_after_load=True,
            )

            asyncio.run(controller.update_weights_from_disk(meta))

            assert meta.clear_checkpoint_after_load is True
            assert not temp_dir.exists()
            await_args = controller._collective_rpc_async.await_args
            assert await_args is not None
            sent_meta = await_args.kwargs["meta"]
            assert sent_meta.clear_checkpoint_after_load is False
            assert sent_meta.path == meta.path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_create_worker_kwargs_clears_tms_env_for_actor_colocated_fork(self):
        controller = RolloutController.__new__(RolloutController)
        job = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=True,
            )
        )

        assert controller._build_create_worker_kwargs(job) == {
            "fork_unset_env_keys": [
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ]
        }

    @pytest.mark.parametrize(
        "strategy",
        [
            SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=False,
            ),
            SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="critic",
                fork=True,
            ),
            SchedulingStrategy(type=SchedulingStrategyType.separation),
        ],
    )
    def test_build_create_worker_kwargs_skips_non_actor_fork_colocation(self, strategy):
        controller = RolloutController.__new__(RolloutController)
        job = SimpleNamespace(scheduling_strategy=strategy)

        assert controller._build_create_worker_kwargs(job) == {}


class TestLocalSchedulerForkEnvPatch:
    def test_apply_env_patch_unsets_and_overrides(self):
        patched = _apply_env_patch(
            {
                "LD_PRELOAD": "/tmp/libtms.so",
                "TMS_INIT_ENABLE": "1",
                "KEEP": "yes",
            },
            env_overrides={"NEW_KEY": "value"},
            unset_env_keys=["LD_PRELOAD", "TMS_INIT_ENABLE", "MISSING_KEY"],
        )

        assert patched == {
            "KEEP": "yes",
            "NEW_KEY": "value",
        }

    def test_create_workers_forwards_fork_env_cleanup(self):
        scheduler = LocalScheduler.__new__(LocalScheduler)
        scheduler._workers = cast(Any, {"actor": [object()]})
        scheduler._colocated_roles = {}
        scheduler._prepare_worker_specs = MagicMock(return_value=[SimpleNamespace()])
        scheduler.fork_workers = MagicMock(return_value=["rollout/0"])

        job = SimpleNamespace(
            role="rollout",
            replicas=1,
            tasks=[SimpleNamespace()],
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=True,
            ),
        )

        worker_ids = scheduler.create_workers(
            job=job,
            fork_unset_env_keys=[
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ],
        )

        assert worker_ids == ["rollout/0"]
        scheduler.fork_workers.assert_called_once_with(
            "rollout",
            "actor",
            env_overrides=None,
            unset_env_keys=[
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ],
        )
        assert scheduler._colocated_roles["rollout"] == "actor"


def _make_validation_trainer(
    *,
    colocated: bool = True,
    weight_update_mode: str = "disk",
) -> Any:
    trainer = cast(Any, PPOTrainer.__new__(PPOTrainer))
    trainer.allocation_mode = SimpleNamespace(gen_backend="sglang")
    scheduling_strategy = SchedulingStrategy(
        type=(
            SchedulingStrategyType.colocation
            if colocated
            else SchedulingStrategyType.separation
        ),
        target="actor" if colocated else None,
    )
    trainer._colocated = colocated
    trainer.config = SimpleNamespace(
        enable_offload=False,
        actor=SimpleNamespace(
            kl_ctl=0,
            weight_update_mode=weight_update_mode,
            scheduling_spec=[SimpleNamespace(env_vars={})],
        ),
        rollout=SimpleNamespace(
            return_routed_experts=False,
            openai=None,
            scheduling_strategy=scheduling_strategy,
            scheduling_spec=[SimpleNamespace(env_vars={})],
        ),
        critic=None,
        ref=None,
        teacher=None,
        cluster=SimpleNamespace(n_nodes=1),
        experiment_name="gsm8k-grpo-colocated",
        trial_name="trial0",
    )
    return trainer


class TestPPOTrainerColocatedScheduling:
    def test_is_colocated_rollout_detects_actor_colocation(self):
        rollout_cfg = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
            )
        )

        assert cast(Any, PPOTrainer)._is_colocated_rollout(rollout_cfg) is True

    def test_is_colocated_rollout_rejects_other_topologies(self):
        rollout_cfg = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="critic",
            )
        )

        assert cast(Any, PPOTrainer)._is_colocated_rollout(rollout_cfg) is False

    def test_validate_cfg_allows_single_controller(self):
        trainer = _make_validation_trainer()

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_multi_node(self):
        trainer = _make_validation_trainer()
        trainer.config.cluster.n_nodes = 2

        with pytest.raises(ValueError, match="single-node runs"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_non_disk_weight_update(self):
        trainer = _make_validation_trainer(weight_update_mode="xccl")

        with pytest.raises(ValueError, match="weight_update_mode='disk'"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_missing_train_dataset(self):
        trainer = _make_validation_trainer()

        with pytest.raises(ValueError, match="requires a train_dataset"):
            trainer._validate_cfg(train_dataset=None)

    def test_validate_cfg_rejects_online_mode(self):
        trainer = _make_validation_trainer()
        trainer.config.rollout.openai = SimpleNamespace(mode="online")

        with pytest.raises(ValueError, match="rollout.openai.mode='online'"):
            trainer._validate_cfg(train_dataset=object())

    @pytest.mark.parametrize(
        ("mutate", "expected_error"),
        [
            (
                lambda trainer: setattr(trainer.config, "critic", object()),
                "critic is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "ref", object()),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config.actor, "kl_ctl", 0.1),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "teacher", object()),
                "teacher is not supported",
            ),
        ],
    )
    def test_validate_cfg_rejects_non_actor_only_components(
        self, mutate, expected_error
    ):
        trainer = _make_validation_trainer()
        mutate(trainer)

        with pytest.raises(ValueError, match=expected_error):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_skips_colocated_restrictions_for_standard_mode(self):
        trainer = _make_validation_trainer(colocated=False)
        trainer.config.cluster.n_nodes = 4
        trainer.config.rollout.openai = SimpleNamespace(mode="online")
        trainer.config.critic = object()
        trainer.config.ref = object()
        trainer.config.teacher = object()

        trainer._validate_cfg(train_dataset=None)

    def test_amend_xccl_weight_update_envvar_injects_tms_for_colocated_controller(self):
        trainer = _make_validation_trainer()
        trainer.allocation_mode = SimpleNamespace(gen_backend="vllm")

        with (
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
            patch(
                "areal.trainer.rl_trainer.get_tms_env_vars",
                return_value={"LD_PRELOAD": "/tmp/libtms.so", "TMS_INIT_ENABLE": "1"},
            ),
        ):
            trainer._amend_xccl_weight_update_envvar()

        assert trainer.config.actor.scheduling_spec[0].env_vars["LD_PRELOAD"] == "/tmp/libtms.so"
        assert "LD_PRELOAD" not in trainer.config.rollout.scheduling_spec[0].env_vars

    def test_publish_disk_weight_update_ready_uses_rollout_version(self):
        trainer = _make_validation_trainer()
        trainer.rollout = SimpleNamespace(get_version=MagicMock(return_value=0))
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v7", version=7)

        with patch("areal.trainer.rl_trainer.name_resolve.add") as mock_add:
            trainer._publish_disk_weight_update_ready(meta)

        mock_add.assert_called_once_with(
            names.update_weights_from_disk(
                "gsm8k-grpo-colocated",
                "trial0",
                7,
            ),
            mock_add.call_args.args[1],
            keepalive_ttl=120,
        )


class TestRemoteInfEngineDiskWeightSync:
    def test_update_weights_from_disk_uses_engine_version_for_rendezvous(self):
        engine = cast(Any, RemoteInfEngine.__new__(RemoteInfEngine))
        engine.backend = MagicMock()
        engine.config = SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            request_retries=2,
            request_timeout=30.0,
        )
        engine.addresses = ["127.0.0.1:8000"]
        engine.get_version = MagicMock(return_value=0)
        engine.logger = MagicMock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v5", version=5)
        fake_future = MagicMock()

        with patch("areal.infra.remote_inf_engine.get_executor") as mock_get_executor:
            mock_get_executor.return_value.submit.return_value = fake_future
            engine.update_weights_from_disk(meta)

        submit_args = mock_get_executor.return_value.submit.call_args.args
        assert submit_args[4] == 0
        fake_future.add_done_callback.assert_called_once()
