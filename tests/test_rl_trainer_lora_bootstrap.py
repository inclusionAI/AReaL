# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import areal.trainer.rl_trainer as rl_trainer_mod
from areal.trainer.rl_trainer import PPOTrainer


def test_prepare_initial_actor_lora_loads_bootstrap_before_save(monkeypatch):
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(
            use_lora=True,
            backend="fsdp:d2p1t1",
            path="/tmp/base-model",
        ),
        cluster=SimpleNamespace(fileroot="/tmp/fileroot"),
        experiment_name="exp",
        trial_name="trial",
    )
    trainer.actor = Mock()
    trainer.actor.load = Mock(
        side_effect=AssertionError("full-model load should not be used")
    )
    trainer.actor.load_lora_adapter = Mock()
    trainer.actor.save = Mock()
    trainer.tokenizer = Mock()
    trainer.processor = Mock()
    trainer._sync_lora_adapter_dir_to_cluster_nodes = Mock()

    path = "/tmp/test-lora-checkpoint"
    monkeypatch.setenv("GUI_LORA_BOOTSTRAP_HF_PATH", path)
    monkeypatch.setattr(
        rl_trainer_mod.Saver,
        "get_model_save_root",
        lambda *args, **kwargs: "/tmp/actor-root",
    )

    initial_lora_path = trainer._prepare_initial_actor_lora()

    assert initial_lora_path == "/tmp/actor-root/initial_lora"
    trainer._sync_lora_adapter_dir_to_cluster_nodes.assert_called_once_with(path)
    trainer.actor.load_lora_adapter.assert_called_once_with(path)
    trainer.actor.load.assert_not_called()
    trainer.actor.save.assert_called_once()
    assert trainer.actor.save.call_args.kwargs["meta"].path == initial_lora_path


def test_lora_bootstrap_hf_path_requires_lora(monkeypatch):
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(use_lora=False, backend="fsdp:d2p1t1")
    )
    trainer.actor = Mock()
    trainer._sync_lora_adapter_dir_to_cluster_nodes = Mock()

    monkeypatch.setenv("GUI_LORA_BOOTSTRAP_HF_PATH", "/tmp/test-lora-checkpoint")

    with pytest.raises(RuntimeError, match="actor.use_lora is false"):
        trainer._get_actor_lora_bootstrap_hf_path()


def test_lora_bootstrap_hf_path_rejects_non_fsdp_backends(monkeypatch):
    trainer = object.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(use_lora=True, backend="megatron:d2p1t1")
    )
    trainer.actor = Mock()
    trainer._sync_lora_adapter_dir_to_cluster_nodes = Mock()

    monkeypatch.setenv("GUI_LORA_BOOTSTRAP_HF_PATH", "/tmp/test-lora-checkpoint")

    with pytest.raises(RuntimeError, match="only supported for FSDP actors"):
        trainer._get_actor_lora_bootstrap_hf_path()


def test_get_actor_worker_ips_uses_current_actor_workers_only():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = SimpleNamespace(
        workers=[
            SimpleNamespace(ip="10.0.0.2"),
            SimpleNamespace(ip="10.0.0.1"),
            SimpleNamespace(ip="10.0.0.2"),
        ]
    )

    assert trainer._get_actor_worker_ips() == ["10.0.0.1", "10.0.0.2"]
