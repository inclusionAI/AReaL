import sys
from types import SimpleNamespace

import pytest

from areal.api.cli_args import (
    BaseExperimentConfig,
    WandBConfig,
    WandBSystemMetricsConfig,
)
from areal.utils.wandb_system_metrics import (
    finish_worker_wandb_system_metrics,
    init_worker_wandb_system_metrics,
    resolve_wandb_run_id,
    worker_system_metrics_enabled,
)


def _make_config(tmp_path, *, roles=None):
    config = BaseExperimentConfig(
        experiment_name="exp",
        trial_name="trial",
        total_train_epochs=1,
    )
    config.stats_logger.experiment_name = "exp"
    config.stats_logger.trial_name = "trial"
    config.stats_logger.fileroot = str(tmp_path)
    config.stats_logger.wandb = WandBConfig(
        mode="shared",
        project="proj",
        entity="entity",
        id_suffix="timestamp",
        system_metrics=WandBSystemMetricsConfig(enabled=True, roles=roles),
    )
    return config


def test_worker_system_metrics_requires_shared_mode():
    with pytest.raises(ValueError, match="requires stats_logger.wandb.mode='shared'"):
        WandBConfig(
            mode="online",
            system_metrics=WandBSystemMetricsConfig(enabled=True),
        )


def test_timestamp_run_id_is_resolved_once(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    timestamps = iter(["2026_05_14_00_00_01", "2026_05_14_00_00_02"])
    monkeypatch.setattr(
        "areal.utils.wandb_system_metrics.time.strftime",
        lambda _: next(timestamps),
    )

    run_id = resolve_wandb_run_id(config.stats_logger)
    assert run_id == "exp_trial_2026_05_14_00_00_01"
    assert resolve_wandb_run_id(config.stats_logger) == run_id


def test_worker_system_metrics_respects_role_filter(tmp_path):
    config = _make_config(tmp_path, roles=["actor"])

    assert worker_system_metrics_enabled(config, "actor")
    assert not worker_system_metrics_enabled(config, "rollout")


def test_worker_wandb_init_uses_non_primary_shared_settings(monkeypatch, tmp_path):
    config = _make_config(tmp_path, roles=["actor"])
    config.stats_logger.wandb.id_suffix = "fixed"
    calls = []
    finishes = []

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_wandb = SimpleNamespace(
        Settings=FakeSettings,
        init=lambda **kwargs: calls.append(kwargs) or object(),
        finish=lambda: finishes.append(True),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert init_worker_wandb_system_metrics(config, role="actor", rank=3)
    assert len(calls) == 1

    call = calls[0]
    assert call["mode"] == "shared"
    assert call["id"] == "exp_trial_fixed"
    assert call["settings"].kwargs == {
        "mode": "shared",
        "x_primary": False,
        "x_label": "actor-3",
        "x_update_finish_state": False,
    }

    finish_worker_wandb_system_metrics()
    assert finishes == [True]
