"""Tests for Trainer-level AgentController integration.

Verifies that after the Wave 2 refactor:
- AgentController lives at the Trainer level, NOT inside RolloutController.
- RolloutController exposes only ``set_agent_service_addr`` (no start/stop).
- ``_ensure_agent_service_started`` is idempotent, creates exactly one controller,
  and propagates the gateway address to both train and eval rollout controllers.
- ``close()`` calls ``AgentController.stop()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from areal.infra.controller.rollout_controller import RolloutController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trainer_state(
    *,
    agent_config: MagicMock | None = None,
    eval_rollout: MagicMock | None = "auto",
):
    """Return a namespace-like MagicMock that mimics the trainer fields
    accessed by ``_ensure_agent_service_started`` and ``close``.

    Setting *eval_rollout* to ``"auto"`` creates a second mock controller.
    Setting it to ``None`` means no eval rollout.
    """
    trainer = MagicMock()
    trainer._agent_service_started = False
    trainer._agent_controller = None
    trainer._proxy_started = True  # assume proxy already started

    # Rollout controller mocks
    rollout = MagicMock(spec=RolloutController)
    sched = MagicMock()
    sched.experiment_name = "test-exp"
    sched.trial_name = "test-trial"
    rollout.scheduler = sched
    trainer.rollout = rollout

    if eval_rollout == "auto":
        trainer.eval_rollout = MagicMock()
    else:
        trainer.eval_rollout = eval_rollout

    # Config with optional agent_service block
    if agent_config is None:
        agent_config = MagicMock()
        agent_config.agent_import_path = "my.Agent"
        agent_config.workers = 2
        agent_config.agent_reuse = False
        agent_config.agent_init_kwargs = None
    trainer.config.rollout.agent_service = agent_config

    return trainer


# ===========================================================================
# 1. Architecture: RolloutController surface area
# ===========================================================================


class TestRolloutControllerSurface:
    """RolloutController should only expose ``set_agent_service_addr``."""

    def test_rollout_controller_has_set_agent_service_addr(self):
        """set_agent_service_addr must exist on RolloutController."""
        from areal.infra.controller.rollout_controller import RolloutController

        assert hasattr(RolloutController, "set_agent_service_addr")

    def test_rollout_controller_no_start_agent_service(self):
        """start_agent_service should NOT exist on RolloutController after Wave 2."""
        from areal.infra.controller.rollout_controller import RolloutController

        assert not hasattr(RolloutController, "start_agent_service"), (
            "start_agent_service should be removed from RolloutController"
        )

    def test_rollout_controller_no_async_start_agent_service(self):
        """_async_start_agent_service should NOT exist on RolloutController."""
        from areal.infra.controller.rollout_controller import RolloutController

        assert not hasattr(RolloutController, "_async_start_agent_service"), (
            "_async_start_agent_service should be removed from RolloutController"
        )


# ===========================================================================
# 2. _ensure_agent_service_started behaviour
# ===========================================================================


class TestEnsureAgentServiceStarted:
    """Tests for ``PPOTrainer._ensure_agent_service_started``."""

    def _call_ensure(self, trainer_mock):
        """Import and invoke ``_ensure_agent_service_started`` bound to *trainer_mock*."""
        from areal.trainer.rl_trainer import PPOTrainer

        # Call the unbound method with our mock as ``self``.
        PPOTrainer._ensure_agent_service_started(trainer_mock)

    # -----------------------------------------------------------------------

    def test_creates_agent_controller(self):
        """_ensure_agent_service_started should create an AgentController."""
        trainer = _make_mock_trainer_state()

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"
            self._call_ensure(trainer)

            MockCtrl.assert_called_once()
            assert trainer._agent_controller is MockCtrl.return_value

    def test_gateway_addr_passed_to_rollout(self):
        """Gateway address must be forwarded to the train rollout controller."""
        trainer = _make_mock_trainer_state()

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"
            self._call_ensure(trainer)

            trainer.rollout.set_agent_service_addr.assert_called_once_with(
                "http://gw:8300"
            )

    def test_gateway_addr_passed_to_eval_rollout(self):
        """Gateway address must be forwarded to eval rollout when it exists."""
        trainer = _make_mock_trainer_state(eval_rollout="auto")

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"
            self._call_ensure(trainer)

            trainer.eval_rollout.set_agent_service_addr.assert_called_once_with(
                "http://gw:8300"
            )

    def test_eval_rollout_none_no_error(self):
        """No error when eval_rollout is None."""
        trainer = _make_mock_trainer_state(eval_rollout=None)

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"
            # Should not raise
            self._call_ensure(trainer)

            # rollout still gets the addr
            trainer.rollout.set_agent_service_addr.assert_called_once()

    def test_idempotent_does_not_create_twice(self):
        """Calling _ensure_agent_service_started twice must not create two controllers."""
        trainer = _make_mock_trainer_state()

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"

            # First call
            self._call_ensure(trainer)
            # Second call — should early-return because _agent_service_started is True
            self._call_ensure(trainer)

            # AgentController constructor called exactly once
            assert MockCtrl.call_count == 1

    def test_single_controller_shared_across_rollouts(self):
        """Only ONE AgentController is created even with both train and eval rollout."""
        trainer = _make_mock_trainer_state(eval_rollout="auto")

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.experimental.agent_service.config.GatewayConfig"),
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            MockCtrl.return_value.start.return_value = "http://gw:8300"
            self._call_ensure(trainer)

            # Exactly one AgentController instantiation
            MockCtrl.assert_called_once()
            # But both rollout controllers receive the address
            trainer.rollout.set_agent_service_addr.assert_called_once()
            trainer.eval_rollout.set_agent_service_addr.assert_called_once()

    def test_skipped_when_agent_config_none(self):
        """_ensure_agent_service_started is a no-op when agent_service config is None."""
        trainer = _make_mock_trainer_state()
        trainer.config.rollout.agent_service = None

        with (
            patch(
                "areal.experimental.agent_service.agent_controller.AgentController"
            ) as MockCtrl,
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        ):
            self._call_ensure(trainer)

            MockCtrl.assert_not_called()
            assert trainer._agent_controller is None


# ===========================================================================
# 3. close() cleanup
# ===========================================================================


class TestTrainerCloseCleanup:
    """Tests for agent controller cleanup during ``PPOTrainer.close``."""

    def test_agent_controller_stop_called(self):
        """close() must call AgentController.stop() when controller exists."""
        from areal.trainer.rl_trainer import PPOTrainer

        trainer = MagicMock(spec=PPOTrainer)
        ctrl = MagicMock()
        trainer._agent_controller = ctrl
        # Provide required attrs that close() accesses
        trainer.eval_rollout = MagicMock()
        trainer.rollout = MagicMock()
        trainer.ref = None
        trainer.critic = None
        trainer.saver = MagicMock()
        trainer.stats_logger = MagicMock()
        trainer.actor = MagicMock()

        PPOTrainer.close(trainer)

        ctrl.stop.assert_called_once()
        assert trainer._agent_controller is None

    def test_close_no_error_when_controller_none(self):
        """close() must not error when _agent_controller is None."""
        from areal.trainer.rl_trainer import PPOTrainer

        trainer = MagicMock(spec=PPOTrainer)
        trainer._agent_controller = None
        trainer.eval_rollout = MagicMock()
        trainer.rollout = MagicMock()
        trainer.ref = None
        trainer.critic = None
        trainer.saver = MagicMock()
        trainer.stats_logger = MagicMock()
        trainer.actor = MagicMock()

        # Should not raise
        PPOTrainer.close(trainer)
