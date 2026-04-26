# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.experimental.agent_service.tau2.workflow import Tau2AgentServiceWorkflow


@pytest.mark.asyncio
@patch("examples.experimental.agent_service.tau2.workflow.workflow_context")
@patch("examples.experimental.agent_service.tau2.workflow.stats_tracker")
async def test_arun_episode_returns_exported_interactions(
    mock_stats_tracker,
    mock_workflow_context,
):
    controller = MagicMock()
    controller.start_session = AsyncMock(
        return_value={
            "session_id": "agent-sess-1",
            "inference_session_id": "inf-sess-1",
            "api_key": "sess-key",
        }
    )
    controller.set_reward = AsyncMock(return_value={"trajectory_id": 3})
    exported = {"last": SimpleNamespace(reward=1.0)}
    controller.export_trajectory = AsyncMock(return_value=exported)

    workflow = Tau2AgentServiceWorkflow(
        agent_controller=controller,
        inference_gateway_addr="http://inference",
        inference_admin_api_key="rollout-admin",
        inference_model="Qwen/Test",
        econfig={"domain": "telecom", "solo_mode": True},
        timeout=10.0,
    )
    workflow._run_dialog = AsyncMock(return_value=1.0)
    mock_scope = object()
    mock_workflow_context.get.return_value = SimpleNamespace(
        task_id="task-from-context"
    )
    mock_workflow_context.stat_scope.return_value = mock_scope
    mock_stats = MagicMock()
    mock_stats_tracker.get.return_value = mock_stats

    result = await workflow.arun_episode(engine=object(), data={"task_id": "task-1"})

    assert result is exported
    controller.start_session.assert_awaited_once_with(
        task_id="task-1",
        inference_gateway_addr="http://inference",
        inference_admin_api_key="rollout-admin",
        inference_model="Qwen/Test",
    )
    workflow._run_dialog.assert_awaited_once_with({"task_id": "task-1"}, "agent-sess-1")
    controller.set_reward.assert_awaited_once_with(1.0, "agent-sess-1")
    controller.export_trajectory.assert_awaited_once_with(
        "agent-sess-1",
        trajectory_id=3,
        discount=1.0,
        style="individual",
    )
    mock_stats.scalar.assert_called_once_with(reward=1.0)
