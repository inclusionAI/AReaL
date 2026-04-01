import os
import re
import subprocess
import time

import pytest

from tests.test_examples import run_example
from tests.utils import get_model_path

from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging

logger = logging.getLogger("InferenceServiceExamples")

pytestmark = pytest.mark.slow


@pytest.mark.ci
@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_inference_service_online_rl(tmp_path_factory):
    """Test inference_service online RL training via persistent online session."""

    GATEWAY_STARTUP_TIMEOUT = 600
    SETTLE_TIME = 5
    HEALTH_CHECK_TIMEOUT = 10
    CHAT_TIMEOUT = 30
    REWARD_TIMEOUT = 10
    ROLLOUT_COMPLETION_TIMEOUT = 300

    def wait_for_pattern(process, pattern, timeout, raise_on_exit=True):
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    if raise_on_exit:
                        raise RuntimeError(
                            f"Process terminated unexpectedly (code {process.returncode})"
                        )
                    return None
                time.sleep(0.1)
                continue
            line = line.strip()
            if line:
                logger.info(f"[Inference Service Online RL] {line}")
            match = pattern.search(line)
            if match:
                return match
        return None

    if current_platform.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs to run.")

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    admin_api_key = "test-admin-key"
    batch_size = 2

    cmd = [
        "python3",
        "examples/experimental/inference_service/online_rollout.py",
        "--config",
        "examples/experimental/inference_service/online_rollout.yaml",
        "rollout.backend=sglang:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=2048",
        f"train_dataset.batch_size={batch_size}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        f"rollout.openai.admin_api_key={admin_api_key}",
        "stats_logger.wandb.mode=disabled",
    ]

    logger.info(f"Starting inference_service online RL service: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    gateway_pattern = re.compile(r"Proxy gateway available at (http://[0-9.:]+)")

    try:
        match = wait_for_pattern(
            process, gateway_pattern, GATEWAY_STARTUP_TIMEOUT, raise_on_exit=True
        )
        if not match:
            raise RuntimeError("Timed out waiting for gateway URL")
        gateway_url = match.group(1)
        logger.info(f"Gateway URL: {gateway_url}")

        time.sleep(SETTLE_TIME)

        import requests

        health_resp = requests.get(
            f"{gateway_url}/health", timeout=HEALTH_CHECK_TIMEOUT
        )
        assert health_resp.status_code == 200, (
            f"Health check failed: {health_resp.text}"
        )

        for trajectory_idx in range(batch_size):
            logger.info(f"Trajectory {trajectory_idx + 1}/{batch_size}")

            chat_resp = requests.post(
                f"{gateway_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {admin_api_key}",
                },
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "max_tokens": 128,
                },
                timeout=CHAT_TIMEOUT,
            )
            assert chat_resp.status_code == 200, (
                f"chat/completions failed (trajectory {trajectory_idx + 1}): {chat_resp.text}"
            )

            reward_resp = requests.post(
                f"{gateway_url}/rl/set_reward",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {admin_api_key}",
                },
                json={"reward": 1.0},
                timeout=REWARD_TIMEOUT,
            )
            assert reward_resp.status_code == 200, (
                f"set_reward failed (trajectory {trajectory_idx + 1}): {reward_resp.text}"
            )
            reward_data = reward_resp.json()
            assert reward_data["session_id"] == "__hitl__"
            assert reward_data["trajectory_id"] == trajectory_idx

        logger.info("Waiting for rollout completion...")
        match = wait_for_pattern(
            process,
            re.compile(r"Rollout complete \(\d+ trajectories\), avg_reward="),
            ROLLOUT_COMPLETION_TIMEOUT,
            raise_on_exit=False,
        )
        assert match is not None, "Rollout did not complete within timeout"
        logger.info("Rollout completed!")

    finally:
        logger.info("Shutting down inference_service online RL service...")
        kill_process_tree(process.pid, graceful=False)


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_tau2_rollout(tmp_path_factory):
    tau2 = pytest.importorskip("tau2")
    del tau2

    tau2_data_dir = os.environ.get("TAU2_DATA_DIR")
    if not tau2_data_dir:
        pytest.skip("TAU2_DATA_DIR environment variable not set. Skipping tau2 test.")
    if not os.path.exists(tau2_data_dir):
        pytest.skip(
            f"TAU2_DATA_DIR ({tau2_data_dir}) does not exist. Skipping tau2 test."
        )

    chat_template_path = "/storage/openpsi/data/qwen3_nonthinking.jinja"
    if not os.path.exists(chat_template_path):
        pytest.skip(f"Chat template not found at {chat_template_path}")

    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for user LLM, 2 for rollout) to run."
        )

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    user_llm_gpu = visible_devices[-1]
    user_llm_port = 30081

    _env = os.environ.copy()
    _env[current_platform.device_control_env_var] = user_llm_gpu

    logger.info(
        f"Launching user LLM server on GPU {user_llm_gpu}, port {user_llm_port}"
    )
    user_llm_proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(user_llm_port),
            "--tool-call-parser",
            "qwen25",
            "--chat-template",
            chat_template_path,
            "--dp-size",
            "1",
            "--mem-fraction-static",
            "0.8",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_env,
    )

    try:
        logger.info("Waiting for user LLM server to start...")
        time.sleep(60)

        user_llm_base_url = f"http://localhost:{user_llm_port}/v1/"
        success = run_async_task(
            run_example,
            "examples/experimental/inference_service/tau2_rollout.py",
            "examples/experimental/inference_service/tau2_rollout.yaml",
            "rollout.backend=sglang:d2",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"model_path={model_path}",
            "train_dataset.batch_size=2",
            "train_dataset.path=tau2/train",
            f"econfig.user_llm_base_url={user_llm_base_url}",
            "econfig.user_llm=openai/self-hosted-qwen3",
            "stats_logger.wandb.mode=disabled",
            timeout=600,
            success_pattern=re.compile(r"Rollout complete"),
        )
        assert success, "Tau2 rollout example failed"
    finally:
        logger.info("Shutting down user LLM server...")
        kill_process_tree(user_llm_proc.pid, graceful=False)
