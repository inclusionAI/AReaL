import asyncio
import os
import re
import shutil
import signal
import subprocess
import time
from typing import Tuple

import pytest

from areal.utils import logging

logger = logging.getLogger(__name__)


async def run_example(
    example_file: str, config_name: str, *additional_args, timeout: int = 300
) -> Tuple[bool, str, str]:
    """
    Run a single example and return the result.

    Args:
        example_file: Path to the example file
        config_name: Name of the config to use

    Returns:
        Tuple of (success, stdout, stderr)
    """
    success_pattern = re.compile(r"Epoch 1/\d+ Step 1/\d+ Train step 1/\d+ done\.")
    # Construct the command
    cmd = [
        "python3",
        "-m",
        "areal.launcher.local",
        example_file,
        "--config",
        config_name,
    ]
    cmd += list(additional_args)

    logger.info(f"Running: {' '.join(cmd)}")

    # Run the command with timeout
    success = False
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    start_time = time.monotonic()

    while True:
        # Read output by line
        line = None
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
            line = line.decode()
        except asyncio.TimeoutError:
            pass

        if line:
            logger.info(f"[Example Output] {line.rstrip()}")
            # Check for success patterns
            success = bool(success_pattern.search(line))

        if success:
            logger.info(f"✓ {example_file} with config {config_name} - SUCCESS")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

        # Check if process has terminated
        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=0.1)
            logger.error(f"Process terminated unexpectedly. Return code: {return_code}")
            break
        except asyncio.TimeoutError:
            pass

        # Check timeout
        if (time.monotonic() - start_time) > timeout:
            logger.error("Process timed out without successful result, terminating...")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

    return_code = await process.wait()  # Wait for the child process to exit
    return return_code, success


@pytest.mark.multi_gpu
def test_countdown_example(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    tmp_path = tmp_path_factory.mktemp("countdown_data")
    data_path = tmp_path / "data/countdown/qwen"
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    os.makedirs(data_path, exist_ok=True)
    test_file_path = data_path / "test_e.jsonl"
    train_file_path = data_path / "train_e.jsonl"
    # generate countdown dataset
    shutil.copy("examples/countdown/countdown.py", tmp_path)
    subprocess.run(
        ["python3", "countdown.py", "--num_samples=10000", "--eval_size=100"],
        cwd=tmp_path,
        check=True,
    )

    example_file = "examples/countdown/train.py"
    config_name = "examples/countdown/train_config.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=128",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={str(train_file_path)}",
            f"valid_dataset.path={str(test_file_path)}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
            "rollout.enable_rollout_tracing=true",
        )
    )
    assert success, f"Countdown example failed, return_code={return_code}"
