import os
import re
import shutil
import subprocess
from typing import Tuple

import pytest

from areal.utils import logging

logger = logging.getLogger(__name__)


def run_example(
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
    # Remove .py extension from example file for the command
    example_name = (
        os.path.splitext(example_file)[0].replace("/", ".").replace("\\", ".")
    )

    # Construct the command
    cmd = [
        "python3",
        "-m",
        "areal.launcher.local",
        example_name,
        "--config",
        config_name,
    ]
    cmd += list(additional_args)

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        # Run the command with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout
        stderr = result.stderr

        # Check if the expected pattern is in the output
        success = bool(success_pattern.search(stdout))

        # Log the result
        logger.info(f"STDOUT: ...{stdout[-10000:]}")  # Truncate long output
        if success:
            logger.info(f"✓ {example_file} with config {config_name} - SUCCESS")
        else:
            logger.warning(f"✗ {example_file} with config {config_name} - FAILED")
            logger.warning(f"Return code: {result.returncode}")
            if stderr:
                logger.warning(f"STDERR: {stderr}")  # Truncate long error messages

        return success, stdout, stderr

    except subprocess.TimeoutExpired:
        error_msg = f"Example {example_file} with config {config_name} timed out after {timeout} seconds"
        logger.error(error_msg)
        return False, "", error_msg

    except Exception as e:
        error_msg = f"Error running {example_file} with config {config_name}: {str(e)}"
        logger.error(error_msg)
        return False, "", error_msg


@pytest.mark.multi_gpu
def test_countdown_example(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    tmp_path = tmp_path_factory.mktemp("countdown_data")
    data_path = tmp_path / "data/countdown/qwen"
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
    success, stdout, stderr = run_example(
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=1024",
        f"train_dataset.path={str(train_file_path)}",
        f"valid_dataset.path={str(test_file_path)}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
    )
    assert success, f"Countdown example failed. STDOUT: {stdout}, STDERR: {stderr}"
