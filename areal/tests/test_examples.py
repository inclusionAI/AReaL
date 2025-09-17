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

SUCCESS_PATTERN = re.compile(r"Epoch 1/\d+ Step 1/\d+ Train step 1/\d+ done\.")


async def run_example(
    example_file: str,
    config_name: str,
    *additional_args,
    timeout: int = 300,
    success_pattern=SUCCESS_PATTERN,
) -> Tuple[bool, str, str]:
    """
    Run a single example and return the result.

    Args:
        example_file: Path to the example file
        config_name: Name of the config to use
        additional_args: Additional command line arguments
        timeout: Timeout in seconds
        success_pattern: Regex pattern to identify successful completion

    Returns:
        Tuple of (success, stdout, stderr)
    """
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
        except (ValueError, asyncio.TimeoutError):
            # NOTE: Here ValueError is raised when the input line is too long
            # that exceeds the buffer size, which will happen if the experiment
            # has tqdm progress bar output.
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
        )
    )
    assert success, f"Countdown example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_gsm8k_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/math/gsm8k_grpo.py"
    config_name = "examples/math/gsm8k_grpo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"GSM8K GRPO example failed, return_code={return_code}"


@pytest.mark.gpu
def test_gsm8k_sft(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen3-1.7B"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen3-1.7B"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/math/gsm8k_sft.py"
    config_name = "examples/math/gsm8k_sft.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=d1",
            "model.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=1",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"model.path={model_path}",
        )
    )
    assert success, f"GSM8K SFT example failed, return_code={return_code}"


@pytest.mark.gpu
def test_gsm8k_eval(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/math/gsm8k_eval.py"
    config_name = "examples/math/gsm8k_grpo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+eval",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            f"valid_dataset.batch_size=16",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=1",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
            success_pattern=re.compile(r"Evaluation results:\n"),
        )
    )
    assert success, f"GSM8K Eval example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_vlm_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_path = "/storage/openpsi/data/BUAADreamer__clevr_count_70k"
    if not os.path.exists(dataset_path):
        dataset_path = "BUAADreamer/clevr_count_70k"

    example_file = "examples/vlm/clevr_count_70k_grpo.py"
    config_name = "examples/vlm/clevr_count_70k_grpo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"CLEVR Count 70k GRPO example failed, return_code={return_code}"


@pytest.mark.skip("Currently SFT dataloading is too slow. Needs to be fixed.")
@pytest.mark.gpu
def test_vlm_sft(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_path = "/storage/openpsi/data/BUAADreamer__clevr_count_70k"
    if not os.path.exists(dataset_path):
        dataset_path = "BUAADreamer/clevr_count_70k"

    example_file = "examples/vlm/clevr_count_70k_sft.py"
    config_name = "examples/vlm/clevr_count_70k_sft.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=d1",
            "model.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=1",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"model.path={model_path}",
            timeout=600,  # tokenizing the VLM dataset for SFT takes a long time
        )
    )
    assert success, f"CLEVR Count 70k SFT example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_gsm8k_grpo_megatron(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/experimental/gsm8k_grpo_megatron.py"
    config_name = "examples/experimental/configs/gsm8k_grpo_megatron.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+megatron:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"GSM8K GRPO Megatron example failed, return_code={return_code}"


@pytest.mark.gpu
def test_gsm8k_sft_megatron(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen3-1.7B"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen3-1.7B"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/experimental/gsm8k_sft_megatron.py"
    config_name = "examples/experimental/configs/gsm8k_sft_megatron.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=megatron:d1",
            "model.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=1",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"model.path={model_path}",
        )
    )
    assert success, f"GSM8K SFT Megatron example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_gsm8k_dapo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/experimental/dapo/gsm8k_dapo.py"
    config_name = "examples/experimental/dapo/gsm8k_dapo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"GSM8K DAPO example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_gsm8k_drgrpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/experimental/dr.grpo/gsm8k_drgrpo.py"
    config_name = "examples/experimental/dr.grpo/gsm8k_drgrpo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"GSM8K DRGRP example failed, return_code={return_code}"


@pytest.mark.multi_gpu
def test_gsm8k_liteppo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path = "/storage/openpsi/data/gsm8k"
    if not os.path.exists(dataset_path):
        dataset_path = "openai/gsm8k"

    example_file = "examples/experimental/lite_ppo/gsm8k_liteppo.py"
    config_name = "examples/experimental/lite_ppo/gsm8k_liteppo.yaml"
    loop = asyncio.get_event_loop()
    return_code, success = loop.run_until_complete(
        run_example(
            example_file,
            config_name,
            "allocation_mode=sglang:d1+fsdp:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=256",
            "actor.mb_spec.max_tokens_per_mb=1024",
            f"train_dataset.batch_size=16",
            f"valid_dataset.batch_size=16",
            f"train_dataset.path={dataset_path}",
            f"valid_dataset.path={dataset_path}",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
        )
    )
    assert success, f"GSM8K LitePPO example failed, return_code={return_code}"
