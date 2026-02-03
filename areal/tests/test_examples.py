import asyncio
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest

from areal.infra.platforms import current_platform
from areal.tests.utils import get_dataset_path, get_model_path
from areal.utils import logging
from areal.utils.concurrent import run_async_task
from areal.utils.proc import kill_process_tree

logger = logging.getLogger("TestExamples")

SUCCESS_PATTERN = re.compile(r"Epoch 1/\d+ Step 1/\d+ Train step 1/\d+ done\.")

pytestmark = pytest.mark.slow


async def run_example(
    example_file: str,
    config_name: str,
    *additional_args,
    timeout: int = 300,
    success_pattern=SUCCESS_PATTERN,
    single_controller: bool = False,
) -> bool:
    """
    Run a single example and return the result.

    Args:
        example_file: Path to the example file
        config_name: Name of the config to use
        additional_args: Additional command line arguments
        timeout: Timeout in seconds
        success_pattern: Regex pattern to identify successful completion
        single_controller: If True, run directly without launcher (single-controller mode)

    Returns:
        Tuple of (success, stdout, stderr)
    """
    # Construct the command
    if single_controller:
        # Single-controller mode: run script directly
        cmd = [
            "python3",
            example_file,
            "--config",
            config_name,
        ]
    else:
        # SPMD mode: use launcher
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
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                line = line.decode()
                logger.info(f"[Example Output] {line.rstrip()}")
                # Check for success patterns
                success = bool(success_pattern.search(line))
                if success:
                    break
            except (TimeoutError, ValueError):
                # NOTE: Here ValueError is raised when the input line is too long
                # that exceeds the buffer size, which will happen if the experiment
                # has tqdm progress bar output.
                break

        if success:
            logger.info(f"✓ {example_file} with config {config_name} - SUCCESS")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

        # Check if process has terminated
        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=0.01)
            logger.error(f"Process terminated unexpectedly. Return code: {return_code}")
            break
        except TimeoutError:
            pass

        # Check timeout
        if (time.monotonic() - start_time) > timeout:
            logger.error("Process timed out without successful result, terminating...")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

    kill_process_tree(process.pid)
    return success


@pytest.mark.multi_gpu
def test_countdown_example(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    tmp_path = tmp_path_factory.mktemp("countdown_data")
    data_path = tmp_path / "data/countdown/qwen"
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    os.makedirs(data_path, exist_ok=True)
    test_file_path = data_path / "test_e.jsonl"
    train_file_path = data_path / "train_e.jsonl"
    # generate countdown dataset
    shutil.copy("examples/countdown/countdown.py", tmp_path)
    subprocess.run(
        [
            "python3",
            "countdown.py",
            "--num_samples=10000",
            "--eval_size=100",
            "--tokenizer_path",
            model_path,
        ],
        cwd=tmp_path,
        check=True,
    )

    example_file = "examples/countdown/train.py"
    config_name = "examples/countdown/train_config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={str(train_file_path)}",
        f"valid_dataset.path={str(test_file_path)}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    assert success, "Countdown example failed"


# vLLM is too slow to launch up in CI environments
# We have tests for vLLM in test_inference_engines.py,
# so we can skip the integration test of vLLM here.
@pytest.mark.parametrize(
    "alloc_mode,single_controller",
    [
        ("sglang:d1+megatron:d1", False),
        ("sglang:d1+megatron:d1", True),
    ],
)
@pytest.mark.multi_gpu
@pytest.mark.ci
def test_gsm8k_grpo(tmp_path_factory, alloc_mode, single_controller):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_grpo.yaml"

    additional_args = [
        f"allocation_mode={alloc_mode}",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    ]
    if single_controller:
        additional_args.append("scheduler.type=local")

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        *additional_args,
        timeout=900,
        single_controller=single_controller,
    )
    assert success, f"GSM8K GRPO example failed (single_controller={single_controller})"


@pytest.mark.parametrize(
    "alloc_mode,single_controller",
    [
        ("fsdp:d1", False),
        ("megatron:d1", False),
        ("fsdp:d1", True),
    ],
)
@pytest.mark.gpu
@pytest.mark.ci
def test_gsm8k_sft(tmp_path_factory, alloc_mode, single_controller):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_sft.py"
    config_name = "examples/math/gsm8k_sft.yaml"

    additional_args = [
        f"allocation_mode={alloc_mode}",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    ]
    if single_controller:
        additional_args.append("scheduler.type=local")

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        *additional_args,
        single_controller=single_controller,
    )
    assert success, f"GSM8K SFT example failed (single_controller={single_controller})"


@pytest.mark.gpu
def test_gsm8k_eval(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_eval.py"
    config_name = "examples/math/gsm8k_grpo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=16",
        "valid_dataset.batch_size=16",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        success_pattern=re.compile(r"Evaluation Results:"),
        single_controller=True,
    )
    assert success, "GSM8K Eval example failed"


@pytest.mark.skip("Currently VLM dataloading is too slow. Needs to be fixed.")
@pytest.mark.multi_gpu
def test_vlm_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/BUAADreamer__clevr_count_70k",
        "BUAADreamer/clevr_count_70k",
    )

    example_file = "examples/vlm/clevr_count_70k_grpo.py"
    config_name = "examples/vlm/clevr_count_70k_grpo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        timeout=1800,
    )
    assert success, "CLEVR Count 70k GRPO example failed"


@pytest.mark.skip("Currently VLM dataloading is too slow. Needs to be fixed.")
@pytest.mark.gpu
def test_vlm_sft(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/BUAADreamer__clevr_count_70k",
        "BUAADreamer/clevr_count_70k",
    )

    example_file = "examples/vlm/clevr_count_70k_sft.py"
    config_name = "examples/vlm/clevr_count_70k_sft.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=d1",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        timeout=600,  # tokenizing the VLM dataset for SFT takes a long time
    )
    assert success, "CLEVR Count 70k SFT example failed"


@pytest.mark.multi_gpu
def test_gsm8k_ppo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_ppo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "critic.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        f"critic.path={model_path}",
    )
    assert success, "GSM8K PPO example failed"


@pytest.mark.multi_gpu
def test_gsm8k_grpo_lora(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/lora/gsm8k_grpo_lora.py"
    config_name = "examples/lora/gsm8k_grpo_lora.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    assert success, "GSM8K GRPO LoRA example failed"


@pytest.mark.multi_gpu
def test_multi_turn_math(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/multi_turn_math/gsm8k_rl_mt.py"
    config_name = "examples/multi_turn_math/gsm8k_grpo_mt.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    assert success, "Multi-turn Math example failed"


@pytest.mark.gpu
def test_hhrlhf_rw(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/Anthropic___hh-rlhf/", "Anthropic/hh-rlhf"
    )

    example_file = "examples/alignment/hhrlhf_rw.py"
    config_name = "examples/alignment/hhrlhf_rw.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=d1",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        timeout=1800,
    )
    assert success, "HH-RLHF Reward Modeling example failed"


@pytest.mark.multi_gpu
def test_tir_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/tir/train_tir.py"
    config_name = "examples/tir/tir_math_config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=64",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "tir.max_length=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    assert success, "TIR GRPO example failed"


@pytest.mark.multi_gpu
def test_search_agent_deepresearch(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for LLM judge, 2 for RL) to run."
        )
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    dataset_path = "/storage/openpsi/data/inclusionAI__Asearcher-train-data/ASearcher-LRM-35k.jsonl"
    if not os.path.exists(dataset_path):
        pytest.skip("Tongyi DeepResearch dataset not available")

    example_file = "examples/search_agent/tongyi_deepresearch/train.py"
    config_name = "examples/search_agent/tongyi_deepresearch/config.yaml"

    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    llm_judge_exp_name = uuid.uuid4().hex
    llm_judge_trial_name = uuid.uuid4().hex
    _env = os.environ.copy()
    _env[current_platform.device_control_env_var] = visible_devices[-1]
    llm_judge_proc = subprocess.Popen(
        " ".join(
            [
                "python3",
                "-m",
                "areal.launcher.local",
                example_file,
                "--config",
                config_name,
                "allocation_mode=sglang:d1",
                f"cluster.fileroot={str(experiments_path)}",
                f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
                f"experiment_name={llm_judge_exp_name}",
                f"trial_name={llm_judge_trial_name}",
                f"actor.path={model_path}",
            ]
        ),
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=_env,
    )

    try:
        time.sleep(20)

        success = run_async_task(
            run_example,
            example_file,
            config_name,
            "allocation_mode=sglang:d1+megatron:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=128",
            "actor.mb_spec.max_tokens_per_mb=2048",
            "train_dataset.batch_size=4",
            f"train_dataset.path={dataset_path}",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
            "max_tokens_per_trajectory=1024",
            "max_llm_calls_per_run=2",
            f"judge_engine.experiment_name={llm_judge_exp_name}",
            f"judge_engine.trial_name={llm_judge_trial_name}",
        )
        if not success:
            raise RuntimeError("Search Agent DeepResearch example failed")
    finally:
        kill_process_tree(llm_judge_proc.pid, graceful=False)


@pytest.mark.multi_gpu
def test_openai_agents(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    example_file = "examples/openai_agents/train_agents.py"
    config_name = "examples/openai_agents/config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=1",
        "gconfig.max_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=4096",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        "valid_dataset.batch_size=16",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    if not success:
        raise RuntimeError("OpenAI Agents example failed")


@pytest.mark.multi_gpu
def test_camel(tmp_path_factory):
    try:
        import camel.agents  # noqa
    except ImportError:
        pytest.skip("camel-ai is not installed. Skipping camel example test.")
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    example_file = "examples/camel/train.py"
    config_name = "examples/camel/config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "allocation_mode=sglang:d1+fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=4096",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
    )
    if not success:
        raise RuntimeError("Camel Math example failed")
