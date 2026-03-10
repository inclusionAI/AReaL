"""Training script for SWE-bench with AReaL proxy mode."""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

from datasets import Dataset

from examples.swe.utils import SWEPPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("SWETrain")


def get_swe_dataset(
    dataset_path: str,
    split: str = "train",
    min_items: int = 64,
) -> Dataset:
    """Create a HuggingFace Dataset from a SWE-bench JSONL file.

    Each line in the JSONL file should be a SWE-bench instance with at minimum:
    - instance_id: The SWE-bench instance ID (e.g., "django__django-10097")
    - problem_statement: The GitHub issue description
    - eval_script: Shell script to evaluate the agent's fix

    Args:
        dataset_path: Path to the SWE-bench JSONL file.
        split: Informational split label (not used for filtering).
        min_items: Minimum dataset size; items are duplicated if fewer exist.

    Returns:
        HuggingFace Dataset of SWE-bench instances.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench dataset not found: {dataset_path}")

    dataset_items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "instance_id" not in item:
                logger.warning(f"Skipping item missing 'instance_id': {line[:100]}")
                continue
            if "problem_statement" not in item:
                logger.warning(
                    "Skipping item missing 'problem_statement': "
                    f"{item.get('instance_id')}"
                )
                continue
            dataset_items.append(item)

    if not dataset_items:
        raise ValueError(f"No valid items found in dataset: {dataset_path}")

    # Duplicate dataset if fewer than min_items for efficient batching
    if len(dataset_items) < min_items:
        original_items = dataset_items.copy()
        while len(dataset_items) < min_items:
            dataset_items.extend(original_items)

    dataset = Dataset.from_list(dataset_items)
    logger.info(
        f"Created SWE dataset with {len(dataset)} items "
        f"from {dataset_path} (split={split})"
    )
    return dataset


def group_filter(x: dict[str, Any]):
    """Filter out groups where all rollouts already solved the task."""
    return x["rewards"].mean() <= 0.95


def _install_swe_deps_on_ray_nodes():
    """Install SWEAgent dependencies on all Ray GPU nodes.

    Each node runs in a separate container with its own venv,
    so we must ensure packages like ``aenv`` are installed everywhere.
    """
    try:
        import ray

        if not ray.is_initialized():
            return

        @ray.remote(num_gpus=0)
        def _install():
            import os
            import socket
            import subprocess

            ip = socket.gethostbyname(socket.gethostname())
            req_path = os.path.join(
                os.environ.get("SWE_AGENT_ROOT", ""),
                "requirements.txt",
            )
            result = subprocess.run(
                ["uv", "pip", "install", "-r", req_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return (
                ip,
                result.returncode,
                result.stderr[-200:] if result.stderr else "",
            )

        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
        ]
        refs = []
        for node in nodes:
            node_ip = node["NodeManagerAddress"]
            refs.append(_install.options(resources={f"node:{node_ip}": 0.01}).remote())

        results = ray.get(refs, timeout=180)
        for ip, rc, err in results:
            if rc != 0:
                logger.warning(f"Failed to install SWE deps on {ip}: {err}")
            else:
                logger.info(f"SWE deps installed on {ip}")
    except Exception as e:
        logger.warning(f"Could not install SWE deps on Ray nodes: {e}")


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, SWEPPOConfig)

    # When using Ray scheduler, ensure SWEAgent deps are on all nodes
    if config.scheduler.type == "ray":
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        _install_swe_deps_on_ray_nodes()

    econfig = config.econfig

    # Resolve dataset paths from config
    train_path = config.train_dataset.path
    valid_path = config.valid_dataset.path

    def resolve_path(p: str) -> str:
        if Path(p).is_absolute() or Path(p).exists():
            return p
        if econfig.dataset_path:
            candidate = Path(econfig.dataset_path) / p
            if candidate.exists():
                return str(candidate)
        return p

    train_dataset = get_swe_dataset(
        dataset_path=resolve_path(train_path),
        split="train",
    )
    valid_dataset = get_swe_dataset(
        dataset_path=resolve_path(valid_path),
        split="test",
    )

    # Build workflow kwargs
    from dataclasses import asdict

    econfig_dict = asdict(econfig)
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=econfig.timeout,
    )

    # Eval workflow with lower temperature for deterministic evaluation
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gen_args"] = dict(
        temperature=0.0,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.swe.agent.SWEAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.swe.agent.SWEAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn="examples.swe.train.group_filter",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
