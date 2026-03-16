"""Training script for CC (Claude Code) agent with AReaL proxy mode."""

import sys
import warnings
from pathlib import Path

from examples.swe.train import get_swe_dataset
from examples.swe.utils import CCPPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("CCTrain")


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

    config, _ = load_expr_config(args, CCPPOConfig)

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
        temperature=0.6,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.swe.agent_cc.CCAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.swe.agent_cc.CCAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn="examples.swe.train.group_filter",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
