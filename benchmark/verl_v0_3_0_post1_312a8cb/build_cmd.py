from pathlib import Path

paths = {
    1: "/storage/testing/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
    7: "/storage/testing/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B",
    32: "/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B",
}

from typing import Any, List, Union

import yaml


def yaml_to_cli_args(yaml_file_path: str) -> List[str]:
    """
    Convert a YAML file to structured CLI configuration arguments.

    Args:
        yaml_file_path (str): Path to the YAML file

    Returns:
        List[str]: List of CLI arguments in the format "key=value" or "nested.key=value"

    Example:
        >>> yaml_to_cli_args("config.yaml")
        ['wandb.mode=disabled', 'mode=ray', 'max_head_offpolicyness=4', ...]
    """
    with open(yaml_file_path, "r") as file:
        data = yaml.safe_load(file)

    return _flatten_dict_to_cli_args(data)


def yaml_dict_to_cli_args(yaml_dict: dict) -> List[str]:
    """
    Convert a YAML dictionary to structured CLI configuration arguments.

    Args:
        yaml_dict (dict): Dictionary loaded from YAML

    Returns:
        List[str]: List of CLI arguments in the format "key=value" or "nested.key=value"
    """
    return _flatten_dict_to_cli_args(yaml_dict)


def _flatten_dict_to_cli_args(data: dict, parent_key: str = "") -> List[str]:
    """
    Recursively flatten a nested dictionary into CLI arguments.

    Args:
        data (dict): Dictionary to flatten
        parent_key (str): Current parent key path

    Returns:
        List[str]: Flattened CLI arguments
    """
    cli_args = []

    for key, value in data.items():
        # Build the full key path
        full_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively process nested dictionaries
            cli_args.extend(_flatten_dict_to_cli_args(value, full_key))
        else:
            # Convert value to string representation
            cli_args.append(f"{full_key}={_format_value(value)}")

    return cli_args


def _format_value(value: Any) -> str:
    """
    Format a value for CLI argument representation.

    Args:
        value: The value to format

    Returns:
        str: Formatted string representation of the value
    """
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        # Convert list to comma-separated string
        return ",".join(str(item) for item in value)
    elif value is None:
        return "null"
    else:
        return str(value)


def get_allocation_mode(n_nodes: int, model_size: int):
    assert n_nodes >= 4, n_nodes
    sglang_tp = 1 if model_size < 32 else 8
    n_sglang_gpus = n_nodes * 8 // 4 * 3
    sglang_alloc = f"sglang.d{n_sglang_gpus // sglang_tp}p1m{sglang_tp}"
    n_train_gpus = n_nodes * 8 - n_sglang_gpus

    if model_size == 1:
        pp = 2
        tp = 1
    elif model_size == 7:
        pp = 4
        tp = 1
    elif model_size == 32:
        pp = 8
        tp = 2
    else:
        raise NotImplementedError(model_size)
    train_alloc = f"d{n_train_gpus // tp // pp}m{tp}p{pp}"
    return sglang_alloc + "+" + train_alloc


def get_trial_name(model_size: int, ctx: int, n_nodes: int):
    trial_name = f"m{model_size}-ctx{ctx}-n{n_nodes}"
    return trial_name


def build_cmd(model_size: int, ctx: int, n_nodes: int):
    trial_name = get_trial_name(model_size, ctx, n_nodes)
    allocation_mode = get_allocation_mode(model_size=model_size, n_nodes=n_nodes)
    config_path = Path(__file__).parent / "areal_config.yaml"
    cli_args = yaml_to_cli_args(str(config_path))
    cmd = (
        ["python3", "training/main_async_ppo.py"]
        + cli_args
        + [
            f"actor.path={paths[model_size]}",
            f"ppo.gen.max_new_tokens={ctx-1024}",
            f"trial_name={trial_name}",
            f"allocation_mode={allocation_mode}",
            f"n_nodes={n_nodes}",
        ]
    )
    if model_size > 1:
        # overwrite to avoid OOM
        cmd += ["actor.sglang.mem_fraction_static=0.7"]
    return " ".join(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--model-size", type=int)
    parser.add_argument("-s", "--ctx", type=int)
    parser.add_argument("-n", "--n-nodes", type=int)
    args = parser.parse_args()
    print(build_cmd(args.model_size, args.ctx, args.n_nodes))
