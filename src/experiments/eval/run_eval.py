#! /usr/bin/env python3
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from sklearn.model_selection import ParameterGrid

from tau2.data_model.simulation import Results
from tau2.run import RunConfig, run_domain


class RunMode(str, Enum):
    DEFAULT = "default"
    NO_USER = "no-user"
    ORACLE_PLAN = "oracle-plan"
    NO_USER_ORACLE_PLAN = "no-user-op"


def is_reasoning_llm(llm: str) -> bool:
    pat = r"^o[1-4]-[a-z0-9-]+$"
    return re.match(pat, llm) is not None


def is_gpt_5(llm: str) -> bool:
    pat = r"^gpt-5"
    return re.match(pat, llm) is not None


def make_config(
    llm: str,
    domain: str,
    mode: RunMode,
    llm_agent_args: dict,
    llm_user: str,
    llm_user_args: dict,
    seed: int,
    max_steps: int,
    max_errors: int,
    max_concurrency: int,
    num_trials: int,
    num_tasks: Optional[int],
    exp_dir: str,
) -> Optional[RunConfig]:
    """
    Creates a RunConfig object for a single experiment configuration.

    This function constructs a RunConfig object based on the provided parameters, handling
    special cases for different LLMs and domains. It sets up:
    - Agent configuration (llm_agent, llm_agent_gt, etc.)
    - User simulator configuration
    - LLM-specific parameters (temperature, reasoning_effort, etc.)
    - Experiment parameters (trials, steps, errors, etc.)

    Args:
        llm (str): The LLM model to use (e.g., "gpt-4.1-2025-04-14")
        domain (str): The domain to run experiments on ("retail", "airline", "telecom")
        mode (RunMode): The experiment mode ("base", "gt", "solo", "solo-gt")
        exp_dir (str): Name of the experiment directory for saving results
        num_tasks (int): Number of tasks to run. Defaults to None.
        llm_user (str): LLM model to use for user simulator.
        llm_agent_args (dict): Arguments for agent LLM.
        llm_user_args (dict): Arguments for user LLM.
        seed (int): Random seed for experiments.
        max_steps (int): Maximum number of steps per trial.
        max_errors (int): Maximum number of errors allowed.
        max_concurrency (int): Maximum number of concurrent simulations.
        num_trials (int): Number of trials per configuration.

    Returns:
        Optional[RunConfig]: A configured RunConfig object, or None if the configuration
                           is not valid for the given domain/mode combination
    """
    if not domain.startswith("telecom") and (
        mode in {RunMode.ORACLE_PLAN, RunMode.NO_USER_ORACLE_PLAN}
    ):
        logger.info(
            f"Mode {mode.value} is not supported for domain {domain}. Skipping..."
        )
        return None

    if mode == RunMode.DEFAULT:
        agent = "llm_agent"
    elif mode == RunMode.ORACLE_PLAN:
        agent = "llm_agent_gt"
    elif mode == RunMode.NO_USER:
        agent = "llm_agent_solo"
    elif mode == RunMode.NO_USER_ORACLE_PLAN:
        agent = "llm_agent_solo_gt"
    else:
        raise ValueError(f"Invalid agent mode: {mode}")

    user = "user_simulator"
    if mode in {RunMode.NO_USER, RunMode.NO_USER_ORACLE_PLAN}:
        user = "dummy_user"

    if is_reasoning_llm(llm):
        logger.info(f"Using reasoning LLM: {llm}. Setting reasoning effort to high...")
        llm_agent_args = deepcopy(llm_agent_args)
        llm_agent_args["reasoning_effort"] = "high"
        tmp = llm_agent_args.pop("temperature")
        logger.warning(
            f"Temperature: {tmp} removed for reasoning LLM: {llm}. Only default temp supported."
        )

    if is_gpt_5(llm):
        logger.info(f"Using GPT-5. Setting reasoning effort to low...")
        llm_agent_args = deepcopy(llm_agent_args)
        # llm_agent_args["reasoning"] = {"effort": "high"}
        tmp = llm_agent_args.pop("temperature")
        logger.warning(
            f"Temperature: {tmp} removed for reasoning LLM: {llm}. Only default temp supported."
        )

    save_to = f"{exp_dir}/{llm}_{domain}_{mode.value}_{llm_user}_{num_trials}trials"
    if num_tasks is not None:
        save_to += f"_{num_tasks}tasks"
    config = RunConfig(
        domain=domain,
        agent=agent,
        llm_agent=llm,
        llm_args_agent=llm_agent_args,
        user=user,
        llm_user=llm_user,
        llm_args_user=llm_user_args,
        num_trials=num_trials,
        seed=seed,
        max_steps=max_steps,
        max_errors=max_errors,
        max_concurrency=max_concurrency,
        save_to=save_to,
        num_tasks=num_tasks,
    )

    return config


def parse_file_name(file_name: str) -> dict:
    """
    Parse the file name into its components.
    """
    pat = r"^(.+)_(.+)_(.+)_(.+)_(\d+)trials(_(\d+)tasks)?\.json"
    match = re.match(pat, file_name)
    if match is None:
        raise ValueError(f"Invalid file name: {file_name}")
    return {
        "llm": match.group(1),
        "domain": match.group(2),
        "mode": match.group(3),
        "llm_user": match.group(4),
        "num_trials": int(match.group(5)),
        "num_tasks": int(match.group(7)) if match.group(7) is not None else None,
    }


def make_configs(
    hyperparams: dict,
    llm_user: str,
    llm_agent_args: dict,
    llm_user_args: dict,
    seed: int,
    max_steps: int,
    max_errors: int,
    max_concurrency: int,
    num_trials: int,
    exp_dir: str,
    num_tasks: Optional[int] = None,
) -> List[RunConfig]:
    """
    Generates all possible experiment configurations based on the provided hyperparameters.

    This function creates a list of RunConfig objects by combining all possible values
    from the hyperparameters dictionary. It handles:
    - All combinations of LLMs, domains, and modes
    - Debug mode configuration (reduces number of trials)
    - Experiment naming and organization

    Args:
        hyperparams (dict): Dictionary containing the hyperparameters to test
        llm_user (str): LLM model to use for user simulator
        llm_agent_args (dict): Arguments for agent LLM
        llm_user_args (dict): Arguments for user LLM
        seed (int): Random seed for experiments
        max_steps (int): Maximum number of steps per trial
        max_errors (int): Maximum number of errors allowed
        max_concurrency (int): Maximum number of concurrent simulations
        num_trials (int): Number of trials per configuration
        num_tasks (int): Number of tasks to run. Defaults to None.
        exp_dir (str, optional): Name of the experiment directory.""

    Returns:
        List[RunConfig]: A list of all valid experiment configurations
    """
    logger.info("Generating run configs from hyperparams...")
    if num_tasks is not None:
        logger.info(f"Running with {num_tasks} tasks per trial.")
    else:
        logger.info("Running with all tasks.")
    configs = []
    for params in list(ParameterGrid(hyperparams)):
        config = make_config(
            llm=params["llm"],
            domain=params["domain"],
            mode=params["mode"],
            exp_dir=exp_dir,
            num_tasks=num_tasks,
            llm_agent_args=deepcopy(llm_agent_args),
            llm_user_args=deepcopy(llm_user_args),
            seed=seed,
            max_steps=max_steps,
            max_errors=max_errors,
            max_concurrency=max_concurrency,
            num_trials=num_trials,
            llm_user=llm_user,
        )
        if config is not None:
            configs.append(config)
    logger.info(f"{len(configs)} run configs generated.")
    return configs


def run_evals(configs: List[RunConfig]):
    """
    Executes a series of evaluation experiments using the provided configurations.

    This function runs each experiment configuration in sequence, executing the domain
    simulations and saving the results. It:
    - Processes each configuration in order
    - Runs the domain simulation for each config
    - Saves results to the specified location
    - Provides progress updates through logging

    Args:
        configs (List[RunConfig]): List of experiment configurations to run
    """
    for i, config in enumerate(configs):
        logger.info(f"{i + 1} / {len(configs)}: Running eval for {config.save_to}...")
        run_domain(config)
        logger.info(f"Eval for {config.save_to} completed.")


def get_simulation_results(exp_dir: Path) -> List[Tuple[dict, Results]]:
    """
    Get the data for the given experiment directory.

    Args:
        exp_dir (Path): Path to the experiment directory containing simulation results

    Returns:
        List[Tuple[dict, Results]]: List of tuples containing parameters and simulation results
    """
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory {exp_dir} does not exist.")
    results = []
    for result_file in exp_dir.glob("*.json"):
        params = parse_file_name(result_file.name)
        results.append((params, Results.load(result_file)))
    return results
