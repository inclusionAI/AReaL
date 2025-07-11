# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import datetime
import getpass
import os
import pathlib
import re
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel

from realhf.api.cli_args import console, highlighter, print_config_help
from realhf.api.quickstart.entrypoint import QUICKSTART_CONFIG_CLASSES, QUICKSTART_FN
from realhf.base.importing import import_module
from realhf.base.prologue import (
    PROLOGUE_EXTERNAL_CONFIG_NAME,
    PROLOGUE_FLAG_NAME,
    PROLOGUE_FLAG_VAR_NAME,
    get_experiment_name,
    get_trial_name,
)
from realhf.version import get_full_version_with_dirty_description

# NOTE: Register all implemented experiments inside ReaL.
import_module(
    str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "common"),
    re.compile(r".*_exp\.py$"),
)
import_module(
    str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "async_exp"),
    re.compile(r".*_exp\.py$"),
)


def print_help(exp_type):
    """Print comprehensive help with rich formatting"""
    config_class = QUICKSTART_CONFIG_CLASSES[exp_type]()

    # Main help panel
    console.print(
        Panel.fit(
            f"[header]Configuration Help for {exp_type}[/header]", border_style="border"
        )
    )

    # Configuration options section
    console.print("\n[title]CONFIGURATION OPTIONS[/title]")
    print_config_help(config_class)

    # Usage section
    console.print("\n[title]USAGE[/title]")
    usage_code = f"python -m realhf.apps.quickstart {exp_type} --config ./your/config.yaml [OPTIONS]"
    console.print(highlighter(usage_code))

    # Examples section
    console.print("\n[title]EXAMPLE OVERRIDES[/title]")
    example_code = f"python -m realhf.apps.quickstart {exp_type} --config ./your/config.yaml dataset.path=/my/dataset.jsonl actor.optimizer.lr=2e-5"
    console.print(highlighter(example_code))

    # Footer
    console.print("\n[dim]Use [bold]--help[/bold] to show this message again[/dim]")


def print_version():
    console.print(f"AReaL Version: {get_full_version_with_dirty_description()}")


def main():
    # Create parser with add_help=False to disable automatic --help
    parser = argparse.ArgumentParser(prog="ReaL Quickstart", add_help=False)

    # Add custom help argument that won't conflict
    parser.add_argument(
        "--help", action="store_true", help="Show this help message and exit"
    )
    parser.add_argument("--version", action="store_true", help="Show AReaL version")

    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    for k, v in QUICKSTART_FN.items():
        # Create subparser with add_help=False
        subparser = subparsers.add_parser(k, add_help=False)

        # Add custom help to subparser
        subparser.add_argument(
            "--help", action="store_true", help="Show help for this command"
        )
        subparser.add_argument(
            PROLOGUE_FLAG_NAME,
            type=str,
            help="Set config (*.yaml) for this experiment.",
        )

        subparser.set_defaults(func=v)

    # Parse known args first to check for help
    args = vars(parser.parse_known_args()[0])

    if args["version"]:
        print_version()
        return

    # Handle help at both main and subcommand levels
    if args["help"]:
        if args["cmd"]:
            # Subcommand help
            print_help(args["cmd"])
        else:
            # Main help
            parser.print_help()
        return

    # Continue with normal execution
    if not args["cmd"]:
        parser.print_help()
    experiment_name = ""
    trial_name = ""

    if args[PROLOGUE_FLAG_VAR_NAME]:
        config_dir, experiment_name, trial_name = prepare_hydra_config(
            args["cmd"], args[PROLOGUE_FLAG_VAR_NAME]
        )
        sys.argv.remove(PROLOGUE_FLAG_NAME)
        sys.argv.remove(args[PROLOGUE_FLAG_VAR_NAME])
        sys.argv += [f"--config-path", f"{config_dir}"]
    else:
        experiment_name = get_experiment_name()
        trial_name = get_trial_name()

    launch_hydra_task(
        args["cmd"], experiment_name, trial_name, QUICKSTART_FN[args["cmd"]]
    )


def prepare_hydra_config(name: str, prologue_path: str):
    config = OmegaConf.load(prologue_path)
    experiment_name = get_experiment_name(config.get("experiment_name"))
    trial_name = get_trial_name(config.get("trial_name"))
    config_dir = f"{config.cluster.fileroot}/configs/{getpass.getuser()}/{experiment_name}/{trial_name}"
    os.makedirs(config_dir, exist_ok=True)

    config.pop(PROLOGUE_EXTERNAL_CONFIG_NAME, {})
    with open(f"{config_dir}/{name}.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    return (config_dir, experiment_name, trial_name)


def launch_hydra_task(
    name: str, experiment_name: str, trial_name: str, func: hydra.TaskFunction
):
    # Disable hydra logging.
    if not any("hydra/job_logging=disabled" in x for x in sys.argv):
        sys.argv.insert(2, "hydra/job_logging=disabled")

    sys.argv.pop(1)

    func()


if __name__ == "__main__":
    main()
