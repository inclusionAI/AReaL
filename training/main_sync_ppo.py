import dataclasses
import datetime
import os
from typing import Dict

import hydra
import yaml
from omegaconf import MISSING, OmegaConf

from realhf.api.quickstart.entrypoint import kind_reminder
from realhf.experiments.common.ppo_math_exp import PPOMATHConfig
from training.utils import run_experiment


@hydra.main(version_base=None, config_path="configs", config_name="sync-ppo")
def main(args):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import realhf.base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    # Overwrite the python dataclass configuration with yaml
    default_args = OmegaConf.structured(PPOMATHConfig)
    args = OmegaConf.merge(default_args, args)
    args: PPOMATHConfig = OmegaConf.to_object(args)

    # Set experiment trial name.
    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = (
            f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )
    else:
        trial_name = args.trial_name

    if args.mode != "ray":
        raise RuntimeError("This script only supports the `ray` mode.")

    from realhf.base.constants import get_log_path

    # Save overwritten configuration to yaml
    config_save_path = os.path.join(get_log_path(args), "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        config_dict: Dict = dataclasses.asdict(args)
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    kind_reminder("ppo-math", logger, args)

    logger.warning(
        f"Synchronous PPO is not recommended for production and customization. "
        "Run asynchronous PPO with `ppo.recompute_logprob=False`, "
        "`ppo.use_decoupled_loss=False`, and `max_head_offpolicyness=0` "
        "will essentially replicate the synchronous PPO behavior, but "
        "is easier to customize and more stable."
    )

    run_experiment(args, exp_name, trial_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="store_true")
    args = parser.parse_known_args()[0]
    if args.help:
        from realhf.api.cli_args import print_config_help

        print_config_help(PPOMATHConfig())
        exit(0)
    main()
