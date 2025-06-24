import dataclasses
import enum
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any, Dict, Optional

import wandb

import realhf.api.core.system_api as config_pkg
from realhf.api.cli_args import BaseExperimentConfig
from realhf.base import constants, logging

try:
    import swanlab
except (ModuleNotFoundError, ImportError):
    swanlab = None

logger = logging.getLogger("AutomaticEvaluator", "colored")


class EvaluationStepStatus(enum.Enum):
    PENDING = 0
    RUNNING = 1
    FAILED = 2
    DONE = 3
    LOGGED = 4


@dataclasses.dataclass
class EvaluationStep:
    args: BaseExperimentConfig
    global_step: int
    status: EvaluationStepStatus
    start_time: Optional[float] = None
    ckpt_dir: Optional[str] = None
    process: Optional[subprocess.Popen] = None

    @staticmethod
    def from_ckpt_dir(args, ckpt_dir):
        # NOTE: ckpt_dir should be absolute path
        if pathlib.Path(ckpt_dir).is_symlink():
            return None
        _dir = os.path.basename(ckpt_dir)
        match = re.match(r"epoch(\d+)epochstep(\d+)globalstep(\d+)", _dir)
        if not match:
            return None
        _, _, global_step = map(int, match.groups())
        return EvaluationStep(
            args=args,
            global_step=global_step,
            status=EvaluationStepStatus.PENDING,
            ckpt_dir=ckpt_dir,
        )

    @staticmethod
    def from_output_dir(args, output_dir):
        # NOTE: output_dir should be absolute path
        # Should only be called in recover.
        _dir = os.path.basename(output_dir)
        match = re.match(r"globalstep(\d+)", _dir)
        if not match:
            return None
        global_step = int(match.groups()[0])
        return EvaluationStep(
            args=args, global_step=global_step, status=EvaluationStepStatus.LOGGED
        )

    @property
    def output_dir(self):
        return os.path.join(
            constants.get_log_path(self.args),
            "eval_output",
            f"globalstep{self.global_step}",
        )

    def slurm_eval_cmd(self, config: config_pkg.AutomaticEvaluator):
        slurm_job_name = f"{constants.experiment_name()}_{constants.trial_name()}:eval_globalstep{self.global_step}"
        cmd = (
            f"srun --mpi=pmi2 -J {slurm_job_name} --ntasks=1 --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G "
            f"singularity exec --no-home --nv --pid --writable-tmpfs --bind /storage:/storage "
            f"{config.eval_job_image or self.args.cluster.gpu_image} "
            f"bash ./evaluation/sh/install_deps_and_eval.sh {self.ckpt_dir} {self.output_dir} "
            f"{config.data_names} {config.max_gen_tokens} {config.prompt_type}"
        )
        return cmd

    def submit(self, config: config_pkg.AutomaticEvaluator):
        os.makedirs(self.output_dir, exist_ok=True)
        log_file = open(os.path.join(self.output_dir, "output.log"), "w")
        if self.args.mode == "slurm":
            cmd = self.slurm_eval_cmd(config)
        else:
            raise NotImplementedError(
                "AutomaticEvaluator does only support slurm cluster."
            )

        logger.info(
            f"Submitting evaluation job of checkpoint at {self.ckpt_dir} (globalstep{self.global_step}), "
            f"command: {cmd}"
        )
        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            shell=True,
        )
        self.start_time = time.perf_counter()
        self.status = EvaluationStepStatus.RUNNING

    def log(self, config: config_pkg.AutomaticEvaluator) -> bool:
        result_path = os.path.join(
            self.output_dir,
            f"math_eval_{config.max_gen_tokens}",
            f"aggregate_parallel_{config.prompt_type}.json",
        )
        # NOTE: If decoding json failed or not found,
        # evaluation step will be marked as failed.
        try:
            with open(result_path, "r") as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            logger.warning(f"JSON file {result_path} decoding failed.")
            self.status = EvaluationStepStatus.FAILED
            return False
        except FileNotFoundError:
            logger.warning(f"JSON file {result_path} does not exist.")
            self.status = EvaluationStepStatus.FAILED
            return False

        log_data = {}
        for data_name, d in data.items():
            for k, v in d.items():
                log_data[f"{data_name}_{k}"] = v
        wandb.log(log_data, step=self.global_step)
        if swanlab is not None:
            swanlab.log(log_data, step=self.global_step)
        self.status = EvaluationStepStatus.LOGGED
        logger.info(f"Logging eval result {log_data} to step {self.global_step}")

        return True

    def check(self):
        assert self.process is not None
        result = self.process.poll()
        if not result is None:
            logger.info(
                f"Evaluation of checkpoint (globalstep{self.global_step}) is done, returncode={self.process.returncode}, "
                f"time passed {time.perf_counter() - self.start_time:.3f} s."
            )
            if self.process.returncode == 0:
                self.status = EvaluationStepStatus.DONE
            else:
                self.status = EvaluationStepStatus.FAILED


class AutomaticEvaluator:

    def __init__(
        self,
        args: BaseExperimentConfig,
        config: config_pkg.AutomaticEvaluator,
        wandb_config: config_pkg.WandBConfig,
        swanlab_config: config_pkg.SwanlabConfig,
    ):
        self.args = args
        self.__eval_steps: Dict[int, EvaluationStep] = {}
        self.__max_concurrent_jobs = config.max_concurrent_jobs
        self.__wandb_config = wandb_config
        self.__swanlab_config = swanlab_config
        self.__config = config
        self.__wandb_initialized = False
        self.__swanlab_initialized = False
        # Check evaluated checkpoints by logs in recover
        # NOTE: All previous evaluation steps with output will be marked
        # as logged, even if it is not really logged in wandb.
        # This is because we do not know the status of evaluation jobs
        # submitted before recover.
        # Resubmiting or waiting for these jobs will probably result in
        # unexpected behaviors.
        output_parent = os.path.join(
            constants.get_log_path(args),
            "eval_output",
        )
        if os.path.exists(output_parent):
            for output_dir in os.listdir(output_parent):
                output_dir = os.path.join(output_parent, output_dir)
                eval_step = EvaluationStep.from_output_dir(self.args, output_dir)
                if eval_step:
                    self.__eval_steps[eval_step.global_step] = eval_step

        logger.info(
            f"Initializing AutomaticEvaluator: \n"
            f"eval_job_image: {config.eval_job_image}\n"
            f"data_names: {config.data_names}\n"
            f"max_gen_tokens: {config.max_gen_tokens}\n"
            f"max_concurrent_jobs: {config.max_concurrent_jobs}\n"
            f"Existing eval outputs for global steps: "
            f"{list(self.__eval_steps.keys())}"
        )
        if self.__config.initial_checkpoint_path and 0 not in self.__eval_steps:
            self.__eval_steps[0] = EvaluationStep(
                args=self.args,
                global_step=0,
                status=EvaluationStepStatus.PENDING,
                ckpt_dir=self.__config.initial_checkpoint_path,
            )

        if not self.args.mode == "slurm":
            raise NotImplementedError(
                "Currently only support automatic evaluation for slurm"
            )

    def __lazy_wandb_init(self):
        # Initializing wandb for evaluator.
        # Here we use lazy init because if this wandb instance is launched
        # with wandb instance on master worker without a time interval,
        # one of them will fail.
        wandb.login()
        wandb.init(
            mode=self.__wandb_config.mode,
            entity=self.__wandb_config.entity,
            project=self.__wandb_config.project or constants.experiment_name(),
            name=self.__wandb_config.name or f"{constants.trial_name()}_eval",
            job_type=self.__wandb_config.job_type,
            group=self.__wandb_config.group
            or f"{constants.experiment_name()}_{constants.trial_name()}",
            notes=self.__wandb_config.notes,
            tags=self.__wandb_config.tags,
            config=self.__wandb_config.config,
            dir=constants.get_log_path(self.args),
            force=True,
            id=f"{constants.experiment_name()}_{constants.trial_name()}_eval",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

    def __lazy_swanlab_init(self):
        if self.__swanlab_config.api_key:
            swanlab.login(self.__swanlab_config.api_key)
        if self.__swanlab_config.config is None:
            import yaml

            with open(
                os.path.join(
                    constants.get_log_path(self.args),
                    "config.yaml",
                ),
                "r",
            ) as f:
                __config = yaml.safe_load(f)
        else:
            __config = self.__swanlab_config.config
        __config["FRAMEWORK"] = "AReaL"
        swanlab.init(
            project=self.__swanlab_config.project or constants.experiment_name(),
            experiment_name=self.__swanlab_config.name
            or f"{constants.trial_name()}_eval",
            config=__config,
            logdir=self.__swanlab_config.logdir
            or os.path.join(
                constants.get_log_path(self.args),
                "swanlab",
            ),
            mode=self.__swanlab_config.mode,
        )

    def step(self):
        # Check whether a new evaluation step should be created
        ckpt_parent = os.path.join(
            constants.get_save_path(self.args),
            "actor",
        )
        if os.path.exists(ckpt_parent):
            for ckpt_dir in os.listdir(ckpt_parent):
                ckpt_dir = os.path.join(ckpt_parent, ckpt_dir)
                eval_step = EvaluationStep.from_ckpt_dir(self.args, ckpt_dir)
                if eval_step is None:
                    continue
                if eval_step.global_step in self.__eval_steps:
                    continue
                self.__eval_steps[eval_step.global_step] = eval_step
                logger.info(
                    f"Found new checkpoint (globalstep{eval_step.global_step}) "
                    f"at {ckpt_dir}"
                )

        # Submit pending evaluation step
        if self.__running_jobs < self.__max_concurrent_jobs:
            # Submit in global_step order
            pending_steps = list(
                filter(
                    lambda x: self.__eval_steps[x].status
                    == EvaluationStepStatus.PENDING,
                    self.__eval_steps.keys(),
                )
            )
            if pending_steps:
                min_pending = min(pending_steps)
                self.__eval_steps[min_pending].submit(self.__config)

        # Check if any eval job is done or failed
        running_steps = filter(
            lambda x: self.__eval_steps[x].status == EvaluationStepStatus.RUNNING,
            self.__eval_steps.keys(),
        )
        for global_step in running_steps:
            self.__eval_steps[global_step].check()

        # Check whether the **minimal global step**, not logged or failed, is done,
        # and log this step to wandb if done.
        # NOTE: LOGGED and FAILED steps have identical behaviors now.
        # But in future versions that supports multi-node eval they could be different.
        log_steps = list(
            filter(
                lambda x: self.__eval_steps[x].status
                not in [
                    EvaluationStepStatus.LOGGED,
                    EvaluationStepStatus.FAILED,
                ],
                self.__eval_steps.keys(),
            )
        )
        if log_steps:
            log_step = min(log_steps)
            if self.__eval_steps[log_step].status == EvaluationStepStatus.DONE:
                if not self.__wandb_initialized:
                    self.__lazy_wandb_init()
                    self.__wandb_initialized = True
                if not self.__swanlab_initialized and swanlab is not None:
                    self.__lazy_swanlab_init()
                    self.__swanlab_initialized = True
                self.__eval_steps[log_step].log(self.__config)

    @property
    def __running_jobs(self):
        return len(
            list(
                filter(
                    lambda x: self.__eval_steps[x].status
                    == EvaluationStepStatus.RUNNING,
                    self.__eval_steps.keys(),
                )
            )
        )
