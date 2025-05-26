# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import copy
import dataclasses
import itertools
import json
import os
from typing import *

from omegaconf import OmegaConf

from realhf.api.cli_args import (
    MFCConfig,
    ModelTrainEvalConfig,
    ParallelismConfig,
    PromptOnlyDatasetConfig,
)
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.system_api import ExperimentConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.base import constants, logging
from realhf.base.topology import decompose_to_three_factors
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("Profiling Experiment", "system")


def default_parallel_config(n_gpus: int) -> List[Dict[str, Any]]:
    factors = decompose_to_three_factors(n_gpus)
    x = [
        {
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "use_sequence_parallel": tp > 1,
        }
        for dp, tp, pp in factors
    ]
    x += [
        {
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "use_sequence_parallel": False,
        }
        for dp, tp, pp in factors
        if tp > 1
    ]
    return x


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d  # Not a dataclass field


@dataclasses.dataclass
class ProfileConfig(CommonExperimentConfig):
    """The experiment configuration for profiling layers and interfaces.

    The `initial_setup` method in this experiment will return a list of
    experiment configurations, which will be run sequentially.
    All configurations share the same experiment name, trial name,
    and the scheduling configuration. They can have different models,
    datasets, or parallel strategies, as long as they always occupy
    a fixed number of GPUs.

    It's important to note that, if any error occurs during the execution,
    the experiment will terminate immediately. In particular, the OOM error
    should not appear because the profiling setup usually uses a small model.
    """

    interfaces_jsonl: str = ""
    allocations_jsonl: Optional[str] = None
    handle_names: Optional[List[str]] = None
    n_mbs: Optional[List[int]] = None
    batch_sizes: Optional[List[int]] = None
    models_jsonl: str = ""
    datasets_jsonl: str = ""

    def __post_init__(self):
        # Check that handle_name belones to ["train_step", "generate", "inference"]
        self.handle_names = list(set(self.handle_names))
        if any(
            k not in ["train_step", "generate", "inference"] for k in self.handle_names
        ):
            raise NotImplementedError(f"Unknown handle_name: {self.handle_name}")

        # Check the configuration of interfaces
        if not os.path.exists(self.interfaces_jsonl):
            raise FileNotFoundError(
                f"File not found: {self.interfaces_jsonl}. "
                "It should be a JSONL file specifying the arguments "
                "for the interface implementation."
            )
        with open(self.interfaces_jsonl, "r") as f:
            self.interface_kwargs = [json.loads(l) for l in f.readlines()]

        # Check the configuration of parallel strategies.
        if self.allocations_jsonl is None:
            self.parallel_kwargs = default_parallel_config(
                self.n_nodes * self.n_gpus_per_node
            )
        else:
            assert self.allocations_jsonl.endswith(".jsonl")
            assert os.path.exists(self.allocations_jsonl)
            with open(self.allocations_jsonl, "r") as f:
                self.parallel_kwargs = [json.loads(l) for l in f.readlines()]
        for pcfg in self.parallel_kwargs:
            assert isinstance(pcfg, dict), type(pcfg)
            assert all(
                k
                in [
                    "data_parallel_size",
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "use_sequence_parallel",
                ]
                for k in pcfg.keys()
            ), pcfg.keys()
            assert (self.n_nodes * self.n_gpus_per_node) == (
                pcfg.get("data_parallel_size", 1)
                * pcfg.get("tensor_parallel_size", 1)
                * pcfg.get("pipeline_parallel_size", 1)
            )

        if self.n_mbs is None:
            self.n_mbs = [1]
        else:
            self.n_mbs = OmegaConf.to_container(self.n_mbs)
            assert isinstance(self.n_mbs, list), type(self.n_mbs)
            assert all(isinstance(x, int) for x in self.n_mbs)

        assert self.batch_sizes is not None

        assert os.path.exists(self.models_jsonl)
        with open(self.models_jsonl, "r") as f:
            self.model_kwargs = [json.loads(l) for l in f.readlines()]

        assert os.path.exists(self.datasets_jsonl)
        with open(self.datasets_jsonl, "r") as f:
            self.dataset_kwargs = [json.loads(l) for l in f.readlines()]
            assert all(x["type_"] == "prompt" for x in self.dataset_kwargs)

    @property
    def allocations(self):
        return dict(default=self._tmp_allocation)

    @property
    def models(self):
        return dict(default=self._tmp_model)

    @property
    def tokenizer_name_or_path(self):
        return self._tmp_model.path

    @property
    def max_prompt_len(self):
        return self._tmp_dataset.args["max_length"]

    @property
    def datasets(self):
        return [self._tmp_dataset]

    @property
    def rpcs(self):
        return dict(default=self._tmp_rpc)

    def initial_setup(self) -> List[ExperimentConfig]:
        self.allocation_mode = "manual"
        setups = []
        setup_log_path = os.path.join(
            constants.LOG_ROOT,
            self.experiment_name,
            self.trial_name,
            "setups.jsonl",
        )
        logger.info(
            f"Experiment setup configurations of the profiling experiment "
            f"will be saved to: {setup_log_path}"
        )
        with open(setup_log_path, "w") as f:
            # batch size in the most outer loop to delay the possible OOM error
            for (
                bs,
                pcfg,
                n_mbs,
                model_cfg,
                dataset_cfg,
                handle_name,
                interface_cfg,
            ) in itertools.product(
                self.batch_sizes,
                self.parallel_kwargs,
                self.n_mbs,
                self.model_kwargs,
                self.dataset_kwargs,
                self.handle_names,
                self.interface_kwargs,
            ):
                if handle_name == "generate" and pcfg["use_sequence_parallel"]:
                    continue

                kwargs_stat = dict(
                    parallel=pcfg,
                    n_mbs=n_mbs,
                    model=model_cfg,
                    dataset=dataset_cfg,
                    interface=interface_cfg,
                    bs=bs,
                )
                f.write(json.dumps(kwargs_stat) + "\n")

                # Create tmp object for constructing experiment setups
                self._tmp_allocation = MFCConfig(
                    parallel=ParallelismConfig(**pcfg), n_mbs=n_mbs
                )
                self._tmp_model = dataclass_from_dict(ModelTrainEvalConfig, model_cfg)
                self._tmp_dataset = DatasetAbstraction(**dataset_cfg)
                if handle_name == "train_step":
                    interface_type = ModelInterfaceType.TRAIN_STEP
                elif handle_name == "inference":
                    interface_type = ModelInterfaceType.INFERENCE
                elif handle_name == "generate":
                    interface_type = ModelInterfaceType.GENERATE
                else:
                    raise NotImplementedError(
                        f"Unknown which handle to run in the interface: {self.handle_name}"
                    )
                self._tmp_rpc = MFCDef(
                    n_seqs=bs,
                    name="default",
                    n_mbs=n_mbs,
                    interface_type=interface_type,
                    interface_impl=ModelInterfaceAbstraction(**interface_cfg),
                    model_name="default",
                    input_keys=["packed_prompts"],
                    log_return_value=False,
                    balanced_dp=True,
                )

                setup = copy.deepcopy(super().initial_setup())
                for m in setup.model_worker:
                    m.profile_mode = True
                setups.append(setup)
        return setups


register_quickstart_exp("profile", ProfileConfig)
