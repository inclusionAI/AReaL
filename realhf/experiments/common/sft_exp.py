# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses

from realhf.api.cli_args import SFTExperimentOptions
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class SFTConfig(CommonExperimentConfig, SFTExperimentOptions):

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="trainDefault",
            mb_spec=self.allocation.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=ModelInterfaceAbstraction("sft"),
            model_name="default",
            input_keys=("packed_input_ids", "prompt_mask"),
            log_return_value=True,
        )
        return {"trainDefault": rpc}

    @property
    def allocations(self):
        return {"trainDefault": self.allocation}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                    fill_to_max_length=self.dataset.fill_to_max_length,
                ),
            )
        ]

    @property
    def eval_dataset(self):
        return DatasetAbstraction(
            "prompt_answer",
            args=dict(
                max_length=self.dataset.max_seqlen,
                dataset_path=self.dataset.valid_path,
            ),
        )

    @property
    def eval_bs(self) -> int:
        return self.dataset.valid_bs_n_seqs

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("sft", SFTConfig)
