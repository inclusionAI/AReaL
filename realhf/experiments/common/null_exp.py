# Copyright 2025 Ant Group Inc.

import dataclasses

from realhf.api.cli_args import (
    MFCConfig,
    ModelTrainEvalConfig,
    NullPPOExperimentOptions,
    PromptOnlyDatasetConfig,
    SFTExperimentOptions,
)
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class NullSFTConfig(CommonExperimentConfig, SFTExperimentOptions):

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
            interface_impl=ModelInterfaceAbstraction("null"),
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
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("null-sft", NullSFTConfig)


@dataclasses.dataclass
class NullPPOConfig(CommonExperimentConfig, NullPPOExperimentOptions):

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        rw = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="reward",
            mb_spec=self.inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ModelInterfaceAbstraction("null"),
            model_name="default",
            input_keys=("packed_prompts",),
            output_keys=("rewards",),
        )
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="trainDefault",
            mb_spec=self.train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=ModelInterfaceAbstraction("null"),
            model_name="default",
            input_keys=("packed_prompts", "rewards"),
            log_return_value=True,
        )
        return {"trainDefault": rpc, "reward": rw}

    @property
    def allocations(self):
        return {"trainDefault": self.train, "reward": self.inf}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "math_code_prompt",
                args=dict(
                    max_length=self.dataset.max_prompt_len,
                    dataset_path=self.dataset.path,
                    filter_threshold=self.dataset_filter_threshold,
                    max_filter_percentage=self.dataset_max_filter_percentage,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("null-ppo", NullPPOConfig)
