# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import dataclasses
import os
from typing import Dict

from realhf.api.cli_args import MathCodeEvalOptions, ModelTrainEvalConfig
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.base import constants, logging
from realhf.experiments.common.common import CommonExperimentConfig
from realhf.experiments.common.utils import asdict

logger = logging.getLogger("Math Cdoe Eval exp", "colored")


@dataclasses.dataclass
class MathCodeEvalConfig(MathCodeEvalOptions, CommonExperimentConfig):

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        return {
            "actor": self.actor,
            "reward": self.rew,
        }

    @property
    def rpcs(self):
        if (
            self.dataset.max_prompt_len + self.gen_config.max_new_tokens
            > self.actor.vllm.max_seq_len_to_capture
        ):
            raise RuntimeError(
                f"vllm max seq len to capture {self.actor.vllm.max_seq_len_to_capture} is "
                f"smaller than the prompt length + generation length {self.dataset.max_prompt_len + self.gen_config.max_new_tokens}"
            )

        # interfaces
        actor_interface = ModelInterfaceAbstraction(
            "ppo_actor",
            args={
                "generation_config": asdict(self.gen_config),
                "group_size": self.group_size,
            },
        )

        rw_interface = ModelInterfaceAbstraction(
            "rw-math-code",
            args=dict(
                dataset_path=self.dataset.path,
                tokenizer_path=self.actor.path,
                rw_type=self.rw_type,
                answer_save_path=os.path.join(
                    constants.get_log_path(self), "generated"
                ),
                check_xml_format=self.check_xml_format,
                group_size=self.group_size,
                check_verifier_status=self.check_verifier_status,
            ),
        )
        rollout = MFCDef(
            name="actor_gen",
            model_name="actor",
            mb_spec=self.actor_gen.mb_spec,
            interface_type=ModelInterfaceType.GENERATE,
            interface_impl=actor_interface,
            input_keys=("packed_prompts", "task_ids"),
            output_keys=("packed_input_ids",),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name="rew_inf",
            model_name="reward",
            mb_spec=self.rew_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=("packed_input_ids", "packed_prompts", "task_ids"),
            output_keys=("rewards",),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        return {
            "actor_gen": rollout,
            "rew_inf": inf_reward,
        }

    @property
    def allocations(self):
        return {
            "actor_gen": self.actor_gen,
            "rew_inf": self.rew_inf,
        }

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "math_code_prompt",
                args=dict(
                    dataset_path=self.dataset.path,
                    max_length=self.dataset.max_prompt_len,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self) -> str:
        return self.actor.path

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len


register_quickstart_exp("math-code-eval", MathCodeEvalConfig)
