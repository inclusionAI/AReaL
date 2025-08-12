import copy
import dataclasses
from typing import Any, Dict, List, Tuple, Optional

import realhf.base.logging as logging
from realhf.api.cli_args import ModelTrainEvalConfig
from realhf.api.core.config import (
    AgentAbstraction,
    DatasetAbstraction,
    EnvServiceAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.experiments.async_exp.async_rl_exp import AsyncRLExperimentConfig
from realhf.experiments.common.ppo_math_exp import PPOMATHConfig
from realhf.experiments.common.utils import asdict

logger = logging.getLogger("Async Werewolf Exp", "colored")


@dataclasses.dataclass
class AsyncWerewolfConfig(AsyncRLExperimentConfig, PPOMATHConfig):
    """Async PPO training configuration for the Werewolf agent."""

    repeat_rules: bool = True
    role: str = "villager"
    opponent: Optional[str] = None

    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction(
            "werewolf_agent",
            args=dict(
                gconfig=self.generation_config,
                tokenizer_path=self.actor.path,
                reward_scaling=self.ppo.reward_output_scaling,
                reward_bias=self.ppo.reward_output_bias,
                role=self.role,
                opponent_path=self.opponent,
            ),
        )

    @property
    def env(self) -> EnvServiceAbstraction:
        return EnvServiceAbstraction(
            "werewolf_env", args=dict(repeat_rules=self.repeat_rules)
        )

    @property
    def gen_backend_args(self) -> Any:
        return self.actor.sglang

    @property
    def generation_config(self) -> GenerationHyperparameters:
        return GenerationHyperparameters(**asdict(self.ppo.gen)).new(n=self.group_size)

    @property
    def rpcs(self):
        if (
            (self._allocation_mode.is_decoupled_vllm() or self.actor.vllm.hybrid_train)
            and self.dataset.max_prompt_len + self.ppo.gen.max_new_tokens
            > self.actor.vllm.max_seq_len_to_capture
            and not self.actor.vllm.enforce_eager
        ):
            raise RuntimeError(
                f"vllm max seq len to capture {self.actor.vllm.max_seq_len_to_capture} is "
                f"smaller than the prompt length + generation length "
                f"{self.dataset.max_prompt_len + self.ppo.gen.max_new_tokens}"
            )

        actor_interface = ModelInterfaceAbstraction(
            "ppo_actor",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "generation_config": asdict(self.ppo.gen),
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "adv_norm": self.ppo.adv_norm,
                "group_size": self.group_size,
                "generation_size": self.generation_size,
                "group_adv_norm": self.group_adv_norm,
                "mask_too_long": self.mask_too_long,
                "sample_reuse": self.ppo.actor_sample_reuse,
                "c_clip": self.ppo.c_clip,
                "behav_imp_weight_cap": self.ppo.behav_imp_weight_cap,
                "logging_keys": (
                    "traj_steps",
                    "traj_len", 
                    "vill_rewards", 
                    "were_rewards",
                    "werewolf_kills",
                    "villager_correct_votes",
                    "villager_wrong_votes",
                    "witch_heals",
                    "witch_poisons",
                    "hunter_shots",
                    ),
            },
        )

        critic_interface = ModelInterfaceAbstraction(
            "ppo_critic",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "group_size": self.group_size,
                "mask_too_long": self.mask_too_long,
                "sample_reuse": self.ppo.critic_sample_reuse,
            },
        )
        critic_interface.args.pop("eps_clip")
        rw_interface = ModelInterfaceAbstraction(
            "rw-math-code",
            args=dict(
                dataset_path=self.dataset.path,
                tokenizer_path=self.actor.path,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
                rw_type=self.rw_type,
                check_xml_format=self.check_xml_format,
                group_size=self.group_size,
                check_verifier_status=self.check_verifier_status,
            ),
        )

        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False
        if self.ppo.fuse_rew_ref:
            ref_interface = ModelInterfaceAbstraction(
                "fused-threading",
                args=dict(interfaces=dict(rew=rw_interface, ref=ref_interface)),
            )

        rollout_output_keys = [
            "seq_no_eos_mask",
            "packed_input_ids",
            "packed_logprobs",
            "prompt_mask",
        ]
        if self.ppo.recompute_logprob and not self.ppo.use_decoupled_loss:
            rollout_output_keys.remove("packed_logprobs")
        rollout = MFCDef(
            name="actor_gen",
            model_name="actor",
            mb_spec=self.actor_gen.mb_spec,
            interface_type=ModelInterfaceType.GENERATE,
            interface_impl=actor_interface,
            input_keys=("packed_prompts", "task_ids"),
            output_keys=tuple(rollout_output_keys),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        actor_inf_outputs = ("packed_logprobs",)
        if self.ppo.use_decoupled_loss:
            actor_inf_outputs = ("proximal_logprobs",)
        actor_inf = MFCDef(
            name="actor_inf",
            model_name="actor",
            mb_spec=self.actor_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=actor_interface,
            input_keys=("packed_input_ids",),
            output_keys=actor_inf_outputs,
            output_key_remap=dict(logprobs=actor_inf_outputs[0]),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name="rew_inf",
            model_name="reward",
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=("packed_input_ids", "packed_prompts", "task_ids"),
            output_keys=("rewards",),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = ["packed_input_ids"]
        inf_ref_outputs = ["packed_ref_logprobs"]
        if self.ppo.fuse_rew_ref:
            inf_ref_inputs += ["packed_prompts", "task_ids"]
            inf_ref_outputs += ["rewards"]

        inf_ref_logits = MFCDef(
            name="ref_inf",
            model_name="ref",
            mb_spec=self.ref_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=tuple(inf_ref_inputs),
            output_keys=tuple(inf_ref_outputs),
            output_key_remap=dict(logprobs="packed_ref_logprobs"),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_values = MFCDef(
            name="critic_inf",
            model_name="critic",
            mb_spec=self.critic_inf.mb_spec,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            min_n_seqs_per_pass=1 / self.group_size,
            input_keys=("packed_input_ids", "seq_no_eos_mask"),
            output_keys=("values",),
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_actor_inputs = [
            "packed_input_ids",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "task_ids",
            "values",
            "prompt_mask",
            "seq_no_eos_mask",
        ]
        if self.ppo.disable_value:
            train_actor_inputs.remove("values")
        if self.ppo.kl_ctl == 0:
            train_actor_inputs.remove("packed_ref_logprobs")
        if self.ppo.use_decoupled_loss:
            train_actor_inputs.append("proximal_logprobs")
        train_actor = MFCDef(
            name="actor_train",
            model_name="actor",
            mb_spec=self.actor_train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=actor_interface,
            input_keys=tuple(train_actor_inputs),
            log_return_value=True,
            min_n_seqs_per_pass=self.ppo.ppo_n_minibatches / self.group_size,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_critic_inputs = [
            "packed_input_ids",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "values",
            "prompt_mask",
            "seq_no_eos_mask",
        ]
        if self.ppo.kl_ctl == 0:
            train_critic_inputs.remove("packed_ref_logprobs")
        train_critic = MFCDef(
            name="critic_train",
            model_name="critic",
            mb_spec=self.critic_train.mb_spec,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            input_keys=tuple(train_critic_inputs),
            log_return_value=True,
            min_n_seqs_per_pass=self.ppo.ppo_n_minibatches / self.group_size,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        rpcs = {
            "actor_gen": rollout,
            "actor_train": train_actor,
            "critic_inf": inf_values,
            "critic_train": train_critic,
            "ref_inf": inf_ref_logits,
            "actor_inf": actor_inf,
            "rew_inf": inf_reward,
        }
        if self.ppo.disable_value:
            rpcs.pop("critic_inf")
            rpcs.pop("critic_train")
        if not self.ppo.recompute_logprob and not self.ppo.use_decoupled_loss:
            rpcs.pop("actor_inf")
        if self.ppo.kl_ctl == 0:
            rpcs.pop("ref_inf")
        if self.ppo.fuse_rew_ref and self.ppo.kl_ctl != 0:
            rpcs.pop("rew_inf")

        rpcs["actor_gen"].output_keys = (
            *rpcs["actor_gen"].output_keys,
            "packed_prompts",
            "version_start",
            "version_end",
            "rewards",
            "birth_time",
            "logging",
        )
        rpcs["actor_train"].input_keys = (
            *rpcs["actor_train"].input_keys,
            "version_start",
            "version_end",
            "logging",
        )
        if "ref_inf" in rpcs:
            actor_interface = rpcs["actor_train"].interface_impl
            rpcs["ref_inf"].interface_impl = copy.deepcopy(actor_interface)
            rpcs["ref_inf"].interface_impl.args["enable_save"] = False
            rpcs["ref_inf"].input_keys = ("packed_input_ids",)
            rpcs["ref_inf"].output_keys = ("packed_ref_logprobs",)
        if "rew_inf" in rpcs:
            rpcs.pop("rew_inf")
        if self.no_training:
            rpcs["actor_train"].interface_impl = ModelInterfaceAbstraction("null")
            rpcs["actor_gen"].interface_impl = ModelInterfaceAbstraction("null")
            if "actor_inf" in rpcs:
                rpcs["actor_inf"].interface_impl = ModelInterfaceAbstraction("null")
        return rpcs

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        reward = copy.deepcopy(self.actor)
        models = {
            "actor": self.actor,
            "critic": self.critic,
            "ref": self.ref,
            "reward": reward,
        }
        if self.ppo.disable_value:
            models.pop("critic")
        if self.ppo.kl_ctl == 0:
            models.pop("ref")
        if self.ppo.fuse_rew_ref and self.ppo.kl_ctl != 0:
            models.pop("reward")
        if "reward" in models:
            models.pop("reward")
        return models

    @property
    def allocations(self):
        allocs = {
            "actor_gen": self.actor_gen,
            "actor_train": self.actor_train,
            "critic_inf": self.critic_inf,
            "critic_train": self.critic_train,
            "ref_inf": self.ref_inf,
            "actor_inf": self.actor_inf,
            "rew_inf": self.rew_inf,
        }
        if self.ppo.disable_value:
            allocs.pop("critic_inf")
            allocs.pop("critic_train")
        if not self.ppo.recompute_logprob and not self.ppo.use_decoupled_loss:
            allocs.pop("actor_inf")
        if self.ppo.kl_ctl == 0:
            allocs.pop("ref_inf")
        if self.ppo.fuse_rew_ref and self.ppo.kl_ctl != 0:
            allocs.pop("rew_inf")
        if "rew_inf" in allocs:
            allocs.pop("rew_inf")
        return allocs

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "werewolf_dataset",
                args=dict(
                    dataset_path=self.dataset.path,
                    max_length=self.dataset.max_prompt_len,
                ),
            )
        ]


register_quickstart_exp("async-werewolf", AsyncWerewolfConfig)