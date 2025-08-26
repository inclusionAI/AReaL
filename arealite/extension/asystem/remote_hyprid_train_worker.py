import gzip
import os
from typing import Any

import requests
import cloudpickle

from arealite.api.cli_args import RemoteMegatronEngineConfig
from arealite.api.engine_api import (
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    Scheduling,
)

import dataclasses
import json
from typing import Dict, List, Callable

import torch

from arealite.dataset.distributed_batch_memory import DistributedBatchMemory

import realhf.impl.model.utils.ppo_functional as ppo_functional
from realhf.api.core.data_api import (
    RL_TASKS,
    MicroBatchSpec,
    SequenceSample,
    SequenceSplitSpec,
)
from realhf.base import logging, stats_tracker
from realhf.base.datapack import flat2d
from realhf.impl.model.utils.functional import (
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("RemoteHypridTrainWorker")


@dataclasses.dataclass
class RemoteMegatronInitConfig:
    server_addrs: list[str]
    global_rank: int
    world_size: int
    recover_dir: str = ""
    enable_colocate_mode: bool = False


class RemoteHypridTrainWorker(TrainEngine):
    def __init__(self, config: RemoteMegatronEngineConfig):
        self.config = config

        self.megatron_addr = None
        self.global_step = config.global_step
        self.global_rank = 0

        # initialization
        self.initialized = False
        self.weight_update_group_initialized = False

        if self.config.wrap_policy.adaptive_kl_ctl:
            assert self.config.wrap_policy.adaptive_kl_target is not None
            assert self.config.wrap_policy.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.config.wrap_policy.kl_ctl, self.config.wrap_policy.adaptive_kl_target,
                self.config.wrap_policy.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.config.wrap_policy.kl_ctl)
        if self.config.wrap_policy.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )
            if self.config.wrap_policy.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.config.wrap_policy.value_norm_beta, epsilon=self.config.wrap_policy.value_norm_eps
                )
            elif self.config.wrap_policy.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.config.wrap_policy.value_norm_type}")

        self.enable_colocate_mode = None
        self.kl_ctl = self.config.wrap_policy.kl_ctl

    def initialize(self, cfg: RemoteMegatronInitConfig):
        global_rank = cfg.global_rank
        self.global_rank = cfg.global_rank
        local_rank = global_rank % 8
        self.enable_colocate_mode = cfg.enable_colocate_mode

        print(f"[megatron] dzq_debug global_rank: {global_rank}, serveraddr len:{len(cfg.server_addrs)}")
        self.megatron_addr = cfg.server_addrs[global_rank]
        master_addr = cfg.server_addrs[0]
        master_ip, master_port = master_addr.split(":", 1)  # ip:port
        logger.info(f"[RemoteHypridTrainWorker] exec initialize, init_config: {cfg}")

        megatron_config = self.config.remote_megatron_config
        # megatron_config["total_train_steps"] = cfg.ft_spec.total_train_steps
        payload = {
            "rank": str(cfg.global_rank),
            "local_rank": str(local_rank),
            "master_port": "57937",
            "master_addr": str(master_ip),
            "world_size": str(cfg.world_size),
            "megatron_config": megatron_config,
            "loss_configs": self.config.loss_configs,
            "recover_dir": cfg.recover_dir,
            "enable_colocate_mode": cfg.enable_colocate_mode,
        }

        try:
            target_url = f"http://{self.megatron_addr}/initialize"
            headers = {"Content-Type": "application/json"}
            logger.info(
                f"[RemoteHypridTrainWorker] initialize begin send request to megatron server, "
                f"target_url: {target_url}, rank: {global_rank}, target_url: {target_url}"
            )
            response = requests.post(
                target_url, data=json.dumps(payload), headers=headers, timeout=7200
            )
            logger.info(
                f"[RemoteHypridTrainWorker] initialize finished send request to megatron server"
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteHypridTrainWorker] rank: {global_rank} Payload sent successfully to {target_url}"
                )
            else:
                raise ValueError(
                    f"[Rank {global_rank}] Failed to send payload. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except ValueError as ve:
            raise ValueError(f"[Rank {global_rank}] Error parsing target address: {ve}")
        except requests.exceptions.RequestException as re:
            raise ValueError(f"[Rank {global_rank}] Error sending HTTP request: {re}")
        except Exception as e:
            raise ValueError(f"[Rank {global_rank}] Unexpected error: {e}")

        logger.info(f"[RemoteHypridTrainWorker] rank: {global_rank} megatron server initialize success")
        self.initialized = True

    def get_scheduling_config(self):
        return Scheduling(
            cpu=4,
            gpu=1,
            mem=8,
            type="engine",
        )

    def destroy(self):
        self.initialized = False

    def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            try:
                logger.info(
                    f"[RemoteHypridTrainWorker] upload_weights begin send request to megatron server, "
                    f"target_url: http://{self.megatron_addr}/update_weights")
                target_url = f"http://{self.megatron_addr}/update_weights"
                headers = {"Content-Type": "application/json"}
                response = requests.post(
                        target_url, data=json.dumps({"path": meta.path}), headers=headers, timeout=7200
                )
                if response.status_code == 200:
                    logger.info(
                        f"[RemoteHypridTrainWorker] upload_weights success")
                else:
                    raise ValueError(
                        f"[RemoteHypridTrainWorker] Failed to upload weights. "
                        f"Status code: {response.status_code}, Response: {response.text}"
                    )
            except requests.exceptions.Timeout:
                raise ValueError("[RemoteHypridTrainWorker] Upload weights request timeout!")
            except requests.exceptions.RequestException as e:
                raise ValueError(
                    f"[RemoteHypridTrainWorker] Send upload weights request, an error occurred: {e}")
        elif meta.type == "disk":
            logger.info(
                f"[RemoteHypridTrainWorker] upload_weights save hf model to disk, path: {meta.path}, step: {self.global_step}.")
            save_load_meta = SaveLoadMeta(
                path=meta.path,
                weight_format="huggingface",
                global_step=self.global_step,
                with_optim=True,
                tokenizer=None,
                base_model_path=None,
            )
            self.save(save_load_meta)
            logger.info(
                f"[RemoteHypridTrainWorker] upload_weights success.")
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format not in ["huggingface", "mcore"]:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        try:
            logger.info(
                f"[RemoteHypridTrainWorker] send save request, "
                f"weight_format: {meta.weight_format}, save_dir: {meta.path},  "
                f"global_step: {meta.global_step}")
            payload = {
                "save_dir_prefix": meta.path,
                "save_type": meta.weight_format,
                "global_step": meta.global_step,
            }
            target_url = f"http://{self.megatron_addr}/save"
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                target_url, data=json.dumps(payload), headers=headers, timeout=7200
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteHypridTrainWorker] save hf request exec success, save_dir: {meta.path}"
                )
            else:
                raise ValueError(
                    f"[RemoteHypridTrainWorker] Failed to send save {meta.weight_format} "
                    f"request. status code: {response.status_code}, response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError(f"[RemoteHypridTrainWorker] save {meta.weight_format} request timed out!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"[RemoteHypridTrainWorker] Send save {meta.weight_format} request, an error occurred: {e}")
        return response

    def load(self, meta: SaveLoadMeta):
        pass

    def step_lr_scheduler(self):
        pass

    def init_distributed_weight_update(self, meta: WeightUpdateMeta):
        raise NotImplementedError(
            "Distributed weight update is not implemented for RemoteMegatronEngine yet. "
        )

    def update_weights_from_distributed(self):
        raise NotImplementedError(
            "Distributed weight update is not implemented for RemoteMegatronEngine yet. "
        )

    def train_distributed_batch(
        self,
        input_: DistributedBatchMemory,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor] = lambda x, y: x.sum(),
        loss_weight_fn: Callable[[Dict], float] = lambda x: 1.0,
    ) -> List[Dict[str, float]]:
        # 0.接受rollout和reward之后的数据
        # - input_ids, prompt_mask, logprobs, versions, seqlen, rewards, task_ids, seq_no_eos_mask
        if not input_ or len(input_.dataset) == 0:
            raise ValueError("input_.dataset is empty")
        first_item = input_[0]
        attrs = list(first_item.keys())

        # 1. 获取所有属性名。input_的key：input_ids, prompt_mask, logprobs, versions, seqlen, rewards, task_ids, seq_no_eos_mask
        # 构造 attr -> tensor
        batch_data = {}
        for attr in attrs:
            batch_data[attr] = input_[attr]
        torch.set_printoptions(threshold=float('inf'))
        logger.info(f"[RemoteHypridTrainWorker] train_distributed_batch rewards: {batch_data['rewards']}")
        # 2. input_的数据转换：prompt_mask, packed_input_ids, seqlens.packed_input_ids, rewards, task_ids, seq_no_eos_mask, packed_logprobs
        # input_ids => packed_input_ids
        # seqlens => seqlens.packed_input_ids
        # logprobs => packed_logprobs
        # input_ids => packed_input_ids
        if "input_ids" in batch_data and "seqlen" in batch_data:
            batch_data["packed_input_ids"] = pack_input_ids(batch_data["input_ids"], batch_data["seqlen"])

        # logprobs => packed_logprobs
        if "logprobs" in batch_data and "seqlen" in batch_data:
            batch_data["packed_logprobs"] = pack_logprobs(batch_data["logprobs"], batch_data["seqlen"])

        if "prompt_mask" in batch_data and "seqlen" in batch_data:
            batch_data["prompt_mask"] = pack_prompt_mask(batch_data["prompt_mask"], batch_data["seqlen"])

        if "ref_logprobs" in batch_data and "seqlen" in batch_data:
            batch_data["packed_ref_logprobs"] = pack_ref_logprobs(batch_data["ref_logprobs"], batch_data["seqlen"])

        # 3.获取{advantages, old_logp, ppo_loss_mask, packed_input_ids, kl_rewards, global_stats}
        train_datas = self.process_training_data(batch_data)
        batch = {"advantages": train_datas["advantages"],
                 "old_logp": train_datas["old_logp"],
                 "rollout_logp": train_datas["rollout_logp"],
                 "ppo_loss_mask": train_datas["ppo_loss_mask"],
                 "packed_input_ids": train_datas["packed_input_ids"],
                 "kl_rewards": train_datas["kl_rewards"]} # batch_data["seqlen"]
        
        if "ref_logprobs" in train_datas:
            batch["ref_logprobs"] = train_datas["ref_logprobs"]

        batch_size = int(input_["seqlen"].shape[0])
        flat_input = SequenceSample.from_default(
            ids=list(range(batch_size)),
            data=batch,
            seqlens=[int(x) for x in input_["seqlen"].cpu().numpy().tolist()],
        )

        flat_input = SequenceSample.shuffled(flat_input)
        bs = flat_input.bs
        n_minibatches = self.config.wrap_policy.n_minibatches
        sizes = [0 for _ in range(n_minibatches)]
        for idx in range(bs):
            sizes[idx % n_minibatches] += 1
        spec = SequenceSplitSpec(sizes=sizes)
        datas = flat_input.split_with_spec(spec)
        all_stats = []
        
        scalar_metrics = {}
        if "global_stats" in train_datas:
            for key, value in train_datas["global_stats"].items():
                if isinstance(value, (int, float)):
                    scalar_metrics[key] = value
        
        for mb_i, data in enumerate(datas):
            train_stats = self.train_batch_sequencesample(data, loss_fn, loss_weight_fn)

            indices = torch.where(train_datas["ppo_loss_mask"] == 1)[0]
            adv = train_datas["advantages"]
            train_stats["advantages"] = adv[indices].mean()
            total_seqlen = input_["seqlen"].sum()
            train_stats[f"rank{self.global_rank}_total_seqlen"] = total_seqlen

            loss = train_stats.get("loss")
            if loss is not None:
                train_stats[f"rank{self.global_rank}_loss"] = loss
            
            for key, value in scalar_metrics.items():
                if key not in train_stats:
                    train_stats[key] = value
                else:
                    logger.warning(f"[RemoteHypridTrainWorker] Duplicate metric key '{key}' found. Keeping existing value: {train_stats[key]}, ignoring global_stats value: {value}")
            
            all_stats.append(train_stats)

        logger.info(f"[RemoteHypridTrainWorker] Train {n_minibatches} minibatches exec success, global_step: {self.global_step}.")
        self.global_step += 1
        return all_stats

    def notify_event(self, event: str, global_step: int) -> None:
        """Handle training start/end events by sending HTTP notification.
        
        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        assert self.enable_colocate_mode is not None, "enable_colocate_mode is not set"
        if event not in ["train_start", "train_end"]:
            raise ValueError(f"Invalid event type: {event}")
            
        logger.info(f"[RemoteHypridTrainWorker] Sending training {event} notification at global_step: {global_step}")
        
        try:
            target_url = f"http://{self.megatron_addr}/events"
            headers = {"Content-Type": "application/json"}
            payload = {
                "event": event,
                "global_step": global_step
            }
            response = requests.post(
                target_url, 
                data=json.dumps(payload), 
                headers=headers,
                timeout=60
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to send training event. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except Exception as e:
            raise ValueError(f"Error sending notify training event: {e}")
        
        return None

    def train_batch_sequencesample(
        self,
        input_: SequenceSample,  # key: str, value: tensor
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        mb_spec = MicroBatchSpec(n_mbs=self.config.n_mbs,
                                 max_tokens_per_mb=self.config.max_tokens_per_mb)
        try:
            target_url = f"http://{self.megatron_addr}/train_batch"
            headers = {"Content-Type": "application/octet-stream"}
            payload = {
                "sequence_sample": input_,
                "micro_batch_spec": mb_spec,
                "global_step": self.global_step,
            }
            data = serialize_and_compress(payload)
            logger.info("[RemoteHypridTrainWorker] send train_batch request to megatron worker....")
            response = requests.post(
                target_url, data=data, headers=headers, timeout=7200
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteHypridTrainWorker] Train batch exec success, response status code: {response.status_code}, response: {response.json()}"
                )
            else:
                raise ValueError(
                    f"[RemoteHypridTrainWorker] Failed to exec train_batch. Status code: {response.status_code}, Response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError("[RemoteHypridTrainWorker] Train request timeout!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                "[RemoteHypridTrainWorker] Send train request, an error occurred:", e
            )

        return response.json()['result']

    def train_batch(
        self,
        input_: Dict,  # key: str, value: tensor
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        # 输入： advantages, old_logp, ppo_loss_mask, packed_input_ids, kl_rewards
        # Dict[str, tensor] to SequenceSample
        logger.info("[RemoteHypridTrainWorker] begin exec train batch...")
        batch_size = int(input_["seqlen"].shape[0])
        flat_input = SequenceSample.from_default(
            ids=list(range(batch_size)),
            data={k: v for k, v in input_.items() if k != "seqlen"},
            seqlens=[int(x) for x in input_["seqlen"].cpu().numpy().tolist()],
        )

        mb_spec = MicroBatchSpec(n_mbs=self.config.n_mbs,
                                 max_tokens_per_mb=self.config.max_tokens_per_mb)
        try:
            target_url = f"http://{self.megatron_addr}/train_batch"
            headers = {"Content-Type": "application/octet-stream"}
            payload = {
                "sequence_sample": flat_input,
                "micro_batch_spec": mb_spec,
                "global_step": self.global_step
            }
            data = serialize_and_compress(payload)
            logger.info("[RemoteHypridTrainWorker] send train_batch request to megatron worker....")
            response = requests.post(
                target_url, data=data, headers=headers, timeout=7200
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteHypridTrainWorker] Train batch exec success, response status code: {response.status_code}, response: {response.json()}"
                )
            else:
                raise ValueError(
                    f"[RemoteHypridTrainWorker] Failed to exec train_batch. Status code: {response.status_code}, Response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError("[RemoteHypridTrainWorker] Train request timeout!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                "[RemoteHypridTrainWorker] Send train request, an error occurred:", e
            )

        return response.json()['result']

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "Eval batch is not implemented for RemoteMegatronEngine yet. "
        )

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        raise NotImplementedError(
            "Forward is not implemented for RemoteMegatronEngine yet. "
        )

    def process_training_data(
        self,
        input_: Dict[str, torch.Tensor],
    ) -> Dict:

        '''
        inputs:
            - prompt_mask, packed_input_ids, seqlens.packed_input_ids, rewards, task_ids, seq_no_eos_mask, packed_logprobs
        outputs:
            - {advantages, old_logp, ppo_loss_mask, packed_input_ids, kl_rewards, global_stats}
        '''
        prompt_mask = input_["prompt_mask"]
        input_lens = torch.tensor(
            input_["seqlen"], device="cpu"
        )
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        prompt_lens = []
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            prompt_lens.append(prompt_mask[s:e].sum())
        prompt_lens = torch.tensor(prompt_lens, device="cpu")
        reward_score = input_["rewards"].float()
        task_mask = input_["task_ids"] != RL_TASKS.index("general")
        reward_score[torch.logical_and(reward_score == 0, task_mask)] = -1
        reward_score = torch.where(
            task_mask,
            (reward_score - self.config.wrap_policy.reward_output_bias) * self.config.wrap_policy.reward_output_scaling,
            reward_score
        )

        logger.info(f"[RemoteHypridTrainWorker] process_training_data reward_score: {reward_score}")
        task_ids = input_["task_ids"]
        # task_ids = task_ids.repeat(self.config.group_size, 1).transpose(0, 1).reshape(-1)

        if "dense_rewards" in input_.keys():
            dense_reward_score = input_["dense_rewards"].float()
        if not self.config.wrap_policy.disable_value:
            values = input_["values"].float()
        else:
            values = torch.zeros_like(
                input_["packed_input_ids"], dtype=torch.float32
            )
        seq_no_eos_mask = input_["seq_no_eos_mask"]
        if self.kl_adapter.value == 0:
            ref_logp: torch.FloatTensor = reward_score.new_zeros(
                int(input_lens.sum()) - len(input_lens)
            )
        else:
            ref_logp: torch.FloatTensor = input_["packed_ref_logprobs"].float()

        old_logp: torch.FloatTensor = input_["packed_logprobs"].float()
        rollout_logp: torch.FloatTensor = input_["packed_logprobs"].float()
        if self.config.wrap_policy.recompute_logp:
            logger.info("[RemoteHypridTrainWorker] enable recompute_logrobs")
            compute_logp_input = {"packed_input_ids": input_["packed_input_ids"],
                                  "seqlen": input_["seqlen"],
                                  "packed_logprobs": input_["packed_logprobs"]}
            logprobs = self.compute_logprobs(compute_logp_input)
            packed_logp = pack_ref_logprobs(logprobs, input_["seqlen"])
            old_logp = packed_logp.float()

        if not self.config.wrap_policy.disable_value:
            if self.config.wrap_policy.value_norm:
                denormalized_values = self.rms.denormalize(values)
            else:
                denormalized_values = values
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                # denormalized_values shape: torch.Size([8, 207]), cu_seqlens shape: torch.Size([9])
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()

        if self.config.wrap_policy.mask_too_long:
            for i in range(seq_no_eos_mask.shape[0]):
                if seq_no_eos_mask[i]:
                    loss_mask[cu_seqlens[i]: cu_seqlens[i + 1]] = False

        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask
        rollout_logp *= loss_mask

        new_reward_score = reward_score

        if not self.config.wrap_policy.adv_norm and self.config.wrap_policy.group_adv_norm:
            n_seqs = len(input_lens)
            reward_score_grpo = reward_score.clone().detach()
            new_reward_score = reward_score_grpo
            for i in range(n_seqs // self.config.group_size):
                group_rewards = reward_score[i * self.config.group_size: (i + 1) * self.config.group_size]
                grouped_std = group_rewards.std(dim=-1)
                normed_rewards = (group_rewards - group_rewards.mean(-1, keepdim=True)) / (grouped_std + 1e-9)
                reward_score_grpo[i * self.config.group_size: (i + 1) * self.config.group_size] = normed_rewards

            logger.info(f"[RemoteHypridTrainWorker] process_training_data new_reward_score: {new_reward_score}")

        # Compute rewards and GAEs.
        use_kl_in_loss = self.config.loss_configs.get('use_kl_in_loss', False)
        kl_ctl_value = 0.0 if use_kl_in_loss else self.kl_adapter.value
        logger.info(
            # f"[RemoteHypridTrainWorker] process_training_data final_token_rewards: {rewards}\n"
            # f"kl_reward: {kl_rewards}\n"
            f"loss_config: {self.config.loss_configs}\n"
            f"kl: {kl_ctl_value=}\n"
            f"use_kl_in_loss: {use_kl_in_loss=}"
        )
        if self.config.wrap_policy.use_dense_reward:
            kl_rewards, rewards = ppo_functional.get_packed_reward_dense(
                kl_ctl=kl_ctl_value,
                clip_reward_value=self.config.wrap_policy.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                dense_reward_score=dense_reward_score,
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                reward_delta=self.config.wrap_policy.reward_delta,
            )
        else:
            kl_rewards, rewards = ppo_functional.get_packed_rewards(
                kl_ctl=kl_ctl_value,
                clip_reward_value=self.config.wrap_policy.max_reward_clip,
                log_probs=old_logp,
                ref_log_probs=ref_logp,
                reward_score=(new_reward_score),
                short1cu_seqlens=short1cu_seqlens,
                seq_no_eos_mask=seq_no_eos_mask,
                mask_no_eos_with_zero=self.config.wrap_policy.mask_no_eos_with_zero,
            )

        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.config.wrap_policy.discount,
            lam=self.config.wrap_policy.gae_lambda,
            values=(
                denormalized_values
                if not self.config.wrap_policy.disable_value
                else denormalized_values.new_zeros(denormalized_values.shape)
            ),
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.config.wrap_policy.value_norm:
            self.rms.update(returns, mask=loss_mask)
        if self.config.wrap_policy.adv_norm:
            if self.config.wrap_policy.group_adv_norm == False:
                advantages = masked_normalization(advantages, loss_mask)
            else:
                logger.info(f"adv_shape: {advantages.shape}")
                logger.info(f"prompt_mask_shape: {prompt_mask.shape}")
                n_samples = len(cu_seqlens) - 1
                assert n_samples % self.config.group_size == 0
                adv_list = []
                for i in range(0, n_samples, self.config.group_size):
                    for j in range(1, self.config.group_size):
                        assert (
                            prompt_mask[cu_seqlens[i]: cu_seqlens[i + 1]].sum()
                            == prompt_mask[
                               cu_seqlens[i + j]: cu_seqlens[i + j + 1]
                               ].sum()
                        )
                    adv_list.append(
                        masked_normalization(
                            advantages[
                            short1cu_seqlens[i]: short1cu_seqlens[
                                i + self.config.group_size
                                ]
                            ],
                            loss_mask[
                            short1cu_seqlens[i]: short1cu_seqlens[
                                i + self.config.group_size
                                ]
                            ],
                            all_reduce=False,
                        )
                    )

                advantages = torch.cat(adv_list, 0)

        # Prepare data to be splitted into mini-batches.
        flat_data = dict(
            advantages=advantages,
            old_logp=old_logp,
            ppo_loss_mask=loss_mask,
            packed_input_ids=input_["packed_input_ids"],
            kl_rewards=kl_rewards,
        )
        use_prox_logp = "proximal_logprobs" in input_.keys()
        if use_prox_logp:
            flat_data["prox_logp"] = input_["proximal_logprobs"].float()

        if self.config.wrap_policy.use_dense_reward:
            dense_reward_score = dense_reward_score[shift_one_indices]

        ### Logging code starts. ###
        with stats_tracker.scope("grpo_actor"):
            assert (
                task_ids.shape == reward_score.shape
            ), f"task_ids ({task_ids.shape}) and reward_score ({reward_score.shape}) must have the same shape"

            task_denominators = {
                f"{task}_n_seqs": (task_ids == idx).bool()
                for idx, task in enumerate(RL_TASKS)
            }

            global_denominators = dict(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                n_tokens=torch.ones_like(prompt_mask, dtype=torch.bool),
                n_valid_tokens=loss_mask.bool(),
                **task_denominators,
            )
            stats_tracker.denominator(**global_denominators)

            for task in RL_TASKS:
                stats_tracker.stat(
                    **{f"{task}_reward": reward_score}, denominator=f"{task}_n_seqs"
                )

            stats = dict(
                advantages=advantages,
                kl_rewards=kl_rewards,
                final_reward=rewards,
            )
            if self.config.wrap_policy.use_dense_reward:
                stats["dense_reward"] = dense_reward_score
            stats_tracker.stat(**stats, denominator="n_valid_tokens")

            seq_stats = dict(
                no_eos_ratios=seq_no_eos_mask.float(),
                task_reward=reward_score,
                prompt_len=prompt_lens.float(),
                seq_len=input_lens.float(),
            )
            if "version_start" in input_.keys():
                seq_stats["head_offpolicyness"] = (
                    self.global_step - input_["version_start"]
                ).float()
            if "version_end" in input_.keys():
                seq_stats["tail_offpolicyness"] = (
                    self.global_step - input_["version_end"]
                ).float()
            stats_tracker.stat(
                **seq_stats,
                denominator="n_seqs",
            )
            scalars = dict(
                disable_value=self.config.wrap_policy.disable_value,
                mask_no_eos_with_zero=self.config.wrap_policy.mask_no_eos_with_zero,
                eps_clip=self.config.wrap_policy.eps_clip,
                use_prox_logp=use_prox_logp,
            )
            if self.config.wrap_policy.c_clip is not None:
                scalars["c_clip"] = self.config.wrap_policy.c_clip
                scalars["use_dual_clip"] = 1
            else:
                scalars["use_dual_clip"] = 0
            stats_tracker.scalar(**scalars)

            global_stats = stats_tracker.export()
            # for k in global_denominators:
            #     global_stats.pop(f"ppo_actor/{k}")

        result = dict(
            advantages=advantages,
            old_logp=old_logp,
            rollout_logp=rollout_logp,
            ppo_loss_mask=loss_mask,
            packed_input_ids=input_["packed_input_ids"],
            kl_rewards=kl_rewards,
            global_stats=global_stats,
        )
        
        if use_kl_in_loss and "packed_ref_logprobs" in input_.keys():
            result["ref_logprobs"] = ref_logp
            
        return result

    def notify_event(self, event: str, global_step: int) -> None:
        """Handle training start/end events by sending HTTP notification.

        Args:
            event: "train_start" or "train_end"
            global_step: Current global step
        """
        if event not in ["train_start", "train_end"]:
            raise ValueError(f"Invalid event type: {event}")

        logger.info(f"[RemoteHypridTrainWorker] Sending training {event} notification at global_step: {global_step}")

        try:
            target_url = f"http://{self.megatron_addr}/events"
            headers = {"Content-Type": "application/json"}
            payload = {
                "event": event,
                "global_step": global_step
            }
            response = requests.post(
                target_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=60
            )
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to send training event. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except Exception as e:
            raise ValueError(f"Error sending notify training event: {e}")

        return None

    def compute_logprobs_with_distributed(
        self,
        input_: DistributedBatchMemory,
    ) -> torch.Tensor | None:
        if not input_ or len(input_.dataset) == 0:
            raise ValueError("input_.dataset is empty")
        first_item = input_[0]
        attrs = list(first_item.keys())

        batch_data = {}
        for attr in attrs:
            batch_data[attr] = input_[attr]
        torch.set_printoptions(threshold=float('inf'))

        if "input_ids" in batch_data and "seqlen" in batch_data:
            batch_data["packed_input_ids"] = pack_input_ids(batch_data["input_ids"], batch_data["seqlen"])

        # logprobs => packed_logprobs
        if "logprobs" in batch_data and "seqlen" in batch_data:
            batch_data["packed_logprobs"] = pack_logprobs(batch_data["logprobs"], batch_data["seqlen"])

        batch = {
            "packed_input_ids": batch_data["packed_input_ids"],
            "seqlen": batch_data["seqlen"],
            "packed_logprobs": batch_data["packed_logprobs"]}
        logger.info(f"[RemoteHypridTrainWorker] compute_logprobs_with_distributed input packed_input_ids data: {batch_data["packed_input_ids"].shape}")
        logprobs = self.compute_logprobs(batch)
        if logprobs is None:
            raise ValueError(
                f"[RemoteHypridTrainWorker] Failed to exec compute_logprobs"
            )
        logger.info(f"[RemoteHypridTrainWorker] compute_logprobs_with_distributed success, logprobs shape: {logprobs.shape}")
        return logprobs

    def compute_logprobs(
        self,
        input_: Dict,  # key: str, value: tensor
    ) -> torch.Tensor | None:
        logger.info("[RemoteHypridTrainWorker] begin exec compute_logprobs...")
        seqlen_tensor = input_["seqlen"]

        if seqlen_tensor.dim() == 1:
            batch_size = int(seqlen_tensor.shape[0])
            group_size = 1
        else:
            batch_size = int(seqlen_tensor.shape[0])
            group_size = int(seqlen_tensor.shape[1])
            
        flat_input = SequenceSample.from_default(
            ids=list(range(batch_size*group_size)),
            data={k: v for k, v in input_.items() if k != "seqlen"},
            seqlens=[int(x) for x in input_["seqlen"].cpu().numpy().tolist()],
        )

        mb_spec = MicroBatchSpec(n_mbs=self.config.n_mbs,
                                 max_tokens_per_mb=self.config.max_tokens_per_mb)
        try:
            target_url = f"http://{self.megatron_addr}/compute_logprobs"
            headers = {"Content-Type": "application/octet-stream"}
            payload = {
                "sequence_sample": flat_input,
                "micro_batch_spec": mb_spec,
            }
            data = serialize_and_compress(payload)
            logger.info("[RemoteHypridTrainWorker] send compute_logprobs request to megatron worker....")
            response = requests.post(
                target_url, data=data, headers=headers, timeout=7200
            )
            if response.status_code == 200:
                sequence_sample_logp = cloudpickle.loads(response.content)
                for k, v in sequence_sample_logp.data.items():
                    sequence_sample_logp.data[k] = v.to("cpu").clone()
                logger.info(
                    f"[RemoteHypridTrainWorker] compute_logprobs exec success, response status code: {response.status_code}"
                )

                logprobs = sequence_sample_logp.data["logprobs"] #[0.11,0.33,0.44]
                seqlens = sequence_sample_logp.seqlens["logprobs"] #[2, 1]

                assert input_["packed_logprobs"].shape == logprobs.shape

                # 将tensor列表转换为(batchsize, max_len)的2D tensor
                batch_result = []
                offset = 0
                max_len = max([seqlen for batch in seqlens for seqlen in batch])
                
                for batch_seqlens in seqlens:
                    for seqlen in batch_seqlens:
                        seq_logprobs = logprobs[offset:offset + seqlen]
                        padded = torch.nn.functional.pad(
                            seq_logprobs, 
                            (0, max_len - seqlen),
                            value=0.0
                        )
                        batch_result.append(padded)
                        offset += seqlen

                stack_res = torch.stack(batch_result)
                return stack_res
            else:
                raise ValueError(
                    f"[RemoteHypridTrainWorker] Failed to exec compute_logprobs. Status code: {response.status_code}, "
                    f"Response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError("[RemoteHybridTrainWorker] compute_logprobs request timeout!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                "[RemoteHybridTrainWorker] Send compute_logprobs request, an error occurred:", e
            )

        return None

def serialize_and_compress(data):
    serialized_data = cloudpickle.dumps(data)
    compressed_data = gzip.compress(serialized_data)
    return compressed_data

def pack_input_ids(input_ids: torch.Tensor, seqlen: torch.Tensor) -> torch.Tensor:
    """
    将input_ids按seqlen拼接成packed_input_ids。
    Args:
        input_ids: shape [batch, seq_len]
        seqlen: shape [batch], 每个样本的有效长度
    Returns:
        packed_input_ids: shape [sum(seqlen)]
    """
    packed = []
    for i in range(input_ids.shape[0]):
        valid_len = seqlen[i].item()
        packed.append(input_ids[i, :valid_len])
    return torch.cat(packed, dim=0)


def pack_logprobs(logprobs: torch.Tensor, seqlen: torch.Tensor) -> torch.Tensor:
    """
    将logprobs按seqlen拼接成packed_logprobs。
    每个样本的有效logprobs长度为seqlen[i]-1。
    处理逻辑：对于每个样本i，跳过第一个元素，取logprobs[i, 1:seqlen[i]]
    Args:
        logprobs: shape [batch, seq_len]
        seqlen: shape [batch], 每个样本的有效长度
    Returns:
        packed_logprobs: shape [sum(seqlen-1)]
    """
    packed = []
    for i in range(logprobs.shape[0]):
        # 跳过第一个元素，取seqlen[i]-1个元素
        packed.append(logprobs[i, 1:seqlen[i].item()])
    return torch.cat(packed, dim=0)

def pack_ref_logprobs(logprobs: torch.Tensor, seqlen: torch.Tensor) -> torch.Tensor:
    packed = []
    for i in range(logprobs.shape[0]):
        ref_len = seqlen[i].item() - 1
        packed.append(logprobs[i, :ref_len])
    return torch.cat(packed, dim=0)

def pack_prompt_mask(prompt_mask: torch.Tensor, seqlen: torch.Tensor) -> torch.Tensor:
    packed = []
    for i in range(prompt_mask.shape[0]):
        valid_len = seqlen[i].item()
        packed.append(prompt_mask[i, :valid_len])
    return torch.cat(packed, dim=0)
