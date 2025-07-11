import dataclasses
import gc
import os
import gzip
import json
from typing import Any, Callable, Dict, List

import torch
import requests
import cloudpickle

from arealite.api.cli_args import RemoteMegatronEngineConfig
from arealite.api.engine_api import (
    FinetuneSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
)
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory

from realhf.base import logging
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample

logger = logging.getLogger("RemoteMegatronEngine")


@dataclasses.dataclass
class RemoteMegatronInitConfig:
    addrs: list[str]
    global_rank: int
    local_rank: int
    world_size: int
    recover_dir: str
    # ft_spec: FinetuneSpec = dataclasses.field(default_factory=FinetuneSpec)
    # megatron_config: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # loss_configs: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class RemoteInferenceInitConfig:
    addrs: list[str]
    global_rank: int
    local_rank: int
    world_size: int

class RemoteMegatronEngine(TrainEngine):
    def __init__(self, config: RemoteMegatronEngineConfig | None):
        self.config = config
        self.megatron_addr = None
        self.rendevzous_ip = None

        # initialization
        self.initialized = False
        self.weight_update_group_initialized = False

    def initialize(self, init_config: RemoteMegatronInitConfig):
        global_rank = init_config.global_rank
        self.megatron_addr = init_config.magatron_addrs[global_rank]
        master_addr = init_config.magatron_addrs[0]
        master_ip, master_port = master_addr.split(":", 1)  # ip:port
        logger.info(f"[RemoteMegatronEngine] _initialize init_config: {init_config}")

        megatron_config = init_config.megatron_config
        megatron_config["total_train_steps"] = init_config.ft_spec.total_train_steps
        payload = {
            "rank": str(init_config.global_rank),
            "local_rank": str(init_config.local_rank),
            "master_port": str(master_port),
            "master_addr": str(master_addr),
            "world_size": str(init_config.world_size),
            "megatron_config": init_config.megatron_config,
            "loss_configs": init_config.loss_configs,
            "recover_dir": init_config.recover_dir,
        }

        try:
            target_url = f"http://{self.magatron_addr}/initialize"
            headers = {"Content-Type": "application/json"}
            logger.info(
                f"[RemoteMegatronEngine] initialize begin send request to megatron server, "
                f"target_url: {target_url}, rank: {global_rank}, target_url: {target_url}"
            )
            response = requests.post(
                target_url, data=json.dumps(payload), headers=headers, timeout=7200
            )
            logger.info(
                f"[RemoteMegatronEngine] initialize finished send request to megatron server"
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteMegatronEngine] rank: {global_rank} Payload sent successfully to {target_url}"
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

        logger.info(f"[RemoteMegatronEngine] rank: {global_rank} megatron server initialize success")
        self.rendevzous_ip = master_addr  # for update_weights
        self.initialized = True

    def get_scheduling_config(self):
        # 获取调度器调度engine所需的资源配置信息
        pass

    def destroy(self):
        self.initialized = False

    async def upload_weights(self, meta: WeightUpdateMeta):
        if meta.type == "nccl":
            if not self.weight_update_group_initialized:
                self.init_distributed_weight_update(meta)
            self.update_weights_from_distributed()
        elif meta.type == "disk":
            save_load_meta = SaveLoadMeta(
                path=meta.path,
                weight_format="huggingface",
                global_step=meta.global_step,
            )
            self.save(save_load_meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format not in ["huggingface", "mcore"]:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        try:
            logger.info(
                f"[RemoteMegatronEngine] send save request, "
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
                    f"[RemoteMegatron] save hf request exec success, save_dir: {meta.path}"
                )
            else:
                raise ValueError(
                    f"[RemoteMegatronEngine] Failed to send save {meta.weight_format} "
                    f"request. status code: {response.status_code}, response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError(f"[RemoteMegatronEngine] save {meta.weight_format} request timed out!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"[RemoteMegatronEngine] Send save {meta.weight_format} request, an error occurred: {e}")
        return

    def load(self, meta: SaveLoadMeta):
        pass

    def step_lr_scheduler(self):
        # 和shaoshang、kuangzhi沟通，offpolicy时不需要控制lr.step(global_step),所以不需要暴露lr sched
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
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        # 1. 获取所有属性名
        if not input_.dataset or len(input_.dataset) == 0:
            raise ValueError("input_.dataset is empty")
        first_item = input_.dataset[0]
        attrs = list(first_item.keys())

        # 2. 构造 attr -> stacked tensor
        batch_data = {}
        for attr in attrs:
            tensor_list = input_[attr]  # list[tensor]
            # 转为大tensor
            batch_tensor = torch.stack(tensor_list)
            batch_data[attr] = batch_tensor

        return self.train_batch(batch_data, mb_spec, loss_fn, loss_weight_fn)

    def train_batch(
        self,
        input_: Dict,  # key: str, value: tensor
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> Dict[str, float]:
        logger.info(f"[RemoteMegatronEngine] mb_spec: {mb_spec}")
        try:
            target_url = f"http://{self.megatron_addr}/train_batch"
            headers = {"Content-Type": "application/octet-stream"}
            payload = {
                "sequence_sample": input_,
                "micro_batch_spec": mb_spec,
            }
            data = serialize_and_compress(payload)
            logger.info("[RemoteMegatronEngine] send train_batch request to megatron worker....")
            response = requests.post(
                target_url, data=data, headers=headers, timeout=7200
            )
            if response.status_code == 200:
                logger.info(
                    f"[RemoteMegatronEngine] Train batch exec success, response status code: {response.status_code}, response: {response.json()}"
                )
            else:
                raise ValueError(
                    f"[RemoteMegatronEngine] Failed to exec train_batch. Status code: {response.status_code}, Response: {response.text}"
                )
        except requests.exceptions.Timeout:
            raise ValueError("[RemoteMegatronEngine] Train request timeout!")
        except requests.exceptions.RequestException as e:
            raise ValueError(
                "[RemoteMegatronEngine] Send train request, an error occurred:", e
            )

        return response.json()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable[[torch.Tensor, Dict], torch.Tensor],
        loss_weight_fn: Callable[[Dict], float],
    ) -> torch.Tensor | None:
        pass

    @torch.no_grad()
    def forward(
        self,
        input_: Dict,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, Dict], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        pass


def serialize_and_compress(data):
    serialized_data = cloudpickle.dumps(data)
    compressed_data = gzip.compress(serialized_data)
    return compressed_data
