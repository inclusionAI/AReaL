from typing import List

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader

from areal.platforms import current_platform
from areal.utils.distributed import init_custom_process_group

logger = init_logger("vllm_worker_extension")


class VLLMWorkerExtension:
    """
    Iherited from vllm codebase
    """

    def sync(self):
        current_platform.synchronize()
        torch.distributed.barrier()

    def update_weights(self, model_path):
        logger.info(f"start update weights, {model_path}", flush=True)
        try:
            # load weight
            self.model_runner.model_config.model = model_path
            model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
            logger.info("Reloading weights inplace...")
            model_loader.load_weights(
                self.model_runner.model, model_config=self.model_runner.model_config
            )
            self.sync()

            return True, "Success"
        except Exception as e:
            error_msg = f"failed to upload weights! {e}"
            logger.error(error_msg)
            return False, error_msg

    def set_weight_meta(
        self, names: List[str], dtypes: List[str], shapes: List[List[int]]
    ):
        logger.info("start set weights meta")
        self.areal_weight_meta_names = names
        self.areal_weight_meta_dtypes = dtypes
        self.areal_weight_meta_shapes = shapes
        return True, "Success"

    def update_weight_xccl(self):
        logger.info("start update weights by nccl or hccl", flush=True)
        names = self.areal_weight_meta_names
        dtypes = self.areal_weight_meta_dtypes
        shapes = self.areal_weight_meta_shapes
        try:
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                tensor = torch.empty(
                    shape, dtype=target_dtype, device=self.model_runner.device
                )
                torch.distributed.broadcast(
                    tensor,
                    src=0,
                    group=self.weight_update_group,
                    async_op=False,
                )
                self.model_runner.model.load_weights(weights=[(name, tensor)])
            self.sync()
            return True, f"Success"
        except Exception as e:
            error_msg = f"Failed to update parameter! {e}."
            logger.error(error_msg)
            return False, error_msg

    def init_update_weight_group(
        self,
        master_address: str,
        master_port: str,
        rank_offset: int,
        world_size: int,
        backend: str,
        group_name: str,
    ):
        if getattr(self, "weight_update_group", None) != None:
            return True, "Success"
        try:
            self.weight_update_group = init_custom_process_group(
                backend=backend,
                world_size=world_size,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=self.rank + rank_offset,
                group_name=group_name,
            )
            return True, f"Success"
        except Exception as e:
            error_msg = f"Failed to init group! {e}."
            logger.error(error_msg)
            return False, error_msg
