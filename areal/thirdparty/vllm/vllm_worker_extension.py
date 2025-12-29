import traceback

import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader

from areal.platforms import current_platform
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
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

    def update_weights_lora(
        self,
        lora_model_path: str,
        lora_name: str,
        lora_int_id: int,
        base_model_name: str,
    ):
        logger.info(
            f"start lora update weights, lora_model_path-{lora_model_path}, lora_name-{lora_name}, lora_int_id-{lora_int_id}, base_model_name-{base_model_name}",
            flush=True,
        )
        try:
            # load lora weight
            self.model_runner.lora_manager.remove_adapter(lora_int_id)
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_model_path,
                base_model_name=base_model_name,
            )
            logger.info(f"Reloading lora weights with request {lora_request}")
            self.model_runner.add_lora(lora_request)

            self.sync()
            return True, "Success"
        except Exception as e:
            error_msg = f"failed to upload lora weights! {e}"
            logger.error(error_msg)
            return False, error_msg

    def set_weight_meta(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
    ):
        logger.info("start set weights meta")
        self.areal_weight_meta_names = names
        self.areal_weight_meta_dtypes = dtypes
        self.areal_weight_meta_shapes = shapes
        self.areal_weight_meta_group_name = group_name
        return True, "Success"

    def set_weight_meta_lora(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        lora_name: str,
        lora_int_id: int,
        lora_target_modules: list[str] | str,
        lora_rank: int,
        lora_alpha: int,
        lora_bias: str,
        base_model_name: str,
    ):
        logger.info(
            f"start set lora weights meta for lora_name={lora_name}, lora_int_id={lora_int_id}"
        )
        self.areal_lora_weight_meta_names = names
        self.areal_lora_weight_meta_dtypes = dtypes
        self.areal_lora_weight_meta_shapes = shapes
        self.areal_weight_meta_group_name = group_name
        self.areal_lora_name = lora_name
        self.areal_lora_int_id = lora_int_id
        self.areal_lora_target_modules = lora_target_modules
        self.areal_lora_rank = lora_rank
        self.areal_lora_alpha = lora_alpha
        self.areal_lora_bias = lora_bias
        self.areal_lora_base_model_name = base_model_name
        return True, "Success"

    def update_weight_xccl(self):
        logger.info("start update weights by nccl or hccl", flush=True)
        names = self.areal_weight_meta_names
        dtypes = self.areal_weight_meta_dtypes
        shapes = self.areal_weight_meta_shapes
        try:
            group = self.weight_update_groups[self.areal_weight_meta_group_name]
        except KeyError:
            raise KeyError(
                f"Weight update group named `{self.areal_weight_meta_group_name}` not found"
            )
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
                    group=group,
                    async_op=False,
                )
                self.model_runner.model.load_weights(weights=[(name, tensor)])
            self.sync()
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to update parameter! {e}."
            logger.error(error_msg)
            return False, error_msg

    def update_weight_lora_xccl(self):
        # NOTE: This code relies on vLLM private APIs: _adapter_manager, _registered_adapters,
        # and _add_adapter/activate_adapter, which may change/ breakdown due to newer vllm versions.

        logger.info(
            f"start update lora weights by xccl, lora_name={self.areal_lora_name}, lora_int_id={self.areal_lora_int_id}",
            flush=True,
        )
        names = self.areal_lora_weight_meta_names
        dtypes = self.areal_lora_weight_meta_dtypes
        shapes = self.areal_lora_weight_meta_shapes
        try:
            group = self.weight_update_groups[self.areal_weight_meta_group_name]
        except KeyError:
            raise KeyError(
                f"Weight update group named `{self.areal_weight_meta_group_name}` not found"
            )
        lora_int_id = self.areal_lora_int_id

        try:
            # Check if LoRA manager and adapter exist
            if self.model_runner.lora_manager is None:
                raise RuntimeError("LoRA manager is not initialized")

            # Check if the LoRA adapter exists
            adapter_ids = self.model_runner.lora_manager.list_adapters()
            if lora_int_id not in adapter_ids:
                raise RuntimeError(
                    f"LoRA adapter {lora_int_id} not found. Available: {adapter_ids}"
                )

            # Get the LoRA model
            lora_model = (
                self.model_runner.lora_manager._adapter_manager._registered_adapters[
                    lora_int_id
                ]
            )
            logger.info(f"Found LoRA model with {len(lora_model.loras)} LoRA modules")

            # Receive all weights via XCCL broadcast
            logger.info(f"Receiving {len(names)} LoRA parameters via XCCL")
            received_weights = {}
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
                    group=group,
                    async_op=False,
                )

                received_weights[name] = tensor

            logger.info(f"Received {len(received_weights)} LoRA parameters via XCCL")

            self.model_runner.lora_manager.remove_adapter(lora_int_id)

            normalized_weights = {
                k.replace("default.", ""): v for k, v in received_weights.items()
            }

            peft_config = {
                "r": self.areal_lora_rank,
                "lora_alpha": self.areal_lora_alpha,
                "target_modules": self.areal_lora_target_modules,
                "bias": self.areal_lora_bias,
            }
            peft_helper = PEFTHelper.from_dict(peft_config)

            new_lora_model = LoRAModel.from_lora_tensors(
                lora_model_id=self.areal_lora_int_id,
                tensors=normalized_weights,
                peft_helper=peft_helper,
                device=self.model_runner.device,
                dtype=self.model_runner.lora_manager.lora_config.lora_dtype,
                embeddings=None,  # Note: Current version does not support lora embeddings (see vllm/lora/models.py)
                target_embedding_padding=self.model_runner.lora_manager.vocab_size
                + self.model_runner.lora_manager.lora_config.lora_extra_vocab_size,
                embedding_modules=self.model_runner.lora_manager.embedding_modules,
                embedding_padding_modules=self.model_runner.lora_manager.embedding_padding_modules,
                weights_mapper=getattr(
                    self.model_runner.model, "hf_to_vllm_mapper", None
                ),
            )

            self.model_runner.lora_manager._adapter_manager._add_adapter(new_lora_model)
            self.model_runner.lora_manager._adapter_manager.activate_adapter(
                new_lora_model.id
            )
            logger.info(
                f"Found LoRA model with {len(new_lora_model.loras)} LoRA modules"
            )

            self.sync()
            return True, "Success"

        except Exception as e:
            error_msg = f"Failed to update LoRA parameter via XCCL!   {e}\n{traceback.format_exc()}"
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
        if not hasattr(self, "weight_update_groups"):
            self.weight_update_groups: dict[str, dist.ProcessGroup] = {}
        try:
            group = init_custom_process_group(
                backend=backend,
                world_size=world_size,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=self.rank + rank_offset,
                group_name=group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.weight_update_groups[group_name] = group
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to init group! {e}."
            logger.error(error_msg)
            return False, error_msg
