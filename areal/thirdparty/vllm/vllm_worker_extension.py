import torch
from vllm.logger import init_logger
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
        self, names: list[str], dtypes: list[str], shapes: list[list[int]]
    ):
        logger.info("start set weights meta")
        self.areal_weight_meta_names = names
        self.areal_weight_meta_dtypes = dtypes
        self.areal_weight_meta_shapes = shapes
        return True, "Success"

    def set_weight_meta_lora(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        lora_name: str,
        lora_int_id: int,
        base_model_name: str,
    ):
        logger.info(
            f"start set lora weights meta for lora_name={lora_name}, lora_int_id={lora_int_id}"
        )
        self.areal_lora_weight_meta_names = names
        self.areal_lora_weight_meta_dtypes = dtypes
        self.areal_lora_weight_meta_shapes = shapes
        self.areal_lora_name = lora_name
        self.areal_lora_int_id = lora_int_id
        self.areal_lora_base_model_name = base_model_name
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
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to update parameter! {e}."
            logger.error(error_msg)
            return False, error_msg

    def update_weight_lora_xccl(self):
        logger.info(
            f"start update lora weights by xccl, lora_name={self.areal_lora_name}, lora_int_id={self.areal_lora_int_id}",
            flush=True,
        )
        names = self.areal_lora_weight_meta_names
        dtypes = self.areal_lora_weight_meta_dtypes
        shapes = self.areal_lora_weight_meta_shapes
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
                    group=self.weight_update_group,
                    async_op=False,
                )

                received_weights[name] = tensor

            logger.info(f"Received {len(received_weights)} LoRA parameters via XCCL")

            # Group PEFT weights by vLLM module and weight type (handles merged modules like qkv_proj and gate_up_proj)
            vllm_module_groups = _group_peft_weights_for_vllm(received_weights)

            logger.info(f"Grouped into {len(vllm_module_groups)} vLLM modules")

            if len(vllm_module_groups) != len(lora_model.loras):
                raise RuntimeError(
                    f"Lora module length mismatch between FSDP and vLLM. Counts {len(vllm_module_groups)} from fsdp vs {len(lora_model.loras)} from vllm "
                )

            updated_modules = 0

            for vllm_module_name, weight_info in vllm_module_groups.items():
                if vllm_module_name not in lora_model.loras:
                    logger.warning(
                        f"vLLM module not found in lora_model.loras: {vllm_module_name}"
                    )
                    continue

                lora_layer = lora_model.loras[vllm_module_name]

                if isinstance(lora_layer.lora_a, list):
                    # Packed module: lora_a and lora_b are lists of tensors
                    if weight_info["lora_a_tensors"]:
                        if len(weight_info["lora_a_tensors"]) != len(lora_layer.lora_a):
                            logger.error(
                                f"Tensor count mismatch for {vllm_module_name}.lora_a: "
                                f"expected {len(lora_layer.lora_a)}, got {len(weight_info['lora_a_tensors'])}"
                            )
                            continue

                        for idx, peft_tensor in enumerate(
                            weight_info["lora_a_tensors"]
                        ):
                            # Transpose PEFT tensor: [rank, in_features] -> [in_features, rank]
                            vllm_tensor = peft_tensor.t().contiguous()

                            if lora_layer.lora_a[idx].shape == vllm_tensor.shape:
                                with torch.no_grad():
                                    lora_layer.lora_a[idx].copy_(vllm_tensor)
                                logger.debug(
                                    f".Updated lora_a[{idx}] for {vllm_module_name}"
                                )
                            else:
                                logger.error(
                                    f"Shape mismatch for {vllm_module_name}.lora_a[{idx}]: "
                                    f"expected {lora_layer.lora_a[idx].shape}, got {vllm_tensor.shape}"
                                )

                    if weight_info["lora_b_tensors"]:
                        if len(weight_info["lora_b_tensors"]) != len(lora_layer.lora_b):
                            logger.error(
                                f"Tensor count mismatch for {vllm_module_name}.lora_b: "
                                f"expected {len(lora_layer.lora_b)}, got {len(weight_info['lora_b_tensors'])}"
                            )
                            continue

                        for idx, peft_tensor in enumerate(
                            weight_info["lora_b_tensors"]
                        ):
                            # Transpose PEFT tensor: [out_features, rank] -> [rank, out_features]
                            vllm_tensor = peft_tensor.t().contiguous()

                            if lora_layer.lora_b[idx].shape == vllm_tensor.shape:
                                with torch.no_grad():
                                    lora_layer.lora_b[idx].copy_(vllm_tensor)
                                logger.debug(
                                    f".Updated lora_b[{idx}] for {vllm_module_name}"
                                )
                            else:
                                logger.error(
                                    f"Shape mismatch for {vllm_module_name}.lora_b[{idx}]: "
                                    f"expected {lora_layer.lora_b[idx].shape}, got {vllm_tensor.shape}"
                                )
                else:
                    # Non-packed module: lora_a and lora_b are single tensors
                    if weight_info["lora_a_tensors"]:
                        if len(weight_info["lora_a_tensors"]) != 1:
                            logger.error(
                                f"Expected 1 lora_a tensor for {vllm_module_name}, got {len(weight_info['lora_a_tensors'])}"
                            )
                            continue

                        peft_tensor = weight_info["lora_a_tensors"][0]
                        # Transpose PEFT tensor: [rank, in_features] -> [in_features, rank]
                        vllm_tensor = peft_tensor.t().contiguous()

                        if lora_layer.lora_a.shape == vllm_tensor.shape:
                            with torch.no_grad():
                                lora_layer.lora_a.copy_(vllm_tensor)
                            logger.debug(f".Updated lora_a for {vllm_module_name}")
                        else:
                            logger.error(
                                f"Shape mismatch for {vllm_module_name}.lora_a: "
                                f"expected {lora_layer.lora_a.shape}, got {vllm_tensor.shape}"
                            )
                            continue

                    # Update lora_B
                    if weight_info["lora_b_tensors"]:
                        if len(weight_info["lora_b_tensors"]) != 1:
                            logger.error(
                                f"Expected 1 lora_b tensor for {vllm_module_name}, got {len(weight_info['lora_b_tensors'])}"
                            )
                            continue

                        peft_tensor = weight_info["lora_b_tensors"][0]
                        # Transpose PEFT tensor: [out_features, rank] -> [rank, out_features]
                        vllm_tensor = peft_tensor.t().contiguous()

                        if lora_layer.lora_b.shape == vllm_tensor.shape:
                            with torch.no_grad():
                                lora_layer.lora_b.copy_(vllm_tensor)
                            logger.debug(f".Updated lora_b for {vllm_module_name}")
                        else:
                            logger.error(
                                f"Shape mismatch for {vllm_module_name}.lora_b: "
                                f"expected {lora_layer.lora_b.shape}, got {vllm_tensor.shape}"
                            )
                            continue

                updated_modules += 1

            logger.info(
                f"LoRA weight update summary via XCCL for lora_int_id={lora_int_id}: "
                f"updated {updated_modules}/{len(lora_model.loras)} modules, "
            )

            self.sync()
            return True, "Success"

        except Exception as e:
            import traceback

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
        if getattr(self, "weight_update_group", None) is not None:
            return True, "Success"
        try:
            self.weight_update_group = init_custom_process_group(
                backend=backend,
                world_size=world_size,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=self.rank + rank_offset,
                group_name=group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to init group! {e}."
            logger.error(error_msg)
            return False, error_msg


# Utility functions for LORA XCCL Update in vLLM
def _group_peft_weights_for_vllm(peft_weights: dict[str, torch.Tensor]) -> dict:
    vllm_groups = {}

    for peft_name, tensor in peft_weights.items():
        # Parse the PEFT name
        vllm_module_name, is_lora_a, component = _parse_peft_name(peft_name)

        if vllm_module_name is None:
            continue

        if vllm_module_name not in vllm_groups:
            vllm_groups[vllm_module_name] = {
                "lora_a_tensors": {},
                "lora_b_tensors": {},
            }

        if is_lora_a:
            vllm_groups[vllm_module_name]["lora_a_tensors"][component] = tensor
        else:
            vllm_groups[vllm_module_name]["lora_b_tensors"][component] = tensor

    # Convert dicts to lists in fixed order
    for module_name, weight_info in vllm_groups.items():
        if "qkv_proj" in module_name:
            order = ["q", "k", "v"]
        elif "gate_up_proj" in module_name:
            order = ["gate", "up"]
        else:
            order = [""]

        weight_info["lora_a_tensors"] = [
            weight_info["lora_a_tensors"].get(comp)
            for comp in order
            if comp in weight_info["lora_a_tensors"]
        ]
        weight_info["lora_b_tensors"] = [
            weight_info["lora_b_tensors"].get(comp)
            for comp in order
            if comp in weight_info["lora_b_tensors"]
        ]

    return vllm_groups


def _parse_peft_name(peft_name: str) -> tuple[str | None, bool, str]:
    # Check if it's a LoRA weight
    if ".lora_A.default.weight" in peft_name:
        is_lora_a = True
        name = peft_name.replace(".lora_A.default.weight", "")
    elif ".lora_B.default.weight" in peft_name:
        is_lora_a = False
        name = peft_name.replace(".lora_B.default.weight", "")
    else:
        return None, False, ""

    # Remove PEFT prefix
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]

    # Determine component and convert to vLLM name
    component = ""

    if ".q_proj" in name:
        component = "q"
        name = name.replace(".q_proj", ".qkv_proj")
    elif ".k_proj" in name:
        component = "k"
        name = name.replace(".k_proj", ".qkv_proj")
    elif ".v_proj" in name:
        component = "v"
        name = name.replace(".v_proj", ".qkv_proj")
    elif ".gate_proj" in name:
        component = "gate"
        name = name.replace(".gate_proj", ".gate_up_proj")
    elif ".up_proj" in name:
        component = "up"
        name = name.replace(".up_proj", ".gate_up_proj")
    elif ".o_proj" in name:
        component = ""
    elif ".down_proj" in name:
        component = ""

    return name, is_lora_a, component
