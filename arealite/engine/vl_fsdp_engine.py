# import os
# import time
# from typing import Optional

# import torch
# import torch.distributed as dist
# import transformers
# from tensordict import TensorDict
# from torch.distributed.checkpoint.state_dict import (
#     StateDictOptions,
#     get_model_state_dict,
# )
# from transformers import AutoModelForImageTextToText

# from arealite.api.cli_args import TrainEngineConfig
# from arealite.api.engine_api import FinetuneSpec, SaveLoadMeta, WeightUpdateMeta
# from arealite.engine.fsdp_engine import FSDPEngine
# from arealite.utils.data import (
#     MicroBatchList,
#     amend_position_ids,
#     pack_tensor_dict,
#     pad_mb_list,
#     split_padded_tensor_dict_into_mb_list,
#     unsqueeze_mb_list,
# )
# from arealite.utils.fsdp import (
#     CPUOffloadPolicy,
#     MixedPrecisionPolicy,
#     apply_fsdp2,
#     create_fsdp_device_mesh,
#     get_cosine_schedule_with_warmup,
# )
# from arealite.utils.model import disable_dropout_in_model
# from realhf.api.core.data_api import load_hf_processor_and_tokenizer
# from realhf.base import logging, name_resolve, names, pkg_version

# logger = logging.getLogger("FSDPEngine")


# class VL_FSDPEngine(FSDPEngine):
#     def __init__(self, config: TrainEngineConfig):
#         super().__init__(config)
#         self.processor=None
#         self.tokenizer = None
        


#     def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
#         # Initialize distributed enviroments and load model.
#         assert addr is None, "FSDPEngine does not support remote initialization."
#         assert pkg_version.is_version_greater_or_equal(
#             "torch", "2.4.0"
#         ), f"arealite only supports FSDP2, which requires torch>=2.4.0"

#         self.create_process_group()

#         # TODO: Handle the condition when LOCAL_RANK is not set in launcher
#         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#         self.device = torch.device(int(os.environ["LOCAL_RANK"]))

#         dtype = torch.bfloat16 
#         self.processor, self.tokenizer = load_hf_processor_and_tokenizer(self.config.path)
#         tik = time.perf_counter()
#         with torch.device("cuda"):
#             # initialize scratch model from config
#             model = AutoModelForImageTextToText.from_pretrained(
#                 pretrained_model_name_or_path=self.config.path,
#                 trust_remote_code=True,
#                 torch_dtype=dtype,
#                 attn_implementation=self.config.attn_impl,
#             )
#             if self.config.disable_dropout:
#                 disable_dropout_in_model(model)
#         if self.config.gradient_checkpointing:
#             model.gradient_checkpointing_enable(
#                 gradient_checkpointing_kwargs={"use_reentrant": False}
#             )
#         logger.info(f"Model creation and loading time: {time.perf_counter() - tik}")
#         self.model = model

#         # Wrap with FSDP2
#         # Simple auto wrap policy
#         self.mixed_precision_policy = MixedPrecisionPolicy(
#             param_dtype=getattr(torch, self.config.dtype),
#             reduce_dtype=torch.float32,
#             cast_forward_inputs=True,
#         )
#         self.device_mesh = create_fsdp_device_mesh(self.world_size, self.world_size)
#         # sharding_strategy = ShardingStrategy.FULL_SHARD
#         self.cpu_offload = (
#             CPUOffloadPolicy() if self.config.fsdp.offload_params else None
#         )
#         fsdp_kwargs = {
#             "mesh": self.device_mesh,
#             "mp_policy": self.mixed_precision_policy,
#             "offload_policy": self.cpu_offload,
#             "reshard_after_forward": True,
#         }
#         tik = time.perf_counter()
#         apply_fsdp2(self.model, fsdp_kwargs, self.config.fsdp.wrap_policy)
#         logger.info(f"Applying FSDP2 time: {time.perf_counter() - tik}")

#         self.create_optimizer(ft_spec)
#         self.initialized = True
    
#     # def save(self, meta: SaveLoadMeta):
#     #     if meta.weight_format == "hf":
#     #         self._save_model_to_hf(meta.path, meta.tokenizer,meta.processor)
#     #     elif meta.weight_format == "dcp":
#     #         # TODO: implement DCP save/load for FSDP
#     #         raise NotImplementedError("DCP format saving is not implemented yet. ")
#     #     else:
#     #         raise ValueError(f"Unknown weight format {meta.weight_format}. ")

#     #     if meta.with_optim:
#     #         self._save_optimizer_state(meta.path)

#     # def _save_model_to_hf(
#     #     self, path: str, tokenizer: Optional[transformers.PreTrainedTokenizerFast], processor: Optional[transformers.AutoProcessor]
#     # ):
#     #     """Save model in HuggingFace format."""
#     #     if self.model is None:
#     #         raise RuntimeError("Model not initialized")
#     #     os.makedirs(path, exist_ok=True)

#     #     # FSDP2 checkpoint saving
#     #     # Get full state dict with FSDP2
#     #     options = StateDictOptions(full_state_dict=True, cpu_offload=True)
#     #     state_dict = get_model_state_dict(self.model, options=options)

#     #     # save huggingface model on rank 0
#     #     if dist.get_rank() == 0:
#     #         os.makedirs(path, exist_ok=True)
#     #         self.model.save_pretrained(path, state_dict=state_dict)
#     #         if tokenizer is not None:
#     #             tokenizer.save_pretrained(path)
#     #         if processor is not None:
#     #             processor.save_pretrained(path)
#     #     dist.barrier()

#     # def upload_weights(self, meta: WeightUpdateMeta):
#     #     if meta.type == "nccl":
#     #         if not self.weight_update_group_initialized:
#     #             self._init_distributed_weight_update(meta)
#     #         self._update_weights_from_distributed()
#     #     elif meta.type == "disk":
#     #         self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
#     #         # dist.barrier() are called when _save_model_to_hf finished
#     #         if dist.get_rank() == 0:
#     #             update_name = names.update_weights_from_disk(
#     #                 self.config.experiment_name,
#     #                 self.config.trial_name,
#     #                 meta.model_version,
#     #             )
#     #             name_resolve.add(update_name, str(time.time_ns()), keepalive_ttl=120)
#     #     else:
#     #         raise ValueError(f"Unknown weight update type {meta.type}")

#     # def _prepare_mb_list(self, input_: TensorDict) -> MicroBatchList:
#     #     assert "attention_mask" in input_ and "input_ids" in input_ and "pixel_values" in input_ and "image_grid_thw" in input_, \
#     #         "Input TensorDict must contain 'attention_mask', 'input_ids', 'pixel_values', and 'image_grid_thw' keys."
 
#     #     if isinstance(input_, dict):
#     #         input_ = TensorDict(input_, batch_size=[input_["input_ids"].shape[0]])
#     #     input_ = amend_position_ids(input_)
#     #     packed_input = pack_tensor_dict(input_)
#     #     mb_list = split_padded_tensor_dict_into_mb_list(
#     #         packed_input,
#     #         self.config.mb_spec,
#     #     )
#     #     mb_list = pad_mb_list(mb_list, pad_value=0.0)
#     #     # NOTE: We unsqueeze here because huggingface transformer models requires
#     #     # packed input to be of shape [1, total_seqlen].
#     #     mb_list = unsqueeze_mb_list(mb_list)
#     #     return mb_list

