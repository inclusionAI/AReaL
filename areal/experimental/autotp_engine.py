import os

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from areal.api.cli_args import TrainEngineConfig
from areal.api.engine_api import FinetuneSpec, SaveLoadMeta, WeightUpdateMeta
from areal.engine.base_hf_engine import BaseHFEngine
from areal.utils.save_load import (
    get_state_dict_from_repo_id_or_path,
    is_existing_local_path,
)
from realhf.base import constants, logging

logger = logging.getLogger("DeepSpeedAutoTPEngine")


class DeepSpeedAutoTPEngine(BaseHFEngine):
    def __init__(self, config: TrainEngineConfig):
        super().__init__(config)

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None):
        """Initialize distributed communication and model."""
        assert (
            addr is None
        ), "DeepSpeedAutoTPEngine does not support remote initialization."
        import deepspeed

        self.create_process_group()

        world_size = int(os.environ.get("WORLD_SIZE"))
        deepspeed.init_distributed(
            dist_backend="nccl",
            world_size=world_size,
            timeout=constants.NCCL_DEFAULT_TIMEOUT,
        )
        self.create_device_model()
        # NOTE: the device context manager does not work here.
        self.model = deepspeed.tp_model_init(
            self.model,
            tp_size=self.config.ds_auto_tp.autotp_size,
            dtype=getattr(torch, self.config.dtype),
        ).to(self.device)
        self.create_optimizer(ft_spec)
        self.initialized = True

    def _check_autotp(self):
        tp_size = self.config.ds_auto_tp.autotp_size
        config = self.model_config
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        return (
            num_attention_heads % tp_size == 0
            and num_key_value_heads % tp_size == 0
            and hidden_size % tp_size == 0
            and intermediate_size % tp_size == 0
        )

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format != "naive_distributed":
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")
        if self.model is None:
            raise RuntimeError("Model not initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            os.makedirs(meta.path, exist_ok=True)
            self.model_config.save_pretrained(
                meta.path,
            )
            if meta.tokenizer is not None:
                meta.tokenizer.save_pretrained(
                    meta.path,
                )

        state_dict = self.model.state_dict()
        if hasattr(self.model, "module"):
            state_dict = {
                k.replace("module.", "", 1) if k.startswith("module.") else k: v.cpu()
                for k, v in state_dict.items()
            }
        else:
            state_dict = {k: v.cpu() for k, v in state_dict.items()}

        # Only support store parameters from model partitions respectively
        gathered_state_dicts = None
        if rank == 0:
            gathered_state_dicts = [None for _ in range(world_size)]
        dist.gather_object(
            obj=state_dict, object_gather_list=gathered_state_dicts, dst=0
        )
        if rank == 0:
            for i, state_dict in enumerate(gathered_state_dicts):
                save_file(state_dict, f"{meta.path}/rank_{i:02d}_model.safetensors")
        if meta.with_optim:
            self.save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format != "naive_distributed":
            raise ValueError(f"Unknown weight format {meta.weight_format}. ")
        rank = dist.get_rank()
        # Only support load full model parameters from huggingface
        # and load model partition locally
        if rank == 0 or is_existing_local_path(meta.path):
            path = f"{meta.path}/rank_{rank:02d}_model.safetensors"
            full_state = get_state_dict_from_repo_id_or_path(meta.path)

            if hasattr(self.model, "module") and not hasattr(full_state):
                full_state = {
                    f"module.{k}" if not k.startswith("module.") else k: v
                    for k, v in full_state.items()
                }
            self.model.load_state_dict(
                full_state, strict=not self.model_config.tie_word_embeddings
            )
            if self.model_config.tie_word_embeddings:
                self.model.tie_weights()

        if meta.with_optim:
            self.load_optimizer_state(meta.path)

    def upload_weights(self, meta: WeightUpdateMeta):
        raise ValueError(f"update weight not implemented {meta.type}")
