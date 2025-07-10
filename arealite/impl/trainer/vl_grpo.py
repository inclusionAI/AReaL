# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from typing import List, Optional
from collections.abc import MutableSequence
import torch
from datasets import Dataset
from arealite.api.cli_args import (
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.io_struct import Trajectory
from arealite.system.rollout_controller import RolloutController
from arealite.utils import (
    compute_varlen_position_indices,
    concat_padded_tensors,
    to_device,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import logging
from .grpo import SpmdGRPOTrainer
from PIL.Image import Image as ImageObject
logger = logging.getLogger("VL GRPO Trainer", "system")


class VL_SpmdGRPOTrainer(SpmdGRPOTrainer):
    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(args, trainer_config, train_dataset, valid_dataset, rollout_controller)
        self.actor_processor, self.actor_tokenizer = load_hf_processor_and_tokenizer(self.config.actor.path)
        
    def _train_step(self, trajs: List[Trajectory]):
        breakpoint()
        rollout = concat_padded_tensors([traj.data for traj in trajs])
        rollout = to_device(rollout, torch.cuda.current_device())

        # Marks which sequence does not has an EOS token, i.e.,
        # generation is truncated by the configured maximum generation length
        batch_tokens = rollout["input_ids"]
        images = [traj.images for traj in trajs]


        if isinstance(images, MutableSequence) and all(isinstance(i, MutableSequence) and all(isinstance(x, str) for x in i) for i in images):
            #paths/url to images
            #convert to double list
            tmp_images=[]
            for image_list in images:
                image_list = [ImageObject.open(image) for image in images]
                tmp_images.append(image_list)
            images = tmp_images

        assert all(isinstance(image, ImageObject) for image_list in images for image in image_list),(
            "All images should be PIL.Image objects, but got: "
            f"{[type(image) for image_list in images for image in image_list]}"
        )
        processed_inputs = self.actor_processor.image_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = processed_inputs["pixel_values"]
        image_grid_thw = processed_inputs["image_grid_thw"]
        pixel_values = pixel_values.to(batch_tokens.device)
        image_grid_thw = image_grid_thw.to(batch_tokens.device)

        seq_no_eos_mask = (
            batch_tokens[:, -1] != self.actor_tokenizer.eos_token_id
        ).logical_and(batch_tokens[:, -1] != self.actor_tokenizer.pad_token_id)

        # Remove padding to use flash-attn
        attn_mask = rollout["attention_mask"]
        input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            rollout["input_ids"], attn_mask
        )
        position_ids = compute_varlen_position_indices(input_ids.shape[0], cu_seqlens)

        # Transformer forward input data
        model_inputs = dict(
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values.unsqueeze(0),
            image_grid_thw=image_grid_thw.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_cache=False,
        )
        old_logp, *_ = unpad_input(rollout["logprobs"], attn_mask)
        prompt_mask, *_ = unpad_input(rollout["prompt_mask"], attn_mask)
        # Shift logprobs and mask for computing loss.
        loss_mask = prompt_mask.logical_not()
        loss_mask = torch.roll(loss_mask, shifts=-1)
        old_logp = torch.roll(old_logp, shifts=-1)

        input_ids = model_inputs["input_ids"].squeeze(0)
        n_seqs = seq_no_eos_mask.shape[0]
        assert n_seqs == self.local_train_batch_size * self.group_size, (
            n_seqs,
            self.group_size,
            self.local_train_batch_size,
        )