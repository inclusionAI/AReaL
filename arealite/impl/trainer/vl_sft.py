import time
from typing import Any, Dict, List

import torch
import torch.utils.data


from arealite.utils import (
    close_wandb_tensorboard,
    compute_varlen_position_indices,
    gather_logprobs,
    init_stats_logging,
    list_of_dict2dict_of_list,
    log_wandb_tensorboard,
    record_timing,
)

from realhf.base import logging
from .sft import SFTTrainer



logger = logging.getLogger("VL_SFT Trainer")


logger = logging.getLogger("VL_SFT Trainer")
 
class VL_SFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_packed_input(self, data: Dict)-> Dict[str, torch.Tensor]:
        """
        Get packed vision-language input tensors.
        data.keys(): vl_prompt_input_ids, vl_prompt_length, pixel_values, image_grid_thw,
        answer_input_ids, answer_length
        
        Output:
        A dictionary with keys:
        input_ids, attention_mask, position_ids, prompt_mask, pixel_values, image_grid_thw, cu_seqlens, max_seqlen, use_cache
        """
        data: Dict[str, List[Any]] = list_of_dict2dict_of_list(data)
        breakpoint()
        tokenized_seqs = data["seq"]
        tokenized_prompts = data["prompt"]
        prompt_lens = [len(prompt) for prompt in tokenized_prompts]
        input_lens = [len(prompt) for prompt in tokenized_seqs]

        input_lens = torch.tensor(input_lens, dtype=torch.int)
        input_ids = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_seqs]

        prompt_mask = []
        for input_len, prompt_len in zip(input_lens, prompt_lens):
            assert input_len >= prompt_len, (input_len, prompt_len)
            pm = [1] * prompt_len + [0] * (input_len - prompt_len)
            prompt_mask.append(torch.tensor(pm, dtype=torch.bool))

        cu_seqlens = torch.nn.functional.pad(
            input_lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        max_seqlen = int(torch.max(input_lens).item())
        packed_input_ids = torch.cat(input_ids, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        total_seqlen = int(cu_seqlens[-1].item())
        position_ids = compute_varlen_position_indices(total_seqlen, cu_seqlens)
        
        pixel_values =[torch.tensor(pixel_value).cuda() for pixel_value in data["pixel_values"]]
        image_grid_thw = [torch.tensor(image_grid_thw_).cuda() for image_grid_thw_ in data["image_grid_thw"]]
        return dict(
            input_ids=packed_input_ids.unsqueeze(0).cuda(),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0).cuda(),
            prompt_mask=prompt_mask.unsqueeze(0).cuda(),
            cu_seqlens=cu_seqlens.cuda(),
            max_seqlen=max_seqlen,
            use_cache=False,
            pixel_values=pixel_values,  
            image_grid_thw=image_grid_thw,  
        )
