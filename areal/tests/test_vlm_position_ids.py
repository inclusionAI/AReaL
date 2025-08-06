import os
import sys

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoModelForImageTextToText
import torch
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.utils.data import pad_sequences_to_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_processor_and_tokenizer

tokenizer_path = "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct"
# tokenizer_path = "/storage/openpsi/models/Qwen2-VL-7B"
data_path = "/storage/openpsi/data/geometry3k/"
def main():
    os.environ["RANK"] = str(0)
    os.environ['WORLD_SIZE'] = str(1)
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = str(7777)
    processor, tokenizer = load_hf_processor_and_tokenizer(tokenizer_path)
    train_dataset = get_custom_dataset(
        path=data_path,
        rank=0,
        world_size=1,
        split="train",
        type="sft",
        tokenizer=tokenizer,
        processor=processor,
    )

    # Create dataset and dataloaders
    bs = 2
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=1,
        collate_fn=pad_sequences_to_tensors,
        drop_last=True,
    )


    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=len(train_dataloader) * bs,
        train_batch_size=bs,
    )

    with torch.device("cuda"):
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        model.eval()
    for step, data in enumerate(train_dataloader):
        for k, v in data.items():
            # print(k, v.shape)
            data[k] = v.cuda()
        data['image_grid_thw']=data['image_grid_thw'].squeeze(1)
        # breakpoint()
        res = model.forward(**data)
        from areal.utils.data import unpad_input
        y1=unpad_input(res.logits, data['attention_mask'])[0]

        from areal.utils.data import unsqueeze_mb_list,amend_position_ids_3d, split_padded_tensor_dict_into_mb_list, pack_tensor_dict, pad_mb_list,amend_position_ids
        from tensordict import TensorDict
        from areal.api.cli_args import MicroBatchSpec
        mb_spec = MicroBatchSpec(1, int(1e12))
        input_ = TensorDict({k: v.clone() for k, v in data.items()})
        # input_ = amend_position_ids(input_)
        input_ = amend_position_ids_3d(input_,model.model.get_rope_index)
        mb_list = split_padded_tensor_dict_into_mb_list(input_, mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=False,
        )
        mb_list = unsqueeze_mb_list(mb_list)
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None)
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
            mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None)
        assert len(mb_list.padded_mbs) == 1
        x = mb_list.padded_mbs[0]
        pad_len = mb_list.padding_lengths[0]
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        print(x['position_ids'])
        res2 = model(**x)
        y2 = res2.logits.squeeze(0)[:-pad_len]
        from torch.testing import assert_close
        assert_close(y1,y2)
        diff = (y1 - y2).abs()
        print(diff)

        # print(res.logits.shape)
        break
    print('finish')

main()