# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
# ]
# ///

"""
# Full training
```
python trl_sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl_sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse

from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def main(
    script_args: ScriptArguments, sft_config: SFTConfig, model_config: ModelConfig
):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )

    # Set default chat template if needed
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer chat template is None")
        # # TODO: source should be passed as an argument
        # model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

    ################
    # Dataset
    ################
    dataset: DatasetDict = load_from_disk(script_args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset[script_args.dataset_train_split].select(range(10)),
        eval_dataset=dataset[script_args.dataset_test_split].select(range(10))
        if sft_config.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # trainer.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # trainer.compute_loss(trainer.model, trainer.train_dataset[0])
    trainer.train()

    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, sft_config, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, sft_config, model_args)
