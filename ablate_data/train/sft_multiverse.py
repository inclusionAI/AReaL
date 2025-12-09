import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import transformers
import trl
from datasets import load_dataset
from utils import (
    MultiverseDataCollatorForCompletionOnlyLM,
    SequentialSFTTrainer,
    add_and_init_special_tokens,
)


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=24576)
    wandb_project: Optional[str] = field(default="Multiverse")
    train_file_path: Optional[str] = field(
        default="Multiverse4FM/Autoregressive-1K-mixed"
    )
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name, **kwargs
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True
    )
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    else:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    add_and_init_special_tokens(model, tokenizer)

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = MultiverseDataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        max_length=config.block_size,
        tokenizer=tokenizer,
        mlm=False,
    )
    args.dataset_text_field = "text"
    # We will truncate the input in our collator, so we set a very large max_seq_length to avoid truncation in tokenizer
    args.max_seq_length = 131072
    trainer = SequentialSFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()