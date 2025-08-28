import argparse
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dotenv
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed import destroy_process_group
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from experiments.model_training.dataset_prep.prepare_tau_dataset import (
    load_as_hf_dataset,
)

dotenv.load_dotenv()

os.environ["WANDB_PROJECT"] = "tau2-bench-agent"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


@dataclass
class TrainingArguments:
    """Additional training arguments not covered by TRL's built-in argument classes."""

    train_dataset_path: str = field(
        default="/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/train_full-v1.jsonl",
        metadata={"help": "Path to the training dataset"},
    )
    test_dataset_path: str = field(
        default="/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/test_full-v1.jsonl",
        metadata={"help": "Path to the test dataset"},
    )
    max_train_datapoints: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training datapoints to use"}
    )
    max_test_datapoints: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of test datapoints to use"}
    )
    use_accelerate: bool = field(
        default=False,
        metadata={"help": "Whether to use accelerate for distributed training"},
    )
    patience: Optional[int] = field(
        default=2, metadata={"help": "Number of epochs to wait for early stopping"}
    )
    delete_existing_output_dir: bool = field(
        default=False,
        metadata={"help": "Whether to automatically delete existing output directory"},
    )


def check_chat_template_supports_assistant_only_loss(chat_template: str) -> bool:
    """
    Check if the chat template supports assistant only loss.
    """
    return "{% generation %}" in chat_template


def get_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    chat_template_path: str | Path = None,
) -> AutoTokenizer:
    """
    Load the HF tokenizer for the given model.
    """
    start_time = time.time()
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, use_fast=use_fast
    )
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    if chat_template_path is not None:
        with open(chat_template_path, mode="r") as fp:
            chat_template = fp.read()
        tokenizer.chat_template = chat_template
        print(f"Tokenizer chat_template loaded from {chat_template_path}")

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for model {model_name} does not have a chat template."
        )
    else:
        print("Tokenizer has a chat template.")
    print(f"Tokenizer eos_token = {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"Tokenizer pad_token = {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
    print(
        f"Tokenizer chat_template supports assistant only loss: {check_chat_template_supports_assistant_only_loss(tokenizer.chat_template)}"
    )
    return tokenizer


def get_model(
    model_name: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    use_accelerate: bool = False,
) -> AutoModelForCausalLM:
    """
    Load the HF model for the given model name.
    """
    start_time = time.time()
    print(f"Loading model for {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,  # Automatically determines the best dtype (e.g., float16, bfloat16)
        device_map=device_map,  # Automatically distributes the model across available devices (e.g., GPUs)
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    if use_accelerate:
        print(f"Using accelerate to move model to the correct device.")
        device_string = PartialState().process_index
        model = model.to(device_string)
        print(f"Model moved to device {device_string}")
    return model


def get_model_and_tokenizer(
    model_name, torch_dtype="auto", device_map="auto", use_accelerate=False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the HF model and tokenizer for the given model name.

    """
    model = get_model(model_name, torch_dtype, device_map, use_accelerate)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer


def load_datasets(
    train_dataset_path: str,
    test_dataset_path: str,
    max_train_datapoints: int | None = None,
    max_test_datapoints: int | None = None,
) -> tuple[Dataset, Dataset]:
    assert os.path.exists(train_dataset_path), (
        f"Train dataset path {train_dataset_path} does not exist."
    )
    assert os.path.exists(test_dataset_path), (
        f"Test dataset path {test_dataset_path} does not exist."
    )
    train_dataset = load_as_hf_dataset(
        train_dataset_path, max_datapoints=max_train_datapoints
    )
    test_dataset = load_as_hf_dataset(
        test_dataset_path, max_datapoints=max_test_datapoints
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    return train_dataset, test_dataset


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, TrainingArguments)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


def train_sft_trl(
    script_args: ScriptArguments,  # Not currently used.
    sft_config: SFTConfig,
    model_config: ModelConfig,
    train_dataset_path: str,
    test_dataset_path: str,
    chat_template_path: str | Path | None = None,
    max_train_datapoints: int | None = None,
    max_test_datapoints: int | None = None,
    use_accelerate: bool = False,
    patience: int | None = 2,
):
    """
    Main function to train a model using the SFTTrainer from trl.
    SFTTrainer is a wrapper around the HF Trainer class that allows for supervised fine-tuning of a model.
    It directly integrates with accelerate.

    Args:
        script_args: ScriptArguments, not currently used.
        sft_config: SFTConfig, configuration for the SFT training.
        model_config: ModelConfig, configuration for the model.
        chat_template_path: str | Path | None, path to the chat template.
        train_dataset_path: str, path to the train dataset.
        test_dataset_path: str, path to the test dataset.
        max_train_datapoints: int | None, maximum number of datapoints to use for training.
        max_test_datapoints: int | None, maximum number of datapoints to use for testing.
        use_accelerate: bool, whether to use accelerate.
        patience: int | None, number of epochs to wait before early stopping.
    """
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    device_map = (
        get_kbit_device_map()
        if use_accelerate or (quantization_config is not None)
        else "auto"
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    # Create tokenizer
    tokenizer = get_tokenizer(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
        chat_template_path=chat_template_path,
    )

    ################
    # Dataset
    ################
    train_dataset, test_dataset = load_datasets(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        max_train_datapoints=max_train_datapoints,
        max_test_datapoints=max_test_datapoints,
    )

    ################
    # Callbacks
    ################
    # Instantiate early stopping callback
    early_stopping_callback = (
        EarlyStoppingCallback(early_stopping_patience=patience)
        if patience is not None
        else None
    )
    callbacks = [early_stopping_callback] if early_stopping_callback is not None else []

    ################
    # Training
    ################

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if sft_config.eval_strategy != "no" else None,
        processing_class=tokenizer,
        callbacks=callbacks,
        peft_config=get_peft_config(model_config),
    )

    # trainer.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # trainer.compute_loss(trainer.model, trainer.train_dataset[0])
    trainer.train()

    # # Save and push to hub
    # trainer.save_model(training_args.output_dir)

    if use_accelerate:
        destroy_process_group()  # For some reason the training process does not end when using accelerate.


def main():
    parser = make_parser()
    script_args, sft_config, model_config, training_args, _ = (
        parser.parse_args_and_config(return_remaining_strings=True)
    )

    # Handle output directory deletion if requested
    if (
        training_args.delete_existing_output_dir
        and Path(sft_config.output_dir).exists()
    ):
        print(f"Deleting existing output directory: {sft_config.output_dir}")
        shutil.rmtree(sft_config.output_dir)
    elif (
        Path(sft_config.output_dir).exists()
        and not training_args.delete_existing_output_dir
    ):
        raise ValueError(
            f"Output directory {sft_config.output_dir} already exists. "
            f"Use --delete_existing_output_dir to automatically delete it."
        )

    train_sft_trl(
        script_args=script_args,
        sft_config=sft_config,
        model_config=model_config,
        chat_template_path=sft_config.chat_template_path,
        train_dataset_path=training_args.train_dataset_path,
        test_dataset_path=training_args.test_dataset_path,
        max_train_datapoints=training_args.max_train_datapoints,
        max_test_datapoints=training_args.max_test_datapoints,
        use_accelerate=training_args.use_accelerate,
        patience=training_args.patience,
    )


if __name__ == "__main__":
    main()
