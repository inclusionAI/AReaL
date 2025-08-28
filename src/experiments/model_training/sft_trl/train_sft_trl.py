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

from experiments.model_training.dataset_prep.prepare_dataset import load_as_hf_dataset

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


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-instruct"
DEFAULT_TRAIN_DATASET_PATH = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/train_full-v1.jsonl"
DEFAULT_TEST_DATASET_PATH = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/test_full-v1.jsonl"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/qwen2.5-0.5b-instruct-sft-full-tau2"
DEFAULT_MAX_TRAIN_DATAPOINTS = None
DEFAULT_MAX_TEST_DATAPOINTS = None
DEFAULT_USE_PEFT = False
DEFAULT_PATIENCE = 2


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


def train(
    model_name: str = DEFAULT_MODEL,
    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH,
    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_train_datapoints: int = DEFAULT_MAX_TRAIN_DATAPOINTS,
    max_test_datapoints: int = DEFAULT_MAX_TEST_DATAPOINTS,
    use_peft: bool = DEFAULT_USE_PEFT,
    patience: int = DEFAULT_PATIENCE,
    report_to_wandb: bool = True,
    run_name: str = None,
    use_accelerate: bool = False,
    chat_template_path: str | Path | None = None,
):
    """
    Train HF model using the SFTTrainer from trl.
    SFTTrainer is a wrapper around the HF Trainer class that allows for supervised fine-tuning of a model.
    It directly integrates with accelerate.
    """
    train_dataset, test_dataset = load_datasets(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        max_train_datapoints=max_train_datapoints,
        max_test_datapoints=max_test_datapoints,
    )

    model, tokenizer = get_model_and_tokenizer(
        model_name, torch_dtype="auto", device_map="cuda", use_accelerate=use_accelerate
    )  # If auto, this is going to create issue with accelerate
    # tokenizer.chat_template = QWEN25_TEMPLATE # NOTE: This is required to be able to compute assistant masks.
    # if hasattr(tokenizer, "model_max_length"):
    #     max_length = tokenizer.model_max_length
    # else:
    #     max_length = 1024
    # max_length = None
    # print(f"Setting max_length to {max_length}")

    if use_peft:
        learning_rate = 1e-4  # Higher learning rate for PEFT?
    else:
        learning_rate = 8e-5

    # NOTE:
    # Issue with assistant_only_loss=True:
    # chat_template does not contain {% generation %} condition so HF cannot compute assistant masks.
    # I updated that but I get no looss if I do assistant_only_loss=True
    assistant_only_loss = True

    sft_config = SFTConfig(
        assistant_only_loss=assistant_only_loss,  # Only compute the loss on the assistant messages, requires jinja2 template with {% generation %} and {% endgeneration %}
        report_to="none" if not report_to_wandb else "wandb",  # disable logging to W&B
        chat_template_path=chat_template_path
        if assistant_only_loss
        else None,  # NOTE: This is required to be able to compute assistant masks.
        run_name=run_name,  #  Optional run name for W&B
        logging_strategy="steps",
        learning_rate=learning_rate,  # Learning rate for training.
        num_train_epochs=3,  #  Set the number of epochs to train the model.
        per_device_train_batch_size=1,  # Batch size for each device (e.g., GPU) during training.
        gradient_accumulation_steps=8,  # Number of steps before performing a backward/update pass to accumulate gradients.
        fp16_full_eval=True,
        eval_accumulation_steps=4,
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed. Not compatible with use_cache=True?
        logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
        eval_strategy="epoch",  # evaluate at end of each epoch
        save_strategy="epoch",  # save checkpoint at end of each epoch
        save_total_limit=1,  # keep only the best/latest model
        load_best_model_at_end=True,  # load best model according to eval loss
        metric_for_best_model="eval_loss",  # use eval loss for best model selection
        greater_is_better=False,  # lower eval_loss is better
        output_dir=output_dir,  # directory to save checkpoints
        auto_find_batch_size=True,  # Automatically find the best batch size for training.
        max_length=8192,
    )

    # Instantiate early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=patience  # Stop if no improvement for 2 evals (epochs)
    )

    if use_peft:  # FIXME: Check what's the right config.
        # lora_config = LoraConfig(
        #     r=64,
        #     lora_alpha=16,
        #     target_modules=["c_attn", "q_proj", "v_proj"],  # adjust to Qwen architecture
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type=TaskType.CAUSAL_LM,
        # )
        lora_config = LoraConfig()
    else:
        lora_config = None

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        callbacks=[early_stopping_callback],
        peft_config=lora_config,
    )
    sft_trainer.train()
    if use_accelerate:
        destroy_process_group()  # For some reason the training process does not end when using accelerate.


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
    chat_template_path: str | Path | None = None,
    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH,
    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH,
    max_train_datapoints: int = DEFAULT_MAX_TRAIN_DATAPOINTS,
    max_test_datapoints: int = DEFAULT_MAX_TEST_DATAPOINTS,
    use_accelerate: bool = False,
    patience: int | None = DEFAULT_PATIENCE,
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
        max_train_datapoints: int, maximum number of datapoints to use for training.
        max_test_datapoints: int, maximum number of datapoints to use for testing.
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
    from pathlib import Path

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
