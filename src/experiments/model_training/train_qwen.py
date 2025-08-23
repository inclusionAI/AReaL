import os
import time

import dotenv
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from experiments.model_training.prepare_dataset import (
    OpenAICompletionDataPoint,
    load_as_hf_dataset,
)

dotenv.load_dotenv()


### QWEN2.5 JINJA2 template with {% generation %} and {% endgeneration %} required to be able to compute assistant masks.
### Example: Smol https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja#L76-L82

with open("qwen2.5_prompt_template.jinja", mode="r") as fp:
    QWEN25_TEMPLATE = fp.read()


def check_chat_template_supports_assistant_only_loss(tokenizer: AutoTokenizer) -> bool:
    """
    Check if the chat template supports assistant only loss.
    """
    return "{% generation %}" in tokenizer.chat_template


def get_model_and_tokenizer(
    model_name, torch_dtype="auto", device_map="auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    start_time = time.time()
    # Load tokenizer and model (4-bit optional)
    print(f"Loading tokenizer and model for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for model {model_name} does not have a chat template."
        )
    else:
        print("Tokenizer has a chat template.")
    print(f"Tokenizer eos_token = {tokenizer.eos_token}")
    print(
        f"Tokenizer eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}"
    )
    print(
        f"Tokenizer chat_template supports assistant only loss: {check_chat_template_supports_assistant_only_loss(tokenizer)}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,  # Automatically determines the best dtype (e.g., float16, bfloat16)
        device_map=device_map,  # Automatically distributes the model across available devices (e.g., GPUs)
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-instruct"
DEFAULT_TRAIN_DATASET_PATH = "data/train_full-v1.jsonl"
DEFAULT_TEST_DATASET_PATH = "data/test_full-v1.jsonl"
DEFAULT_OUTPUT_DIR = "./data/qwen2.5-0.5b-instruct-sft-full-tau2"
DEFAULT_MAX_TRAIN_DATAPOINTS = None
DEFAULT_MAX_TEST_DATAPOINTS = None
DEFAULT_USE_PEFT = False
DEFAULT_PATIENCE = 2


def train(
    qwen_model: str = DEFAULT_QWEN_MODEL,
    train_dataset_path: str = DEFAULT_TRAIN_DATASET_PATH,
    test_dataset_path: str = DEFAULT_TEST_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_train_datapoints: int = DEFAULT_MAX_TRAIN_DATAPOINTS,
    max_test_datapoints: int = DEFAULT_MAX_TEST_DATAPOINTS,
    use_peft: bool = DEFAULT_USE_PEFT,
    patience: int = DEFAULT_PATIENCE,
    report_to_wandb: bool = True,
    run_name: str = None,
):
    os.environ["WANDB_PROJECT"] = "tau2-bench-agent"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    model_name = qwen_model

    train_dataset = load_as_hf_dataset(
        train_dataset_path, max_datapoints=max_train_datapoints
    )
    test_dataset = load_as_hf_dataset(
        test_dataset_path, max_datapoints=max_test_datapoints
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    ## NOTE
    ## Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures.
    ## Those are cleaned up by SFTTraining using remove_non_values
    ## https://github.com/huggingface/trl/blob/30576d2ddcf2c0e17c399399e2465fbe81446ade/trl/trainer/sft_trainer.py#L71
    ## Pushed to prepare_dataset.py -> remove_none_values
    ## Warning: Those None values will make generation fail if not cleaned up.

    model, tokenizer = get_model_and_tokenizer(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer.chat_template = (
        QWEN25_TEMPLATE  # NOTE: This is required to be able to compute assistant masks.
    )

    if use_peft:
        learning_rate = 1e-4  # Higher learning rate for PEFT?
    else:
        learning_rate = 8e-5

    # NOTE:
    # Issue with assistant_only_loss=True:
    # chat_template does not contain {% generation %} condition so HF cannot compute assistant masks.
    # I updated that but I get no looss if I do assistant_only_loss=True

    sft_config = SFTConfig(
        assistant_only_loss=True,  # Only compute the loss on the assistant messages, requires jinja2 template with {% generation %} and {% endgeneration %}
        report_to="none" if not report_to_wandb else "wandb",  # disable logging to W&B
        run_name=run_name,  #  Optional run name for W&B
        logging_strategy="steps",
        learning_rate=learning_rate,  # Learning rate for training.
        num_train_epochs=20,  #  Set the number of epochs to train the model.
        per_device_train_batch_size=2,  # Batch size for each device (e.g., GPU) during training.
        gradient_accumulation_steps=8,  # Number of steps before performing a backward/update pass to accumulate gradients.
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
        logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
        eval_strategy="epoch",  # evaluate at end of each epoch
        save_strategy="epoch",  # save checkpoint at end of each epoch
        save_total_limit=1,  # keep only the best/latest model
        load_best_model_at_end=True,  # load best model according to eval loss
        metric_for_best_model="eval_loss",  # use eval loss for best model selection
        greater_is_better=False,  # lower eval_loss is better
        output_dir=output_dir,  # directory to save checkpoints
        auto_find_batch_size=True,  # Automatically find the best batch size for training.
        max_length=None,
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


if __name__ == "__main__":
    from pathlib import Path

    model_name = "Qwen2.5-3B-instruct"
    qwen_model = f"Qwen/{model_name}"
    train_dataset_path = "data/train_full-v1.jsonl"
    test_dataset_path = "data/test_full-v1.jsonl"
    trained_model_name = f"{model_name}-sft-full-tau2-assistant-only-loss"
    output_dir = f"./data/{trained_model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    report_to_wandb = True
    run_name = f"{trained_model_name}"
    if Path(output_dir).exists():
        raise ValueError(
            f"Output directory {output_dir} already exists. Please delete it or choose a different output directory."
        )

    max_train_datapoints = None
    max_test_datapoints = None
    use_peft = False
    patience = 2

    train(
        qwen_model=qwen_model,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        output_dir=output_dir,
        max_train_datapoints=max_train_datapoints,
        max_test_datapoints=max_test_datapoints,
        use_peft=use_peft,
        patience=patience,
        report_to_wandb=report_to_wandb,
        run_name=run_name,
    )
