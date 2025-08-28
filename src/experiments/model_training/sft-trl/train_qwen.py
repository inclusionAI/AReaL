### QWEN2.5 JINJA2 template with {% generation %} and {% endgeneration %} required to be able to compute assistant masks.
### Example: Smol https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja#L76-L82

QWEN25_TEMPLATE_PATH = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/qwen2.5_prompt_template.jinja"
QWEN25_TEMPLATE = None
with open(QWEN25_TEMPLATE_PATH, mode="r") as fp:
    QWEN25_TEMPLATE = fp.read()

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-instruct"
DEFAULT_TRAIN_DATASET_PATH = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/train_full-v1.jsonl"
DEFAULT_TEST_DATASET_PATH = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/test_full-v1.jsonl"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/qwen2.5-0.5b-instruct-sft-full-tau2"
DEFAULT_MAX_TRAIN_DATAPOINTS = None
DEFAULT_MAX_TEST_DATAPOINTS = None
DEFAULT_USE_PEFT = False
DEFAULT_PATIENCE = 2


if __name__ == "__main__":
    from pathlib import Path

    model_name = "Qwen2.5-0.5B-instruct"
    qwen_model = f"Qwen/{model_name}"
    train_dataset_path = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/train_full-v1.jsonl"
    test_dataset_path = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/test_full-v1.jsonl"
    trained_model_name = f"{model_name}-sft-full-tau2-assistant-only-loss"
    output_dir = f"/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/{trained_model_name}"
    report_to_wandb = True
    run_name = f"{trained_model_name}"
    if Path(output_dir).exists():
        print(
            f"Output directory {output_dir} already exists. Do you want to delete it? (y/n)"
        )
        answer = input()
        if answer == "y":
            print(f"Deleting output directory {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                f"Output directory {output_dir} already exists. Please delete it or choose a different output directory."
            )

    max_train_datapoints = None
    max_test_datapoints = None
    use_peft = False
    patience = 2
    use_accelerate = True

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
        use_accelerate=use_accelerate,
    )
