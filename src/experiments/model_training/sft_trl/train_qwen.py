from pathlib import Path
import shutil
from .train_sft_trl import train



model_name = "Qwen2.5-0.5B-instruct"
model = f"Qwen/{model_name}"
train_dataset_path = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/train_full-v1.jsonl"
test_dataset_path = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/test_full-v1.jsonl"
trained_model_name = f"{model_name}-sft-full-tau2-assistant-only-loss"
output_dir = f"/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/data/{trained_model_name}"
chat_template_path = "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/qwen2.5_prompt_template.jinja"
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
    model_name=model,
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
    chat_template_path=chat_template_path,
)