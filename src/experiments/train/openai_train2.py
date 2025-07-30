import json
import os
from pathlib import Path
from typing import Dict, List, Literal

from dotenv import load_dotenv
from openai import OpenAI
from tau2.utils.utils import DATA_DIR

DATA_DIR = DATA_DIR / "exp" / "data" / "test_sft_dataset"


def prepare_training_data(
    file_name: str,
    input_dir: str,
    mode: Literal["dpo", "sft"],
) -> List[Dict]:
    """
    Prepare training data for fine-tuning.

    Args:
        file_name: Name of the training file
        input_dir: Directory containing train.jsonl file
        mode: Fine-tuning mode, either "dpo" or "sft"

    Returns:
        List of training examples
    """
    input_path = Path(input_dir)
    training_data = []

    # Read training data
    with open(input_path / file_name, "r") as f:
        print(f"Reading training data from {input_path / file_name}")
        for line in f:
            example = json.loads(line)
            if mode == "sft":
                sft_example = {
                    "messages": [
                        example["user_simulator_formatted_messages"][0],
                        example["user_simulator_formatted_messages"][1],
                        example["annotated_user_simulator_preferred_messages"],
                    ]
                }
                training_data.append(sft_example)
            else:
                dpo_example = {
                    "input": {
                        "messages": example["user_simulator_formatted_messages"],
                    },
                    "preferred_output": [
                        example["annotated_user_simulator_preferred_messages"],
                    ],
                    "non_preferred_output": [
                        example["user_simulator_non_preferred_messages"],
                    ],
                }
                training_data.append(dpo_example)

    return training_data


def upload_training_file(client: OpenAI, training_data: List[Dict]) -> str:
    """
    Upload training data to OpenAI and return the file ID.

    Args:
        client: OpenAI client instance
        training_data: List of training examples
        training_run_dir: Directory to save training data

    Returns:
        File ID from OpenAI
    """
    output_file = DATA_DIR / "training_data.jsonl"
    with open(output_file, "w") as f:
        for example in training_data:
            f.write(json.dumps(example) + "\n")

    # Upload the file to OpenAI
    with open(output_file, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")

    return response.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str,
    mode: Literal["dpo", "sft"] = "sft",
    hyperparams: Dict = None,
    auto: bool = True,
) -> str:
    """
    Create a fine-tuning job with OpenAI.

    Args:
        client: OpenAI client instance
        training_file_id: ID of the uploaded training file
        model: Base model to fine-tune
        mode: Fine-tuning mode, either "dpo" or "sft"
        hyperparams: Dictionary of hyperparameters to use
        auto: Whether to use auto-parameters
    Returns:
        Job ID from OpenAI
    """
    if mode == "dpo":
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            method={
                "type": "dpo",
                "dpo": {
                    "hyperparameters": hyperparams if not auto else {},
                },
            },
        )
    else:  # sft mode
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            hyperparameters=hyperparams if not auto else {},
        )

    return job.id


def monitor_fine_tuning_job(client: OpenAI, job_id: str):
    """
    Monitor the progress of a fine-tuning job.

    Args:
        client: OpenAI client instance
        job_id: ID of the fine-tuning job
    """
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status == "succeeded":
            print(f"Fine-tuning completed! Model: {job.fine_tuned_model}")
            break
        elif status == "failed":
            print("Fine-tuning failed!")
            print(f"Error: {job.error}")
            break
        else:
            print(f"Status: {status}")
            # Wait for 60 seconds before checking again
            import time

            time.sleep(60)


def main(mode, model, monitor, file_name, auto):
    """
    Main function to run fine-tuning.

    Args:
        mode: Fine-tuning mode, either "dpo" or "sft"
        model: Base model to fine-tune
        monitor: Whether to monitor job progress
        file_name: Name of the training file
        auto: Whether to use auto-parameters
    """
    # Load environment variables from .env file
    load_dotenv()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        raise ValueError("Please set OPENAI_API_KEY in .env file")

    print("\nPreparing training data...")
    training_data = prepare_training_data(
        file_name=file_name, mode=mode, input_dir=DATA_DIR
    )
    hyperparams = {"learning_rate_multiplier": 0.5, "n_epochs": 3}
    if auto:
        print("\nUsing auto hyperparameters")
        hyperparams = {"learning_rate_multiplier": -1, "n_epochs": -1}
    else:
        print(
            f"\nLaunching fine-tuning jobs with learning rate multiplier {hyperparams['learning_rate_multiplier']} and n_epochs {hyperparams['n_epochs']}"
        )

    print("\nUploading training file to OpenAI...")
    training_file_id = upload_training_file(client, training_data)

    training_metadata = {
        "model": model,
        "mode": mode,
        "hyperparams": hyperparams,
        "using_auto_hyperparams": auto,
        "training_data_file_name": str(DATA_DIR / file_name),
        "training_file_id": training_file_id,
    }

    print(f"Starting fine-tuning with metadata: {training_metadata}")

    job_id = create_fine_tuning_job(
        client,
        training_file_id,
        model=model,
        mode=mode,
        hyperparams=hyperparams,
        auto=auto,
    )
    training_metadata.update({"job_id": job_id})
    training_metadata_file = DATA_DIR / "training_metadata.json"
    with open(training_metadata_file, "w") as f:
        json.dump(training_metadata, f)
    all_trains_metadata_file = DATA_DIR / "all_trains_metadata.jsonl"
    with open(all_trains_metadata_file, "a") as f:
        f.write(json.dumps(training_metadata) + "\n")

    if monitor:
        print("\nMonitoring fine-tuning progress...")
        monitor_fine_tuning_job(client, job_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fine-tuning with either DPO or SFT mode"
    )
    parser.add_argument(
        "--mode",
        choices=["dpo", "sft"],
        default="dpo",
        help="Fine-tuning mode (default: sft)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        choices=[
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
        ],
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Do not monitor job progress"
    )
    parser.add_argument(
        "--file-name",
        default="train.jsonl",
        help="Name of the training file",
    )
    parser.add_argument(
        "--auto", action="store_true", default=True, help="use auto-parameters"
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        model=args.model,
        monitor=not args.no_monitor,
        file_name=args.file_name,
        auto=args.auto,
    )