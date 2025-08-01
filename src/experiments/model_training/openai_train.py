# Copyright Sierra

import base64
from io import StringIO
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from openai.types import FileObject
from openai.types.fine_tuning import FineTuningJob

client = OpenAI()


def create_training_file(file: str | Path) -> FileObject:
    """
    Create a training file.
    """
    with open(file, mode="rb") as fp:
        created_file = client.files.create(file=fp, purpose="fine-tune")
        print(f"FILE ID: {created_file.id}")
        print(created_file.model_dump_json(indent=4))
    return created_file


def start_fine_tuning_job(
    training_file: str,
    model: str,
    suffix: str,
    seed: int = 42,
    validation_file: Optional[str] = None,
    mode: Literal["dpo", "sft"] = "sft",
    hyperparams: Optional[dict] = None,
    auto: bool = True,
) -> FineTuningJob:
    """
    Start a fine-tuning job.
    """
    if hyperparams is None or auto:
        hyperparams = {}
    if mode == "dpo":
        fine_tuning_job = client.fine_tuning.jobs.create(
            training_file=training_file,
            model=model,
            suffix=suffix,
            seed=seed,
            validation_file=validation_file,
            method={
                "type": "dpo",
                "dpo": {
                    "hyperparameters": hyperparams,
                },
            },
        )
    else:  # sft mode
        fine_tuning_job = client.fine_tuning.jobs.create(
            training_file=training_file,
            model=model,
            suffix=suffix,
            seed=seed,
            validation_file=validation_file,
            hyperparameters=hyperparams,
        )
    print(f"JOB ID: {fine_tuning_job.id}")
    print(fine_tuning_job.model_dump_json(indent=4))
    return fine_tuning_job


def list_fine_tuning_jobs(limit: int = 10) -> None:
    """
    List fine-tuning jobs.
    """
    job_list = client.fine_tuning.jobs.list(limit=limit)
    for job in job_list:
        print(job.model_dump_json(indent=4))


def get_fine_tuning_job_state(job_id: str) -> None:
    """
    Get the state of a fine-tuning job.
    """
    fine_tuning_job = client.fine_tuning.jobs.retrieve(job_id)
    print(fine_tuning_job.model_dump_json(indent=4))


def list_jobs_events(job_id: str, limit: int = 10) -> None:
    """
    List events for a fine-tuning job.
    """
    job_events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id, limit=limit
    )
    for job_event in job_events.data:
        print(job_event.model_dump_json(indent=4))


def retrieve_ft_metrics(metrics_file_id: str, show_plot: bool = True) -> pd.DataFrame:
    """
    Retrieve fine-tuning metrics from a file.
    """
    content = client.files.content(metrics_file_id)
    res = base64.b64decode(content.content).decode("utf-8")
    df = pd.read_csv(StringIO(res))
    if show_plot:
        x = "step"
        y = [c for c in df.columns if c != x]
        df.plot(x=x, y=y)
        plt.title("FT metrics")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.show()
    return df


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
