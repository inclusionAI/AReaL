"""Convert Terminal Bench tasks to RLLM/VERL format."""

import argparse
import json
from pathlib import Path

import pandas as pd
from terminal_bench_task import TBenchTrainingTask, load_terminal_bench_tasks
from tqdm import tqdm


def create_prompt_from_task(
    task: TBenchTrainingTask, system_prompt: str | None = None
) -> str:
    """Create a prompt from a terminal bench task."""
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant helping to complete terminal-based tasks. "
            "Follow the instructions carefully and use appropriate commands to accomplish the goal."
        )

    # Convert to string format (you might want to use a specific chat template)
    prompt = (
        f"<|system|>\n{system_prompt}\n<|user|>\n{task.instruction}\n<|assistant|>\n"
    )

    return prompt


def convert_tasks_to_parquet(
    tasks_dir: Path,
    output_dir: Path,
    train_split: float | None = None,
    system_prompt: str | None = None,
    task_names: list[str] | None = None,
    test_tasks_dir: Path | None = None,
) -> None:
    """Convert terminal bench tasks to parquet format for VERL training.

    Args:
        tasks_dir: Directory containing terminal bench tasks (or train tasks if test_tasks_dir is provided)
        output_dir: Output directory for parquet files
        train_split: Fraction of data for training (ignored if test_tasks_dir is provided)
        system_prompt: System prompt to use
        task_names: Specific task names to convert
        test_tasks_dir: Directory containing test tasks for validation set
    """

    # Load tasks
    if test_tasks_dir is not None:
        # Load train and test tasks separately
        print(f"Loading training tasks from {tasks_dir}")
        train_tasks = load_terminal_bench_tasks(tasks_dir, task_names)
        print(f"Loaded {len(train_tasks)} training tasks")

        print(f"Loading validation tasks from {test_tasks_dir}")
        val_tasks = load_terminal_bench_tasks(test_tasks_dir, task_names)
        print(f"Loaded {len(val_tasks)} validation tasks")

        tasks = train_tasks + val_tasks
    else:
        # Load all tasks from single directory
        print(f"Loading tasks from {tasks_dir}")
        tasks = load_terminal_bench_tasks(tasks_dir, task_names)
        print(f"Loaded {len(tasks)} tasks")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for parquet
    data_records = []

    for task in tqdm(tasks, desc="Converting tasks"):
        record = {
            "prompt": create_prompt_from_task(task, system_prompt),
            "task_name": task.task_name,
            "task_path": str(task.task_path),
            "instruction": task.instruction,
            "data_source": "terminal_bench",  # For reward_fn_key
            "metadata": {
                "test_weights": task.test_weights,
                "max_test_timeout_sec": task.max_test_timeout_sec,
            },
        }

        # Always include extra_info with task configuration
        extra_info_dict = {
            "task_name": task.task_name,
            "task_path": str(task.task_path),
            "instruction": task.instruction,
            "test_weights": task.test_weights,
            "dockerfile_contents": task.dockerfile_contents,
            "py_test_file_contents": task.py_test_file_contents,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        }

        # Include additional files if present
        if task.additional_files:
            extra_info_dict["additional_files"] = task.additional_files

        record["extra_info"] = json.dumps(extra_info_dict)

        data_records.append(record)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Split into train and validation
    if test_tasks_dir is not None:
        # Use pre-defined split based on directories
        n_train = len(train_tasks)
        train_df = df[:n_train]
        val_df = df[n_train:]
    else:
        # Use train_split parameter
        if train_split is None:
            train_split = 0.9
        n_train = int(len(df) * train_split)
        train_df = df[:n_train]
        val_df = df[n_train:]

    # Save to parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")

    # Also save task information for the reward function
    tasks_info = {}
    for task in tasks:
        task_info = {
            "task_path": str(task.task_path),
            "test_weights": task.test_weights,
            "dockerfile_contents": task.dockerfile_contents,
            "py_test_file_contents": task.py_test_file_contents,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        }
        # Include additional files if present
        if task.additional_files:
            task_info["additional_files"] = task.additional_files

        tasks_info[task.task_name] = task_info

    tasks_info_path = output_dir / "tasks_info.json"
    with open(tasks_info_path, "w") as f:
        json.dump(tasks_info, f, indent=2)

    print(f"Saved task information to {tasks_info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Terminal Bench tasks to RLLM/VERL format."
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="tasks",
        help="Directory containing training tasks (default: tasks)",
    )
    parser.add_argument(
        "--test-tasks-dir",
        type=str,
        default="",
        help="Directory containing test/validation tasks (default: empty)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for parquet files (default: data)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction of data for training (only used if --test-tasks-dir is not provided)",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="Custom system prompt to use"
    )
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help="Specific task names to convert (optional)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    tasks_dir = Path(args.tasks_dir)
    test_tasks_dir = (
        Path(args.test_tasks_dir)
        if args.test_tasks_dir and args.test_tasks_dir.strip()
        else None
    )
    output_dir = Path(args.output_dir)

    # Check if directories exist
    if not tasks_dir.exists():
        print(f"Error: {tasks_dir} directory not found")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if test_tasks_dir:
        print(
            f"Converting {tasks_dir}/ -> train.parquet and {test_tasks_dir}/ -> val.parquet"
        )
    else:
        print(f"Converting {tasks_dir}/ with train_split={args.train_split}")

    # Convert to parquet only (with extra_info)
    convert_tasks_to_parquet(
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        train_split=args.train_split,
        system_prompt=args.system_prompt,
        task_names=args.task_names,
        test_tasks_dir=test_tasks_dir,
    )


if __name__ == "__main__":
    main()
