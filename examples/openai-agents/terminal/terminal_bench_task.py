import json
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel
from tqdm import tqdm


class TBenchTaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNRATED = "unrated"


class TBenchTrainingTask(BaseModel):
    """Data model for a task which follows the same format as terminal-bench."""

    task_name: str
    task_path: Path
    instruction: str
    difficulty: TBenchTaskDifficulty
    test_weights: dict
    dockerfile_contents: str
    py_test_file_contents: str
    max_test_timeout_sec: int = 300  # Default timeout
    additional_files: dict | None = None  # Maps file paths to contents


def load_terminal_bench_tasks(
    tasks_dir: Path,
    task_names: list[str] | None = None,
) -> list[TBenchTrainingTask]:
    if task_names is None:
        task_names = [p.name for p in tasks_dir.iterdir() if p.is_dir()]

    tasks = []
    for task_name in tqdm(task_names, desc="Loading tasks"):
        task_path = tasks_dir / task_name
        task_yaml = task_path / "task.yaml"

        if not task_yaml.exists():
            tqdm.write(f"Task YAML file not found: {task_yaml}")
            continue

        with open(task_yaml, encoding="utf-8") as f:
            task_data = yaml.safe_load(f)

        instruction = task_data.get("instruction")
        if not instruction:
            tqdm.write(f"Instruction not found in task YAML: {task_yaml}")
            continue

        # Get max test timeout if specified
        max_test_timeout_sec = task_data.get("max_test_timeout_sec", 300)
        difficulty = task_data.get("difficulty", TBenchTaskDifficulty.UNRATED)

        # Load test weights
        test_weights_path = task_path / "test_weights.json"
        if test_weights_path.exists():
            with open(test_weights_path, encoding="utf-8") as f:
                test_weights = json.load(f)
        else:
            # Use placeholder to avoid PyArrow empty struct error
            test_weights = {"_no_weights": 1.0}

        # Load Dockerfile
        dockerfile_path = task_path / "Dockerfile"
        if not dockerfile_path.exists():
            tqdm.write(f"Dockerfile not found for task: {task_name}")
            continue
        with open(dockerfile_path, encoding="utf-8") as f:
            dockerfile_contents = f.read()
        if not dockerfile_contents:
            tqdm.write(f"Dockerfile is empty for task: {task_name}")
            continue

        # Load Python test file if it exists
        py_test_file_path = task_path / "tests" / "test_outputs.py"
        if not py_test_file_path.exists():
            tqdm.write(f"Python test file not found for task: {task_name}")
            continue
        with open(py_test_file_path, encoding="utf-8") as f:
            py_test_file_contents = f.read()
        if not py_test_file_contents:
            tqdm.write(f"Python test file is empty for task: {task_name}")
            continue

        # Load additional files if they exist
        additional_files = {}
        # List all files in the task directory (excluding standard files)
        standard_files = {"Dockerfile", "task.yaml", "test_weights.json"}
        standard_dirs = {"tests", "__pycache__"}

        for item in task_path.iterdir():
            if item.is_file() and item.name not in standard_files:
                # Read the file and store with relative path
                rel_path = item.relative_to(task_path)
                try:
                    with open(item, encoding="utf-8") as f:
                        additional_files[str(rel_path)] = f.read()
                except UnicodeDecodeError:
                    tqdm.write(f"Binary file found for task: {task_name}")
            elif item.is_dir() and item.name not in standard_dirs:
                # Recursively read files from subdirectories
                for subfile in item.rglob("*"):
                    if subfile.is_file():
                        rel_path = subfile.relative_to(task_path)
                        try:
                            with open(subfile, encoding="utf-8") as f:
                                additional_files[str(rel_path)] = f.read()
                        except UnicodeDecodeError:
                            # Skip binary files for now
                            tqdm.write(f"Binary file found for task: {task_name}")

        tasks.append(
            TBenchTrainingTask(
                task_name=task_name,
                task_path=task_path,
                instruction=instruction,
                difficulty=difficulty,
                test_weights=test_weights,
                dockerfile_contents=dockerfile_contents,
                py_test_file_contents=py_test_file_contents,
                max_test_timeout_sec=max_test_timeout_sec,
                additional_files=additional_files if additional_files else None,
            )
        )

    return tasks
