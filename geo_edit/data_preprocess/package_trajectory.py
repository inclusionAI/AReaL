"""Package multi-turn trajectory data into HuggingFace parquet dataset.

This module reads trajectory.json files from subfolders and packages them into
a parquet dataset for testing tool call effectiveness.

Usage:
gpt-5_grounding_dino  gpt-5_label  gpt-5_ocr  gpt-5_ocr.parquet  gpt-5_only_crop  gpt-5_only_highlight  gpt-5_sam  gpt-5_test  gpt-5_testset
    python -m geo_edit.data_preprocess.package_trajectory \
        --data_dir /storage/openpsi/data/lcy_image_edit/CartoMapQA_output_0303/gpt-5_ocr/ \
        --out_path /storage/openpsi/data/lcy_image_edit/CartoMapQA_output_0303/gpt-5_ocr.parquet
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, Features, Image as HFImage, Sequence, Value

from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def _load_trajectory(trajectory_path: Path) -> List[Dict[str, Any]]:
    """Load trajectory.json file."""
    with trajectory_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_meta_info(meta_path: Path) -> Dict[str, Any]:
    """Load first record from meta_info.jsonl."""
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _extract_turns_except_last(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all turns except the last assistant turn containing the answer.

    The last turn is identified as:
    - role == "assistant"
    - contains "<answer>...</answer>" pattern
    """
    if not trajectory:
        return []

    last_turn = trajectory[-1]
    if (last_turn.get("role") == "assistant" and
        "<answer>" in str(last_turn.get("content", ""))):
        return trajectory[:-1]

    return trajectory


def _resolve_image_path(url: str, subfolder: Path) -> Optional[Path]:
    """Resolve image path from file:// URL.

    Args:
        url: File URL like "file:///storage/.../image.png"
        subfolder: Current subfolder path for relative resolution

    Returns:
        Resolved Path object or None if not resolvable
    """
    if not url.startswith("file://"):
        return None

    # Remove file:// prefix
    path_str = url[7:]

    # Handle Windows paths that might start with /C:/
    if path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
        path_str = path_str[1:]

    path = Path(path_str)

    if path.exists():
        return path

    # Try relative to subfolder
    relative_path = subfolder / path.name
    if relative_path.exists():
        return relative_path

    return None


def _extract_and_replace_images(
    trajectory: List[Dict[str, Any]],
    subfolder: Path
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """Extract images from trajectory and replace URLs with placeholders.

    Traverses messages, extracts file:// images, and replaces with [IMAGE_N] placeholders.

    Args:
        trajectory: List of conversation turns
        subfolder: Path to the subfolder containing images

    Returns:
        Tuple of (modified trajectory with placeholders, list of image bytes)
    """
    # Deep copy to avoid modifying original
    trajectory = copy.deepcopy(trajectory)
    images_bytes: List[bytes] = []
    image_idx = 0

    for turn in trajectory:
        content = turn.get("content", [])
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get("type") == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    else:
                        url = str(image_url_data)

                    if url.startswith("file://"):
                        image_path = _resolve_image_path(url, subfolder)
                        if image_path and image_path.exists():
                            images_bytes.append(image_path.read_bytes())
                            # Replace URL with placeholder
                            if isinstance(image_url_data, dict):
                                part["image_url"]["url"] = f"[IMAGE_{image_idx}]"
                            else:
                                part["image_url"] = {"url": f"[IMAGE_{image_idx}]"}
                            image_idx += 1
                        else:
                            logger.warning("Image not found: %s", url)

    return trajectory, images_bytes


def _count_tool_calls(trajectory: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """Count tool calls and extract tool names from trajectory.

    Returns:
        Tuple of (total tool call count, list of unique tool names)
    """
    count = 0
    names: List[str] = []

    for turn in trajectory:
        tool_calls = turn.get("tool_calls", [])
        for tc in tool_calls:
            count += 1
            func = tc.get("function", {})
            name = func.get("name", "")
            if name and name not in names:
                names.append(name)

    return count, names


def _process_subfolder(subfolder: Path) -> Optional[Dict[str, Any]]:
    """Process a single subfolder and return a dataset record.

    Args:
        subfolder: Path to subfolder containing trajectory.json and meta_info.jsonl

    Returns:
        Dataset record dict or None if processing failed
    """
    trajectory_path = subfolder / "trajectory.json"
    meta_path = subfolder / "meta_info.jsonl"

    if not trajectory_path.exists():
        logger.warning("Missing trajectory.json in %s", subfolder.name)
        return None

    if not meta_path.exists():
        logger.warning("Missing meta_info.jsonl in %s", subfolder.name)
        return None

    try:
        # Load data
        trajectory = _load_trajectory(trajectory_path)
        meta_info = _load_meta_info(meta_path)

        # Extract turns (excluding final answer)
        turns = _extract_turns_except_last(trajectory)

        # Extract and replace images
        processed_turns, images_bytes = _extract_and_replace_images(turns, subfolder)

        # Count tool calls
        num_tool_calls, tool_names = _count_tool_calls(turns)

        # Get answer from meta_info
        answer = meta_info.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(x) for x in answer)

        # Build record
        record = {
            "id": subfolder.name,
            "messages": json.dumps(processed_turns, ensure_ascii=False),
            "images": [{"bytes": b, "path": None} for b in images_bytes],
            "num_turns": len(processed_turns),
            "num_tool_calls": num_tool_calls,
            "tool_names": tool_names,
            "answer": str(answer),
            "meta_info": json.dumps(meta_info, ensure_ascii=False),
        }

        return record

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", subfolder.name, e)
        return None
    except Exception as e:
        logger.error("Error processing %s: %s", subfolder.name, e)
        return None


def package_trajectory_dataset(data_dir: Path, out_path: Path) -> None:
    """Package trajectory data from multiple subfolders into parquet dataset.

    Args:
        data_dir: Parent directory containing trajectory subfolders
        out_path: Output parquet file path
    """
    records: List[Dict[str, Any]] = []

    # Traverse all subfolders
    subfolders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info("Found %d subfolders in %s", len(subfolders), data_dir)

    for subfolder in subfolders:
        record = _process_subfolder(subfolder)
        if record:
            records.append(record)

    if not records:
        logger.error("No valid records found")
        return

    # Define features
    features = Features({
        "id": Value("string"),
        "messages": Value("string"),
        "images": Sequence(HFImage()),
        "num_turns": Value("int64"),
        "num_tool_calls": Value("int64"),
        "tool_names": Sequence(Value("string")),
        "answer": Value("string"),
        "meta_info": Value("string"),
    })

    # Create and save dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(records, features=features)
    ds.to_parquet(str(out_path))

    logger.info("Saved parquet: %s (%d records)", out_path, len(ds))
    print(f"Saved parquet: {out_path} ({len(ds)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package multi-turn trajectory data into HuggingFace parquet dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Parent directory containing trajectory subfolders."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output parquet file path."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out_path).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    package_trajectory_dataset(data_dir, out_path)


if __name__ == "__main__":
    main()
