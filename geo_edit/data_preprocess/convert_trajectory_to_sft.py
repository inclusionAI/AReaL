#!/usr/bin/env python3
"""Convert trajectories to LLaMA Factory SFT (ShareGPT) format.

Consumes the output of ``augment_traj_data.py`` (which already filters and
diversifies) and converts each trajectory into step-level ShareGPT samples.

Usage:
    python -m geo_edit.data_preprocess.convert_trajectory_to_sft \
        --src_dir /storage/openpsi/data/lcy_image_edit/reasonmap_plus_sft_selected \
        --dst_dir /storage/openpsi/data/lcy_image_edit/reasonmap_plus_sft_selected_data 
"""

import argparse
import copy
import glob
import json
import os
import re
import shutil
from collections import Counter

from geo_edit.data_preprocess.trajectory_utils import get_text_from_content
from geo_edit.prompts.system_prompts import (
    build_tool_system_prompt,
    build_user_message,
    DATASET_TASK_TYPES,
    DEFAULT_OUTPUT_FORMAT,
)
from geo_edit.tool_definitions import ToolRouter, format_tool_declarations_text


def has_image_in_content(content):
    """Check if the content contains an image_url part."""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def get_image_url_from_content(content):
    """Extract image URL(s) from content."""
    urls = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                urls.append(url)
    return urls


def _get_assistant_text(content) -> str:
    """Robustly extract text from assistant message content.

    Handles both string content (chat_completions mode) and list-of-parts
    content (responses mode) so that downstream regex extraction works
    regardless of the API mode used during trajectory generation.
    """
    if isinstance(content, str):
        return content
    return get_text_from_content(content)


def convert_trajectory(
    trajectory, src_dir, dst_images_dir, task_id, system_prompt, tool_definitions_text,
    data_source="chartqa",
):
    """Convert a single trajectory to step-level ShareGPT samples.

    Each gpt turn produces one training sample containing all context up to
    (and including) that turn.  A 3-step trajectory yields 3 samples.

    Returns a list of dicts (each with keys: conversations, images, system),
    or an empty list if conversion fails.
    """
    conversations = []
    images = []  # relative paths for the output
    output_image_counter = 0

    # Extract input images from the first user message in the trajectory.
    # Supports both file:// URLs (shared images) and inline image_url references.
    input_image_paths = []
    first_msg = trajectory[0] if trajectory else {}
    if first_msg.get("role") == "user":
        urls = get_image_url_from_content(first_msg.get("content", []))
        for url in urls:
            path = url.replace("file://", "") if url.startswith("file://") else url
            if os.path.exists(path):
                input_image_paths.append(path)

    for idx, input_src in enumerate(input_image_paths):
        basename = os.path.basename(input_src)
        dst_path = os.path.join(dst_images_dir, basename)
        # Deduplicate: only copy if not already present (shared images like dubai.png)
        if not os.path.exists(dst_path):
            shutil.copy2(input_src, dst_path)
        images.append(f"images/{basename}")

    observation_counter = len(input_image_paths)

    i = 0
    n = len(trajectory)

    # --- First user message (with image + question) ---
    if i >= n or trajectory[i].get("role") != "user":
        return None

    first_user = trajectory[i]
    question_text = get_text_from_content(first_user["content"])
    # Strip all "Observation N:" prefixes from the extracted text
    question_text_clean = re.sub(r"Observation\s+\d+:\s*\n?", "", question_text).strip()
    # Remove duplicate "Question:" prefix (original dataset may already contain it)
    question_text_clean = re.sub(r"^(Question:\s*)+", "", question_text_clean).strip()

    # Build first user message using unified prompt template
    task_type = DATASET_TASK_TYPES.get(data_source, "visual question answering")
    first_user_value = build_user_message(
        question=question_text_clean,
        num_images=len(input_image_paths),
        task_type=task_type,
        output_format=DEFAULT_OUTPUT_FORMAT,
    )
    conversations.append({"from": "human", "value": first_user_value})
    i += 1

    # --- Process remaining messages ---
    while i < n:
        msg = trajectory[i]
        role = msg.get("role")

        if role == "assistant":
            # Collect consecutive assistant messages (think + tool_calls or think + answer)
            think_content = ""
            tool_call_json = None
            answer_content = None

            assistant_texts = []
            while i < n and trajectory[i].get("role") == "assistant":
                amsg = trajectory[i]
                content = amsg.get("content", "")
                tool_calls = amsg.get("tool_calls")

                if tool_calls:
                    tc = tool_calls[0]
                    func = tc.get("function", {})
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        tool_call_json = {
                            "name": func.get("name", ""),
                            "arguments": json.loads(raw_args),
                        }
                    else:
                        tool_call_json = {
                            "name": func.get("name", ""),
                            "arguments": raw_args,
                        }
                elif content:
                    text = _get_assistant_text(content)
                    if text:
                        assistant_texts.append(text)

                i += 1

            merged_text = "\n".join(assistant_texts)

            # Extract the last think block (most refined reasoning)
            think_blocks = re.findall(r"<think>(.*?)</think>", merged_text, re.DOTALL)
            if think_blocks:
                think_content = think_blocks[-1].strip()

            answer_match = re.search(r"<answer>(.*?)</answer>", merged_text, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()

            if tool_call_json is not None:
                value = f"<think>{think_content}</think>\n<action>{json.dumps(tool_call_json)}</action>"
                conversations.append({"from": "gpt", "value": value})
            elif answer_content is not None:
                value = (
                    f"<think>{think_content}</think>\n<answer>{answer_content}</answer>"
                )
                conversations.append({"from": "gpt", "value": value})
            else:
                if merged_text.strip():
                    conversations.append({"from": "gpt", "value": merged_text.strip()})

        elif role == "tool":
            # Tool response - collect tool content and subsequent user messages
            tool_content = msg.get("content", "")
            i += 1

            # Check if tool produced an image
            tool_produced_image = False
            try:
                tool_data = json.loads(tool_content)
                if "image_ref" in tool_data:
                    tool_produced_image = True
            except (json.JSONDecodeError, TypeError):
                pass

            # Collect subsequent user messages (feedback + possible new image)
            user_texts = []
            user_has_image = False
            image_url = None

            while i < n and trajectory[i].get("role") == "user":
                umsg = trajectory[i]
                ucontent = umsg["content"]

                if has_image_in_content(ucontent):
                    user_has_image = True
                    urls = get_image_url_from_content(ucontent)
                    if urls:
                        image_url = urls[0]
                    utext = get_text_from_content(ucontent)
                    utext = re.sub(r"^Observation\s+\d+:\s*$", "", utext).strip()
                    if utext:
                        user_texts.append(utext)
                else:
                    utext = get_text_from_content(ucontent)
                    if utext:
                        user_texts.append(utext)

                i += 1

            # Build the human message
            if tool_produced_image or user_has_image:
                # Copy the output image
                output_image_counter += 1
                img_filename = None
                if image_url:
                    img_filename = os.path.basename(image_url.replace("file://", ""))
                    img_src = os.path.join(src_dir, "images", img_filename)
                else:
                    try:
                        refs = json.loads(tool_content).get("image_ref", {})
                        for obs_key, fname in refs.items():
                            img_filename = fname
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
                    img_src = (
                        os.path.join(src_dir, "images", img_filename)
                        if img_filename
                        else None
                    )

                if img_src and os.path.exists(img_src):
                    ext = os.path.splitext(img_filename)[1] if img_filename else ".jpg"
                    dst_name = f"{task_id}_output_{output_image_counter}{ext}"
                    dst_path = os.path.join(dst_images_dir, dst_name)
                    shutil.copy2(img_src, dst_path)
                    images.append(f"images/{dst_name}")

                    feedback = "\n".join(user_texts).strip()
                    value = f"Tool executed successfully. New image produced.\nObservation {observation_counter}:\n<image>"
                    observation_counter += 1
                    if feedback:
                        value += f"\n{feedback}"
                    conversations.append({"from": "human", "value": value})
                else:
                    # Image file not found, fall back to text-only
                    result_text = tool_content
                    try:
                        parsed = json.loads(tool_content)
                        if "analysis" in parsed:
                            result_text = parsed["analysis"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                    feedback = "\n".join(user_texts).strip()
                    value = f"Tool executed successfully.\nResult: {result_text}"
                    if feedback:
                        value += f"\n{feedback}"
                    conversations.append({"from": "human", "value": value})
            else:
                # Text-only tool result
                result_text = tool_content
                try:
                    parsed = json.loads(tool_content)
                    if "analysis" in parsed:
                        result_text = parsed["analysis"]
                except (json.JSONDecodeError, TypeError):
                    pass
                feedback = "\n".join(user_texts).strip()
                value = f"Tool executed successfully.\nResult: {result_text}"
                if feedback:
                    value += f"\n{feedback}"
                conversations.append({"from": "human", "value": value})

        elif role == "user":
            # Standalone user message not following a tool response
            i += 1
        else:
            i += 1

    # Validate: must have at least 2 turns and end with gpt
    if len(conversations) < 2:
        return []

    # Fix consecutive same-role messages by merging
    fixed = []
    for turn in conversations:
        if fixed and fixed[-1]["from"] == turn["from"]:
            fixed[-1]["value"] += "\n" + turn["value"]
        else:
            fixed.append(turn)
    conversations = fixed

    if conversations[0]["from"] != "human":
        return []

    # Step-level split: one sample per gpt turn
    gpt_indices = [j for j, turn in enumerate(conversations) if turn["from"] == "gpt"]
    if not gpt_indices:
        return []

    results = []
    for gpt_idx in gpt_indices:
        sub_conv = copy.deepcopy(conversations[: gpt_idx + 1])
        # Count <image> tags in truncated conversation to determine needed images
        image_count = sum(turn["value"].count("<image>") for turn in sub_conv)
        sub_images = images[:image_count]
        results.append(
            {
                "conversations": sub_conv,
                "images": sub_images,
                "system": system_prompt,
            }
        )

    return results


def _discover_subdirs(src_root: str) -> list:
    """Discover task subdirectories containing trajectory.json + meta_info.jsonl.

    Accepts ANY directory name (not limited to numeric IDs) and also handles
    nested multi-trajectory layouts (task_id/traj_N/).
    """
    subdirs = []
    for name in sorted(os.listdir(src_root)):
        subdir = os.path.join(src_root, name)
        if not os.path.isdir(subdir):
            continue

        traj_path = os.path.join(subdir, "trajectory.json")
        meta_path = os.path.join(subdir, "meta_info.jsonl")

        if os.path.exists(traj_path) and os.path.exists(meta_path):
            # Direct layout: src_root/{task_id}/trajectory.json
            subdirs.append((name, subdir))
        else:
            # Check for nested multi-trajectory layout: src_root/{task_id}/traj_N/
            for child_name in sorted(os.listdir(subdir)):
                child_dir = os.path.join(subdir, child_name)
                if not os.path.isdir(child_dir):
                    continue
                child_traj = os.path.join(child_dir, "trajectory.json")
                child_meta = os.path.join(child_dir, "meta_info.jsonl")
                if os.path.exists(child_traj) and os.path.exists(child_meta):
                    composite_id = f"{name}_{child_name}"
                    subdirs.append((composite_id, child_dir))

    return subdirs


def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectories to LLaMA Factory SFT (ShareGPT) format"
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Source directory containing trajectory subdirectories",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Output directory for SFT data",
    )
    parser.add_argument(
        "--enable_tools",
        type=str,
        nargs="+",
        default=None,
        help="Override enabled tools (default: read from config.yaml)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="trajectory_sft",
        help="Dataset name used in dataset_info.json (default: trajectory_sft)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="chartqa",
        help="Data source name for task_type lookup (default: chartqa)",
    )
    args = parser.parse_args()

    src_root = args.src_dir
    dst_root = args.dst_dir
    dst_images = os.path.join(dst_root, "images")
    os.makedirs(dst_images, exist_ok=True)

    # Build system prompt and tool definitions matching RL (verl-agent) format
    tool_router = ToolRouter(
        tool_mode="auto",
        enable_tools=args.enable_tools,
        skip_agent_init=True,
    )
    declarations = tool_router.get_available_declarations()
    tool_definitions_text = format_tool_declarations_text(declarations)
    system_prompt = build_tool_system_prompt(tool_definitions_text)

    print(f"Enabled tools: {[d['name'] for d in declarations]}")

    # Discover all valid task subdirectories (any name, flat or nested layout)
    subdirs = _discover_subdirs(src_root)

    print(f"Found {len(subdirs)} task directories")

    results = []
    stats = Counter()

    for task_id, subdir in subdirs:
        traj_path = os.path.join(subdir, "trajectory.json")

        if not os.path.exists(traj_path):
            stats["missing_files"] += 1
            continue

        try:
            with open(traj_path) as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            stats["bad_trajectory"] += 1
            print(f"  [WARN] Task {task_id}: failed to load trajectory: {e}")
            continue

        entries = convert_trajectory(
            trajectory,
            subdir,
            dst_images,
            task_id,
            system_prompt,
            tool_definitions_text,
            data_source=args.data_source,
        )
        if not entries:
            stats["conversion_failed"] += 1
            continue

        results.extend(entries)
        stats["kept"] += 1
        stats["total_steps"] += len(entries)

    # Save train.json
    train_path = os.path.join(dst_root, "train.json")
    with open(train_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save dataset_info.json (LLaMA-Factory compatible)
    dataset_name = args.dataset_name
    dataset_info = {
        dataset_name: {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    info_path = os.path.join(dst_root, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Conversion Summary")
    print(f"{'=' * 50}")
    print(f"Total task directories:  {len(subdirs)}")
    print(f"Kept (trajectories):     {stats['kept']}")
    print(f"Step-level samples:      {stats['total_steps']}")
    if stats["kept"] > 0:
        print(f"Avg steps/trajectory:    {stats['total_steps'] / stats['kept']:.1f}")
    print(f"Skipped - missing files: {stats['missing_files']}")
    print(f"Skipped - bad traj:      {stats['bad_trajectory']}")
    print(f"Skipped - conv failed:   {stats['conversion_failed']}")
    print(f"\nOutput: {train_path}")
    print(f"Images: {dst_images}")
    print(f"Dataset info: {info_path}")

    # Spot check
    if results:
        print(f"\n{'=' * 50}")
        print("Spot Check: First entry")
        print(f"{'=' * 50}")
        first = results[0]
        print(f"  Num turns: {len(first['conversations'])}")
        print(f"  Num images: {len(first['images'])}")
        print(f"  Images: {first['images']}")
        for j, turn in enumerate(first["conversations"]):
            preview = turn["value"][:120].replace("\n", "\\n")
            print(f"  Turn {j} [{turn['from']}]: {preview}...")


if __name__ == "__main__":
    main()
