#!/usr/bin/env python3
"""Convert MapQA iterative trajectories to LLaMA Factory SFT (ShareGPT) format."""

import json
import os
import re
import shutil
import argparse
from collections import Counter
from pathlib import Path

from geo_edit.data_preprocess.trajectory_utils import (
    extract_answer_values,
    get_text_from_content,
    is_brute_force,
    load_meta_info,
)
from geo_edit.prompts.system_prompts import TOOL_CALL_SYSTEM_PROMPT
from geo_edit.tool_definitions import ToolRouter, format_tool_declarations_text


def is_correct(meta):
    """Check if the model's output matches the expected answer."""
    if meta is None:
        return False
    output = str(meta.get("output_text", "")).strip()
    answer = str(meta.get("answer", "")).strip()
    return output == answer


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


def convert_trajectory(trajectory, meta, src_dir, dst_images_dir, task_id,
                       system_prompt, tool_definitions_text):
    """Convert a single trajectory to ShareGPT format.

    Returns a dict with keys: conversations, images, system
    or None if conversion fails.
    """
    conversations = []
    images = []  # relative paths for the output
    output_image_counter = 0

    # Copy input image
    input_src = os.path.join(src_dir, "input_image.png")
    if not os.path.exists(input_src):
        return None
    input_dst_name = f"{task_id}_input.png"
    input_dst = os.path.join(dst_images_dir, input_dst_name)
    shutil.copy2(input_src, input_dst)
    images.append(f"images/{input_dst_name}")

    i = 0
    n = len(trajectory)

    # --- First user message (with image + question) ---
    if i >= n or trajectory[i].get("role") != "user":
        return None

    first_user = trajectory[i]
    question_text = get_text_from_content(first_user["content"])
    # Remove "Observation 0:" prefix
    question_text = re.sub(r"^Observation\s+\d+:\s*\n?", "", question_text).strip()

    first_user_value = (
        f"Available tools:\n{tool_definitions_text}\n\n"
        f"Use this format for tool calls:\n"
        f'<action>{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}</action>\n\n'
        f"When you have the final answer:\n"
        f"<answer>your answer here</answer>\n\n"
        f"Task: {question_text}\n"
        f"<image>"
    )
    conversations.append({
        "from": "human",
        "value": first_user_value
    })
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
                    tool_call_json = {
                        "name": func.get("name", ""),
                        "arguments": json.loads(func.get("arguments", "{}"))
                    }
                elif content:
                    assistant_texts.append(content)

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
                value = f"<think>{think_content}</think>\n<answer>{answer_content}</answer>"
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
                    img_src = os.path.join(src_dir, "images", img_filename) if img_filename else None

                if img_src and os.path.exists(img_src):
                    ext = os.path.splitext(img_filename)[1] if img_filename else ".jpg"
                    dst_name = f"{task_id}_output_{output_image_counter}{ext}"
                    dst_path = os.path.join(dst_images_dir, dst_name)
                    shutil.copy2(img_src, dst_path)
                    images.append(f"images/{dst_name}")

                    feedback = "\n".join(user_texts).strip()
                    value = "Tool executed successfully. New image produced.\n<image>"
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
        return None

    # Fix consecutive same-role messages by merging
    fixed = []
    for turn in conversations:
        if fixed and fixed[-1]["from"] == turn["from"]:
            fixed[-1]["value"] += "\n" + turn["value"]
        else:
            fixed.append(turn)
    conversations = fixed

    if conversations[-1]["from"] != "gpt":
        return None
    if conversations[0]["from"] != "human":
        return None

    return {
        "conversations": conversations,
        "images": images,
        "system": system_prompt
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MapQA trajectories to SFT format")
    parser.add_argument(
        "--src_dir",
        default="/storage/openpsi/data/lcy_image_edit/mapqa_iterative_0330",
        help="Source directory containing trajectory subdirectories",
    )
    parser.add_argument(
        "--dst_dir",
        default="/storage/openpsi/data/lcy_image_edit/mapqa_sft_0330",
        help="Output directory for SFT data",
    )
    parser.add_argument(
        "--enable_tools",
        type=str,
        nargs="+",
        default=None,
        help="Override enabled tools (default: read from config.yaml)",
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
    system_prompt = TOOL_CALL_SYSTEM_PROMPT.strip()

    print(f"Enabled tools: {[d['name'] for d in declarations]}")

    # Collect all numbered subdirectories
    subdirs = []
    for name in os.listdir(src_root):
        subdir = os.path.join(src_root, name)
        if os.path.isdir(subdir) and name.isdigit():
            subdirs.append((int(name), subdir))
    subdirs.sort()

    print(f"Found {len(subdirs)} task directories")

    results = []
    stats = Counter()

    for task_id, subdir in subdirs:
        meta_path = os.path.join(subdir, "meta_info.jsonl")
        traj_path = os.path.join(subdir, "trajectory.json")

        if not os.path.exists(meta_path) or not os.path.exists(traj_path):
            stats["missing_files"] += 1
            continue

        meta = load_meta_info(Path(meta_path))
        if not meta:
            stats["bad_meta"] += 1
            continue

        if not is_correct(meta):
            stats["incorrect"] += 1
            continue

        try:
            with open(traj_path, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            stats["bad_trajectory"] += 1
            print(f"  [WARN] Task {task_id}: failed to load trajectory: {e}")
            continue

        if is_brute_force(trajectory, meta):
            stats["brute_force"] += 1
            continue

        entry = convert_trajectory(trajectory, meta, subdir, dst_images, task_id,
                                   system_prompt, tool_definitions_text)
        if entry is None:
            stats["conversion_failed"] += 1
            continue

        results.append(entry)
        stats["kept"] += 1

    # Save train.json
    train_path = os.path.join(dst_root, "train.json")
    with open(train_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save dataset_info.json
    dataset_info = {
        "mapqa_sft": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system"
            }
        }
    }
    info_path = os.path.join(dst_root, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Conversion Summary")
    print(f"{'='*50}")
    print(f"Total task directories:  {len(subdirs)}")
    print(f"Kept (converted):        {stats['kept']}")
    print(f"Filtered - incorrect:    {stats['incorrect']}")
    print(f"Filtered - brute force:  {stats['brute_force']}")
    print(f"Filtered - missing files:{stats['missing_files']}")
    print(f"Filtered - bad meta:     {stats['bad_meta']}")
    print(f"Filtered - bad traj:     {stats['bad_trajectory']}")
    print(f"Filtered - conv failed:  {stats['conversion_failed']}")
    print(f"\nOutput: {train_path}")
    print(f"Images: {dst_images}")
    print(f"Dataset info: {info_path}")

    # Spot check
    if results:
        print(f"\n{'='*50}")
        print(f"Spot Check: First entry")
        print(f"{'='*50}")
        first = results[0]
        print(f"  Num turns: {len(first['conversations'])}")
        print(f"  Num images: {len(first['images'])}")
        print(f"  Images: {first['images']}")
        for j, turn in enumerate(first['conversations']):
            preview = turn['value'][:120].replace('\n', '\\n')
            print(f"  Turn {j} [{turn['from']}]: {preview}...")


if __name__ == "__main__":
    main()
