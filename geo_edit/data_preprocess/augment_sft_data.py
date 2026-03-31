#!/usr/bin/env python3
"""Augment MapQA/ChartQA SFT data: diversify <think> expressions without increasing data volume."""

import json
import os
import re
import copy
import random
import argparse
from pathlib import Path


def fix_tool_names(text: str) -> str:
    return re.sub(r"functions\.(\w+)", r"\1", text)


TOOL_REASON_TEMPLATES = {
    "map_text_ocr": [
        "To answer this question, I need to extract text labels visible on the map. The map_text_ocr tool specializes in reading place names, venue names, and geographic labels from map images, which is exactly what's needed here.",
        "This question requires identifying specific locations or features on the map. I'll use map_text_ocr to extract all readable text from the map, including street names, landmark labels, and point-of-interest names.",
        "Since the answer depends on what locations are shown on this map, I should first extract all visible text. map_text_ocr is optimized for reading map-specific labels while filtering noise like route numbers.",
        "I need to read the text on this map to find the relevant information. Let me use map_text_ocr which is designed to accurately extract place names and labels from map imagery.",
        "The question refers to specific places on the map. To identify them, I'll apply map_text_ocr to pull out all location names, landmarks, and points of interest visible in the image.",
        "Looking at this map question, I need to identify what places and features are labeled. map_text_ocr will give me a structured extraction of all text elements on the map.",
        "To determine the answer from this map, I first need a comprehensive list of all labeled locations. map_text_ocr is the right tool as it focuses on geographic text recognition.",
        "Before I can reason about locations on this map, I need to know what's actually labeled there. Let me use map_text_ocr to read all visible place names and landmarks.",
    ],
    "text_spotting": [
        "This question requires knowing both the text content and its position on the map. text_spotting will give me text labels with their bounding box coordinates, allowing me to reason about spatial relationships between locations.",
        "I need to locate specific text on the map and understand where things are relative to each other. text_spotting provides both the detected text and its coordinates, which is essential for this spatial reasoning task.",
        "Since the answer involves understanding where things are positioned on the map, I'll use text_spotting to get both text content and location data for each label.",
        "To answer this question about relative positions or distances, I need text labels with their spatial coordinates. text_spotting is the ideal tool for this as it returns bounding boxes alongside detected text.",
        "The spatial arrangement of locations matters here. text_spotting will extract text along with precise positional information, enabling me to compare relative positions of different places.",
        "I need both the names and locations of features on this map. text_spotting provides coordinate-level text detection that helps me reason about proximity and spatial layout.",
    ],
    "text_ocr": [
        "I need a general text extraction from this image. text_ocr provides comprehensive optical character recognition that captures all visible text, suitable for detailed analysis.",
        "Let me perform a general OCR scan of this image to capture all readable text, including any details that might be relevant to answering this question.",
        "Using text_ocr for a thorough extraction of all text in the image, which will help me identify the specific information needed for this question.",
        "A general text extraction will help gather all readable content from this image. text_ocr handles various text styles and sizes effectively.",
    ],
    "grounding_dino": [
        "I need to detect and locate specific objects or features in this image. grounding_dino performs open-vocabulary object detection, perfect for finding particular elements mentioned in the question.",
        "The question asks about specific visual elements that need to be detected and located. grounding_dino can identify objects based on natural language descriptions.",
        "To find the specific feature mentioned in the question, I'll use grounding_dino which excels at locating objects described in natural language within images.",
    ],
    "image_crop": [
        "I need to focus on a specific region of the map for closer analysis. image_crop will isolate the area of interest, allowing more detailed examination.",
        "To get better detail on the relevant part of this map, I'll crop the specific region using image_crop for closer inspection.",
        "Let me zoom into the relevant area by cropping the image. This will help me see finer details that might be obscured at the full map scale.",
    ],
    "auto_segment": [
        "I need to segment different regions in this image to understand the spatial layout better. auto_segment will identify and separate distinct areas for analysis.",
        "Using auto_segment to decompose the image into meaningful regions, which will help me identify and analyze specific areas relevant to the question.",
    ],
    "image_highlight": [
        "I'll highlight specific detected regions to better visualize and confirm the relevant areas on the map before answering.",
        "Let me use image_highlight to mark the relevant regions, making it easier to verify the spatial relationships in this map.",
    ],
    "chart_data_extract": [
        "I need to extract data points and values from this chart. chart_data_extract specializes in reading numerical data, labels, and legend entries from chart images.",
        "The question requires understanding the data presented in this chart. I'll use chart_data_extract to pull out all data points, axis labels, and values.",
        "To answer this chart-based question, I first need structured data extraction. chart_data_extract will identify data series, labels, and values from the chart.",
    ],
}

FINAL_ANSWER_STARTERS = [
    "Based on the extracted information, ",
    "From the OCR results, ",
    "Looking at the tool output, ",
    "The extracted text reveals that ",
    "Analyzing the results, ",
    "According to the map data, ",
    "The tool results show that ",
    "Having reviewed the extracted content, ",
    "The analysis indicates that ",
    "From the available evidence, ",
]


def diversify_think_tool_selection(think_text: str) -> str:
    tool_match = re.search(r"Tool:\s*(functions\.)?(\w+)", think_text)
    if not tool_match:
        return think_text

    tool_name = tool_match.group(2)
    templates = TOOL_REASON_TEMPLATES.get(tool_name, None)
    if not templates:
        return fix_tool_names(think_text)

    new_reason = random.choice(templates)
    return f"Tool: {tool_name}\nReason: {new_reason}"


def diversify_final_think(think_text: str) -> str:
    if "Tool:" in think_text:
        return think_text

    if random.random() < 0.3:
        starter = random.choice(FINAL_ANSWER_STARTERS)
        for prefix in ["The ", "Based on ", "From the ", "Looking at "]:
            if think_text.startswith(prefix):
                think_text = (
                    think_text[len(prefix)].lower() + think_text[len(prefix) + 1 :]
                )
                break
        think_text = starter + think_text

    return think_text


def augment_example(example: dict) -> dict:
    augmented = copy.deepcopy(example)

    for conv in augmented["conversations"]:
        if conv["from"] == "gpt":
            conv["value"] = fix_tool_names(conv["value"])

            def replace_think(match):
                think_content = match.group(1)
                if "Tool:" in think_content:
                    new_content = diversify_think_tool_selection(think_content)
                else:
                    new_content = diversify_final_think(think_content)
                return f"<think>{new_content}</think>"

            conv["value"] = re.sub(
                r"<think>(.*?)</think>", replace_think, conv["value"], flags=re.DOTALL
            )

    return augmented


def augment_dataset_rulebased(data: list) -> list:
    """Diversify expressions in-place. Output size == input size."""
    return [augment_example(example) for example in data]


def augment_with_llm(
    data: list,
    api_base: str = "https://api.minimax.chat/v1",
    model: str = "MiniMax-M1-40k",
    api_key: str = None,
) -> list:
    """Rephrase <think> blocks in-place via LLM. No data volume increase."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not available, skipping LLM augmentation")
        return data

    api_key = (
        api_key
        or os.environ.get("MINIMAX_API_KEY")
        or os.environ.get("LLM_API_KEY", "dummy")
    )
    client = OpenAI(base_url=api_base, api_key=api_key)

    failed = 0
    for i, example in enumerate(data):
        for conv in example["conversations"]:
            if conv["from"] != "gpt":
                continue

            thinks = re.findall(r"<think>(.*?)</think>", conv["value"], re.DOTALL)
            for think_text in thinks:
                if len(think_text) < 20:
                    continue

                prompt = (
                    "Rephrase the following reasoning text to be more diverse and natural "
                    "while preserving the same logical content and conclusions. "
                    "Keep the same structure (if it mentions a Tool, keep the tool name). "
                    "Output ONLY the rephrased text, nothing else.\n\n"
                    f"Original:\n{think_text}\n\nRephrased:"
                )

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.8,
                    )
                    new_think = response.choices[0].message.content.strip()
                    if new_think and len(new_think) > 10:
                        conv["value"] = conv["value"].replace(
                            f"<think>{think_text}</think>",
                            f"<think>{new_think}</think>",
                            1,
                        )
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"LLM rephrase failed for example {i}: {e}")
                    continue

        if (i + 1) % 50 == 0:
            print(f"  LLM rephrased {i + 1}/{len(data)} examples ({failed} failures)")

    print(f"  LLM rephrase done: {len(data)} examples, {failed} failures")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Augment SFT data: diversify <think> expressions without increasing data volume"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input train.json (sharegpt format)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output train.json",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to rephrase <think> blocks (after rule-based diversification)",
    )
    parser.add_argument(
        "--api-base",
        default="https://api.minimax.chat/v1",
        help="LLM API base URL (default: Minimax)",
    )
    parser.add_argument(
        "--model",
        default="MiniMax-M1-40k",
        help="LLM model name (default: MiniMax-M1-40k)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key (or set MINIMAX_API_KEY / LLM_API_KEY env var)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading data from {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    print("Running rule-based diversification...")
    augmented = augment_dataset_rulebased(data)
    print(f"Diversified {len(augmented)} examples (same count as input)")

    if args.use_llm:
        print(f"Running LLM rephrase via {args.api_base} model={args.model}...")
        augmented = augment_with_llm(
            augmented,
            api_base=args.api_base,
            model=args.model,
            api_key=args.api_key,
        )

    random.shuffle(augmented)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(augmented)} examples to {args.output}")


if __name__ == "__main__":
    main()
