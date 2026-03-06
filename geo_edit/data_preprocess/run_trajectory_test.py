"""Run trajectory test on vLLM server and output results for evaluation.

This module reads a parquet dataset created by package_trajectory.py,
sends multi-turn conversations to a vLLM server, and saves results
in a format compatible with geo_edit.evaluation.openai_as_judge.

Usage:
    # 1. Start vLLM server
    bash geo_edit/scripts/launch_vllm_generate.sh

    # 2. Run test
    python -m geo_edit.data_preprocess.run_trajectory_test \
        --parquet_path /storage/.../trajectory_dataset.parquet \
        --output_path /storage/.../test_results/ \
        --api_base http://127.0.0.1:8000

    # 3. Evaluate with openai_as_judge
    python -m geo_edit.evaluation.openai_as_judge \
        --api_key $OPENAI_API_KEY \
        --result_path /storage/.../test_results/ \
        --output_path /storage/.../eval_results/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from openai import OpenAI
from PIL import Image

from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def _bytes_to_data_url(image_bytes: bytes) -> str:
    """Convert image bytes to data URL.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Data URL string like "data:image/png;base64,..."
    """
    # Detect image format
    image = Image.open(io.BytesIO(image_bytes))
    fmt = image.format or "PNG"

    # Encode to base64
    encoded = base64.b64encode(image_bytes).decode("ascii")
    mime = "image/jpeg" if fmt.upper() in {"JPG", "JPEG"} else "image/png"

    return f"data:{mime};base64,{encoded}"


def _inject_images_into_messages(
    messages: List[Dict[str, Any]],
    images: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Replace [IMAGE_N] placeholders with actual base64 data URLs.

    Args:
        messages: List of conversation messages with placeholders
        images: List of image dicts with "bytes" key from HuggingFace dataset

    Returns:
        Messages with image placeholders replaced by data URLs
    """
    # Build placeholder to data URL mapping
    image_map: Dict[str, str] = {}
    for idx, img_data in enumerate(images):
        placeholder = f"[IMAGE_{idx}]"
        img_bytes = img_data.get("bytes")
        if img_bytes:
            image_map[placeholder] = _bytes_to_data_url(img_bytes)

    # Replace placeholders in messages
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                        if url in image_map:
                            part["image_url"]["url"] = image_map[url]
                    elif isinstance(image_url_data, str) and image_url_data in image_map:
                        part["image_url"] = {"url": image_map[image_url_data]}

    return messages


def _extract_final_answer(output_text: str) -> str:
    """Extract final answer from model output.

    Looks for content within <answer>...</answer> tags.

    Args:
        output_text: Raw model output text

    Returns:
        Extracted answer or full output if no tags found
    """
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(output_text)
    if matches:
        return matches[-1].strip()
    return output_text.strip()


def _save_result(
    sample: Dict[str, Any],
    output_text: str,
    output_path: Path
) -> None:
    """Save result in format compatible with openai_as_judge.

    Creates a subfolder with meta_info.jsonl containing question, answer, and output_text.

    Args:
        sample: Dataset sample dict
        output_text: Model output text
        output_path: Base output directory
    """
    sample_id = sample["id"]
    sample_dir = output_path / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Load meta_info
    meta_info = json.loads(sample["meta_info"]) if sample["meta_info"] else {}

    # Build result record
    result = {
        "question": meta_info.get("question", ""),
        "answer": sample["answer"],
        "output_text": output_text,
        "image_path": meta_info.get("image_path", ""),
    }

    # Write to meta_info.jsonl
    meta_path = sample_dir / "meta_info.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_trajectory_test(
    parquet_path: Path,
    output_path: Path,
    api_base: str,
    model_name: str,
    api_mode: str = "chat_completions",
    max_tokens: int = 16384,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
) -> None:
    """Run trajectory test on vLLM server.

    Args:
        parquet_path: Path to parquet dataset
        output_path: Output directory for results
        api_base: vLLM server base URL
        model_name: Model name/path
        api_mode: API mode ("chat_completions" or "responses")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt to prepend
    """
    # Load dataset
    logger.info("Loading dataset from %s", parquet_path)
    ds = Dataset.from_parquet(str(parquet_path))
    logger.info("Loaded %d samples", len(ds))

    # Create OpenAI client for vLLM
    client = OpenAI(
        base_url=f"{api_base.rstrip('/')}/v1",
        api_key="none",
    )

    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = (
            "You are an advanced AI assistant capable of complex reasoning. "
            "Based on the conversation history, provide the final answer. "
            "Wrap your reasoning in <think>...</think> tags and your final answer in <answer>...</answer> tags."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    # Process each sample
    for idx, sample in enumerate(ds):
        sample_id = sample["id"]
        logger.info("Processing sample %d/%d: %s", idx + 1, len(ds), sample_id)

        try:
            # Parse messages from JSON string
            messages = json.loads(sample["messages"])

            # Inject images into messages
            images = sample["images"]
            messages = _inject_images_into_messages(messages, images)

            # Prepend system message
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            # Call vLLM
            if api_mode == "chat_completions":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                output_text = response.choices[0].message.content or ""
            else:
                # responses API
                response = client.responses.create(
                    model=model_name,
                    input=messages,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
                output_text = getattr(response, "output_text", "") or ""

            # Save result
            _save_result(sample, output_text, output_path)
            logger.info("Sample %s completed, output length: %d", sample_id, len(output_text))

        except Exception as e:
            logger.error("Error processing sample %s: %s", sample_id, e)
            # Save error result
            _save_result(sample, f"ERROR: {str(e)}", output_path)

    logger.info("Test completed. Results saved to %s", output_path)
    print(f"Test completed. Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trajectory test on vLLM server and output results for evaluation."
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Path to parquet dataset."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for results."
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8000",
        help="vLLM server base URL."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/storage/openpsi/models/Qwen3-VL-32B-Thinking",
        help="Model name/path."
    )
    parser.add_argument(
        "--api_mode",
        type=str,
        default="chat_completions",
        choices=["chat_completions", "responses"],
        help="API mode."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt."
    )
    args = parser.parse_args()

    run_trajectory_test(
        parquet_path=Path(args.parquet_path),
        output_path=Path(args.output_path),
        api_base=args.api_base,
        model_name=args.model_name,
        api_mode=args.api_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
