"""Run trajectory test on vLLM server and output results for evaluation.

This module reads a parquet dataset created by package_trajectory.py,
sends multi-turn conversations to a vLLM server, and saves results
in a format compatible with geo_edit.evaluation.openai_as_judge.

Usage:
    # 1. Start vLLM server
    bash geo_edit/scripts/launch_vllm_generate.sh

    # 2. Run test
    python -m geo_edit.scripts.run_trajectory_test \
        --parquet_path /storage/openpsi/data/lcy_image_edit/CartoMapQA_output_0303/gpt-5_ocr.parquet \
        --output_path /storage/openpsi/data/lcy_image_edit/CartoMapQA_output_0303/Qwen3-VL-8B-Thinking/gpt-5_ocr \
        --model_name /storage/openpsi/models/Qwen3-VL-8B-Thinking \
        --api_base http://127.0.0.1:8000 \
        --num_workers 8

    # 3. Evaluate with openai_as_judge
    python -m geo_edit.evaluation.openai_as_judge \
        --api_key $OPENAI_API_KEY \
        --result_path /storage/.../test_results/ \
        --output_path /storage/.../eval_results/
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from openai import OpenAI
from PIL import Image

from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def _pil_image_to_data_url(image: Image.Image) -> str:
    """Convert PIL Image to data URL.

    Args:
        image: PIL Image object

    Returns:
        Data URL string like "data:image/png;base64,..."
    """
    # Save to buffer
    buffer = io.BytesIO()
    fmt = image.format or "PNG"
    save_fmt = "JPEG" if fmt.upper() in {"JPG", "JPEG"} else "PNG"
    image.save(buffer, format=save_fmt)

    # Encode to base64
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if save_fmt == "JPEG" else "image/png"

    return f"data:{mime};base64,{encoded}"


def _bytes_to_data_url(image_bytes: bytes) -> str:
    """Convert image bytes to data URL.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Data URL string like "data:image/png;base64,..."
    """
    image = Image.open(io.BytesIO(image_bytes))
    return _pil_image_to_data_url(image)


def _image_to_data_url(img_data: Any) -> Optional[str]:
    """Convert various image formats to data URL.

    Handles:
    - PIL Image objects (from HuggingFace auto-decode)
    - Dict with "bytes" key
    - Raw bytes

    Args:
        img_data: Image data in various formats

    Returns:
        Data URL string or None if conversion failed
    """
    try:
        if isinstance(img_data, Image.Image):
            # PIL Image object (HuggingFace auto-decoded)
            return _pil_image_to_data_url(img_data)
        elif isinstance(img_data, dict):
            # Dict with "bytes" key
            img_bytes = img_data.get("bytes")
            if img_bytes:
                return _bytes_to_data_url(img_bytes)
        elif isinstance(img_data, bytes):
            # Raw bytes
            return _bytes_to_data_url(img_data)
    except Exception as e:
        logger.warning("Failed to convert image to data URL: %s", e)
    return None


def _inject_images_into_messages(
    messages: List[Dict[str, Any]],
    images: List[Any]
) -> List[Dict[str, Any]]:
    """Replace [IMAGE_N] placeholders with actual base64 data URLs.

    Args:
        messages: List of conversation messages with placeholders
        images: List of images (PIL Image, dict with bytes, or raw bytes)

    Returns:
        Messages with image placeholders replaced by data URLs
    """
    # Deep copy to avoid modifying original
    messages = copy.deepcopy(messages)

    # Build placeholder to data URL mapping
    image_map: Dict[str, str] = {}
    for idx, img_data in enumerate(images):
        placeholder = f"[IMAGE_{idx}]"
        data_url = _image_to_data_url(img_data)
        if data_url:
            image_map[placeholder] = data_url

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
    sample_id: str,
    meta_info_str: str,
    answer: str,
    output_text: str,
    output_path: Path
) -> None:
    """Save result in format compatible with openai_as_judge.

    Creates a subfolder with meta_info.jsonl containing question, answer, and output_text.
    """
    sample_dir = output_path / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Load meta_info
    meta_info = json.loads(meta_info_str) if meta_info_str else {}

    # Build result record
    result = {
        "question": meta_info.get("question", ""),
        "answer": answer,
        "output_text": output_text,
        "image_path": meta_info.get("image_path", ""),
    }

    # Write to meta_info.jsonl
    meta_path = sample_dir / "meta_info.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _process_single_sample(
    sample_id: str,
    messages_json: str,
    images: List[Any],
    meta_info_str: str,
    answer: str,
    client: OpenAI,
    model_name: str,
    api_mode: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
    output_path: Path,
) -> Tuple[str, bool, str]:
    """Process a single sample.

    Returns:
        Tuple of (sample_id, success, message)
    """
    try:
        # Parse messages from JSON string
        messages = json.loads(messages_json)

        # Inject images into messages
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
        _save_result(sample_id, meta_info_str, answer, output_text, output_path)
        return (sample_id, True, f"output length: {len(output_text)}")

    except Exception as e:
        # Save error result
        _save_result(sample_id, meta_info_str, answer, f"ERROR: {str(e)}", output_path)
        return (sample_id, False, str(e))


def run_trajectory_test(
    parquet_path: Path,
    output_path: Path,
    api_base: str,
    model_name: str,
    api_mode: str = "responses",
    max_tokens: int = 16384,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
    num_workers: int = 8,
) -> None:
    """Run trajectory test on vLLM server with parallel processing.

    Args:
        parquet_path: Path to parquet dataset
        output_path: Output directory for results
        api_base: vLLM server base URL
        model_name: Model name/path
        api_mode: API mode ("chat_completions" or "responses")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_prompt: Optional system prompt to prepend
        num_workers: Number of parallel workers
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

    # Collect samples for parallel processing
    samples = []
    for idx, sample in enumerate(ds):
        samples.append({
            "sample_id": sample["id"],
            "messages_json": sample["messages"],
            "images": sample["images"],
            "meta_info_str": sample["meta_info"],
            "answer": sample["answer"],
            "idx": idx,
        })

    logger.info("Starting parallel processing with %d workers", num_workers)

    # Process samples in parallel
    completed = 0
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for sample_data in samples:
            future = executor.submit(
                _process_single_sample,
                sample_data["sample_id"],
                sample_data["messages_json"],
                sample_data["images"],
                sample_data["meta_info_str"],
                sample_data["answer"],
                client,
                model_name,
                api_mode,
                max_tokens,
                temperature,
                system_prompt,
                output_path,
            )
            futures[future] = sample_data["idx"]

        for future in as_completed(futures):
            idx = futures[future]
            sample_id, success, message = future.result()
            completed += 1

            if success:
                success_count += 1
                logger.info(
                    "[%d/%d] Sample %s completed: %s",
                    completed, len(samples), sample_id, message
                )
            else:
                error_count += 1
                logger.error(
                    "[%d/%d] Sample %s failed: %s",
                    completed, len(samples), sample_id, message
                )

    logger.info(
        "Test completed. Success: %d, Errors: %d, Total: %d",
        success_count, error_count, len(samples)
    )
    logger.info("Results saved to %s", output_path)
    print(f"Test completed. Success: {success_count}, Errors: {error_count}")
    print(f"Results saved to {output_path}")


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
        required=True,
        help="Model name/path (e.g., /storage/openpsi/models/Qwen3-VL-8B-Thinking)."
    )
    parser.add_argument(
        "--api_mode",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
        help="API mode (default: responses)."
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers."
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
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
