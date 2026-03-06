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
    """Convert PIL Image to data URL."""
    buffer = io.BytesIO()
    fmt = image.format or "PNG"
    save_fmt = "JPEG" if fmt.upper() in {"JPG", "JPEG"} else "PNG"
    image.save(buffer, format=save_fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if save_fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def _image_to_data_url(img_data: Any) -> Optional[str]:
    """Convert various image formats to data URL."""
    try:
        if isinstance(img_data, Image.Image):
            return _pil_image_to_data_url(img_data)
        elif isinstance(img_data, dict):
            img_bytes = img_data.get("bytes")
            if img_bytes:
                image = Image.open(io.BytesIO(img_bytes))
                return _pil_image_to_data_url(image)
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data))
            return _pil_image_to_data_url(image)
    except Exception as e:
        logger.warning("Failed to convert image to data URL: %s", e)
    return None


def _convert_to_responses_format(messages: List[Dict[str, Any]], images: List[Any]) -> List[Dict[str, Any]]:
    """Convert messages from chat_completions format to responses API format.

    chat_completions format:
        {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "...", "content": "..."}

    responses format (flat list of items):
        {"type": "message", "role": "user", "content": [{"type": "input_text", ...}, {"type": "input_image", ...}]}
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
        {"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}
        {"type": "function_call_output", "call_id": "...", "output": "..."}
    """
    # Build placeholder to data URL mapping
    image_map: Dict[str, str] = {}
    for idx, img_data in enumerate(images):
        placeholder = f"[IMAGE_{idx}]"
        data_url = _image_to_data_url(img_data)
        if data_url:
            image_map[placeholder] = data_url

    converted_items = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        # Handle tool messages -> function_call_output
        if role == "tool":
            call_id = msg.get("tool_call_id", "")
            # Content can be string or list
            if isinstance(content, str):
                output = content
            elif isinstance(content, list):
                # Extract text from content parts
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        texts.append(part.get("text", ""))
                output = "\n".join(texts) if texts else json.dumps(content)
            else:
                output = json.dumps(content) if content else ""

            converted_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            })
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant" and "tool_calls" in msg:
            # First add the assistant message content if any
            if content:
                text_content = content if isinstance(content, str) else ""
                if isinstance(content, list):
                    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
                    text_content = "\n".join(texts)
                if text_content:
                    converted_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text_content}],
                    })

            # Then add function_call items
            for tool_call in msg.get("tool_calls", []):
                call_id = tool_call.get("id", "")
                func = tool_call.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                converted_items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                })
            continue

        # Handle regular user/assistant messages
        new_content = []

        if isinstance(content, str):
            new_content.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    new_content.append({"type": "input_text", "text": str(part)})
                    continue

                part_type = part.get("type", "")

                if part_type in ("text", "input_text"):
                    new_content.append({
                        "type": "input_text",
                        "text": part.get("text", "")
                    })
                elif part_type == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict):
                        url = image_url_data.get("url", "")
                    else:
                        url = str(image_url_data)

                    # Replace placeholder with actual data URL
                    if url in image_map:
                        url = image_map[url]

                    new_content.append({
                        "type": "input_image",
                        "image_url": url,
                        "detail": "auto"
                    })
                elif part_type == "input_image":
                    # Already in responses format
                    new_content.append(part)
                else:
                    new_content.append({"type": "input_text", "text": str(part)})

        converted_items.append({
            "type": "message",
            "role": role,
            "content": new_content,
        })

    return converted_items


def _save_result(
    sample_id: str,
    meta_info_str: str,
    answer: str,
    output_text: str,
    output_path: Path
) -> None:
    """Save result in format compatible with openai_as_judge."""
    sample_dir = output_path / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta_info = json.loads(meta_info_str) if meta_info_str else {}

    result = {
        "question": meta_info.get("question", ""),
        "answer": answer,
        "output_text": output_text,
        "image_path": meta_info.get("image_path", ""),
    }

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
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
    output_path: Path,
) -> Tuple[str, bool, str]:
    """Process a single sample using responses API."""
    try:
        # Parse messages from JSON string
        messages = json.loads(messages_json)

        # Convert to responses API format and inject images
        messages = _convert_to_responses_format(messages, images)

        # Build API kwargs
        api_kwargs = {
            "model": model_name,
            "input": messages,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            api_kwargs["instructions"] = system_prompt

        # Call vLLM responses API
        response = client.responses.create(**api_kwargs)
        output_text = getattr(response, "output_text", "") or ""

        # Save result
        _save_result(sample_id, meta_info_str, answer, output_text, output_path)
        return (sample_id, True, f"output length: {len(output_text)}")

    except Exception as e:
        _save_result(sample_id, meta_info_str, answer, f"ERROR: {str(e)}", output_path)
        return (sample_id, False, str(e))


def run_trajectory_test(
    parquet_path: Path,
    output_path: Path,
    api_base: str,
    model_name: str,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
    num_workers: int = 8,
) -> None:
    """Run trajectory test on vLLM server with parallel processing."""
    logger.info("Loading dataset from %s", parquet_path)
    ds = Dataset.from_parquet(str(parquet_path))
    logger.info("Loaded %d samples", len(ds))

    # Create OpenAI client for vLLM
    client = OpenAI(
        base_url=f"{api_base.rstrip('/')}/v1",
        api_key="none",
    )

    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are an advanced AI assistant capable of complex reasoning. "
            "Based on the conversation history, provide the final answer. "
            "Wrap your reasoning in <think>...</think> tags and your final answer in <answer>...</answer> tags."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    # Collect samples
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
                logger.info("[%d/%d] Sample %s completed: %s", completed, len(samples), sample_id, message)
            else:
                error_count += 1
                logger.error("[%d/%d] Sample %s failed: %s", completed, len(samples), sample_id, message)

    logger.info("Test completed. Success: %d, Errors: %d, Total: %d", success_count, error_count, len(samples))
    print(f"Test completed. Success: {success_count}, Errors: {error_count}")
    print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trajectory test on vLLM server and output results for evaluation."
    )
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to parquet dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for results.")
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1:8000", help="vLLM server base URL.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name/path.")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    run_trajectory_test(
        parquet_path=Path(args.parquet_path),
        output_path=Path(args.output_path),
        api_base=args.api_base,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
