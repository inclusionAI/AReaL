from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from openai import OpenAI

from geo_edit.utils.text_utils import extract_response_text

DEFAULT_API_BASE = "https://matrixllm.alipay.com/v1"
DEFAULT_MODEL = "gpt-5-2025-08-07"
DEFAULT_PROMPT = "Please describe this image briefly."

REPO_RELATIVE_IMAGE_PATH = Path("geo_edit/images/input_image.png")

def _to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

@pytest.mark.slow
def test_describe_single_image_via_chat_completions() -> None:
    api_key = os.getenv("MATRIX_API_KEY")
    if not api_key:
        raise ValueError("Please set MATRIX_API_KEY before running this test.")

    api_base = DEFAULT_API_BASE.rstrip("/")
    model = DEFAULT_MODEL
    prompt = DEFAULT_PROMPT
    image_path = REPO_RELATIVE_IMAGE_PATH

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=60,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_path)}},
                ],
            }
        ],
        max_completion_tokens=1024,
        reasoning_effort="low"
    )

    print(response)
    content = extract_response_text(response, "chat_completions")

    assert content.strip(), "Model returned empty content."
    print("\nModel output:\n", content.strip())
