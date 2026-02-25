from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest
from openai import OpenAI

# To avoid accidental paid API calls in normal test runs, this test is disabled by default.
RUN_ENV_FLAG = "RUN_GEO_EDIT_IMAGE_API_TEST"

DEFAULT_API_BASE = "https://matrixllm.alipay.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROMPT = "Please describe this image briefly."

REPO_RELATIVE_IMAGE_PATH = Path("images/input_image.png")

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
@pytest.mark.skipif(
    os.getenv(RUN_ENV_FLAG, "0") != "1",
    reason=(
        "Live API test is disabled by default. "
        f"Set {RUN_ENV_FLAG}=1 to run."
    ),
)
def test_describe_single_image_via_chat_completions() -> None:
    api_key =os.getenv("MATRIX_API_KEY")
    api_base = DEFAULT_API_BASE.rstrip("/")
    model =DEFAULT_MODEL
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
        temperature=0.0,
        max_tokens=256,
    )

    content = ""
    if response.choices and response.choices[0].message:
        content = response.choices[0].message.content or ""

    assert content.strip(), "Model returned empty content."
    print("\nModel output:\n", content.strip())
