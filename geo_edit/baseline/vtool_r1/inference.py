"""VTool-R1 inference on geo_edit datasets.

Faithfully reproduces VTool-R1's two-turn inference loop:
  Turn 1: model sees image + question + tool descriptions
          → outputs reasoning + optional ```python code block
  If code found: execute it → get edited image
  Turn 2: model sees original image + edited image + observation
          → outputs FINAL ANSWER: ... TERMINATE

VTool-R1's visual tools (highlight / mask / draw on bboxes) are ported
from the upstream repo and executed in a sandboxed context.  Datasets
that lack bbox metadata will naturally fall through to direct answering
(the model learned when NOT to call tools during RL training).

Prerequisites:
  - vLLM server running with VTool-R1 checkpoint:
      bash geo_edit/scripts/launch_vllm_generate.sh /path/to/VTool-R1-7B 8000
  - geo_edit installed (pip install -e .)

Usage:
  python geo_edit/baseline/vtool_r1/inference.py \
      --dataset_path /path/to/data.parquet \
      --dataset_name chartqa \
      --model_name_or_path /path/to/VTool-R1-7B \
      --api_base http://localhost:8000 \
      --output_dir ./vtool_r1_results
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import datetime
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

from geo_edit.datasets.task_registry import DATASET_SPECS, get_dataset_spec
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# VTool-R1 tool functions (ported from VTool-R1/verl/tooluse/tools.py)
# ---------------------------------------------------------------------------
from PIL import ImageDraw


def focus_on_columns_with_mask(image, columns_to_focus_on, all_columns_bounding_boxes):
    if not all_columns_bounding_boxes or not columns_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    to_mask = [c for c in all_columns_bounding_boxes if c not in columns_to_focus_on]
    if len(to_mask) == len(all_columns_bounding_boxes):
        return image
    for name in to_mask:
        bb = all_columns_bounding_boxes[name]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), fill="white")
    return image


def focus_on_rows_with_mask(image, rows_to_focus_on, all_rows_bounding_boxes):
    if not rows_to_focus_on or not all_rows_bounding_boxes:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    keys = list(all_rows_bounding_boxes.keys())
    to_mask = [r for r in keys[1:] if r not in rows_to_focus_on]
    if len(to_mask) == len(keys) - 1:
        return image
    for name in to_mask:
        bb = all_rows_bounding_boxes[name]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), fill="white")
    return image


def focus_on_columns_with_draw(image, columns_to_focus_on, all_columns_bounding_boxes):
    if not all_columns_bounding_boxes or not columns_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    for name in columns_to_focus_on:
        if name not in all_columns_bounding_boxes:
            continue
        bb = all_columns_bounding_boxes[name]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), outline="red", width=5)
    return image


def focus_on_rows_with_draw(image, rows_to_focus_on, all_rows_bounding_boxes):
    if not rows_to_focus_on or not all_rows_bounding_boxes:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    for name in rows_to_focus_on:
        if name not in all_rows_bounding_boxes:
            continue
        bb = all_rows_bounding_boxes[name]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), outline="red", width=5)
    return image


def _highlight_bboxes(image, keys, bbox_map):
    if not keys or not bbox_map:
        return image
    mask = image.convert('RGBA').copy()
    mask_draw = ImageDraw.Draw(mask)
    for k in keys:
        if k not in bbox_map:
            continue
        bb = bbox_map[k]
        mask_draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), fill=(255, 0, 0, 50))
    return Image.alpha_composite(image.convert('RGBA'), mask)


def focus_on_columns_with_highlight(image, columns_to_focus_on, all_columns_bounding_boxes):
    return _highlight_bboxes(image, columns_to_focus_on, all_columns_bounding_boxes)


def focus_on_rows_with_highlight(image, rows_to_focus_on, all_rows_bounding_boxes):
    return _highlight_bboxes(image, rows_to_focus_on, all_rows_bounding_boxes)


def focus_on_x_values_with_mask(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    if not all_x_values_bounding_boxes or not x_values_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    to_mask = [v for v in all_x_values_bounding_boxes if v not in x_values_to_focus_on]
    if len(to_mask) == len(all_x_values_bounding_boxes):
        return image
    for v in to_mask:
        bb = all_x_values_bounding_boxes[v]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), fill="white")
    return image


def focus_on_y_values_with_mask(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    if not all_y_values_bounding_boxes or not y_values_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    to_mask = [v for v in all_y_values_bounding_boxes if v not in y_values_to_focus_on]
    if len(to_mask) == len(all_y_values_bounding_boxes):
        return image
    for v in to_mask:
        bb = all_y_values_bounding_boxes[v]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), fill="white")
    return image


def focus_on_x_values_with_draw(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    if not all_x_values_bounding_boxes or not x_values_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    for v in x_values_to_focus_on:
        if v not in all_x_values_bounding_boxes:
            continue
        bb = all_x_values_bounding_boxes[v]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), outline="red", width=5)
    return image


def focus_on_y_values_with_draw(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    if not all_y_values_bounding_boxes or not y_values_to_focus_on:
        return image
    draw = ImageDraw.Draw(image, "RGBA")
    for v in y_values_to_focus_on:
        for existing_k in all_y_values_bounding_boxes:
            if v in existing_k or existing_k in v:
                v = existing_k
                break
        if v not in all_y_values_bounding_boxes:
            continue
        bb = all_y_values_bounding_boxes[v]
        draw.rectangle(((int(bb['x1'])+2, int(bb['y1'])+2), (int(bb['x2'])-2, int(bb['y2'])-2)), outline="red", width=5)
    return image


def focus_on_x_values_with_highlight(image, x_values_to_focus_on, all_x_values_bounding_boxes):
    return _highlight_bboxes(image, x_values_to_focus_on, all_x_values_bounding_boxes)


def focus_on_y_values_with_highlight(image, y_values_to_focus_on, all_y_values_bounding_boxes):
    return _highlight_bboxes(image, y_values_to_focus_on, all_y_values_bounding_boxes)


ALL_TOOL_FUNCTIONS = {
    "focus_on_columns_with_mask": focus_on_columns_with_mask,
    "focus_on_rows_with_mask": focus_on_rows_with_mask,
    "focus_on_columns_with_draw": focus_on_columns_with_draw,
    "focus_on_rows_with_draw": focus_on_rows_with_draw,
    "focus_on_columns_with_highlight": focus_on_columns_with_highlight,
    "focus_on_rows_with_highlight": focus_on_rows_with_highlight,
    "focus_on_x_values_with_mask": focus_on_x_values_with_mask,
    "focus_on_y_values_with_mask": focus_on_y_values_with_mask,
    "focus_on_x_values_with_draw": focus_on_x_values_with_draw,
    "focus_on_y_values_with_draw": focus_on_y_values_with_draw,
    "focus_on_x_values_with_highlight": focus_on_x_values_with_highlight,
    "focus_on_y_values_with_highlight": focus_on_y_values_with_highlight,
}

# ---------------------------------------------------------------------------
# VTool-R1 Parser (ported from VTool-R1/verl/tooluse/parse.py)
# ---------------------------------------------------------------------------


class VToolParser:
    def parse(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        text: str = str(response)
        content = text.replace("\\_", "_").replace("\\", "")

        start_pos = content.find("```python")
        if start_pos == -1:
            return {'status': False, 'content': content, 'message': 'No tool call', 'error_code': 'NOTOOL'}

        content = content[start_pos + len("```python"):]
        end_pos = content.find("```")
        if end_pos == -1:
            return {'status': False, 'content': content, 'message': 'No closing ```', 'error_code': 'unknown'}

        content = content[:end_pos]
        if not content.strip():
            return {'status': False, 'content': content, 'message': 'Empty code block', 'error_code': 'unknown'}

        try:
            compile(content, "prog.py", "exec")
            return {'status': True, 'content': content, 'message': 'OK', 'error_code': ''}
        except Exception as err:
            return {'status': False, 'content': content, 'message': str(err), 'error_code': 'unknown'}

    def trim_to_action_end(self, text: str) -> str:
        last = text.rfind("```")
        if last == -1:
            return text
        prev = text.rfind("```", 0, last)
        if prev == -1:
            return text
        return text[:last + 3]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img
    img_rgb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def load_image_from_item(item: Dict, image_key: str) -> Optional[Image.Image]:
    raw = item.get(image_key)
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(BytesIO(raw["bytes"]))
    if isinstance(raw, bytes):
        return Image.open(BytesIO(raw))
    if isinstance(raw, str) and os.path.isfile(raw):
        return Image.open(raw)
    return None


# ---------------------------------------------------------------------------
# Build VTool-R1 prompt (follows chartQA.jinja template structure)
# ---------------------------------------------------------------------------

VTOOL_R1_PROMPT_TEMPLATE = """
Here are some tools that can help you. All are python codes. They are in tools.py and will be imported for you.
You will be given a figure: image_1 and a question.
You should determine which tools to use based on the question.
Below are the tools in tools.py:
```python
def focus_on_columns_with_highlight(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"Focus on specific columns by adding transparent red highlight.\"\"\"

def focus_on_rows_with_highlight(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"Focus on specific rows by adding transparent red highlight.\"\"\"

def focus_on_columns_with_mask(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"Focus on specific columns by masking out others with white.\"\"\"

def focus_on_rows_with_mask(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"Focus on specific rows by masking out others with white.\"\"\"

def focus_on_columns_with_draw(image, columns_to_focus_on, all_columns_bounding_boxes):
    \"\"\"Focus on specific columns by drawing red boxes around them.\"\"\"

def focus_on_rows_with_draw(image, rows_to_focus_on, all_rows_bounding_boxes):
    \"\"\"Focus on specific rows by drawing red boxes around them.\"\"\"
```
# GOAL #: Based on the above tools, reason about how to solve the request and generate actions (python function calls).
You may use the tools to process images and make decisions based on visual outputs.

# REQUIREMENTS #:
1. The generated actions can resolve the given user request perfectly.
2. If you think you got the answer, use ANSWER: <your answer> Please extract the final answer in FINAL ANSWER: <final answer> and ends with TERMINATE.
3. All images in the initial user request are stored in PIL Image objects named image_1. Use display() to show images.
4. Use as few tools as possible. Only one turn of action, ACTION 0, is allowed.
5. If you do not think you have enough information to answer from tool outputs, answer based on the original image.
"""


def build_vtool_r1_user_prompt(question: str, bbox_info: Optional[str] = None) -> str:
    parts = [question]
    if bbox_info:
        parts.append(bbox_info)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def run_vtool_r1_single(
    client: OpenAI,
    model: str,
    image: Image.Image,
    question: str,
    bbox_metadata: Optional[Dict[str, Any]] = None,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run VTool-R1's two-turn inference on a single example."""
    parser = VToolParser()

    bbox_info = None
    if bbox_metadata:
        bbox_info = json.dumps(bbox_metadata, ensure_ascii=False)

    user_prompt = build_vtool_r1_user_prompt(question, bbox_info)

    image_url = pil_to_data_url(image)

    # ---- Turn 1: model sees image + question + tool descriptions ----
    messages_t1 = [
        {"role": "system", "content": VTOOL_R1_PROMPT_TEMPLATE.strip()},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    response_t1 = client.chat.completions.create(
        model=model,
        messages=messages_t1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.99,
    )
    text_t1 = response_t1.choices[0].message.content or ""

    result = {
        "first_rollout_response": text_t1,
        "second_rollout_response": None,
        "model_response": text_t1,
        "code": None,
        "code_error": None,
        "tool_used": False,
        "edited_image": None,
    }

    # ---- Parse tool call ----
    parsed = parser.parse(text_t1)
    if not parsed["status"]:
        # No valid tool call → direct answer from Turn 1
        return result

    code = parsed["content"]
    result["code"] = code
    result["tool_used"] = True

    # ---- Execute tool code ----
    captured_output = None

    def display(obj):
        nonlocal captured_output
        captured_output = obj

    exec_context = {
        "display": display,
        "image_1": image.copy(),
        "Image": Image,
    }
    exec_context.update(ALL_TOOL_FUNCTIONS)

    if bbox_metadata:
        if "columns_bbox" in bbox_metadata:
            exec_context["columns_bbox"] = bbox_metadata["columns_bbox"]
        if "rows_bbox" in bbox_metadata:
            exec_context["rows_bbox"] = bbox_metadata["rows_bbox"]
        if "x_values_bbox" in bbox_metadata:
            exec_context["columns_bbox"] = bbox_metadata["x_values_bbox"]
            exec_context["rows_bbox"] = bbox_metadata.get("y_values_bbox", {})

    try:
        f = StringIO()
        with redirect_stdout(f):
            exec(code, exec_context)
    except Exception as e:
        result["code_error"] = str(e)
        return result

    if not isinstance(captured_output, Image.Image):
        result["code_error"] = f"Tool did not produce an image (got {type(captured_output)})"
        return result

    edited_image = captured_output
    result["edited_image"] = edited_image

    # ---- Turn 2: model sees original + edited image + observation ----
    trimmed_t1 = parser.trim_to_action_end(text_t1)
    observation_text = (
        trimmed_t1
        + "\nOBSERVATION: Execution success. The output is as follows:"
        + "\n<the image outputs of the code is added as the second image>"
    )

    edited_url = pil_to_data_url(edited_image)

    messages_t2 = [
        {"role": "system", "content": VTOOL_R1_PROMPT_TEMPLATE.strip()},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": edited_url}},
                {"type": "text", "text": user_prompt + "\n\n" + observation_text},
            ],
        },
    ]

    response_t2 = client.chat.completions.create(
        model=model,
        messages=messages_t2,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.99,
    )
    text_t2 = response_t2.choices[0].message.content or ""

    result["second_rollout_response"] = text_t2
    result["model_response"] = text_t2
    return result


# ---------------------------------------------------------------------------
# Answer extraction (follows VTool-R1's format)
# ---------------------------------------------------------------------------
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?:\.\s*TERMINATE|TERMINATE|$)", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str:
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")

    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VTool-R1 inference on geo_edit datasets")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_SPECS.keys()))
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Local model path or HF id (must match the model served by vLLM)")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000",
                        help="vLLM server base URL")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--max_concurrent_requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = os.path.basename(args.model_name_or_path.rstrip("/"))
    output_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}.jsonl")
    stats_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}_stats.json")
    img_dir = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_{timestamp}_images")
    os.makedirs(img_dir, exist_ok=True)

    client = OpenAI(
        base_url=args.api_base.rstrip("/") + "/v1",
        api_key="none",
    )

    dataset_spec = get_dataset_spec(args.dataset_name)
    dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]

    if args.sample_rate < 1.0:
        n = int(len(dataset) * args.sample_rate)
        dataset = dataset.shuffle(seed=args.seed).select(range(n))
        logger.info("Sampled %d examples", n)

    logger.info("Dataset: %s (%d examples)", args.dataset_name, len(dataset))
    logger.info("Model: %s at %s", args.model_name_or_path, args.api_base)
    logger.info("Output: %s", output_path)

    num_tool_calls = 0
    num_tool_success = 0
    num_tool_failed = 0
    num_direct = 0
    write_lock = __import__("threading").Lock()
    counter_lock = __import__("threading").Lock()

    def process_one(item):
        nonlocal num_tool_calls, num_tool_success, num_tool_failed, num_direct
        item_id = str(item[dataset_spec.id_key])
        image_key = dataset_spec.image_key or "image"
        image = load_image_from_item(item, image_key)
        if image is None:
            logger.warning("[%s] no image, skipping", item_id)
            return

        question = dataset_spec.build_prompt(item, use_tools=True)
        answer_gt = dataset_spec.get_answer(item)

        bbox_metadata = None
        meta_extra = dataset_spec.build_task_kwargs(item).get("meta_info_extra", {})
        if meta_extra:
            bbox_metadata = meta_extra
        for bbox_key in ("columns_bbox", "rows_bbox", "x_values_bbox", "y_values_bbox"):
            if bbox_key in item:
                if bbox_metadata is None:
                    bbox_metadata = {}
                val = item[bbox_key]
                bbox_metadata[bbox_key] = json.loads(val) if isinstance(val, str) else val

        try:
            result = run_vtool_r1_single(
                client=client,
                model=args.model_name_or_path,
                image=image,
                question=question,
                bbox_metadata=bbox_metadata,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            logger.error("[%s] inference failed: %s", item_id, e)
            return

        with counter_lock:
            if result["tool_used"]:
                num_tool_calls += 1
                if result["code_error"]:
                    num_tool_failed += 1
                else:
                    num_tool_success += 1
            else:
                num_direct += 1

        predicted = extract_answer(result["model_response"])
        edited_img_path = None
        if result["edited_image"] is not None:
            edited_img_path = os.path.join(img_dir, f"{item_id}_edited.png")
            result["edited_image"].convert("RGB").save(edited_img_path)

        record = {
            "id": item_id,
            "question": question,
            "ground_truth": str(answer_gt),
            "predicted": predicted,
            "first_rollout_response": result["first_rollout_response"],
            "second_rollout_response": result["second_rollout_response"],
            "model_response": result["model_response"],
            "code": result["code"],
            "code_error": result["code_error"],
            "tool_used": result["tool_used"],
            "edited_image_path": edited_img_path,
        }
        with write_lock, open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as pool:
        futures = [pool.submit(process_one, item) for item in dataset]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="VTool-R1 Inference"):
            pass

    stats = {
        "total": len(dataset),
        "num_tool_calls": num_tool_calls,
        "num_tool_success": num_tool_success,
        "num_tool_failed": num_tool_failed,
        "num_direct": num_direct,
    }
    logger.info("Stats: %s", json.dumps(stats))

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
