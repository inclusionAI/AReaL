"""LLM-driven think block diversification for trajectory data.

Provides extraction, classification, and LLM rephrasing of ``<think>``
blocks inside assistant messages.
"""

from __future__ import annotations

import copy
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from geo_edit.data_preprocess.trajectory_utils import get_text_from_content
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

def _format_message(msg: Dict[str, Any]) -> str:
    """Format a single trajectory message as readable text for LLM context."""
    role = msg.get("role", "unknown")
    content = msg.get("content")

    if role == "assistant" and msg.get("tool_calls"):
        parts = []
        for tc in msg["tool_calls"]:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "")
            if isinstance(args, dict):
                import json as _json

                args = _json.dumps(args, ensure_ascii=False)
            parts.append(f"[Tool Call] {name}({args})")
        return "Assistant: " + " ".join(parts)

    text = get_text_from_content(content) if content is not None else ""
    if role == "tool":
        return f"Tool Result: {text}"

    label = role.capitalize()
    return f"{label}: {text}"


def _build_context(trajectory: List[Dict[str, Any]], up_to: int) -> str:
    """Build formatted conversation context from trajectory[0:up_to]."""
    parts = []
    for msg in trajectory[:up_to]:
        formatted = _format_message(msg)
        if formatted:
            parts.append(formatted)
    return "\n\n".join(parts)


class PromptType(Enum):
    A = "first_tool"
    B = "subsequent_tool"
    C = "final_reasoning"


@dataclass
class ThinkBlock:
    msg_index: int
    part_index: Optional[int]  # None if content is str; int if content is list
    start_pos: int  # Char offset of ``<think>``
    end_pos: int  # Char offset right after ``</think>``
    text: str  # Inner text between the tags
    prompt_type: Optional[PromptType] = None
    tool_name: Optional[str] = None  # Extracted from "Tool: xxx"
    context: str = ""  # Formatted conversation history preceding this block


# ── Prompt Templates ─────────────────────────────────────────────────────────

DIVERSIFY_SYSTEM_PROMPT = (
    "You are a text rephrasing assistant. Rephrase the given reasoning text "
    "while preserving its core meaning and key technical information. "
    "Output ONLY the rephrased text, nothing else."
)

PROMPT_FIRST_TOOL = """\
Rephrase the following reasoning text where a model decides to use the tool \
"{tool_name}" as its first analysis step.

Requirements:
- Keep the tool name "{tool_name}" exactly as-is in the output
- Vary how and why this tool is chosen as the first step
- Change sentence structure, word choice, and perspective
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rephrased reasoning text, no explanations

Original text:
{text}"""

PROMPT_SUBSEQUENT_TOOL = """\
Below is the conversation history of a geometric analysis model solving a problem, \
followed by one of its intermediate reasoning blocks where it decides to call \
the tool "{tool_name}" after analyzing previous results.

Conversation so far:
{context}

Reasoning block to rewrite:
{text}

Requirements:
- Start with a critical analysis of the previous tool call's result: \
is it correct, reasonable, and useful for solving the problem?
- If the result seems questionable, note what might be off and how to account for it
- Then explain why the current information is insufficient and why \
calling "{tool_name}" next is the right decision
- Keep the tool name "{tool_name}" exactly as-is in the output
- Use natural self-reflection phrases (e.g. "Wait, let me verify...", \
"Hmm, this result suggests...", "I should double-check...")
- Change sentence structure and word choice compared to the original
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rewritten reasoning text, no explanations"""

PROMPT_FINAL_REASONING = """\
Below is the full conversation history of a geometric analysis model solving \
a problem, followed by its final reasoning block that synthesizes all results.

Full conversation:
{context}

Final reasoning block to rewrite:
{text}

Requirements:
- Review each tool call and its result: was it necessary? Was the result \
correct and reliable? Did it contribute useful information?
- If any tool result appears wrong or unhelpful, explicitly reason about \
how to correct for it and derive the right answer despite the error
- Use self-reflection to verify the logical chain from observations to conclusion
- Preserve ALL factual data, numbers, measurements, and the final conclusion EXACTLY
- Use natural self-reflection phrases (e.g. "Let me verify this makes sense...", \
"Looking back at the results...", "I need to reconsider whether...")
- Change sentence structure and word choice compared to the original
- Do NOT include any XML tags like <think>, </think>, <answer>, or <action>
- Output ONLY the rewritten reasoning text, no explanations"""


# ── Block Extraction & Classification ────────────────────────────────────────


def extract_think_blocks(trajectory: List[Dict[str, Any]]) -> List[ThinkBlock]:
    """Extract all ``<think>...</think>`` blocks from assistant messages.

    Handles both string content and list-of-parts content formats.
    Each block receives the formatted conversation history preceding it.
    """
    blocks: List[ThinkBlock] = []
    think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    for msg_idx, msg in enumerate(trajectory):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content")
        if content is None:
            continue

        ctx = _build_context(trajectory, msg_idx)

        if isinstance(content, str):
            for m in think_re.finditer(content):
                blocks.append(
                    ThinkBlock(
                        msg_index=msg_idx,
                        part_index=None,
                        start_pos=m.start(),
                        end_pos=m.end(),
                        text=m.group(1),
                        context=ctx,
                    )
                )

        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                elif isinstance(part, str):
                    text = part
                else:
                    continue

                for m in think_re.finditer(text):
                    blocks.append(
                        ThinkBlock(
                            msg_index=msg_idx,
                            part_index=part_idx,
                            start_pos=m.start(),
                            end_pos=m.end(),
                            text=m.group(1),
                            context=ctx,
                        )
                    )

    return blocks


def classify_think_blocks(blocks: List[ThinkBlock]) -> None:
    """Classify think blocks into prompt types A / B / C **in-place**."""
    tool_re = re.compile(r"Tool:\s*(?:functions\.)?(\w+)")
    tool_think_count = 0

    for block in blocks:
        match = tool_re.search(block.text)
        if match:
            tool_think_count += 1
            block.tool_name = match.group(1)
            block.prompt_type = PromptType.A if tool_think_count == 1 else PromptType.B
        else:
            block.prompt_type = PromptType.C


# ── DiversificationClient ────────────────────────────────────────────────────


class DiversificationClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.7,
        requests_per_minute: int = 0,
    ):
        from openai import OpenAI  # lazy: allow --skip-diversify without openai

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

        # Rate limiting: 0 means no limit
        self._rate_limit = requests_per_minute
        if requests_per_minute > 0:
            self._min_interval = 60.0 / requests_per_minute
        else:
            self._min_interval = 0.0
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()

    def _build_prompt(self, block: ThinkBlock) -> str:
        if block.prompt_type == PromptType.A:
            return PROMPT_FIRST_TOOL.format(
                tool_name=block.tool_name, text=block.text
            )
        if block.prompt_type == PromptType.B:
            return PROMPT_SUBSEQUENT_TOOL.format(
                tool_name=block.tool_name,
                text=block.text,
                context=block.context,
            )
        return PROMPT_FINAL_REASONING.format(
            text=block.text, context=block.context
        )

    @staticmethod
    def _validate_result(block: ThinkBlock, rephrased: str) -> bool:
        if not rephrased or len(rephrased) <= 10:
            return False
        if len(rephrased) > len(block.text) * 3:
            return False
        if block.prompt_type in (PromptType.A, PromptType.B):
            if block.tool_name and block.tool_name not in rephrased:
                return False
        rephrased_lower = rephrased.lower()
        for tag in ("<think>", "</think>", "<answer>", "</answer>", "<action>", "</action>"):
            if tag in rephrased_lower:
                return False
        return True

    def _wait_for_rate_limit(self) -> None:
        """Block until enough time has passed since the last request."""
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

    def diversify_block(self, block: ThinkBlock) -> str:
        prompt = self._build_prompt(block)

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": DIVERSIFY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                rephrased = (resp.choices[0].message.content or "").strip()

                if self._validate_result(block, rephrased):
                    return rephrased

                logger.debug(
                    "Validation failed (attempt %d/%d, len=%d)",
                    attempt + 1,
                    self.max_retries,
                    len(rephrased),
                )
            except Exception as e:
                logger.warning(
                    "API error diversifying block (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )

            # Exponential back-off: 1 s → 2 s → 4 s
            time.sleep(2**attempt)

        logger.warning("All retries exhausted for block, keeping original text")
        return block.text

    def diversify_blocks_batch(
        self,
        blocks: List[ThinkBlock],
        max_workers: int = 8,
    ) -> Dict[int, str]:
        results: Dict[int, str] = {}
        eligible: List[Tuple[int, ThinkBlock]] = []

        for i, b in enumerate(blocks):
            if len(b.text) < 20:
                results[i] = b.text
            else:
                eligible.append((i, b))

        if not eligible:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.diversify_block, block): idx
                for idx, block in eligible
            }

            pbar = tqdm(
                total=len(eligible),
                desc="Diversifying think blocks",
                unit="block",
            )
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Failed to diversify block %d: %s", idx, e)
                    results[idx] = blocks[idx].text
                pbar.update(1)
            pbar.close()

        return results


# ── Apply diversified blocks back ────────────────────────────────────────────


def apply_diversified_blocks(
    trajectory: List[Dict[str, Any]],
    blocks: List[ThinkBlock],
    new_texts: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Splice diversified text back into the trajectory.

    Blocks are grouped by ``(msg_index, part_index)`` and replaced in
    **reverse** start-position order so that character offsets stay valid.
    """
    traj = copy.deepcopy(trajectory)

    groups: Dict[Tuple[int, Optional[int]], List[Tuple[ThinkBlock, str]]] = {}
    for block_idx, block in enumerate(blocks):
        if block_idx not in new_texts:
            continue
        key = (block.msg_index, block.part_index)
        groups.setdefault(key, []).append((block, new_texts[block_idx]))

    for (msg_idx, part_idx), pairs in groups.items():
        pairs.sort(key=lambda p: p[0].start_pos, reverse=True)

        for block, new_text in pairs:
            replacement = f"<think>{new_text}</think>"

            if part_idx is None:
                content = traj[msg_idx].get("content", "")
                if isinstance(content, str):
                    traj[msg_idx]["content"] = (
                        content[: block.start_pos]
                        + replacement
                        + content[block.end_pos :]
                    )
            else:
                parts = traj[msg_idx].get("content")
                if not isinstance(parts, list) or part_idx >= len(parts):
                    continue

                part = parts[part_idx]
                if isinstance(part, dict) and part.get("type") == "text":
                    old = part.get("text", "")
                    part["text"] = (
                        old[: block.start_pos] + replacement + old[block.end_pos :]
                    )
                elif isinstance(part, str):
                    parts[part_idx] = (
                        part[: block.start_pos] + replacement + part[block.end_pos :]
                    )

    return traj
