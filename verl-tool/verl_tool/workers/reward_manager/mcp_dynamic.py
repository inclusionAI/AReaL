"""
Dynamic MCP reward manager.

For samples carrying `validation_calls`, it will invoke the specified MCP
tools at evaluation time and compare model outputs against the live tool
results. If no validation_calls are provided, falls back to static match.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Dict, List, Optional

import torch
from verl import DataProto
from verl.workers.reward_manager import register  # type: ignore
from verl_tool.servers.tools.utils.mcp_client import call_mcp_tool, get_content_text, MCPClientError


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _from_json_like(text: str) -> Optional[str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("result", "answer", "final_answer"):
                if key in obj:
                    return str(obj[key]).strip()
            return json.dumps(obj, ensure_ascii=False)
        return str(obj).strip()
    except Exception:
        return None


def _extract_answer_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    matches = list(ANSWER_RE.finditer(text))
    candidate = matches[-1].group(1).strip() if matches else text.strip()

    from_json = _from_json_like(candidate)
    if from_json is not None:
        return from_json

    num = _extract_numeric_value(candidate)
    if num is not None:
        return str(int(num)) if num.is_integer() else str(num)

    return candidate


def _normalize(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    try:
        num = float(s.replace(",", ""))
        if math.isfinite(num):
            if num.is_integer():
                return str(int(num))
            return f"{num:.6f}".rstrip("0").rstrip(".")
    except Exception:
        pass

    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\n\t")
    return s


def _extract_numeric_value(text: str) -> Optional[float]:
    if not isinstance(text, str):
        text = str(text)
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _compute_score(pred: str, target: str, tolerance: float = 1e-4) -> float:
    npred = _normalize(pred)
    ntarget = _normalize(target)

    try:
        vp = float(npred)
        vt = float(ntarget)
        if math.isfinite(vp) and math.isfinite(vt) and abs(vp - vt) <= tolerance:
            return 1.0
    except Exception:
        pass

    if ntarget and ntarget in npred:
        return 1.0

    vp_alt = _extract_numeric_value(pred)
    vt_alt = _extract_numeric_value(target)
    if vp_alt is not None and vt_alt is not None and math.isfinite(vp_alt) and math.isfinite(vt_alt):
        if abs(vp_alt - vt_alt) <= tolerance:
            return 1.0

    return 1.0 if npred == ntarget else 0.0


def _extract_alert_keywords(text: str) -> List[str]:
    """Lightweight heuristics for weather alerts; returns normalized keywords."""
    if not text:
        return []
    lower = text.lower()
    keywords: List[str] = []
    if "no active alerts" in lower:
        keywords.extend(
            [
                "no active alerts",
                "no alerts",
                "no active weather alerts",
                "no weather alerts",
            ]
        )
    for line in text.splitlines():
        if line.lower().startswith("event:"):
            ev = line.split(":", 1)[1].strip()
            if ev:
                keywords.append(ev.lower())
    return [_normalize(k) for k in keywords if k]


async def _run_validation_call(call_spec: Dict[str, Any]) -> str:
    server = call_spec.get("server")
    tool_name = call_spec.get("name")
    args = call_spec.get("arguments") or {}
    if not server or not tool_name:
        return ""
    try:
        res = await call_mcp_tool(tool_name, args, server=server)
        return get_content_text(res)
    except Exception:
        return ""


def _execute_validation_calls(call_specs: List[Dict[str, Any]]) -> str:
    async def _run_all():
        outs = []
        for spec in call_specs:
            outs.append(await _run_validation_call(spec))
        return outs

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # In case an event loop is already running, fallback to new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(_run_all())
        loop.close()
    return "\n".join([r for r in results if r])


@register("mcp_dynamic")
class MCPDynamicRM:
    """Dynamic MCP answer evaluator (uses validation_calls)."""

    name = "mcp_dynamic"

    def __init__(self, tokenizer, num_examine: int, tolerance: float = 1e-4, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.tolerance = tolerance

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch:
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        resp_ids = data.batch["responses"]
        attn = data.batch["attention_mask"]
        prompt_ids = data.batch["prompts"]

        reward_tensor = torch.zeros_like(resp_ids, dtype=torch.float32)
        reward_extra: Dict[str, List[Any]] = {"score": []}

        for i in range(len(data)):
            prompt_len = prompt_ids.shape[-1]
            v_prompt = int(attn[i, :prompt_len].sum().item())
            v_resp = int(attn[i, prompt_len:].sum().item())

            resp_str = self.tokenizer.decode(resp_ids[i, :v_resp], skip_special_tokens=True)
            extracted = _extract_answer_text(resp_str)

            ntb = data[i].non_tensor_batch
            validation_calls = ntb.get("validation_calls") or ntb.get("extra_info", {}).get("validation_calls")

            dynamic_target = ""
            if validation_calls:
                dynamic_target = _execute_validation_calls(validation_calls)

            gt = None
            if isinstance(ntb.get("reward_model"), dict) and "ground_truth" in ntb["reward_model"]:
                gt = ntb["reward_model"]["ground_truth"]
            if gt is None:
                gt = ""

            target_text = dynamic_target if dynamic_target else gt
            score = _compute_score(extracted, target_text, tolerance=self.tolerance) if target_text else 0.0

            if score == 0.0 and dynamic_target:
                # Try relaxed keyword match for weather-like alerts and summaries.
                pred_norm = _normalize(extracted)
                for kw in _extract_alert_keywords(dynamic_target):
                    if kw and kw in pred_norm:
                        score = 1.0
                        break

            reward_tensor[i, v_resp - 1] = score
            reward_extra["score"].append(score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra}
        return reward_tensor
