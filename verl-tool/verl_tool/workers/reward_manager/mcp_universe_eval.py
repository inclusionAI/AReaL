import os
from pathlib import Path
from datetime import datetime
import asyncio
from collections import defaultdict
from typing import Dict, Any, List

from verl import DataProto
from verl.workers.reward_manager import register  # type: ignore


def _decode_answer_blocks(text: str) -> str | None:
    import re
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not m:
        return None
    return m[-1].group(1).strip()


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        # Fallback: create a new task and wait synchronously
        # Note: reward managers are usually called in a non-async context.
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()


@register("mcp_universe_eval")
class MCPUniverseEvaluatorRM:
    """
    Reward manager that proxies scoring to MCP‑Universe evaluators.

    For each sample, it:
    - Decodes the model response
    - Locates the original MCP‑Universe task JSON via extra_info["task"]
    - Runs the task's evaluators (mcpuniverse.evaluator)
    - Scores = (#passed / #total); writes to the last response token

    Args (via reward_kwargs):
    - configs_root: Root folder to resolve relative task paths. Defaults to
      "verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"
    """

    name = "mcp_universe_eval"

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        configs_root: str | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        if configs_root:
            self.configs_root = configs_root
        else:
            # Resolve absolute configs root based on repo layout, without hardcoding host paths
            try:
                repo_root = Path(__file__).resolve().parents[3]
                default_root = repo_root / "benchmarks" / "MCP-Universe" / "mcpuniverse" / "benchmark" / "configs"
                self.configs_root = str(default_root)
            except Exception:
                # Fallback: derive from current file location
                self.configs_root = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "benchmarks",
                        "MCP-Universe",
                        "mcpuniverse",
                        "benchmark",
                        "configs",
                    )
                )

    def _resolve_task_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.configs_root, rel_or_abs)

    def __call__(self, data: DataProto, return_dict: bool = False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        # Lazy import to avoid hard dependency when not used
        from mcpuniverse.benchmark.task import Task
        from mcpuniverse.common.context import Context

        prompt_ids = data.batch["prompts"]
        resp_ids = data.batch["responses"]
        attn = data.batch["attention_mask"]

        import torch
        reward_tensor = torch.zeros_like(resp_ids, dtype=torch.float32)
        reward_extra: Dict[str, List[Any]] = defaultdict(list)

        for i in range(len(data)):
            prompt_len = prompt_ids.shape[-1]
            v_prompt = int(attn[i, :prompt_len].sum().item())
            v_resp = int(attn[i, prompt_len:].sum().item())

            # Decode strings
            prompt_str = self.tokenizer.decode(
                prompt_ids[i, prompt_len - v_prompt : prompt_len],
                skip_special_tokens=True,
            )
            resp_str = self.tokenizer.decode(
                resp_ids[i, :v_resp], skip_special_tokens=True
            )

            payload = _decode_answer_blocks(resp_str) or resp_str

            try:
                import json as _json
                _obj = _json.loads(payload)
                if isinstance(_obj, dict) and "answer" in _obj:
                    payload = str(_obj.get("answer", ""))
            except Exception:
                pass

            # Locate original task file
            extra = data[i].non_tensor_batch.get("extra_info", {}) or {}
            task_rel = extra.get("task")
            score = 0.0
            n_total = 0
            n_pass = 0
            if task_rel:
                task_path = self._resolve_task_path(str(task_rel))
                # One-shot JSON write per sample call: before/after/error in a single file
                debug_dir = os.environ.get("MCP_EVAL_DEBUG_DIR", "mcp_eval_debug")
                try:
                    import json as _json
                    import time as _time
                    # Build subdirectories by task group and session time
                    # task group: try the folder after 'test/' in task_rel, else parent folder name
                    try:
                        _parts = Path(task_rel).parts if isinstance(task_rel, str) else []
                        if "test" in _parts and _parts.index("test") + 1 < len(_parts):
                            task_group = _parts[_parts.index("test") + 1]
                        else:
                            task_group = Path(task_rel).parent.name if isinstance(task_rel, str) else "unknown"
                    except Exception:
                        task_group = "unknown"

                    # session folder: YYYY-MM-DD:H:MM (e.g., 2025-09-16:1:20)
                    _now = datetime.now()
                    session_folder = f"{_now.year:04d}-{_now.month:02d}-{_now.day:02d}:{_now.hour}:{_now.minute:02d}"

                    target_dir = os.path.join(debug_dir, task_group, session_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    # extract first correct_answer from task config (if any)
                    correct_answer = None
                    try:
                        with open(task_path, "r", encoding="utf-8") as _f:
                            _cfg = _json.load(_f)
                        for _ev in _cfg.get("evaluators", []) or []:
                            _args = _ev.get("op_args", {}) or {}
                            if "correct_answer" in _args:
                                correct_answer = _args.get("correct_answer")
                                break
                    except Exception:
                        correct_answer = None

                    # run evaluation
                    status = "ok"
                    error_msg = None
                    eval_results = []
                    n_total = 0
                    n_pass = 0
                    score = 0.0
                    try:
                        task = Task(task_path, context=Context())
                        eval_results = _run_async(task.evaluate(payload))
                        n_total = len(eval_results)
                        n_pass = sum(1 for r in eval_results if getattr(r, "passed", False))
                        score = (n_pass / n_total) if n_total > 0 else 0.0
                        reward_extra["eval_error"].append(0)
                    except Exception as e:
                        status = "error"
                        error_msg = str(e)
                        reward_extra["eval_error"].append(1)

                    # assemble single debug record and persist immediately
                    ts = int(_time.time() * 1000)
                    record = {
                        "sample_index": i,
                        "timestamp_ms": ts,
                        "status": status,
                        "error": error_msg,
                        "task_rel": task_rel,
                        "task_path": task_path,
                        "payload": payload,
                        "correct_answer": correct_answer,
                        "n_total": int(n_total),
                        "n_pass": int(n_pass),
                        "score": float(score),
                        "results": [
                            {
                                "passed": bool(getattr(r, "passed", False)),
                                "reason": getattr(r, "reason", ""),
                                "func": getattr(getattr(r, "config", None), "func", ""),
                                "op": getattr(getattr(r, "config", None), "op", ""),
                            }
                            for r in (eval_results or [])
                        ],
                    }
                    out_path = os.path.join(target_dir, f"{ts}_sample{i}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        _json.dump(record, f, ensure_ascii=False, indent=2)
                        f.flush()
                    # proceed with score already computed (or 0.0 if error)
                except Exception:
                    # avoid breaking training/eval if debug writing fails
                    pass
            else:
                # If no task path is available, try an EM fallback using <answer>
                ans = payload or ""
                gt = (data[i].non_tensor_batch.get("reward_model", {}) or {}).get(
                    "ground_truth"
                )
                if isinstance(gt, str) and gt.strip():
                    score = 1.0 if ans.strip() == gt.strip() else 0.0
                else:
                    reward_extra["skipped"].append(1)

            reward_tensor[i, v_resp - 1] = float(score)
            reward_extra["acc"].append(float(score))
            reward_extra["eval_pass"].append(int(n_pass))
            reward_extra["eval_total"].append(int(n_total))

            if self.num_examine > 0 and i < self.num_examine:
                try:
                    print("[prompt]", prompt_str)
                    print("[response]", resp_str)
                    print("[task]", task_rel if task_rel else "<none>")
                    print("[score]", score, "(pass/total=", n_pass, "/", n_total, ")")
                except Exception:
                    pass

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra}
        return reward_tensor


# ---------- Public helper for external test harness ----------
DEFAULT_CONFIGS_ROOT = "verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs"


def eval_answer_for_task(
    answer_or_response: str,
    task_rel_path: str,
    configs_root: str | None = None,
) -> dict:
    """
    Evaluate a single <answer> payload against an MCP‑Universe task and
    return a summary dict. This exposes the same core logic used by the
    reward manager, for test harnesses to reuse.

    Args:
        answer_or_response: The model's final output. Can be a full response
            with <answer>...</answer> tags, or only the inner payload (JSON or text).
        task_rel_path: Task JSON path, relative to configs_root (or absolute).
        configs_root: The configs root folder. If None, defaults to
            verl-tool/benchmarks/MCP-Universe/mcpuniverse/benchmark/configs.

    Returns:
        dict with keys: score (float in [0,1]), passed (int), total (int),
        details (list of {func, op, passed, reason}).
    """
    from mcpuniverse.benchmark.task import Task
    from mcpuniverse.common.context import Context
    import os as _os

    payload = _decode_answer_blocks(answer_or_response) or answer_or_response

    root = configs_root or DEFAULT_CONFIGS_ROOT
    # Resolve task path robustly. Accept either:
    # - absolute path
    # - relative to configs_root
    # - relative to configs_root/test
    if _os.path.isabs(task_rel_path):
        task_path = task_rel_path
    else:
        cand1 = _os.path.join(root, task_rel_path)
        cand2 = _os.path.join(root, "test", task_rel_path)
        task_path = cand1 if _os.path.exists(cand1) else (cand2 if _os.path.exists(cand2) else cand1)

    # Run async evaluation synchronously
    async def _run():
        task = Task(task_path, context=Context())
        return await task.evaluate(payload)

    results = _run_async(_run())
    total = len(results)
    passed = sum(1 for r in results if getattr(r, "passed", False))
    score = (passed / total) if total > 0 else 0.0

    def _get_cfg_attr(cfg, name, default=""):
        try:
            return getattr(cfg, name, default)
        except Exception:
            return default

    details = [
        {
            "func": _get_cfg_attr(r.config, "func"),
            "op": _get_cfg_attr(r.config, "op"),
            "passed": bool(getattr(r, "passed", False)),
            "reason": getattr(r, "reason", ""),
        }
        for r in results
    ]

    return {"score": score, "passed": passed, "total": total, "details": details}
