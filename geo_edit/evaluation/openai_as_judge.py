import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Union
from geo_edit.prompts import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT
from geo_edit.utils.io_utils import iter_meta_info_files, load_records
from geo_edit.utils.stats import compute_tool_combination_statistics, get_input_tokens_total, get_output_tokens_total, get_total_tokens
from geo_edit.utils.text_utils import extract_response_text, get_final_prediction, parse_score
from openai import OpenAI


@dataclass
class EvalConfig:
    model: str = "gpt-5-mini-2025-08-07"
    extract_answer_tags: Optional[str] = "split"
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class OpenAIJudge:
    """Judge using OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini", api_base: Optional[str] = None):
        client_kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
        if api_base is not None:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.api_mode = self._resolve_api_mode(api_base)

    @staticmethod
    def _resolve_api_mode(api_base: Optional[str]) -> str:
        if api_base and "matrixllm.alipay.com" in api_base.lower():
            return "chat"
        return "responses"

    def judge_correctness(self, question: str, ground_truth: str, prediction: str) -> str:
        prompt = EVAL_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        if self.api_mode == "chat":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            output_text = extract_response_text(resp, "chat_completions")
            return parse_score(output_text)
        resp = self.client.responses.create(
            model=self.model,
            instructions=EVAL_SYSTEM_PROMPT,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        return parse_score(resp.output_text or "")


def _normalize_for_match(s) -> str:
    """Normalize a string for exact matching: lowercase, strip, collapse whitespace."""
    return " ".join(str(s).strip().lower().split())


def _try_exact_match(prediction: str, ground_truth: str) -> Optional[float]:
    """Return 1.0/0.0 if we can determine correctness without LLM, else None."""
    pred = _normalize_for_match(prediction)
    gt = _normalize_for_match(ground_truth)
    if not pred or not gt:
        return None
    if pred == gt:
        return 1.0
    try:
        if float(pred) == float(gt):
            return 1.0
    except (ValueError, OverflowError):
        pass
    return None


def evaluate_final_answer(
    question: str,
    predict_str_list: List[str],
    ground_truth: str,
    cfg: EvalConfig,
) -> Union[float, dict]:
    """
    只评价最终答案正确性：
      - 返回 1.0 / 0.0
      - 若没解析到 Score，则返回 dict 方便你上层过滤
    """

    final_pred = get_final_prediction(predict_str_list, cfg.extract_answer_tags)

    exact = _try_exact_match(final_pred, ground_truth)
    if exact is not None:
        print(f"[ExactMatch] Question: {question}, Ground Truth: {ground_truth}, Prediction: {final_pred}, Score: {exact}")
        return exact

    judge = OpenAIJudge(api_key=cfg.api_key, model=cfg.model, api_base=cfg.api_base)
    score_str = judge.judge_correctness(question, ground_truth, final_pred)
    print(f"Question: {question}, Ground Truth: {ground_truth}, Prediction: {final_pred}, Score: {score_str}")
    if score_str == "":
        return {"is_filter": True, "info": "no_score_returned", "raw_final_pred": final_pred}
    return 1.0 if score_str == "1" else 0.0


def load_baseline_results(path: str) -> dict:
    """Load eval_result.jsonl from path or directory."""
    if os.path.isdir(path):
        path = os.path.join(path, "eval_result.jsonl")
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                results[str(record["id"])] = record
    return results


def compare_with_baseline(eval_results: list, baseline_path: str) -> None:
    """Compare current results with baseline (current as primary)."""
    baseline = load_baseline_results(baseline_path)
    current_by_id = {str(r["id"]): r for r in eval_results}

    both_correct = both_wrong = current_only = baseline_only = 0
    current_correct = baseline_correct = 0

    for sid, cur in current_by_id.items():
        if sid not in baseline:
            continue
        cur_result = cur.get("result")
        base_result = baseline[sid].get("result")
        # Handle filtered results (dict means filtered)
        cur_ok = cur_result == 1.0
        base_ok = base_result == 1.0
        current_correct += cur_ok
        baseline_correct += base_ok
        if cur_ok and base_ok:
            both_correct += 1
        elif not cur_ok and not base_ok:
            both_wrong += 1
        elif cur_ok:
            current_only += 1
        else:
            baseline_only += 1

    total = both_correct + both_wrong + current_only + baseline_only
    if total == 0:
        print("\nNo common samples for comparison!")
        return

    cur_acc = current_correct / total
    base_acc = baseline_correct / total

    print("\n" + "=" * 50)
    print("COMPARISON (current as primary)")
    print("=" * 50)
    print(f"Common samples: {total}")
    print(f"Current:  {current_correct}/{total} ({cur_acc:.4f})")
    print(f"Baseline: {baseline_correct}/{total} ({base_acc:.4f})")
    print("-" * 50)
    print(f"Diff: {cur_acc - base_acc:+.4f}")
    print(f"Both correct: {both_correct}, Both wrong: {both_wrong}")
    print(f"Current only: {current_only}, Baseline only: {baseline_only}")
    print("=" * 50)


def _build_eval_item(record: dict, record_id: str, cfg: EvalConfig, final_pred: str, result) -> dict:
    question = record["question"]
    ground_truth = record["answer"]
    if isinstance(ground_truth, list):
        ground_truth = "\n".join(ground_truth)
    item = {
        "id": record_id,
        "question": question,
        "image_path": record.get("image_path"),
        "total_steps": record.get("total_steps"),
        "function_call_each_count": record.get("function_call_each_count"),
        "function_call_total_count": record.get("function_call_total_count"),
        "function_call_per_step": record.get("function_call_per_step"),
        "tokens_used_total": record.get("tokens_used_total"),
        "tokens_used_per_step": record.get("tokens_used_per_step", record.get("tokens_output_per_step")),
        "tokens_output_per_step": record.get("tokens_output_per_step", record.get("tokens_used_per_step")),
        "tokens_output_total": record.get("tokens_output_total"),
        "tokens_input_total": record.get("tokens_input_total"),
        "tokens_input_per_step": record.get("tokens_input_per_step"),
        "tokens_total_per_step": record.get("tokens_total_per_step"),
        "output_text": record["output_text"],
        "ground_truth": ground_truth,
        "prediction": final_pred,
        "result": result,
    }
    if record.get("category"):
        item["category"] = record["category"]
    return item


def evaluate_record(record: dict, cfg: EvalConfig, record_id: str) -> dict:
    question = record["question"]
    ground_truth = record["answer"]
    if isinstance(ground_truth, list):
        ground_truth = "\n".join(ground_truth)
    output_text = record["output_text"]
    if isinstance(output_text, list):
        predict_str_list = [str(x) for x in output_text]
    else:
        predict_str_list = [str(output_text)]
    result = evaluate_final_answer(question, predict_str_list, ground_truth, cfg)
    item = {
        "id": record_id,
        "question": question,
        "image_path": record.get("image_path"),
        "total_steps": record.get("total_steps"),
        "function_call_each_count": record.get("function_call_each_count"),
        "function_call_total_count": record.get("function_call_total_count"),
        "function_call_per_step": record.get("function_call_per_step"),
        "tokens_used_total": record.get("tokens_used_total"),
        "tokens_used_per_step": record.get("tokens_used_per_step", record.get("tokens_output_per_step")),
        "tokens_output_per_step": record.get("tokens_output_per_step", record.get("tokens_used_per_step")),
        "tokens_output_total": record.get("tokens_output_total"),
        "tokens_input_total": record.get("tokens_input_total"),
        "tokens_input_per_step": record.get("tokens_input_per_step"),
        "tokens_total_per_step": record.get("tokens_total_per_step"),
        "output_text": output_text,
        "ground_truth": ground_truth,
        "prediction": get_final_prediction(predict_str_list, cfg.extract_answer_tags),
        "result": result,
    }
    if record.get("category"):
        item["category"] = record["category"]
    return item


def main(
    *,
    default_model: str = "gpt-5-mini-2025-08-07",
    default_api_base: Optional[str] = None,
    description: str = "Batch evaluate results with OpenAI judge.",
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path containing subdirectories with meta_info.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write eval_result.jsonl and summary.txt.",
    )
    parser.add_argument("--model", type=str, default=default_model, help="Model name used for evaluation.")
    parser.add_argument(
        "--api_base",
        type=str,
        default=default_api_base,
        help="Optional OpenAI-compatible base URL, e.g. https://matrixllm.alipay.com/v1 or https://llm-proxy.perflab.nvidia.com/openai/v1.",
    )
    parser.add_argument(
        "--additional_prompt",
        type=str,
        default="",
        help="Additional prompt to append to the evaluation prompt.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name from task_registry to auto-load judge_prompt.",
    )
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        help="Path to baseline eval results for comparison.",
    )
    parser.add_argument("--max_workers", type=int, default=32, help="Max concurrent judge threads.")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: skip LLM judge when exact match cannot determine, score as 0.0.",
    )
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.makedirs(args.output_path, exist_ok=True)

    additional_prompt = args.additional_prompt.strip()
    if not additional_prompt and args.dataset_name:
        from geo_edit.datasets.task_registry import get_dataset_spec

        spec = get_dataset_spec(args.dataset_name)
        additional_prompt = spec.get_judge_prompt() or ""
    if additional_prompt:
        global EVAL_QUERY_PROMPT
        EVAL_QUERY_PROMPT += "\n" + additional_prompt + "\n"

    cfg = EvalConfig(model=args.model, api_key=args.api_key, api_base=args.api_base)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    done_ids = set()
    eval_results = []
    if os.path.exists(eval_output_path):
        with open(eval_output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    eval_results.append(item)
                    done_ids.add(str(item["id"]))
        print(f"[Resume] Loaded {len(done_ids)} already-evaluated records from {eval_output_path}")

    total = 0
    correct = 0
    filtered = 0
    output_tokens_sum = 0.0
    input_tokens_sum = 0.0
    total_tokens_sum = 0.0
    output_tokens_count = 0
    input_tokens_count = 0
    total_tokens_count = 0

    for eval_item in eval_results:
        result = eval_item["result"]
        is_filter = isinstance(result, dict) and result.get("is_filter")
        if is_filter:
            filtered += 1
        else:
            total += 1
            if result == 1.0:
                correct += 1
        output_total = get_output_tokens_total(eval_item)
        if output_total is not None:
            output_tokens_sum += output_total
            output_tokens_count += 1
        input_total = get_input_tokens_total(eval_item)
        if input_total is not None:
            input_tokens_sum += input_total
            input_tokens_count += 1
        total_total = get_total_tokens(eval_item)
        if total_total is not None:
            total_tokens_sum += float(total_total)
            total_tokens_count += 1

    def _account_and_write(eval_item, out_f):
        nonlocal total, correct, filtered, output_tokens_sum, output_tokens_count
        nonlocal input_tokens_sum, input_tokens_count, total_tokens_sum, total_tokens_count
        eval_results.append(eval_item)
        result = eval_item["result"]
        is_filter = isinstance(result, dict) and result.get("is_filter")
        if is_filter:
            filtered += 1
        else:
            total += 1
            if result == 1.0:
                correct += 1
        out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
        out_f.flush()
        output_total = get_output_tokens_total(eval_item)
        if output_total is not None:
            output_tokens_sum += output_total
            output_tokens_count += 1
        input_total = get_input_tokens_total(eval_item)
        if input_total is not None:
            input_tokens_sum += input_total
            input_tokens_count += 1
        total_total = get_total_tokens(eval_item)
        if total_total is not None:
            total_tokens_sum += float(total_total)
            total_tokens_count += 1

    max_workers = args.max_workers
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(eval_output_path, "a", encoding="utf-8") as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))
            if str(record_id) in done_ids:
                continue
            for record in load_records(meta_path):
                question = record["question"]
                ground_truth = record["answer"]
                if isinstance(ground_truth, list):
                    ground_truth = "\n".join(ground_truth)
                output_text = record["output_text"]
                if isinstance(output_text, list):
                    predict_str_list = [str(x) for x in output_text]
                else:
                    predict_str_list = [str(output_text)]
                final_pred = get_final_prediction(predict_str_list, cfg.extract_answer_tags)
                exact = _try_exact_match(final_pred, ground_truth)
                if exact is not None:
                    print(f"[ExactMatch] Question: {question}, GT: {ground_truth}, Pred: {final_pred}, Score: {exact}")
                    eval_item = _build_eval_item(record, record_id, cfg, final_pred, exact)
                    _account_and_write(eval_item, out_f)
                elif args.fast and len(final_pred.split()) <= 1 and len(ground_truth.split()) <= 1:
                    print(f"[Fast] Question: {question}, GT: {ground_truth}, Pred: {final_pred}, Score: 0.0")
                    eval_item = _build_eval_item(record, record_id, cfg, final_pred, 0.0)
                    _account_and_write(eval_item, out_f)
                else:
                    from time import sleep
                    sleep(1)
                    futures.append(executor.submit(evaluate_record, record, cfg, record_id))

        for future in as_completed(futures):
            eval_item = future.result()
            _account_and_write(eval_item, out_f)

    tool_stats_text = compute_tool_combination_statistics(eval_results)

    accuracy = (correct / total) if total else 0.0
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={accuracy:.6f}\n")
        avg_output = output_tokens_sum / output_tokens_count if output_tokens_count else 0.0
        avg_input = input_tokens_sum / input_tokens_count if input_tokens_count else 0.0
        avg_total = total_tokens_sum / total_tokens_count if total_tokens_count else 0.0
        f.write(f"total_output_tokens={output_tokens_sum:.0f}\n")
        f.write(f"total_input_tokens={input_tokens_sum:.0f}\n")
        f.write(f"total_tokens={total_tokens_sum:.0f}\n")
        f.write(f"avg_output_tokens={avg_output:.2f}\n")
        f.write(f"avg_input_tokens={avg_input:.2f}\n")
        f.write(f"avg_total_tokens={avg_total:.2f}\n")

        cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
        for item in eval_results:
            cat = item.get("category")
            if not cat:
                continue
            result = item.get("result")
            if isinstance(result, dict) and result.get("is_filter"):
                continue
            cat_stats[cat]["n"] += 1
            if result == 1.0:
                cat_stats[cat]["correct"] += 1
        if cat_stats:
            f.write("\n--- Per-Category Accuracy ---\n")
            for cat in sorted(cat_stats):
                s = cat_stats[cat]
                cat_acc = s["correct"] / s["n"] if s["n"] else 0.0
                f.write(f"  {cat}: {s['correct']}/{s['n']} ({cat_acc:.4f})\n")
            if len(cat_stats) > 1:
                all_n = sum(s["n"] for s in cat_stats.values())
                all_c = sum(s["correct"] for s in cat_stats.values())
                f.write(f"  overall: {all_c}/{all_n} ({all_c/all_n:.4f})\n")

        f.write(tool_stats_text)

    # Compare with baseline if provided
    if args.compare_with:
        compare_with_baseline(eval_results, args.compare_with)


if __name__ == "__main__":
    main()
