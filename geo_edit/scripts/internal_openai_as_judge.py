import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union
from geo_edit.constants import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT
from openai import OpenAI

ANSWER_TEMPLATE = "<answer>{}</answer>"


@dataclass
class EvalConfig:
    model: str = "gpt-4o"
    extract_answer_tags: Optional[str] = "split"


def parse_score(text: str) -> str:
    """Extract Score: 0/1 from model output, return "0" or "1"; empty string if not found."""
    m = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return m.group(1) if m else ""


def extract_answer(text: str, mode: str) -> Optional[str]:
    """
    Extract answer from ANSWER_TEMPLATE, return None if not found.
    mode:
      - "split": loose split
      - "strict": exact position match
    """
    parts = ANSWER_TEMPLATE.split("{}")
    if mode == "split":
        if parts[0] not in text or parts[1] not in text:
            return None
        return text.split(parts[0])[-1].split(parts[1])[0].strip()

    if mode == "strict":
        start = text.find(parts[0])
        if start == -1:
            return None
        start += len(parts[0])
        if parts[1]:
            end = text.find(parts[1], start)
            if end == -1:
                return None
            return text[start:end].strip()
        return text[start:].strip()

    raise ValueError(f"Unknown extract mode: {mode}")


def get_final_prediction(predict_str_list: List[str], extract_mode: Optional[str]) -> str:
    """Get final prediction from the last round of output."""
    if not predict_str_list:
        return ""
    last = predict_str_list[-1].strip()
    if not extract_mode:
        return last
    extracted = extract_answer(last, extract_mode)
    return extracted if extracted is not None else last


def get_output_tokens_total(item: Dict) -> Optional[float]:
    tokens_output_total = item.get("tokens_output_total")
    if isinstance(tokens_output_total, (int, float)):
        return float(tokens_output_total)
    per_step = item.get("tokens_used_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    return None


def get_input_tokens_total(item: Dict) -> Optional[float]:
    input_total = item.get("tokens_input_total")
    if isinstance(input_total, (int, float)):
        return float(input_total)
    tokens_used_total = item.get("tokens_used_total")
    output_total = get_output_tokens_total(item)
    if isinstance(tokens_used_total, (int, float)) and isinstance(output_total, (int, float)):
        value = float(tokens_used_total) - float(output_total)
        if value < 0:
            value = 0.0
        return value
    per_step = item.get("tokens_input_per_step")
    if isinstance(per_step, list):
        last_idx = None
        last_input = None
        for idx in range(len(per_step) - 1, -1, -1):
            value = per_step[idx]
            if isinstance(value, (int, float)):
                last_idx = idx
                last_input = float(value)
                break
        if last_input is None:
            return None
        outputs = item.get("tokens_used_per_step")
        if isinstance(outputs, list) and last_idx is not None:
            output_before = sum(
                v for v in outputs[:last_idx] if isinstance(v, (int, float))
            )
            input_total = last_input - float(output_before)
            if input_total < 0:
                input_total = 0.0
            return float(input_total)
        return float(last_input)
    return None


def get_total_tokens(item: Dict) -> Optional[float]:
    tokens_used_total = item.get("tokens_used_total")
    if isinstance(tokens_used_total, (int, float)):
        return float(tokens_used_total)
    per_step = item.get("tokens_total_per_step")
    if isinstance(per_step, list):
        values = [v for v in per_step if isinstance(v, (int, float))]
        if values:
            return float(sum(values))
    output_total = get_output_tokens_total(item)
    input_total = get_input_tokens_total(item)
    if output_total is not None and input_total is not None:
        return float(output_total + input_total)
    return None


class InternalOpenAIJudge:
    """Judge using Internal Matrix LLM API (OpenAI compatible format)."""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", base_url: str = "https://matrixllm.alipay.com/v1"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("MATRIX_API_KEY"),
            base_url=base_url
        )
        self.model = model

    def judge_correctness(self, question: str, ground_truth: str, prediction: str) -> str:
        prompt = EVAL_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        output_text = resp.choices[0].message.content if resp.choices else ""
        return parse_score(output_text or "")


def compute_tool_combination_statistics(eval_results: List[Dict]) -> str:
    """
    Compute accuracy statistics grouped by tool combination.
    Returns formatted summary text.
    """
    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "token_total_sum": 0.0,
            "token_total_count": 0,
            "token_correct_sum": 0.0,
            "token_correct_count": 0,
        }
    )

    for item in eval_results:
        result = item.get("result")
        if isinstance(result, dict) and result.get("is_filter"):
            continue
        func_counts = item.get("function_call_each_count", {})
        used = sorted([t for t, c in func_counts.items() if c > 0])
        category = "+".join(used) if used else "no_tool"
        s = stats[category]
        s["total"] += 1
        output_total = get_output_tokens_total(item)
        if isinstance(output_total, (int, float)):
            s["token_total_sum"] += float(output_total)
            s["token_total_count"] += 1
        if result == 1.0:
            s["correct"] += 1
            if isinstance(output_total, (int, float)):
                s["token_correct_sum"] += float(output_total)
                s["token_correct_count"] += 1

    lines = ["\n" + "=" * 60, "Tool Combination Statistics", "=" * 60]
    for cat in sorted(stats.keys(), key=lambda x: (x != "no_tool", x)):
        s = stats[cat]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        avg_tokens = (
            s["token_total_sum"] / s["token_total_count"] if s["token_total_count"] > 0 else 0.0
        )
        avg_tokens_correct = (
            s["token_correct_sum"] / s["token_correct_count"] if s["token_correct_count"] > 0 else 0.0
        )
        lines.append(
            "  "
            + f"{cat}: total={s['total']}, correct={s['correct']}, accuracy={acc:.4f}, "
            + f"avg_tokens={avg_tokens:.2f}, avg_tokens_correct={avg_tokens_correct:.2f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def evaluate_final_answer(
    question: str,
    predict_str_list: List[str],
    ground_truth: str,
    cfg: Optional[EvalConfig] = None,
) -> Union[float, dict]:
    """
    只评价最终答案正确性：
      - 返回 1.0 / 0.0
      - 若没解析到 Score，则返回 dict 方便你上层过滤
    """
    final_pred = get_final_prediction(predict_str_list, cfg.extract_answer_tags)

    judge = InternalOpenAIJudge(model=cfg.model)
    score_str = judge.judge_correctness(question, ground_truth, final_pred)
    print(f"Question: {question}, Ground Truth: {ground_truth}, Prediction: {final_pred}, Score: {score_str}")
    if score_str == "":
        return {"is_filter": True, "info": "no_score_returned", "raw_final_pred": final_pred}
    return 1.0 if score_str == "1" else 0.0


def iter_meta_info_files(result_path: str) -> Iterable[str]:
    for name in os.listdir(result_path):
        subdir = os.path.join(result_path, name)
        if not os.path.isdir(subdir):
            continue
        meta_path = os.path.join(subdir, "meta_info.jsonl")
        if os.path.isfile(meta_path):
            yield meta_path


def load_records(meta_path: str) -> Iterable[dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON line in {meta_path}: {line}")


def evaluate_record(record: dict, cfg: EvalConfig, record_id: str) -> dict:
    question = record["question"]
    ground_truth = record["answer"]
    output_text = record["output_text"]
    if isinstance(output_text, list):
        predict_str_list = [str(x) for x in output_text]
    else:
        predict_str_list = [str(output_text)]
    result = evaluate_final_answer(question, predict_str_list, ground_truth, cfg)
    return {
        "id": record_id,
        "question": question,
        "image_path":record["image_path"],
        "total_steps":record["total_steps"],
        "function_call_each_count":record["function_call_each_count"],
        "function_call_total_count":record["function_call_total_count"],
        "function_call_per_step":record["function_call_per_step"],
        "tokens_used_total":record["tokens_used_total"],
        "tokens_used_per_step":record["tokens_used_per_step"],
        "tokens_output_total": record.get("tokens_output_total"),
        "tokens_input_total": record.get("tokens_input_total"),
        "tokens_input_per_step": record.get("tokens_input_per_step"),
        "tokens_total_per_step": record.get("tokens_total_per_step"),
        "output_text": output_text,
        "ground_truth": ground_truth,
        "prediction": get_final_prediction(predict_str_list, cfg.extract_answer_tags),
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate results with Internal Matrix LLM judge.")
    parser.add_argument("--api_key", type=str, required=True, help="Matrix LLM API key.")
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
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use for evaluation (default: gpt-4o).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://matrixllm.alipay.com/v1",
        help="Base URL for the API endpoint (default: https://matrixllm.alipay.com/v1).",
    )
    parser.add_argument(
        "--additional_prompt",
        type=str,
        default="",
        help="Additional prompt to append to the evaluation prompt.",
    )
    args = parser.parse_args()

    os.environ["MATRIX_API_KEY"] = args.api_key
    os.makedirs(args.output_path, exist_ok=True)

    additional_prompt = args.additional_prompt.strip()
    if additional_prompt:
        global EVAL_QUERY_PROMPT
        EVAL_QUERY_PROMPT += "\n" + additional_prompt + "\n"

    cfg = EvalConfig(model=args.model)
    eval_output_path = os.path.join(args.output_path, "eval_result.jsonl")
    summary_path = os.path.join(args.output_path, "summary.txt")

    total = 0
    correct = 0
    filtered = 0
    eval_results = []
    output_tokens_sum = 0.0
    input_tokens_sum = 0.0
    total_tokens_sum = 0.0
    output_tokens_count = 0
    input_tokens_count = 0
    total_tokens_count = 0

    max_workers = 32
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(
        eval_output_path, "w", encoding="utf-8"
    ) as out_f:
        for meta_path in iter_meta_info_files(args.result_path):
            record_id = os.path.basename(os.path.dirname(meta_path))
            for record in load_records(meta_path):
                from time import sleep
                sleep(1)
                futures.append(executor.submit(evaluate_record, record, cfg, record_id))

        for future in as_completed(futures):
            eval_item = future.result()
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

    # Compute tool combination statistics
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
        f.write(tool_stats_text)


if __name__ == "__main__":
    main()
