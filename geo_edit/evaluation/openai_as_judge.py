import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from geo_edit.constants import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT
from geo_edit.evaluation.utils import (
    compute_tool_combination_statistics,
    get_final_prediction,
    get_input_tokens_total,
    get_output_tokens_total,
    get_total_tokens,
    iter_meta_info_files,
    load_records,
    parse_score,
)
from openai import OpenAI


@dataclass
class EvalConfig:
    model: str = "gpt-5-mini"
    extract_answer_tags: Optional[str] = "split"

class OpenAIJudge:
    """
    使用 OpenAI 官方 SDK + Responses API 做 0/1 判分。
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def judge_correctness(self, question: str, ground_truth: str, prediction: str) -> str:
        prompt = EVAL_QUERY_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        resp = self.client.responses.create(
            model=self.model,
            instructions=EVAL_SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        )
        return parse_score(resp.output_text or "")


def evaluate_final_answer(
    question: str,
    predict_str_list: List[str],
    ground_truth: str,
    cfg: Optional[dict] = None,
) -> Union[float, dict]:
    """
    只评价最终答案正确性：
      - 返回 1.0 / 0.0
      - 若没解析到 Score，则返回 dict 方便你上层过滤
    """

    final_pred = get_final_prediction(predict_str_list, cfg.extract_answer_tags)

    judge = OpenAIJudge(model=cfg.model)
    score_str = judge.judge_correctness(question, ground_truth, final_pred)
    print(f"Question: {question}, Ground Truth: {ground_truth}, Prediction: {final_pred}, Score: {score_str}")
    if score_str == "":
        return {"is_filter": True, "info": "no_score_returned", "raw_final_pred": final_pred}
    return 1.0 if score_str == "1" else 0.0


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
    return {
        "id": record_id,
        "question": question,
        "image_path": record.get("image_path"),
        "total_steps": record.get("total_steps"),
        "function_call_each_count": record.get("function_call_each_count"),
        "function_call_total_count": record.get("function_call_total_count"),
        "function_call_per_step": record.get("function_call_per_step"),
        "tokens_used_total": record.get("tokens_used_total"),
        "tokens_used_per_step": record.get("tokens_used_per_step"),
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
    parser = argparse.ArgumentParser(description="Batch evaluate results with OpenAI judge.")
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
    parser.add_argument(
        "--additional_prompt",
        type=str,
        default="",
        help="Additional prompt to append to the evaluation prompt.",
    )
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key
    os.makedirs(args.output_path, exist_ok=True)
    
    additional_prompt = args.additional_prompt.strip()
    if additional_prompt:
        global EVAL_QUERY_PROMPT
        EVAL_QUERY_PROMPT += "\n" + additional_prompt + "\n"

    cfg = EvalConfig()
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
