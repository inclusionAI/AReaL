import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union
from ..constants import EVAL_QUERY_PROMPT, EVAL_SYSTEM_PROMPT
from openai import OpenAI

ANSWER_TEMPLATE = "<answer>{}</answer>"


@dataclass
class EvalConfig:
    model: str = "gpt-5-mini"
    extract_answer_tags: Optional[str] = "split"


def parse_score(text: str) -> str:
    """
    从模型输出里抽取 Score: 0/1，返回 "0" 或 "1"；找不到返回空串。
    """
    m = re.search(r"\bscore\s*:\s*([01])\b", text, re.IGNORECASE)
    return m.group(1) if m else ""


def extract_answer(text: str, mode: str) -> Optional[str]:
    """
    从模型输出中抽取ANSWER_TEMPLATE，抽不到返回 None。
    mode:
      - "split": 宽松 split
      - "strict": 任意位置匹配 ANSWER_TEMPLATE
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
    """
    只取最后一轮作为最终输出；若 extract_mode 非空，则尝试抽取 <answer>...</answer>。
    抽取失败时，退回使用最后一轮原文（strip 后）。
    """
    if not predict_str_list:
        return ""
    last = predict_str_list[-1].strip()
    if not extract_mode:
        return last
    extracted = extract_answer(last, extract_mode)
    return extracted if extracted is not None else last

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
            result = eval_item["result"]
            is_filter = isinstance(result, dict) and result.get("is_filter")
            if is_filter:
                filtered += 1
            else:
                total += 1
                if result == 1.0:
                    correct += 1
            out_f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")

    accuracy = (correct / total) if total else 0.0
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"evaluated={total}\n")
        f.write(f"correct={correct}\n")
        f.write(f"filtered={filtered}\n")
        f.write(f"accuracy={accuracy:.6f}\n")


if __name__ == "__main__":
    main()
