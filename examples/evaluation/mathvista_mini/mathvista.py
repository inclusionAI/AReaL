import argparse
import asyncio
import base64
import io
import json
import random
import re
import traceback
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional

import aiohttp
from datasets import load_dataset
from PIL import Image
from tqdm.asyncio import tqdm  # 使用 async 版本的 tqdm 或者直接用标准 tqdm


def color_kv(d: Dict[str, Any]) -> str:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    return "{ " + ", ".join(f"{GREEN}{k}{RESET}={CYAN}{v}{RESET}" for k, v in d.items()) + " }"


def pil_to_data_url(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_image_any(x: Any) -> Image.Image:
    try:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, dict):
            if "path" in x and isinstance(x["path"], str):
                return Image.open(x["path"]).convert("RGB")
            if "bytes" in x and isinstance(x["bytes"], (bytes, bytearray)):
                return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
        return Image.open(x).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        # 返回一个纯黑图片防止程序崩溃
        return Image.new("RGB", (224, 224), (0, 0, 0))


_FINAL_RE = re.compile(r"(final answer|final|answer)\s*[:：]\s*(.+)", re.IGNORECASE)
_LATEX_FRAC_RE = re.compile(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}")

_NUM_FRAC_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_NUM_INT_RE = re.compile(r"^\s*[+-]?\d+\s*$")
_NUM_DEC_RE = re.compile(r"^\s*[+-]?\d+\.\d+\s*$")
_NUM_SCI_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?[eE][+-]?\d+\s*$")

_CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
_LAST_NUM_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+)?|[+-]?\d+(?:\.\d+)?[eE][+-]?\d+)"
)


def strip_wrappers(s: str) -> str:
    if not s:
        return ""
    s = s.strip().replace("$", "").replace("，", ",").replace("\u00a0", " ").replace("✅", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("`*_~ ")


def latex_to_simple(s: str) -> str:
    s2 = _LATEX_FRAC_RE.sub(r"\1/\2", s)
    s2 = s2.replace("\\,", "").replace("\\ ", "").replace("\\left", "").replace("\\right", "")
    return s2.strip()


def extract_choice_letter(text: str) -> Optional[str]:
    hits = _CHOICE_RE.findall(text.upper())
    return hits[-1].upper() if hits else None


def extract_boxed_balanced(text: str) -> Optional[str]:
    key = r"\boxed{"
    start = text.find(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    out: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_final_answer(text: str) -> str:
    t = text.strip()
    boxed = extract_boxed_balanced(t)
    if boxed is not None:
        return strip_wrappers(boxed)

    m = _FINAL_RE.search(t)
    if m:
        return strip_wrappers(m.group(2))

    lines = [x.strip() for x in t.splitlines() if x.strip()]
    tail = strip_wrappers(lines[-1]) if lines else strip_wrappers(t)

    if len(tail) > 48:
        nums = _LAST_NUM_RE.findall(t)
        if nums:
            return strip_wrappers(nums[-1])

    return tail


def normalize_text(s: str) -> str:
    x = latex_to_simple(strip_wrappers(s)).lower()
    x = x.replace("*", "").replace("_", "").replace("~", "")
    x = x.replace(" ", "").replace(",", "").replace(".", "").replace(":", "")
    return x


def parse_number(s: str) -> Optional[Fraction]:
    x = latex_to_simple(strip_wrappers(s)).replace(" ", "").replace("{", "").replace("}", "")

    m = _NUM_FRAC_RE.match(x)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return None if b == 0 else Fraction(a, b)

    if _NUM_INT_RE.match(x):
        return Fraction(int(x), 1)

    if _NUM_DEC_RE.match(x):
        return Fraction(x)

    if _NUM_SCI_RE.match(x):
        try:
            sign = 1
            y = x
            if y.startswith("+"):
                y = y[1:]
            elif y.startswith("-"):
                sign = -1
                y = y[1:]
            base, exp = re.split(r"[eE]", y)
            exp_i = int(exp)

            if "." in base:
                int_part, frac_part = base.split(".", 1)
            else:
                int_part, frac_part = base, ""
            digits = (int_part + frac_part).lstrip("0") or "0"
            frac_len = len(frac_part)

            pow10 = exp_i - frac_len
            if pow10 >= 0:
                return Fraction(sign * int(digits + ("0" * pow10)), 1)

            k = -pow10
            if len(digits) <= k:
                val = "0." + ("0" * (k - len(digits))) + digits
            else:
                val = digits[:-k] + "." + digits[-k:]
            if sign < 0:
                val = "-" + val
            return Fraction(val)
        except Exception:
            return None

    return None


def extract_last_number_token(text: str) -> Optional[str]:
    t = latex_to_simple(strip_wrappers(text))
    nums = _LAST_NUM_RE.findall(t)
    return strip_wrappers(nums[-1]) if nums else None


def infer_choice_from_options(pred_text: str, options: List[str]) -> Optional[str]:
    if not options:
        return None
    pnorm = normalize_text(pred_text)
    letters = "ABCDE"
    for i, opt in enumerate(options[:5]):
        onorm = normalize_text(str(opt))
        if onorm and onorm in pnorm:
            return letters[i]
    return None


def is_correct(pred: str, gold: str, tol: Fraction, options: Optional[List[str]] = None) -> bool:
    try:
        g = strip_wrappers(gold).upper()

        if len(g) == 1 and g in "ABCDE":
            pl = extract_choice_letter(pred)
            if pl is not None:
                return pl == g
            if options is not None:
                inferred = infer_choice_from_options(pred, options)
                return inferred == g if inferred is not None else False
            return False

        p = extract_final_answer(pred)
        p_num = extract_last_number_token(p)

        gn = parse_number(gold)
        pn = parse_number(p_num) if p_num is not None else parse_number(p)

        if pn is not None and gn is not None:
            d = pn - gn
            if d < 0:
                d = -d
            return d <= tol

        if gn is not None and p_num is not None:
            return normalize_text(p_num) == normalize_text(gold)

        return normalize_text(p) == normalize_text(gold)
    except Exception:
        return False


@dataclass
class ReqItem:
    i: int
    ex: Dict[str, Any]


async def infer_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    question: str,
    image: Image.Image,
    options: List[str],
    max_tokens: int,
    temperature: float,
) -> str:
    data_url = pil_to_data_url(image)
    if len(options) > 0:
        assert len(options) == 5, options
        if ''.join(options) != 'ABCDE':
            options = f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\n"
    else:
        options=''

    input_text = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": input_text},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"API Error {resp.status}: {text}")
        out = await resp.json()
        return out["choices"][0]["message"]["content"]


async def run(args: argparse.Namespace) -> None:
    ds = load_dataset("parquet", data_files=args.dataset, split=args.split)
    tol = Fraction(str(args.tol))

    url = args.base_url.rstrip("/") + "/chat/completions"
    n = len(ds)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    
    q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 2)

    # 创建进度条
    pbar = tqdm(total=n, desc="infer", ncols=100)

    async def worker(worker_id: int) -> None:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, connector_owner=False) as session:
            while True:
                item = await q.get()
                if item is None:
                    # 收到 Sentinel，退出
                    q.task_done()
                    break

                try:
                    ex = item.ex
                    # 处理图片
                    img = load_image_any(ex["decoded_image"])
                    question = ex["query"]
                    gold = ex["answer"]
                    options = ex.get("choices", []) or []

                    # 推理
                    pred_text = await infer_one(
                        session=session,
                        url=url,
                        model=args.model,
                        question=question,
                        image=img,
                        options=options,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )

                    # 判题
                    final_ans = extract_final_answer(pred_text)
                    correct_now = is_correct(pred_text, str(gold), tol, options=options)
                    reward = 1.0 if correct_now else 0.0

                    result = {
                        "i": item["pid"],
                        "question": question,
                        "prediction": pred_text,
                        "options": options,
                        "answer": str(gold),
                        "final": final_ans,
                        "correct": bool(correct_now),
                        "reward": reward,
                    }

                    if random.random() < 0.01:
                        # 使用 tqdm.write 避免打乱进度条
                        pbar.write(color_kv(result))

                    results[item.i] = result

                except Exception as e:
                    # 捕获所有异常，确保不会卡死
                    err_msg = f"Error in item {item.i}: {str(e)}"
                    # 打印简短错误日志
                    pbar.write(f"\033[91m{err_msg}\033[0m")
                    # 记录失败结果，保证索引对应
                    results[item.i] = {
                        "i": item.i,
                        "question": item.ex.get("question", ""),
                        "prediction": f"ERROR: {str(e)}",
                        "answer": str(item.ex.get("answer", "")),
                        "correct": False,
                        "reward": 0.0,
                        "error": str(e)
                    }
                finally:
                    # 无论成功还是失败，都要更新进度和标记任务完成
                    pbar.update(1)
                    q.task_done()

    # 启动 Workers
    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]

    # 填充队列
    for i in range(n):
        await q.put(ReqItem(i=i, ex=ds[i]))
    
    # 放入结束信号
    for _ in workers:
        await q.put(None)

    # 等待所有任务处理完毕 (queue为空)
    await q.join()
    
    # 关闭进度条
    pbar.close()

    # 等待 Worker 正常退出
    for w in workers:
        await w
    await connector.close()
    # 写入结果
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            if r is None:
                # 理论上不应该发生，因为有 try/except 填充，但以防万一
                r = {"correct": False, "error": "Unknown missing result"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid_results = [r for r in results if r is not None and "error" not in r]
    correct = sum(1 for r in valid_results if r["correct"])
    # 注意：计算准确率时分母应该是总数 n，还是有效数？通常是总数
    acc = correct / n * 100.0 if n > 0 else 0.0

    score_obj = {
        "dataset": args.dataset,
        "split": args.split,
        "n": n,
        "valid_n": len(valid_results),
        "correct": correct,
        "accuracy": acc,
        "tol": float(args.tol),
        "output": args.output,
    }
    with open(args.score_json, "w", encoding="utf-8") as f:
        json.dump(score_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(score_obj, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MathLLMs/MathVision")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", type=str, required=True)

    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout-s", type=int, default=3600)

    p.add_argument("--max-tokens", type=int, default=25600)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    p.add_argument("--tol", type=float, default=1e-6)

    p.add_argument("--output", type=str, default="mathvision_preds.jsonl")
    p.add_argument("--score-json", type=str, default="mathvision_score.json")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")