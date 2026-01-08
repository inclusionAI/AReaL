import argparse
import base64
import io
import json
import random
import re
import time
import concurrent.futures
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# --------------------------
# Utils / Helper Functions (保持不变)
# --------------------------

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
        return Image.new("RGB", (224, 224), (0, 0, 0))

# --------------------------
# Math Extraction & Verification Logic (保持不变)
# --------------------------

_FINAL_RE = re.compile(r"(final answer|final|answer)\s*[:：]\s*(.+)", re.IGNORECASE)
_LATEX_FRAC_RE = re.compile(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}")
_NUM_FRAC_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")
_NUM_INT_RE = re.compile(r"^\s*[+-]?\d+\s*$")
_NUM_DEC_RE = re.compile(r"^\s*[+-]?\d+\.\d+\s*$")
_NUM_SCI_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?[eE][+-]?\d+\s*$")
_CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+)?|[+-]?\d+(?:\.\d+)?[eE][+-]?\d+)")

def strip_wrappers(s: str) -> str:
    if not s: return ""
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
    if start == -1: return None
    i = start + len(key)
    depth = 1
    out: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{": depth += 1; out.append(ch)
        elif ch == "}": depth -= 1; out.append(ch) if depth > 0 else None
        else: out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None

def extract_final_answer(text: str) -> str:
    t = text.strip()
    boxed = extract_boxed_balanced(t)
    if boxed is not None: return strip_wrappers(boxed)
    m = _FINAL_RE.search(t)
    if m: return strip_wrappers(m.group(2))
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    tail = strip_wrappers(lines[-1]) if lines else strip_wrappers(t)
    if len(tail) > 48:
        nums = _LAST_NUM_RE.findall(t)
        if nums: return strip_wrappers(nums[-1])
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
        a, b = int(m.group(1)), int(m.group(2))
        return None if b == 0 else Fraction(a, b)
    if _NUM_INT_RE.match(x): return Fraction(int(x), 1)
    if _NUM_DEC_RE.match(x): return Fraction(x)
    return None

def extract_last_number_token(text: str) -> Optional[str]:
    t = latex_to_simple(strip_wrappers(text))
    nums = _LAST_NUM_RE.findall(t)
    return strip_wrappers(nums[-1]) if nums else None

def infer_choice_from_options(pred_text: str, options: List[str]) -> Optional[str]:
    if not options: return None
    pnorm = normalize_text(pred_text)
    letters = "ABCDE"
    for i, opt in enumerate(options[:5]):
        onorm = normalize_text(str(opt))
        if onorm and onorm in pnorm: return letters[i]
    return None

def is_correct(pred: str, gold: str, tol: Fraction, options: Optional[List[str]] = None) -> bool:
    try:
        g = strip_wrappers(gold).upper()
        if len(g) == 1 and g in "ABCDE":
            pl = extract_choice_letter(pred)
            if pl is not None: return pl == g
            if options is not None:
                inferred = infer_choice_from_options(pred, options)
                return inferred == g if inferred is not None else False
            return False
        p = extract_final_answer(pred)
        p_num = extract_last_number_token(p)
        gn = parse_number(gold)
        pn = parse_number(p_num) if p_num is not None else parse_number(p)
        if pn is not None and gn is not None:
            return abs(pn - gn) <= tol
        return normalize_text(p) == normalize_text(gold)
    except Exception:
        return False

def create_robust_session(pool_size=50):
    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[], 
        allowed_methods=["POST"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_size, 
        pool_maxsize=pool_size      
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def infer_one(
    session: requests.Session, 
    url: str,
    api_key: str,
    model: str,
    question: str,
    image: Image.Image,
    options: List[str],
    max_tokens: int,
    temperature: float,
) -> str:
    data_url = pil_to_data_url(image)
    
    if len(options) > 0:
        if ''.join(options) != 'ABCDE':
            options_str = f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\n"
        else:
            options_str = ""
    else:
        options_str = ''

    input_text = (
        'Please solve the problem step by step and put your answer in one "\\boxed{}". '
        'If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'
        f"{question}\n{options_str}"
    )

    payload = {
        "stream": False,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": {"google": {"thinking_config": {"include_thoughts": True,"thinking_level":"high"}, "thought_tag_marker": "think"}},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_retries = 4
    base_delay = 2.0
    backoff_factor = 2.0

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"[Retry] Starting attempt {attempt + 1}/{max_retries + 1}...")

            resp = session.post(url, json=payload, headers=headers, timeout=600)
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    if choice.get("finish_reason") == "length" and "message" not in choice:
                        return "Error: Output truncated (finish_reason=length)."
                    return choice["message"]["content"]
                except Exception as e:
                    raise RuntimeError(f"Unexpected response format: {resp.text}") from e
            
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                sleep_time = (base_delay * (backoff_factor ** attempt)) + random.uniform(0, 1)
                # 输出 HTTP 错误重试状态
                print(f"API Warning: Received status {resp.status_code}. Retrying in {sleep_time:.2f}s (Attempt {attempt + 1}/{max_retries + 1})...")
                time.sleep(sleep_time)
                continue
            
            # 输出不可重试的错误信息
            print(f"API Critical Error: Status {resp.status_code} | Body: {resp.text[:200]}")
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError, requests.exceptions.SSLError) as e:

            if attempt < max_retries:
                sleep_time = (base_delay * (backoff_factor ** attempt)) + random.uniform(0, 1)
                # 输出网络错误重试状态
                print(f"Network Exception ({type(e).__name__}): {e}. Retrying in {sleep_time:.2f}s (Attempt {attempt + 1}/{max_retries + 1})...")
                time.sleep(sleep_time)
            else:
                print(f"Max retries exceeded due to network error: {e}")
                raise e
    
    raise RuntimeError("Max retries exceeded")


def process_item(item: Dict[str, Any], args: argparse.Namespace, tol: Fraction, url: str) -> Dict[str, Any]:
    with create_robust_session(pool_size=1) as session:
        try:
            ex = item['ex']
            img = load_image_any(ex.get("decoded_image", ex.get("image", ex.get("img"))))
            
            question = ex["question"]
            gold = ex["answer"]
            options = ex.get("options", []) or []

            pred_text = infer_one(
                session=session,
                url=url,
                api_key=args.api_key,
                model=args.model,
                question=question,
                image=img,
                options=options,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            final_ans = extract_final_answer(pred_text)
            correct_now = is_correct(pred_text, str(gold), tol, options=options)
            reward = 1.0 if correct_now else 0.0

            return {
                "i": item['i'],
                "question": question,
                "prediction": pred_text,
                "options": options,
                "answer": str(gold),
                "final": final_ans,
                "correct": bool(correct_now),
                "reward": reward,
            }

        except Exception as e:
            return {
                "i": item['i'],
                "question": item['ex'].get("question", ""),
                "prediction": f"ERROR: {str(e)}",
                "answer": str(item['ex'].get("answer", "")),
                "correct": False,
                "reward": 0.0,
                "error": str(e)
            }

def run(args: argparse.Namespace) -> None:
    ds = load_dataset(args.dataset, split=args.split)
    tol = Fraction(str(args.tol))

    base = args.base_url.rstrip("/")
    url = f"{base}/chat/completions" if not base.endswith("/chat/completions") else base
    
    print(f"Target URL: {url}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Saving incremental results to: {args.output}")

    n = len(ds)
    results: List[Optional[Dict[str, Any]]] = [None] * n
    
    tasks = [{'i': i, 'ex': ds[i]} for i in range(n)]

    with open(args.output, "w", encoding="utf-8") as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_item = {
                executor.submit(process_item, item, args, tol, url): item 
                for item in tasks
            }
            
            pbar = tqdm(total=n, desc="infer", ncols=100)
            
            for future in concurrent.futures.as_completed(future_to_item):
                result = future.result()
                idx = result['i']
                
                results[idx] = result
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
                if random.random() < 0.1:
                     pbar.write(color_kv({
                        "question": result["question"],
                        "prediction": result["prediction"],
                        "id": idx,
                        "correct": result["correct"],
                        "ans": result["answer"],
                        "pred_short": result["final"]
                    }))
                elif "error" in result:
                     pbar.write(f"\033[91mError in item {idx}: {result['error']}\033[0m")
                
                pbar.update(1)
            
            pbar.close()


    # 计算分数
    valid_results = [r for r in results if r is not None and "error" not in r]
    correct = sum(1 for r in valid_results if r["correct"])
    acc = correct / n * 100.0 if n > 0 else 0.0

    score_obj = {
        "dataset": args.dataset,
        "split": args.split,
        "n": n,
        "valid_n": len(valid_results),
        "correct": correct,
        "accuracy": acc,
        "model": args.model
    }
    
    with open(args.score_json, "w", encoding="utf-8") as f:
        json.dump(score_obj, f, ensure_ascii=False, indent=2)

    print("\nEvaluation Result:")
    print(json.dumps(score_obj, ensure_ascii=False, indent=2))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MathLLMs/MathVision")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--base-url", type=str, default="http://matrixllm-pool.global.alipay.com/v1")
    p.add_argument("--api-key", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--concurrency", type=int, default=32) 
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--output", type=str, default="mathvision_preds.jsonl")
    p.add_argument("--score-json", type=str, default="mathvision_score.json")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")