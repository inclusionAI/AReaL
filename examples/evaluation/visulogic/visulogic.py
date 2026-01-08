import argparse, asyncio, base64, io, json, random, re, aiohttp, traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from PIL import Image
from tqdm.asyncio import tqdm

# --- 辅助工具 ---
def color_kv(d: Dict[str, Any]) -> str:
    GREEN, CYAN, RESET = "\033[92m", "\033[96m", "\033[0m"
    return "{ " + ", ".join(f"{GREEN}{k}{RESET}={CYAN}{v}{RESET}" for k, v in d.items()) + " }"

def pil_to_data_url(img: Image.Image) -> str:
    if img.mode != "RGB": img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def extract_letter_answer(text: str) -> str:
    """提取单选题答案：优先匹配 \boxed{}，其次是 Final Answer，最后是文中末尾字母"""
    boxed = re.search(r"\\boxed\s*\{\s*([A-E])\s*\}", text, re.I)
    if boxed: return boxed.group(1).upper()
    final = re.search(r"(?:answer|final)\s*[:：]\s*([A-E])", text, re.I)
    if final: return final.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else ""

@dataclass
class ReqItem:
    i: int
    ex: Dict[str, Any]

# --- 推理函数 ---
async def infer_one(session, url, model, question, image, max_tokens, temperature) -> str:
    data_url = pil_to_data_url(image)
    input_text = f"Question: {question}\nSolve the problem step by step and put your final choice (A, B, C, D, or E) in \\boxed{{}}."
    
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": input_text},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"API Error {resp.status}: {await resp.text()}")
        out = await resp.json()
        return out["choices"][0]["message"]["content"]

# --- 主逻辑 ---
async def run(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    # 使用关键字参数 data_files 加载，避免路径黑名单报错
    ds = load_dataset("parquet", data_files=args.dataset, split=args.split)
    n = len(ds)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    url = args.base_url.rstrip("/") + "/chat/completions"
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 2)
    pbar = tqdm(total=n, desc="infer", ncols=100)

    async def worker() -> None:
        # connector_owner=False 保证 session 关闭时不销毁共享的 connector
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, connector_owner=False) as session:
            while True:
                item = await q.get()
                if item is None:
                    q.task_done()
                    break
                try:
                    ex = item.ex
                    img = ex["image"].convert("RGB")
                    gold = str(ex["label"]).strip().upper()

                    pred_text = await infer_one(
                        session=session, url=url, model=args.model,
                        question=ex["question"], image=img,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                    )

                    pred_ans = extract_letter_answer(pred_text)
                    is_ok = (pred_ans == gold)

                    res = {
                        "i": item.i,
                        "question": ex["question"],
                        "prediction": pred_text,
                        "extracted": pred_ans,
                        "answer": gold,
                        "correct": is_ok,
                    }
                    results[item.i] = res

                    if random.random() < 0.05:
                        pbar.write(color_kv({"question": ex["question"],"prediction": pred_text,"id": item.i, "gold": gold, "pred": pred_ans, "ok": is_ok}))

                except Exception as e:
                    pbar.write(f"\033[91mError in item {item.i}: {str(e)}\033[0m")
                    results[item.i] = {
                        "i": item.i,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "correct": False
                    }
                finally:
                    pbar.update(1)
                    q.task_done()

    # 启动 Workers
    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]

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
    
    # 关闭连接器
    await connector.close()

    # 写入结果
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            if r is None: r = {"correct": False, "error": "Unknown missing result"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    valid_results = [r for r in results if r and "error" not in r]
    correct = sum(1 for r in valid_results if r["correct"])
    acc = correct / n * 100.0 if n > 0 else 0.0

    score_obj = {"dataset": args.dataset, "n": n, "correct": correct, "accuracy": acc}
    with open(args.score_json, "w", encoding="utf-8") as f:
        json.dump(score_obj, f, indent=2)
    print(json.dumps(score_obj, indent=2))

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Path to parquet file")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout_s", type=int, default=3600)
    p.add_argument("--max-tokens", type=int, default=10240)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", type=str, default="preds.jsonl")
    p.add_argument("--score-json", type=str, default="score.json")
    return p

if __name__ == "__main__":
    try:
        asyncio.run(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")