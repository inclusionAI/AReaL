import os, base64, argparse, asyncio, re, json, hashlib, time
from typing import List, Dict, Any
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm
from PIL import Image
import random

def hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def summarize_ious(ious: List[float]) -> Dict[str, float]:
    valid = [x for x in ious if x is not None]
    if not valid:
        return {"mean_iou": 0.0, "pass_rate_05": 0.0}
    mean_iou = sum(valid) / len(valid)
    pass_rate_05 = sum(1 for x in valid if x > 0.5) / len(valid)
    return {"mean_iou": mean_iou, "pass_rate_05": pass_rate_05}

def scale_bbox(bbox, w, h):
    import math
    x1, y1, x2, y2 = bbox
    cleaned = []
    for v in (x1, y1, x2, y2):
        v = float(v)
        if not math.isfinite(v):
            v = 0.0
        if v < 0.0:
            v = 0.0
        if v > 1000.0:
            v = 1000.0
        cleaned.append(v)
    x1, y1, x2, y2 = cleaned

    x1 = round(x1 * w / 1000.0); x2 = round(x2 * w / 1000.0)
    y1 = round(y1 * h / 1000.0); y2 = round(y2 * h / 1000.0)

    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

    if x1 == x2 and w > 1:
        x2 = min(x1 + 1, w - 1)
    if y1 == y2 and h > 1:
        y2 = min(y1 + 1, h - 1)

    return [float(x1), float(y1), float(x2), float(y2)]


def inv_scale_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    if w > 0 and h > 0:
        x1 = max(0, min(round(x1), w - 1)); x2 = max(0, min(round(x2), w - 1))
        y1 = max(0, min(round(y1), h - 1)); y2 = max(0, min(round(y2), h - 1))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    nx1 = round(x1 * 1000.0 / w) if w > 0 else 0
    nx2 = round(x2 * 1000.0 / w) if w > 0 else 1000
    ny1 = round(y1 * 1000.0 / h) if h > 0 else 0
    ny2 = round(y2 * 1000.0 / h) if h > 0 else 1000
    nx1 = max(0, min(int(nx1), 1000)); nx2 = max(0, min(int(nx2), 1000))
    ny1 = max(0, min(int(ny1), 1000)); ny2 = max(0, min(int(ny2), 1000))
    if nx1 == nx2 and w > 1: nx2 = min(nx1 + 1, 1000)
    if ny1 == ny2 and h > 1: ny2 = min(ny1 + 1, 1000)
    return [float(nx1), float(ny1), float(nx2), float(ny2)]

def apply_remap(bbox, w, h, remap_mode):
    mode = remap_mode.strip().lower() if isinstance(remap_mode, str) else remap_mode
    if mode == "scale": return scale_bbox(bbox, w, h)
    if mode == "inverse": return inv_scale_bbox(bbox, w, h)
    if mode == "keep" or mode in (None, "", False): return bbox
    raise ValueError(f"Unknown remap mode: {remap_mode}")

def compute_iou(boxA, boxB):
    if boxA is None or boxB is None: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA); inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = areaA + areaB - inter_area
    if denom == 0: return 0.0
    return inter_area / denom

async def call_one(client: AsyncOpenAI, model: str, rec: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    base_image_dir = os.getenv("BASE_IMG_DIR")
    rec_path = rec["image"].lstrip("/")
    path = os.path.join(base_image_dir, rec_path)
    with Image.open(path) as img:
        w, h = img.size
    if not (w == rec["width"] and h == rec["height"]):
        raise ValueError(f"width/height mismatch: image.size=({w},{h}), rec=({rec['width']},{rec['height']}), image={rec['image']}, sent={rec['sent']}")
    gt_bbox = rec["bbox"]

    prompt = PROMPT_TEMPLATE.format(sent=rec["sent"])
    content = [
        {"type": "image_url", "image_url": {"url": to_data_url(path), "detail": "high"}},
        {"type": "text", "text": prompt},
    ]

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,              
            "repetition_penalty": 1.0,
        },
    )
    t1 = time.perf_counter()
    latency_s = t1 - t0
    ch = resp.choices[0]
    token_count = getattr(resp.usage, "completion_tokens", None)
    txt = ch.message.content

    m = re.search(r'"bbox_2d"\s*:\s*\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]', txt, flags=re.S)
    m2 = re.search(r'\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]', txt, flags=re.S)
    nums = re.findall(r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?', txt, flags=re.S)

    if m:
        bbox = [float(m.group(i)) for i in range(1, 5)]
    elif m2:
        bbox = [float(m2.group(i)) for i in range(1, 5)]
    elif len(nums) >= 4:
        bbox = list(map(float, nums[-4:]))
    else:
        bbox = None

    scaled_bbox = apply_remap(bbox, w, h, REMAP_MODE) if bbox is not None else None
    iou = compute_iou(scaled_bbox, gt_bbox) if scaled_bbox is not None else 0.0
    result = {
        "generation": txt,
        "scaled_bbox": scaled_bbox,
        "iou": iou,
        "latency_s": latency_s,
        "response_tokens": token_count,
    }

    if random.random() < 0.003:
        print(result)
    return result

async def bounded_call(idx: int, rec: Dict[str, Any], sem: asyncio.Semaphore, client: AsyncOpenAI, model: str, max_tokens: int):
    async with sem:
        out = await call_one(client, model, rec, max_tokens)
        return idx, out

async def main_async(args):
    ds = load_dataset("json", split='train', data_files=args.data_json)
    cols = ds.column_names

    if "sent" not in cols:
        def extract_sent(row):
            text = " ".join(x.get("value", "") for x in row["conversations"])
            m = re.findall(r'<ref>\s*["“]?(.+?)["”]?\s*</ref>', text, flags=re.S)
            return m[-1].strip()
        ds = ds.add_column("sent", [extract_sent(row) for row in ds])
        cols = ds.column_names

    if "bbox" not in cols and "modal_bbox" in cols:
        ds = ds.add_column("bbox", ds["modal_bbox"])
        cols = ds.column_names
    elif "bbox" not in cols and "conversations" in cols:
        def extract_bbox(row):
            text = " ".join(x.get("value", "") for x in row["conversations"])
            m = re.findall(r'<box>\s*\[([^\]]+)\]\s*</box>', text, flags=re.S)
            nums = re.findall(r'-?\d+', m[-1])
            return {"bbox": list(map(float, nums))}
        ds = ds.map(extract_bbox, num_proc=64, desc="Adding column: bbox")
        cols = ds.column_names

    for need in ["image", "sent", "bbox", "width", "height"]:
        if need not in ds.column_names:
            raise ValueError(f"Missing field '{need}'; available columns: {ds.column_names}")

    if not args.output_dir:
        raise ValueError("--output_dir is required")
    os.makedirs(args.output_dir, exist_ok=True)

    client = AsyncOpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key="none")
    sem = asyncio.Semaphore(args.concurrency)

    global PROMPT_TEMPLATE
    if args.prompt_template.lower() == "grounding":
        PROMPT_TEMPLATE = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref> "
    elif args.prompt_template.lower() == "amodal":
        PROMPT_TEMPLATE = "Please provide the amodal bounding box coordinate of the region this sentence describes: <ref>{sent}</ref>"
    elif args.prompt_template.lower() == "qwen3":
        PROMPT_TEMPLATE = "Locate {sent}, output its bbox coordinates using JSON format"
    elif args.prompt_template.lower() == "qwen3_amodal":
        PROMPT_TEMPLATE = "Locate {sent} amodally, output its bbox coordinates using JSON format"
    else:
        raise ValueError(f"Unknown prompt_template: {args.prompt_template}")

    if args.box_remap not in ("keep", "scale", "inverse"):
        raise ValueError(f"Unknown box_remap: {args.box_remap}")
    else:
        global REMAP_MODE
        REMAP_MODE = args.box_remap
    start_time = time.perf_counter()
    tasks = []
    for i in range(len(ds)):
        rec = {"image": ds[i]["image"], "sent": ds[i]["sent"], "bbox": ds[i]["bbox"], "height": ds[i]["height"], "width": ds[i]["width"]}
        tasks.append(asyncio.create_task(bounded_call(i, rec, sem, client, args.model, args.max_tokens)))

    generations = [None] * len(ds); scaled_bbox = [None] * len(ds); iou = [None] * len(ds)
    all_lat = []; all_tokens = []
    pbar = tqdm(total=len(tasks), desc="vLLM async infer (no chunk)")
    for coro in asyncio.as_completed(tasks):
        idx, out = await coro
        generations[idx] = out.get("generation")
        scaled_bbox[idx] = out.get("scaled_bbox")
        latency = out.get("latency_s", None); tokens = out.get("response_tokens", None)
        if latency is not None: all_lat.append(latency)
        if tokens is not None: all_tokens.append(tokens)
        iou[idx] = out.get("iou"); pbar.update(1)
    pbar.close()
    end_time = time.perf_counter() 
    whole_time = end_time - start_time
    stats = summarize_ious(iou)
    mean_iou = stats["mean_iou"]; pass_rate = stats["pass_rate_05"]
    total_tokens = int(sum(x for x in all_tokens if x is not None)) if all_tokens else 0
    total_time = int(sum(all_lat)) if all_lat else 0
    case_num = int(len(ds))

    print("case_num", case_num)
    print("total_time", total_time)
    print("total_tokens", total_tokens)
    print("mean_time", total_time / case_num if case_num > 0 else 0)
    print("mean_tokens", total_tokens / case_num if case_num > 0 else 0)
    print("mean_iou", mean_iou)
    print("pass_rate@0.5", pass_rate)
    print("whole_time", whole_time)

    detail = {}
    for idx in range(len(ds)):
        v = 0.0 if iou[idx] is None else float(iou[idx])
        detail[idx] = {"v": v, "hash": hash(ds[idx]["sent"])}

    with open(os.path.join(args.output_dir, "iou_detail.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "iou.txt"), "w", encoding="utf-8") as f:
        f.write(f"# IOU scores (N={len(iou)})\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"pass_rate@0.5: {pass_rate:.6f}\n")

    with open(os.path.join(args.output_dir, "latency.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_time_sec: {total_time:.3f}\n")
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"case_number: {case_num}\n")
        f.write(f"whole_time_sec: {whole_time:.3f}\n") 

    ds_out = (ds.add_column("generated_result", generations)
                .add_column("scaled_bbox", scaled_bbox)
                .add_column("iou", iou))
    ds_out.save_to_disk(args.output_dir)
    print(f"Finish: Output saved to {args.output_dir}")

def parse_args():
    ap = argparse.ArgumentParser("Async concurrent VLM inference via vLLM (OpenAI-compatible)")
    ap.add_argument("--data_json", required=True, help="COCO2014 train style JSON file path")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000", help="vLLM OpenAI service address (excluding /v1)")
    ap.add_argument("--model", required=True, help="Model name loaded on the vLLM server")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--prompt_template", type=str, required=True, help="grounding|amodal|qwen3")
    ap.add_argument("--box_remap", type=str, required=True, help="keep|scale|inverse")
    return ap.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
