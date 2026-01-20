import asyncio
import io
import json

import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

API_KEY = ""
MODEL_NAME = "gemini-3-pro-preview"
CONCURRENCY_LIMIT = 32
DATASET_PATH = r"..\..\data\MathVision\data\test-00000-of-00001-3532b8d3f1b4047a.parquet"
INPUT_TEMPLATE = f"Please solve the problem step by step and put your answer in one '\\boxed{{}}'. If it is a multiple choice question, only one letter is allowed in the '\\boxed{{}}'.\n{{question}}\n{{options}}"
OUTPUT_JSONL_PATH = "MathVision_gemini_clean.jsonl"  
OUTPUT_JSON_PATH = "MathVision_gemini_clean.json"    

client = genai.Client(api_key=API_KEY)

def construct_prompt(row):
    question = row["question"]
    options = row["options"]

    if len(options) > 0:
        if len(options) == 5:
            options_str = "".join(options)
            if options_str != "ABCDE":
                formatted_options = (
                    f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\n"
                )
            else:
                formatted_options = ""
        else:
            formatted_options = "\n".join(options)
    else:
        formatted_options = ""

    input_text = INPUT_TEMPLATE.format(question=question, options=formatted_options)
    return input_text

async def call_model_async(row, semaphore):
    async with semaphore:
        retries = 3
        for attempt in range(retries):
            try:
                text_prompt = construct_prompt(row)

                decoded_image = row["decoded_image"]
                if isinstance(decoded_image, dict):
                    if "bytes" in decoded_image and isinstance(decoded_image["bytes"], (bytes, bytearray)):
                        decoded_image = Image.open(io.BytesIO(decoded_image["bytes"])).convert("RGB")
                image_input = decoded_image

                contents = [text_prompt]
                if image_input:
                    contents.append(image_input)

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinkingLevel="high",
                            include_thoughts=True
                        ),
                        temperature=0.0,
                    ),
                )

                thinking_process = ""
                final_answer = ""

                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "thought") and part.thought:
                            thinking_process += part.text
                        else:
                            final_answer += part.text

                result = {
                    "id": row["id"],
                    "generated_thinking": thinking_process,
                    "generated_answer": final_answer,
                    "len_thinking": len(thinking_process),
                    "len_answer": len(final_answer),
                    "len_total": len(thinking_process) + len(final_answer),
                    "status": "success",
                    "original_response": str(response),
                }
                return result

            except Exception as e:
                if attempt == retries - 1:
                    print(f"Error processing ID {row['id']}: {e}")
                    return {
                        "id": row["id"],
                        "generated_thinking": "",
                        "generated_answer": "",
                        "len_thinking": 0,
                        "len_answer": 0,
                        "len_total": 0,
                        "status": f"error: {str(e)}",
                        "original_response": None,
                    }
                await asyncio.sleep(2 * (attempt + 1))

async def process_and_save(row, semaphore, fp, write_lock):

    result = await call_model_async(row, semaphore)

    record = dict(row)
    record.pop("decoded_image", None)  
    record.update(result)

    line = json.dumps(record, ensure_ascii=False) + "\n"
    async with write_lock:
        fp.write(line)
        fp.flush()

    return result

async def main():
    print("Loading dataset...")
    ds = load_dataset("parquet", data_files=DATASET_PATH, split="train")
    ## read processed data from jsonl to skip
    try:
        with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as f:
            processed_ids = {json.loads(line)["id"] for line in f if line.strip()}
        ds = ds.filter(lambda x: x["id"] not in processed_ids)
        print(f"Skipped {len(processed_ids)} already processed examples.")
    except FileNotFoundError:
        pass
    
    
    total = len(ds)
    print(f"Total examples: {total}")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    write_lock = asyncio.Lock()

    total_chars = 0
    total_thinking = 0
    total_answer = 0
    success_cnt = 0

    print("Starting concurrent generation (immediate saving to JSONL)...")
    with open(OUTPUT_JSONL_PATH, "a", encoding="utf-8") as fp:
        tasks = [
            asyncio.create_task(process_and_save(row, semaphore, fp, write_lock))
            for row in ds
        ]

        for fut in tqdm(asyncio.as_completed(tasks), total=total):
            r = await fut
            total_chars += r["len_total"]
            total_thinking += r["len_thinking"]
            total_answer += r["len_answer"]
            if r["status"] == "success":
                success_cnt += 1

    avg_thinking = total_thinking / total if total else 0.0
    success_rate = success_cnt / total * 100 if total else 0.0

    print(f"\n======== Statistics ========")
    print(f"Total Characters: {total_chars}")
    print(f"  - Thinking: {total_thinking}")
    print(f"  - Answer:   {total_answer}")
    print(f"Average Thinking Length: {avg_thinking:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"JSONL saved to: {OUTPUT_JSONL_PATH}")

    print(f"Converting JSONL -> JSON array: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
