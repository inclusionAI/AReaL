import argparse
import hashlib
import json
from glob import glob

import numpy as np
from tqdm import tqdm


def get_hash(text):
    text_bytes = text.encode("utf-8")  # 将文本转换为字节串
    md5_hash = hashlib.md5(text_bytes).hexdigest()
    return md5_hash


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    all_results = {}
    for fname in glob(f"{args.log_path}/*.jsonl"):
        with open(fname) as f:
            cur_results = [json.loads(x) for x in f]

        for x in cur_results:
            # query_id = x['query_id'].split("@")[0]
            if "query_id" in x:
                query_id = x["query_id"].split("@")[0]
            else:
                query_id = get_hash(x["prompt"])
            if query_id not in all_results:
                all_results[query_id] = []

            all_results[query_id].append(x["reward"] > 0)

    all_acc = []
    for query_id, results in sorted(all_results.items(), key=lambda x: x[0]):
        print(query_id, len(results), np.mean(results))
        all_acc.append(sum(results) / len(results))

    print(len(all_acc))
    print(f"Mean accuracy: {np.mean(all_acc)}")
