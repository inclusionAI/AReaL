import os
import shutil
import json


def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def custom_collate_fn(batch):
    all_keys = set().union(*(d.keys() for d in batch))
    collated_batch = {}
    for key in all_keys:
        collated_batch[key] = [d.get(key) for d in batch]
    return collated_batch

def worker_dump_rollout_output(sample_info: dict):
    worker_log_root_dir = os.environ.get("WORKER_LOG_DIR", None)
    assert worker_log_root_dir is not None, "WORKER_LOG_DIR environment variable must be set"

    version = sample_info["versions"][-1]
    query_id = sample_info["query_id"]

    json_log_dir = f"{worker_log_root_dir}/rollouts/{version}/json"
    human_log_dir = f"{worker_log_root_dir}/rollouts/{version}/human"
    os.makedirs(json_log_dir, exist_ok=True)
    os.makedirs(human_log_dir, exist_ok=True)
    json_log_file = os.path.join(json_log_dir, f"{query_id}.jsonl")

    with open(json_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample_info, ensure_ascii=False) + "\n")

    human_format_log = f"################# prompt #################\n"
    human_format_log += f"{sample_info['prompt']}\n\n"
    human_format_log += f"----------------- stop reason ------------------\n"
    human_format_log += f"{sample_info['stop_reason']}\n"
    human_format_log += f"---------------- reward ------------------\n"
    human_format_log += f"{sample_info['reward']}\n"
    human_format_log += f"----------------- seq_len -------------------\n"
    human_format_log += f"{sample_info['seq_len']}\n"
    human_format_log += f"----------------- completion -------------------\n"
    human_format_log += f"{sample_info['completion']}\n\n"
    human_format_log += f"################# end #################\n\n"

    human_format_log_file = os.path.join(human_log_dir, f"{query_id}.txt")
    with open(human_format_log_file, "a", encoding="utf-8") as f:
        f.write(human_format_log)
