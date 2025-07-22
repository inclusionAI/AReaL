import argparse
import json
import os
import subprocess
from glob import glob

import numpy as np
from cf_elo_caculator import calculate_cf_elo_from_samples
from rm_maj_eval import group_pred
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", required=True, type=lambda x: x.split(","))
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
    )
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_sample_nodes", default=8, type=int)
    parser.add_argument("--samples_per_node", default=4, type=int)
    parser.add_argument("--n_sampling", default=32, type=int)
    parser.add_argument("--prompt_type", default="r1-distilled-qwen", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate_train", action="store_true")
    parser.add_argument("--max_gen_tokens", default=32768, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--task", default="math", type=str)

    parser.add_argument(
        "--cf_cache_file", type=str, default="./data/codeforces/all_contest_data.json"
    )
    parser.add_argument(
        "--cf_metadata_path", type=str, default="./data/codeforces/metadata_cf.json"
    )
    parser.add_argument(
        "--cf_ratings_path", type=str, default="./data/codeforces/sorted_rating.json"
    )

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.model_path

    args.cf_pass_n = args.num_sample_nodes
    return args


def eval_maj_k_metrics(data_list, k=8):
    # print(f"evaluating maj@{k}")

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        groups, majority_pred = group_pred(
            sample["pred"][:k], strip=False, use_symbol=False
        )
        idx = groups[majority_pred][0]
        right_count += sample["score"][idx]
        count += 1

    task_acc = right_count / count * 100
    # print(f"maj@{k}: {task_acc:.1f}")
    return task_acc


def pass_at_k(data_list, k=8):

    def cur_pass_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    # count, right_count = 0, 0
    pass_at_ks = []
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        correct = sum(sample["score"])
        pass_at_ks.append(cur_pass_k(len(sample["score"]), correct, k))

    return np.mean(pass_at_ks) * 100


def calculate_cf_elo_for_dataset(data_name, results, args):
    if "codeforces" not in data_name.lower():
        return None
    try:
        print(f"\nCalculating CF ELO rating for dataset {data_name}...")
        print(f"Dataset contains {len(results)} problems")

        all_samples = []
        for idx, sample_data in results.items():
            sample = {"idx": idx, "score": sample_data["score"]}
            all_samples.append(sample)

        cf_result = calculate_cf_elo_from_samples(
            all_samples=all_samples,
            pass_n=args.cf_pass_n,
            metadata_path=args.cf_metadata_path,
            ratings_path=args.cf_ratings_path,
            cache_file_path=args.cf_cache_file,
            verbose=True,
        )

        if cf_result:
            print(f"✓ {data_name} CF ELO calculation success:")
            print(f"  Estimated percentile: {cf_result['estimated_percentile']:.1f}%")
            print(f"  Estimated CF rating: {cf_result['estimated_rating']:.0f}")

            return {
                "cf_percentile": cf_result["estimated_percentile"],
                "cf_rating": cf_result["estimated_rating"],
            }
        else:
            print(f"✗ {data_name} CF ELO calculation failed")
            return None

    except Exception as e:
        print(f"✗ {data_name} CF ELO calculation error: {e}")
        return None


def get_metrics(fname_pattern, tokenizer, is_greedy):

    generated = []
    lengths = []
    results = {}

    for fname in glob(fname_pattern):
        datas = load_jsonl(fname)
        for data in tqdm(datas, desc=fname):

            # tokenize
            generated.extend(data["code"])
            if len(generated) > 2000:
                encodings = tokenizer(generated, return_length=True)
                lengths.extend(encodings["length"])
                generated = []

            # answer score
            cur_idx = data["idx"]
            if cur_idx not in results:
                results[cur_idx] = {"pred": [], "score": []}

            results[cur_idx]["pred"] += data["pred"]
            results[cur_idx]["score"] += data["score"]

    if generated:
        encodings = tokenizer(generated, return_length=True)
        lengths.extend(encodings["length"])

    print(len(lengths))
    assert len(lengths) != 0
    if is_greedy:
        return {
            "greedy_length": np.mean(lengths),
            "greedy_acc": pass_at_k(results.values(), 1),
            "num_questions": len(lengths),
        }, results
    else:
        return {
            "sample_length": np.mean(lengths),
            "sample_pass@1": pass_at_k(results.values(), 1),
            "pass@8": (
                pass_at_k(results.values(), 8)
                if args.num_sample_nodes * args.samples_per_node >= 8
                else "-"
            ),
            "pass@16": (
                pass_at_k(results.values(), 16)
                if args.num_sample_nodes * args.samples_per_node >= 16
                else "-"
            ),
            "maj@32": (
                eval_maj_k_metrics(results.values(), 32)
                if args.num_sample_nodes * args.samples_per_node >= 32
                else "-"
            ),
            "pass@32": (
                pass_at_k(results.values(), 32)
                if args.num_sample_nodes * args.samples_per_node >= 32
                else "-"
            ),
        }, results


def process_single_data_name(args, data_name, base_dir, tokenizer):
    cur_dir = os.path.join(base_dir, data_name)
    greedy_prefix = f"test_{args.prompt_type}_-1_seed0_t0.0_topp1.00_topk-1_s0_e-1_n1"
    sampling_prefix = f"test_{args.prompt_type}_-1_seed*_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}_s0_e-1_n{args.samples_per_node}"

    greedy_length_metrics, _ = get_metrics(
        os.path.join(cur_dir, greedy_prefix + ".jsonl"), tokenizer, True
    )
    sampling_metrics, sampling_results = get_metrics(
        os.path.join(cur_dir, sampling_prefix + ".jsonl"), tokenizer, False
    )

    sample_length = sampling_metrics.pop("sample_length")
    output = dict(
        num_questions=greedy_length_metrics["num_questions"],
        greedy_length=greedy_length_metrics["greedy_length"],
        sample_length=sample_length,
        greedy_acc=greedy_length_metrics["greedy_acc"],
        **sampling_metrics,
    )

    cf_elo_result = calculate_cf_elo_for_dataset(data_name, sampling_results, args)
    if cf_elo_result:
        output.update(cf_elo_result)

    return output


if __name__ == "__main__":
    args = parse_args()
    print(f"Evaluation output to {args.output_path}")
    assert args.num_sample_nodes * args.samples_per_node >= args.n_sampling

    eval_dir = f"math_eval_{args.max_gen_tokens}"

    base_dir = os.path.join(args.output_path, eval_dir)
    os.makedirs(base_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    result_path = os.path.join(
        base_dir,
        f"aggregate_parallel_{args.prompt_type}_{args.temperature:.1f}_{args.top_p:.2f}_{args.top_k}.json",
    )

    if not os.path.exists(result_path) or args.overwrite or args.evaluate_train:
        log_path = os.path.join(base_dir, "logs")
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "greedy.log"), "w") as f:
            subprocess.run(
                [
                    "sh",
                    "sh/eval_greedy.sh",
                    args.model_path,
                    str(args.max_gen_tokens),
                    ",".join(args.data_names),
                    args.prompt_type,
                    args.output_path,
                    args.task,
                ],
                text=True,
                stdout=f,
                stderr=f,
            )
        print(f"Evaluation: greedy finished!")

        for i in range(args.num_sample_nodes):
            with open(
                os.path.join(
                    log_path, f"seed-{i+1}-sample-{args.samples_per_node}.log"
                ),
                "w",
            ) as f:
                subprocess.run(
                    [
                        "sh",
                        "sh/eval_sample_with_seed.sh",
                        args.model_path,
                        str(i + 1),
                        str(args.samples_per_node),
                        str(args.max_gen_tokens),
                        ",".join(args.data_names),
                        args.prompt_type,
                        args.output_path,
                        str(args.temperature),
                        str(args.top_p),
                        str(args.top_k),
                        args.task,
                    ],
                    text=True,
                    stdout=f,
                    stderr=f,
                )
            print(f"Evaluation: seed {i + 1} finished!")

        all_results = dict()
        for data_name in args.data_names:
            all_results[data_name] = process_single_data_name(
                args, data_name, base_dir, tokenizer
            )

        if not args.evaluate_train:
            with open(result_path, "w") as f:
                json.dump(all_results, f, indent=2)

    else:
        with open(result_path) as f:
            all_results = json.load(f)

    cf_elo_summary = {}
    for data_name, result in all_results.items():
        if any(key.startswith("cf_") for key in result.keys()):
            cf_elo_summary[data_name] = {
                "percentile": result.get("cf_percentile"),
                "rating": result.get("cf_rating"),
            }

    try:
        from prettytable import PrettyTable

        table = PrettyTable()
        field_names = ["dataset"] + list(all_results[args.data_names[0]].keys())
        table.field_names = field_names
        for k, v in all_results.items():
            formatted_values = []
            for field in field_names[1:]:
                value = v.get(field, "-")
                if isinstance(value, (int, float)) and value != "-":
                    if field.startswith("cf_") and "rating" in field:
                        formatted_values.append(f"{value:.0f}")
                    elif field.startswith("cf_"):
                        formatted_values.append(f"{value:.1f}")
                    else:
                        formatted_values.append(f"{value:.1f}")
                else:
                    formatted_values.append(str(value))
            table.add_row([k] + formatted_values)

        if cf_elo_summary:
            print("\n" + "=" * 50)
            print("CODEFORCES ELO rating summary")
            print("=" * 50)
            cf_table = PrettyTable()
            cf_table.field_names = ["Dataset", "Percentile (%)", "CF Rating"]
            for data_name, cf_data in cf_elo_summary.items():
                cf_table.add_row(
                    [
                        data_name,
                        (
                            f"{cf_data['percentile']:.1f}"
                            if cf_data["percentile"] is not None
                            else "-"
                        ),
                        (
                            f"{cf_data['rating']:.0f}"
                            if cf_data["rating"] is not None
                            else "-"
                        ),
                    ]
                )
            print(cf_table)
            print("=" * 50)

        print(table)
    except ModuleNotFoundError as e:

        print(json.dumps(all_results, indent=2))
