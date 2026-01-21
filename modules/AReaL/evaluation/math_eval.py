import argparse
import itertools
import json
import os
import random
import time
from datetime import datetime
from parser import *

import ray
import torch
from data_loader import load_data
from evaluate import evaluate
from model_utils import generate_completions, load_hf_lm_and_tokenizer
from python_executor import PythonExecutor
from rm_maj_eval import group_pred
from tqdm import tqdm
from trajectory import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    args.top_k = -1 if args.temperature == 0 else args.top_k

    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    args.data_parallel_size = len(available_gpus) // args.tensor_parallel_size
    return args


def eval_maj_k_metrics(data_list, k=8):
    print(f"evaluating maj@{k}")

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
    print(f"maj@{k}: {task_acc:.1f}")
    return task_acc


def pass_at_k(data_list, k=8):
    print(f"evaluating pass@{k}")

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        # assert sum(sample['score']) in [0, k]
        for x in sample["score"][:k]:
            if x:
                right_count += 1
                break
        count += 1

    task_acc = right_count / count * 100
    print(f"pass@{k}: {task_acc:.1f}")
    return task_acc


def pass_at_1(data_list, k=1):
    assert k == 1
    print(f"evaluating pass@{k}")

    count, right_count = 0, 0
    for sample in data_list:
        assert len(sample["score"]) >= k, sample
        for x in sample["score"]:
            right_count += x
            count += 1

    print("pass_at_1", right_count, count)
    task_acc = right_count / count * 100
    print(f"pass@{k}: {task_acc:.1f}")
    return task_acc


def pass_at_k_v2(data_list, k=8):

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


def generate_in_parallel(requests, model_args, sampling_params, data_parallel_size):
    @ray.remote
    def run_inference_one_model(
        model_args: dict, sampling_params, requests, cuda_visisble_devices
    ):
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in cuda_visisble_devices]
        )
        # print("OS.ENVIRON", json.dumps({x: os.environ[x]  for x in sorted(dict(os.environ))}))
        llm = LLM(**model_args)
        return llm.generate(requests, sampling_params=sampling_params)

    # print("OUT_OS_ENVIRON", json.dumps({x: os.environ[x]  for x in sorted(dict(os.environ))}))
    all_cuda_visisble_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    requests = [list(x) for x in distribute(data_parallel_size, requests)]
    inputs = (
        (model_args, sampling_params, req, cuda_visisble_devices)
        for req, cuda_visisble_devices in zip(
            requests, np.array_split(all_cuda_visisble_devices, data_parallel_size)
        )
    )
    object_refs = [run_inference_one_model.remote(*x) for x in inputs]
    results = ray.get(object_refs)
    ray.shutdown()
    return undistribute(results)


# from more_itertools import distribute
from itertools import islice, tee


def distribute(n, iterable):
    """Distribute the items from *iterable* among *n* smaller iterables.

        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]

    If the length of *iterable* is not evenly divisible by *n*, then the
    length of the returned iterables will not be identical:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]

    If the length of *iterable* is smaller than *n*, then the last returned
    iterables will be empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]

    This function uses :func:`itertools.tee` and may require significant
    storage.

    If you need the order items in the smaller iterables to match the
    original iterable, see :func:`divide`.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    children = tee(iterable, n)
    return [islice(it, index, None, n) for index, it in enumerate(children)]


def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        raise RuntimeError
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"

    eval_dir = f"math_eval_{args.max_tokens_per_call}"

    out_file = f"{output_dir}/{eval_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_n{args.n_sampling}.jsonl"
    os.makedirs(f"{output_dir}/{eval_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{eval_dir}/{data_name}/")
            # if f.endswith(".jsonl") and f.startswith(out_file_prefix)
            if f == os.path.basename(out_file)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{eval_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        # breakpoint()
        if args.data_parallel_size <= 1:
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=args.tensor_parallel_size,
                # distributed_executor_backend="ray",
                trust_remote_code=True,
                enforce_eager=True,
                # dtype="float16",
                disable_custom_all_reduce=True,
                disable_sliding_window=True,
                max_model_len=32768,
                enable_chunked_prefill=False,
                swap_space=32,
            )
        else:
            print(
                f"TP = {args.tensor_parallel_size}\n",
                f"DP = {args.data_parallel_size}",
            )
            llm = dict(
                model=args.model_name_or_path,
                tensor_parallel_size=args.tensor_parallel_size,
                # distributed_executor_backend="ray",
                trust_remote_code=True,
                enforce_eager=True,
                # dtype="float16",
                disable_custom_all_reduce=True,
                disable_sliding_window=True,
                max_model_len=32768,
                enable_chunked_prefill=False,
                swap_space=32,
            )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    # max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] or args.use_vllm else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                seed=args.seed,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens_per_call,
                n=args.n_sampling,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                ),
            )
            if args.data_parallel_size <= 1:

                outputs = llm.generate(prompts[:: args.n_sampling], sampling_params)
            else:
                outputs = generate_in_parallel(
                    prompts[:: args.n_sampling],
                    llm,
                    sampling_params,
                    args.data_parallel_size,
                )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [x.text for output in outputs for x in output.outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time
    print("time_use", time_use)
    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    if args.n_sampling > 1:
        result_json[f"maj@{args.n_sampling}"] = eval_maj_k_metrics(
            all_samples, k=args.n_sampling
        )
        result_json[f"pass@{args.n_sampling}"] = pass_at_k_v2(
            all_samples, k=args.n_sampling
        )
        if args.n_sampling > 16:
            result_json[f"pass@16"] = pass_at_k_v2(all_samples, k=16)
        if args.n_sampling > 8:
            result_json[f"pass@8"] = pass_at_k_v2(all_samples, k=8)
        result_json["pass@1"] = pass_at_k_v2(all_samples, k=1)
        result_json["acc"] = result_json["pass@1"]

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
