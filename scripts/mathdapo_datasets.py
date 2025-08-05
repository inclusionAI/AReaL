from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node




def mathdapo_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from realhf.impl.dataset.math_parser_greso import process_results_greso_v2

    return int(process_results_greso_v2(completions, answer)[0])

def eval_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    from realhf.impl.dataset.math_parser_greso import process_results_v2

    return int(process_results_v2(completions, answer)[0])


def _apply_greso_template(problem):
    return f"""Please solve the following math problem: {problem}. The assistant first thinks about the reasoning process step by step and then provides the user with the answer. Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step. """


#####
# math+dapo
#####

def _process_mathdapo_rl_dataset(math_dataset: Dataset, dapo_dataset: Dataset, cfg):
    def process_math(sample):
        messages = [{"role": "user", "content": _apply_greso_template(sample["problem"])}]
        return {"messages": messages, "answer": sample["solution"]}

    def process_dapo(sample):
        messages = [{"role": "user", "content": _apply_greso_template(sample["prompt"])}]
        return {"messages": messages, "answer": sample["target"]}

    math_dataset = math_dataset.map(process_math, remove_columns=math_dataset.column_names)
    dapo_dataset = dapo_dataset.map(process_dapo, remove_columns=dapo_dataset.column_names)
    dataset = concatenate_datasets([math_dataset, dapo_dataset])
    
    if cfg.max_prompt_length:
        dataset = dataset.filter(lambda r: len(r["messages"]) <= cfg.max_prompt_length)
    
    return dataset

def get_mathdapo_dataset(split, rank, world_size, cfg=None):
    math_dataset = load_dataset(path="xDAN2099/lighteval-MATH", split=split)
    dapo_dataset = load_dataset(path="haizhongzheng/DAPO-Math-17K-cleaned", split=split)

    math_dataset = split_dataset_by_node(math_dataset, rank=rank, world_size=world_size)
    dapo_dataset = split_dataset_by_node(dapo_dataset, rank=rank, world_size=world_size)

    return _process_mathdapo_rl_dataset(math_dataset, dapo_dataset, cfg)


#####
# math500
#####

def _preprocess_math500_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": _apply_greso_template(sample["problem"])}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])
    return dataset

def get_math500_dataset(split, rank, world_size, cfg=None):
    dataset = load_dataset(path="HuggingFaceH4/MATH-500", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return _preprocess_math500_rl_dataset(dataset)


#####
# amc
#####

def _preprocess_amc_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": _apply_greso_template(sample["problem"])}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])
    return dataset

def get_amc_dataset(split, rank, world_size, cfg=None):
    dataset = load_dataset(path="AI-MO/aimo-validation-amc", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return _preprocess_amc_rl_dataset(dataset)

