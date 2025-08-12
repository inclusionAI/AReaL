from datasets import Split, load_dataset
from datasets.distributed import split_dataset_by_node


def get_werewolf_rl_dataset(path, split, rank, world_size):
    """
    Load werewolf prompts for RL training.

    Note that this dataset is actually a placeholder. The only use is to decide the length
    of training for werewolf experiment.

    Each row in the JSONL dataset should contain:
        {
            "id": str,
            "prompt": str,
      }
    """
    dataset = load_dataset("json", data_files=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        message = [{"role": "user", "content": sample["prompt"]}]
        res = {"messages": message}
        if "id" in sample:
            res["query_id"] = sample["id"]
        return res

    dataset = dataset.map(process).remove_columns([col for col in dataset.column_names if col not in ["messages", "query_id"]])
    return dataset
