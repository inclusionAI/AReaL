from datasets import Dataset
import torch
from torch.utils.data import Sampler

def process_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["prompt"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["prompt"])
    return dataset


def get_shuffle_indices(size: int, seed: int):
    """Generate shuffled indices given seed and (dataset) size."""
    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_idx = torch.randperm(size, generator=g).tolist()  # type: ignore[arg-type]
    return shuffle_idx


class ShuffleSampler(Sampler):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.shuffle_indices = get_shuffle_indices(size=len(data_source), seed=seed)

    def __iter__(self):
        for idx in self.shuffle_indices:
            yield idx
        
    def __len__(self):
        return len(self.data_source)
