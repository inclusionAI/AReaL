# Dataset Customization

This guide provides detailed examples of how to create custom datasets in AReaL for model training.

## Define Your Dataset

Create a new file under `realhf/impl/dataset/`, for example, `my_custom_dataset.py`. Your `Dataset` must implement the `torch.utils.data.Dataset` interface and follow the framework's conventions.

```python
class MyCustomDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        # Your custom parameters
        custom_param: float = 1.0,
    ):
        """Custom dataset initialization
        
        Args:
            util: Dataset utility class containing tokenizer, seed, distributed info, etc.
            max_length: Maximum sequence length
            dataset_path: Path to dataset file (optional)
            dataset_builder: Data construction function (optional, alternative to dataset_path)
            custom_param: Your custom parameter
        """
        self._util = util
        self.max_length = max_length
        
        # Load and split dataset
        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)
        
        # Your custom data processing logic
        ...
```

## Implement Core Methods

Every dataset class must implement the following three core methods:

### 1. `__len__` Method

Returns the size of the dataset:

```python
def __len__(self):
    return len(self.data_samples)
```

### 2. `__getitem__` Method

Returns the sample at the specified index, must return a `SequenceSample` object:

```python
def __getitem__(self, idx):
    # Get raw data
    sample = self.data_samples[idx]
    
    # Process data
    ...
    
    # Return SequenceSample object
    return data_api.SequenceSample.from_default(
        ids=[sample["id"]],
        seqlens=[len(processed_data["input_ids"])],
        data=dict(
            packed_prompts=torch.tensor(processed_data["input_ids"], dtype=torch.long),
            # Other necessary data fields
        ),
    )
```

### 3. `filter` Property

The `filter` method enables dynamic data filtering during training based on evaluation scores. This is particularly useful for curriculum learning or removing samples that the model has already mastered.

```python
def filter(self, eval_scores: Dict[str, float]):
    """Filter data samples based on evaluation scores
    
    Args:
        eval_scores: Dictionary mapping sample IDs to their evaluation scores.
                    Higher scores typically indicate better performance or easier samples.
    
    This method allows you to:
    - Remove samples that exceed a certain performance threshold
    - Implement curriculum learning by filtering out mastered content
    - Maintain training efficiency by focusing on challenging samples
    """
    scores_to_remove = {}
    
    for pop_idx, actual_idx in enumerate(self.active_indices):
        data_id = self.ids[actual_idx]
        if data_id in eval_scores and eval_scores[data_id] > self.filter_threshold:
            scores_to_remove[pop_idx] = eval_scores[data_id]
    
    # Control filtering quantity based on max_filter_percentage
    max_remove = int(len(self.active_indices) * self.max_filter_percentage)
    indices_to_remove = sorted(
        scores_to_remove.keys(),
        key=lambda x: scores_to_remove[x],
        reverse=True
    )[:max_remove]
    
    # Remove samples from active indices
    for pop_idx in sorted(indices_to_remove, reverse=True):
        self.active_indices.pop(pop_idx)
        
    logger.info(f"Filtered {len(indices_to_remove)} samples, "
               f"{len(self.active_indices)} samples remain.")
```

#### Configuration Parameters:

- `filter_threshold`: Score threshold above which samples may be removed
- `max_filter_percentage`: Maximum percentage of dataset to filter in one call
- `active_indices`: Internal list tracking which samples are currently active

### Dataset Examples

We provide some examples of dataset under `realhf/impl/dataset/`:
- For SFT, please refer `prompt_answer_dataset.py`.
- For Reward model training, please refer `rw_paired_dataset.py`
- For RL training, please refer `math_code_dataset.py`

## Data Format Requirements

### JSONL File Format

Your data file should be in JSONL format, with one JSON object per line:

- Math Data
```json
{"qid": "sample_1", "prompt": "Solve this math problem: 2+2=", "solutions": ["\\boxed{4}"]}
```
- Code Data
```json
{"qid": "sample_2", "prompt": "Code problem", "input_output": "{\"inputs\": [\"5\\n2 3 5 10 12\\n\"], \"outputs\": [\"17\\n\"]}"}
```

### Required Fields
- `qid`: Unique identifier for the sample
- `prompt`: Input prompt text
- `task`: Task type ("math" and "code" are supported now.)
- `solutions`: (Required for Math task) List of solutions
- `input_output`: (Required for Code task) test cases for code problem

## Registration and Configuration

### Register Dataset

Register your dataset at the end of your dataset file:

```python
# in realhf/impl/dataset/my_custom_dataset.py
data_api.register_dataset("my-custom", MyCustomDataset)
```

### Modify Experiment Configuration

Use your new dataset in the experiment configuration (refer to `realhf/experiments/common/*_exp.py`):

```python
# in your experiment config file
@property
def datasets(self) -> List[DatasetAbstraction]:
    return [
        DatasetAbstraction(
            "my-custom",  # Your registered name
            args=dict(
                dataset_path=self.dataset_path,
                max_length=self.max_length,
                custom_param=self.custom_param,
                # Other initialization parameters
            ),
        )
    ]
```
