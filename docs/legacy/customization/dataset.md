# Dataset (Legacy)

> **Note**: While this legacy approach works, we strongly recommend using the new AReaL
> for new projects. It provides better flexibility, cleaner abstractions, and easier
> maintenance.

## Define Your Dataset

Create a new file under `realhf/impl/dataset/`, for example, `my_custom_dataset.py`.
Your `Dataset` must implement the `torch.utils.data.Dataset` interface and follow the
framework's conventions.

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

Every dataset class must implement the following two core methods:

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

### Dataset Examples

We provide some examples of dataset under `realhf/impl/dataset/`:

- For SFT, please refer `prompt_answer_dataset.py`.
- For Reward model training, please refer `rw_paired_dataset.py`
- For RL training, please refer `math_code_dataset.py`

## Data Format Requirements

### JSONL File Format

Your data file should be in JSONL format, with one JSON object per line. If you are
using our PromptDataset implementation, your data should be like:

- Math Data

```json
{"qid": "sample_1", "prompt": "Solve this math problem: 2+2=", "solutions": ["\\boxed{4}"]}
```

- Code Data

```json
{"qid": "sample_2", "prompt": "Code problem", "input_output": "{\"inputs\": [\"5\\n2 3 5 10 12\\n\"], \"outputs\": [\"17\\n\"]}"}
```

- `qid`: Unique identifier for the sample
- `prompt`: Input prompt text
- `task`: Task type, used to distinguish how to calculate the reward. ("math" and "code"
  are supported now.)

Note: There is no format restriction for a customized dataset as long as it can be
loaded by your custom code.

## Registration and Configuration

### Register Dataset

Register your dataset at the end of your dataset file:

```python
# in realhf/impl/dataset/my_custom_dataset.py
data_api.register_dataset("my-custom", MyCustomDataset)
```

### Modify Experiment Configuration

Use your new dataset in the experiment configuration (refer to
`realhf/experiments/common/*_exp.py`):

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
