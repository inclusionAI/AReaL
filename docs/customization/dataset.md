# Dataset

## Option 1: Using AReaLite (Recommended)

AReaLite directly integrates with the `Dataset` class from the HuggingFace `datasets`
package. This gives you full flexibility to load, process, and filter your data before
training.

The required columns in your dataset depend on the specific implementation of the
`RolloutWorkflow` (for online reinforcement learning) or the training engines (for
offline training, such as `LMEngine` for Supervised Fine-Tuning (SFT)).

Here are two concrete examples from the existing implementation:

### SFT (Offline Training)

In the SFT example, we see that the loaded data is directly passed to the `train_lm`
method:

```python
# examples/arealite/gsm8k_sft.py
def main(args):
    ...
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    ...
    # Run training loop
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            stats = engine.train_lm(data)
            engine.step_lr_scheduler()
            stats_tracker.scalar(**stats)
```

In this case, the `train_lm` method requires the keys "input_ids", "attention_mask", and
"loss_mask" to function. We first tokenize the dataset to extract the "input_ids" and
"loss_mask". Then, the `pad_sequences_to_tensors` method is used to batch multiple
sequences and append the "attention_mask":

```python
def process_gsm8k_sft_dataset(dataset: Dataset, tokenizer):
    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    # Remove unnecessary columns to avoid errors during collation
    dataset = dataset.map(process).remove_columns(["question", "answer"])
    return dataset

def get_gsm8k_dataset(split, tokenizer, rank, world_size):
    dataset = load_dataset(path="openai/gsm8k", name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_sft_dataset(dataset, tokenizer)
```

### GRPO (Online Training)

In the GRPO example, the loaded data is passed to the `InferenceEngine`, rather than the
`TrainEngine`:

```python
# examples/arealite/gsm8k_ppo.py
def main(args):
    ...
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    # Run training loop
    ...
    for global_step in range(max_steps):
        batch = rollout.rollout_batch(data, workflow=workflow)
        ...
```

Note that the `collate_fn` here is an identity function, meaning it simply returns the
list of individual data items as a batch. In the `InferenceEngine`, the data is then
dispatched to multiple concurrent executions of `workflow.arun_episode`, where each
dispatched data corresponds to a single episode.

The `RLVRWorkflow` implementation extracts the "messages" field from the data dictionary
as the prompt for generating a response. Additionally, this data is passed to the
`reward_fn` as keyword arguments, which allows the reward function to make use of other
dataset fields, like "answers". Here’s an example:

```python
class RLVRWorkflow(RolloutWorkflow):

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
        ...
        reward = self.reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **data,
        )
```

Thus, the "messages" field must be constructed when loading the dataset, and the reward
function should be defined to handle the dataset's specific fields. Here’s how you can
process the dataset for this example:

```python
def process_gsm8k_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["question"]}]
        return {"messages": messages}

    # The dataset has two fields "messages" and "answer"
    dataset = dataset.map(process).remove_columns(["question"])
    return dataset

def get_gsm8k_dataset(split, rank, world_size):
    dataset = load_dataset(path="openai/gsm8k", name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_rl_dataset(dataset)

def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    # "answer" is passed in through "**data"
    from realhf.impl.dataset.math_parser import process_results

    return int(process_results(completions, answer)[0])
```

## Option 2: Using the Legacy Dataset (NOT Recommended)

### Define Your Dataset

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

### Implement Core Methods

Every dataset class must implement the following two core methods:

#### 1. `__len__` Method

Returns the size of the dataset:

```python
def __len__(self):
    return len(self.data_samples)
```

#### 2. `__getitem__` Method

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

#### Dataset Examples

We provide some examples of dataset under `realhf/impl/dataset/`:

- For SFT, please refer `prompt_answer_dataset.py`.
- For Reward model training, please refer `rw_paired_dataset.py`
- For RL training, please refer `math_code_dataset.py`

### Data Format Requirements

#### JSONL File Format

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

### Registration and Configuration

#### Register Dataset

Register your dataset at the end of your dataset file:

```python
# in realhf/impl/dataset/my_custom_dataset.py
data_api.register_dataset("my-custom", MyCustomDataset)
```

#### Modify Experiment Configuration

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
