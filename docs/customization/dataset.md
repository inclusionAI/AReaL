# Dataset

AReaL directly integrates with the `Dataset` class from the HuggingFace `datasets`
package. This gives you full flexibility to load, process, and filter your data before
training.

The required columns in your dataset depend on the specific implementation of the
`RolloutWorkflow` (for online reinforcement learning) or the training engines (for
offline training, such as `LMEngine` for Supervised Fine-Tuning (SFT)).

Here are two concrete examples from the existing implementation:

## SFT (Offline Training)

In the SFT example, we see that the loaded data is directly passed to the `train_lm`
method:

```python
# examples/lite/gsm8k_sft.py
def main(args):
    ...
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", tokenizer, rank, world_size),
        collate_fn=pad_sequences_to_tensors,
    )
    ...
    # Run training loop
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            stats = engine.train_lm(data)
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

## GRPO (Online Training)

In the GRPO example, the loaded data is passed to the `InferenceEngine`, rather than the
`TrainEngine`:

```python
# examples/lite/gsm8k_ppo.py
def main(args):
    ...
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", rank, world_size),
        collate_fn=lambda x: x,
    )
    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    workflow = RLVRWorkflow(
        reward_fn=gsm8k_reward_fn,
        ...
    )
    # Run training loop
    ...
    for global_step in range(max_steps):
        batch = rollout.rollout_batch(data, workflow=workflow)
        ...
```

Note that the `collate_fn` here is an identity function, meaning it simply returns the
list of individual data items as a batch. In `rollout_batch`, the data is then
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
        req = LLMRequest(
            input_ids=input_ids,
            ...
        )
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
