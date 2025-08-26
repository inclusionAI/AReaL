import pytest
import torch
import random
from tensordict import TensorDict, NonTensorData
from arealite.workflow.partial_rollout import PartialRolloutWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import LLMRequest, LLMResponse
from transformers import PreTrainedTokenizerFast

class DummyTokenizer(PreTrainedTokenizerFast):
    def __init__(self):
        self.eos_token_id = 0
        pass
    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=True):
        return " ".join([str(i) for i in ids])
    def __call__(self, text, truncation=True, max_length=None, padding=False, return_length=True, return_attention_mask=False):
        tokens = [ord(c) for c in text]
        return {"input_ids": tokens, "length": len(tokens)}

class DummyEngine:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
    
    async def agenerate(self, req):
        # Randomly generate different token sequences with varying lengths
        min_tokens = 1
        max_tokens = 8
        output_length = self.rng.randint(min_tokens, max_tokens)
        
        # Generate random tokens (simulating different vocabulary ranges)
        output_tokens = [self.rng.randint(100, 200) for _ in range(output_length)]
        
        # Generate corresponding logprobs (negative values)
        output_logprobs = [self.rng.uniform(-2.0, -0.1) for _ in range(output_length)]
        
        # Generate versions (usually 1 for simplicity, but could vary)
        output_versions = [self.rng.randint(1, 3) for _ in range(output_length)]
        
        # Randomly choose stop reason
        stop_reasons = ["stop", "length", "interrupt"]
        stop_reason = self.rng.choice(stop_reasons)
        
        return LLMResponse(
            input_tokens=req.input_ids,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
            stop_reason=stop_reason
        )

@pytest.fixture
def dummy_reward_fn():
    def reward_fn(prompt, completion, prompt_ids, completion_ids, **kwargs):
        # Return the length of completion_ids as reward
        return float(len(completion_ids))
    return reward_fn

@pytest.fixture
def dummy_gconfig():
    class DummyGConfig:
        n_samples = 1
        max_new_tokens = 2
        min_new_tokens = 1
        greedy = True
        top_p = 1.0
        top_k = 0
        temperature = 1.0
        stop_token_ids = []
        def new(self, **kwargs):
            return self
    return DummyGConfig()

@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()

@pytest.mark.asyncio
async def test_arun_episode_new_prompt(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    engine = DummyEngine(seed=42)
    data = {
        "prompt": ["abc"],
        "task": ["math"],
        "solutions": ["solution"],
        "query_id": ["qid"],
        "index_in_group": [0]
    }
    results = await workflow.arun_episode(engine, data.copy())
    assert isinstance(results, TensorDict)
    td = results
    
    # Check that input_ids contains original prompt + generated tokens
    assert td["input_ids"].shape[0] == 1  # batch size 1
    assert td["input_ids"][0][:3].tolist() == [97, 98, 99]  # original prompt "abc"
    
    # Check that rewards equals the length of generated tokens
    generated_tokens_count = td["input_ids"].shape[1] - 3  # total - prompt length
    assert td["rewards"].item() == float(generated_tokens_count)
    
    assert td["task_ids"].item() == 0
    assert td["query_id"][0] == "qid"
    assert td["index_in_group"][0] == 0
    assert td["task"][0] == "math"

@pytest.mark.asyncio
async def test_arun_episode_previous_ids(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    engine = DummyEngine(seed=123)
    data = {
        "previous_ids": [[97, 98, 99]],
        "previous_version": [[1, 1, 1]],
        "previous_logprobs": [[-0.5, -0.2, -0.1]],
        "previous_seq_no_eos_mask": [[1]],
        "previous_prompt_len": [3],
        "previous_rewards": [1],
        "task": ["math"],
        "solutions": ["solution"],
        "query_id": ["qid"],
        "index_in_group": [1]
    }
    results = await workflow.arun_episode(engine, data.copy())
    assert isinstance(results, TensorDict)
    td = results
    
    # Check that input_ids contains previous_ids + generated tokens
    assert td["input_ids"][0][:3].tolist() == [97, 98, 99]  # previous_ids
    
    # Check that rewards equals the length of generated tokens
    generated_tokens_count = td["input_ids"].shape[1] - 3  # total - previous_ids length
    assert td["rewards"].item() == float(generated_tokens_count)
    
    assert td["task_ids"].item() == 0
    assert td["query_id"][0] == "qid"
    assert td["index_in_group"][0] == 1

@pytest.mark.asyncio
async def test_arun_episode_variable_length_outputs(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    """Test that different seeds produce different output lengths and the reward reflects this"""
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    
    data_template = {
        "prompt": ["test prompt"],
        "task": ["math"],
        "solutions": ["solution"],
        "query_id": ["qid"],
        "index_in_group": [0]
    }
    
    results_list = []
    seeds = [1, 2, 3, 4, 5]
    
    for seed in seeds:
        engine = DummyEngine(seed=seed)
        data = data_template.copy()
        results = await workflow.arun_episode(engine, data)
        results_list.append(results)
    
    # Check that we get different output lengths (hence different rewards)
    rewards = [result["rewards"].item() for result in results_list]
    sequence_lengths = [result["input_ids"].shape[1] for result in results_list]
    
    # There should be some variation in outputs
    assert len(set(rewards)) > 1, "Expected some variation in reward values"
    assert len(set(sequence_lengths)) > 1, "Expected some variation in sequence lengths"
    
    # Verify reward equals generated token count for each result
    prompt_length = len(dummy_tokenizer("test prompt")["input_ids"])
    for result, reward in zip(results_list, rewards):
        generated_length = result["input_ids"].shape[1] - prompt_length
        assert reward == float(generated_length)

@pytest.mark.asyncio
async def test_arun_episode_different_tasks(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    """Test with different task types"""
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    
    engine = DummyEngine(seed=42)
    
    # Test different tasks (assuming these exist in RL_TASKS)
    tasks = ["math", "code"]  # Add more if available
    
    for i, task in enumerate(tasks):
        data = {
            "prompt": [f"prompt for {task}"],
            "task": [task],
            "solutions": [f"solution for {task}"],
            "query_id": [f"qid_{task}"],
            "index_in_group": [i]
        }
        
        results = await workflow.arun_episode(engine, data.copy())
        td = results
        
        assert td["task"][0] == task
        assert td["query_id"][0] == f"qid_{task}"
        assert td["index_in_group"][0] == i
        
        # Verify reward is still based on completion length
        prompt_length = len(dummy_tokenizer(f"prompt for {task}")["input_ids"])
        generated_length = td["input_ids"].shape[1] - prompt_length
        assert td["rewards"].item() == float(generated_length)

@pytest.mark.asyncio 
async def test_arun_episode_stop_reasons(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    """Test different stop reasons and their effects"""
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    
    data = {
        "prompt": ["test stop reasons"],
        "task": ["math"],
        "solutions": ["solution"],
        "query_id": ["qid_stop"],
        "index_in_group": [0]
    }
    
    # Test multiple times to potentially hit different stop reasons
    stop_reasons_seen = set()
    for seed in range(10, 20):  # Use different seeds
        engine = DummyEngine(seed=seed)
        results = await workflow.arun_episode(engine, data.copy())
        
        # We can't directly check stop_reason from the result, but we can verify
        # that seq_no_eos_mask is set correctly based on the implementation
        seq_no_eos_mask = results["seq_no_eos_mask"].item()
        assert isinstance(seq_no_eos_mask, bool)
        
        # Verify basic structure
        assert results["input_ids"].shape[0] == 1
        assert results["rewards"].item() >= 1.0  # At least 1 token generated

@pytest.mark.asyncio
async def test_arun_episode_complex_scenarios(dummy_reward_fn, dummy_gconfig, dummy_tokenizer):
    """Test complex scenarios with longer prompts and continuation"""
    workflow = PartialRolloutWorkflow(
        reward_fn=dummy_reward_fn,
        gconfig=dummy_gconfig,
        tokenizer=dummy_tokenizer
    )
    
    # Test with longer prompt
    long_prompt = "This is a much longer prompt that should result in more input tokens " * 3
    engine = DummyEngine(seed=999)
    
    data = {
        "prompt": [long_prompt],
        "task": ["math"],
        "solutions": ["complex solution"],
        "query_id": ["qid_complex"],
        "index_in_group": [0]
    }
    
    results = await workflow.arun_episode(engine, data.copy())
    td = results
    
    # Verify that prompt length is correctly handled
    prompt_length = len(dummy_tokenizer(long_prompt)["input_ids"])
    total_length = td["input_ids"].shape[1]
    generated_length = total_length - prompt_length
    
    assert td["rewards"].item() == float(generated_length)
    assert generated_length >= 1  # At least some tokens were generated
    
    # Test continuation scenario
    first_part_ids = td["input_ids"][0][:prompt_length + 2].tolist()  # Take first 2 generated tokens
    first_part_versions = [1] * len(first_part_ids)
    
    continuation_data = {
        "previous_ids": [first_part_ids],
        "previous_version": [first_part_versions],
        "previous_logprobs": [[-0.5] * len(first_part_ids)],
        "previous_seq_no_eos_mask": [[1] * len(first_part_ids)],
        "previous_prompt_len": [len(first_part_ids)],
        "previous_rewards": [1],
        "task": ["math"],
        "solutions": ["continued solution"],
        "query_id": ["qid_complex"],
        "index_in_group": [1]
    }
    
    engine2 = DummyEngine(seed=888)
    continuation_results = await workflow.arun_episode(engine2, continuation_data.copy())
    
    # Verify continuation
    cont_td = continuation_results
    assert cont_td["input_ids"][0][:len(first_part_ids)].tolist() == first_part_ids
    
    # Reward should be based on newly generated tokens only
    new_generated_length = cont_td["input_ids"].shape[1] - len(first_part_ids)
    assert cont_td["rewards"].item() == float(new_generated_length)

def test_init_tokenizer_required(dummy_reward_fn, dummy_gconfig):
    with pytest.raises(ValueError):
        PartialRolloutWorkflow(
            reward_fn=dummy_reward_fn,
            gconfig=dummy_gconfig
        )