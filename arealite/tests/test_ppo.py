"""Test script for PPO Trainer implementation."""

from typing import Dict

import pytest
import torch
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    EngineConfig,
    FSDPConfig,
    LLMClientConfig,
    MicroBatchSpec,
    ModelFamily,
    OptimizerConfig,
    PPOTrainerConfig,
    RLVRConfig,
    RolloutControllerConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec
from arealite.api.rollout_api import RolloutWorkflowFactory
from arealite.api.trainer_api import TrainerFactory
from arealite.impl.rollout_controller import RolloutController
from arealite.impl.trainer.ppo import SpmdPPOTrainer, UnpaddedRolloutOutput
from arealite.utils import (
    compute_varlen_position_indices,
    split_dict_tensor_with_cu_seqlens,
)
from realhf.base import constants, name_resolve, seeding
from realhf.impl.model.utils.padding import unpad_input

EXPR_NAME = "test_ppo"
TRIAL_NAME = "test_ppo"
MODEL_PATH = "Qwen/Qwen2.5-0.5B"


def create_mock_input(bs: int = 2, min_seqlen: int = 3, max_seqlen: int = 12) -> Dict:
    """Create mock input data for testing."""
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (bs,), dtype=torch.int, device="cuda"
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(0, 100, (bs, max_seqlen), dtype=torch.long, device="cuda")

    attn_mask = torch.zeros((bs, max_seqlen), dtype=torch.bool, device="cuda")
    attn_mask[
        torch.arange(0, max_seqlen, device="cuda").unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1

    packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attn_mask
    )

    assert torch.allclose(
        cu_seqlens, torch.nn.functional.pad(seqlens.cumsum(0, dtype=torch.int), (1, 0))
    )
    position_ids = compute_varlen_position_indices(int(sum(seqlens)), cu_seqlens)
    prompt_lens = torch.randint(1, min_seqlen, (bs,), dtype=torch.int, device="cuda")
    prompt_mask = torch.arange(max_seqlen, device="cuda").unsqueeze(
        0
    ) < prompt_lens.unsqueeze(1)
    prompt_mask, *_ = unpad_input(prompt_mask, attn_mask)

    return dict(
        input_ids=packed_input_ids.unsqueeze(0),
        attention_mask=None,
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        prompt_mask=prompt_mask.unsqueeze(0),
        use_cache=False,
    )


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


def create_dataset(cfg: DatasetConfig):
    # select five data for test
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
    )
    dataset = dataset.select(range(5))
    return dataset


@pytest.mark.skip("")
def test_engine():
    """Test engine creation and basic functionality."""
    print("Testing PPO train creation...")

    train_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="train",
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    valid_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="test",
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    engine_config = EngineConfig(
        type=ModelFamily("qwen2", False),
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    sglang_client_config = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path=MODEL_PATH,
    )

    ppo_config = PPOTrainerConfig(
        actor=engine_config,
    )

    train_config = TrainerConfig(
        type="ppo",
        ppo=ppo_config,
    )

    rollout_controller_config = RolloutControllerConfig(
        llm_client=sglang_client_config,
    )

    args = TrainingArgs(
        mode="local",
        n_nodes=1,
        n_gpus_per_node=1,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        trainer=train_config,
        rollout=rollout_controller_config,
    )

    rollout_controller = None
    if args.rollout is not None:
        rollout_controller = RolloutController(args, args.rollout)
    train_dataset = create_dataset(args.train_dataset)
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = create_dataset(args.valid_dataset)
    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            rollout_controller=rollout_controller,
        )
        trainer.train()

    print("All tests passed!")


@pytest.fixture(scope="module")
def args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    args.rollout.llm_client.tokenizer_path = MODEL_PATH
    args.train_dataset = DatasetConfig(
        path="openai/gsm8k",
        name="main",
        split="train",
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    args.trainer = TrainerConfig(type="ppo", ppo=PPOTrainerConfig())
    args.trainer.ppo.actor = EngineConfig(
        type=ModelFamily("qwen2", False),
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )
    args.trainer.ppo.ref = EngineConfig(
        type=ModelFamily("qwen2", False),
        path=MODEL_PATH,
        gradient_checkpointing=False,
        backend=EngineBackendConfig(type="hf"),
    )
    args.rollout.llm_client = LLMClientConfig(
        server_backend="sglang",
        tokenizer_path=MODEL_PATH,
    )
    args.rollout.workflow.rlvr = RLVRConfig(solution_path="nothing")
    args.rollout.gconfig.max_new_tokens = 16
    name_resolve.reconfigure(args.cluster.name_resolve)
    yield args
    name_resolve.reset()


@pytest.mark.parametrize("kl_ctl", [0.0, 0.1])
@pytest.mark.parametrize("bs", [2, 4])
@pytest.mark.parametrize("n_samples", [1, 2])
def test_train_step(args, kl_ctl, bs, n_samples):
    args.rollout.gconfig.n_samples = n_samples
    args.trainer.ppo.kl_ctl = kl_ctl
    args.train_dataset.batch_size = bs
    # Create mock rollout controller and trainer
    rollout_factory = RolloutWorkflowFactory(args)
    workflow = rollout_factory.make_workflow(args.rollout.workflow)
    rollout_controller = RolloutController(args, args.rollout, workflow=workflow)
    dataset = load_dataset("openai/gsm8k", name="main", split="train").select(range(10))

    trainer = SpmdPPOTrainer(
        args=args,
        trainer_config=args.trainer,
        train_dataset=dataset,
        rollout_controller=rollout_controller,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=100,
        train_batch_size=args.train_dataset.batch_size,
    )
    trainer.actor.init_distributed(None, ft_spec)
    trainer.actor.eval()
    if trainer.ref is not None:
        trainer.ref.init_distributed(None, ft_spec)
        trainer.ref.eval()

    # Create mock UnpaddedRolloutOutput
    min_seqlen, max_seqlen = 8, 16
    mock_inputs = create_mock_input(bs * n_samples, min_seqlen, max_seqlen)
    input_ids = mock_inputs["input_ids"]
    prompt_mask = mock_inputs["prompt_mask"].squeeze(0)
    device = input_ids.device
    logprobs = (
        -torch.randn_like(
            input_ids.squeeze(0), dtype=torch.float32, device=device
        ).abs()
        * 0.1
    )
    rewards = torch.randn(bs * n_samples, dtype=torch.float32, device=device)
    seq_no_eos_mask = torch.randint(
        0, 2, (bs * n_samples,), dtype=torch.bool, device=device
    )
    rollout_output = UnpaddedRolloutOutput(
        loaded_data={},
        model_inputs=mock_inputs,
        prompt_mask=prompt_mask,
        rewards=rewards,
        seq_no_eos_mask=seq_no_eos_mask,
        logprobs=logprobs,
    )
    stats_list = trainer._train_step(rollout_output)

    # Verify the output
    assert isinstance(stats_list, list), "Should return a list of stats"
    assert len(stats_list) > 0, "Should return non-empty stats list"
    assert isinstance(stats_list[0], dict), "Each stat should be a dictionary"

    # Check that stats contain expected keys
    expected_keys = [
        "ppo_actor/advantages",
        "ppo_actor/kl_rewards",
        "ppo_actor/final_reward",
    ]
    for key in expected_keys:
        assert any(key in stats for stats in stats_list), f"Missing expected key: {key}"
