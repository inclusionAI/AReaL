import os
from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    DatasetPreprocessor,
    EngineConfig,
    GRPOTrainerConfig,
    OptimizerConfig,
    RLVRConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.io_struct import FinetuneSpec
from arealite.api.rollout_api import RolloutCollectorFactory
from arealite.system.rollout_controller import RolloutController
from arealite.tests.utils import mock_rollout_output
from realhf.base import constants, name_resolve, seeding
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from arealite.impl.dataset.VL_dataset import VLDataset
from arealite.api.trainer_api import TrainerFactory
from arealite.api.dataset_api import Multimodal_DatasetFactory

EXPR_NAME = "test_vlm_grpo"
TRIAL_NAME = "test_vlm_grpo"
MODEL_PATH = "/storage/openpsi/models/Qwen2-VL-7B"


def create_args():
    args = TrainingArgs(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME)
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    seeding.set_random_seed(args.seed, EXPR_NAME)
    args.train_dataset = DatasetConfig(
        path="/storage/openpsi/data/clevr_count_70k/",
        preprocessor=DatasetPreprocessor("clevr_count_70k_rl"),
        split="train",
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    args.trainer = TrainerConfig(type="vl_grpo", grpo=GRPOTrainerConfig())
    args.trainer.grpo.actor = EngineConfig(
        path=MODEL_PATH,
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )
    args.trainer.grpo.ref = EngineConfig(
        path=MODEL_PATH,
        gradient_checkpointing=False,
        backend=EngineBackendConfig(type="hf"),
    )
    args.rollout.model_path = MODEL_PATH
    args.rollout.server_backend = "vl_sglang"
    args.rollout.collector.rlvr = RLVRConfig(solution_path="nothing")
    args.rollout.gconfig.max_new_tokens = 16
    name_resolve.reconfigure(args.cluster.name_resolve)
    return args


def test_train_step(args, kl_ctl, bs, n_samples, recompute, use_decoupled_loss):
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    args.rollout.gconfig.n_samples = n_samples
    args.trainer.grpo.kl_ctl = kl_ctl
    args.trainer.grpo.recompute_logprobs = recompute
    args.trainer.grpo.use_decoupled_loss = use_decoupled_loss
    args.train_dataset.batch_size = bs
    args.rollout.collector.rlvr.reward_type = "clevr_count_70k"
    
    # Create mock rollout controller and trainer
    rollout_factory = RolloutCollectorFactory(args)
    collector = rollout_factory.make_collector(args.rollout.collector)
    rollout_controller = RolloutController(args, args.rollout, collector=collector)
    dataset_factory = Multimodal_DatasetFactory(args)
    dataset = dataset_factory.make_dataset(args.train_dataset, 0, 1)

    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=dataset,
            valid_dataset=None,
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

    example_image=dataset[0]["images"]
    rollout_output = mock_rollout_output(bs, n_samples)
    for traj in rollout_output:
        traj.images=example_image
    stats_list = trainer._train_step(rollout_output)

    # Verify the output
    breakpoint()
    assert isinstance(stats_list, list)
    assert len(stats_list) == args.trainer.grpo.ppo_n_minibatches
    for stats in stats_list:
        assert isinstance(stats, dict)
        for k, v in stats.items():
            assert isinstance(v, float)


# 参数化调用示例
def run_tests():
    args = create_args()
    
    kl_ctl_values = [0.0, 0.1]
    bs_values = [4]
    n_samples_values = [2]
    recompute_values = [False, True]
    use_decoupled_loss_values = [False, True]

    for kl_ctl in kl_ctl_values:
        for bs in bs_values:
            for n_samples in n_samples_values:
                for recompute in recompute_values:
                    for use_decoupled_loss in use_decoupled_loss_values:
                        test_train_step(
                            args,
                            kl_ctl,
                            bs,
                            n_samples,
                            recompute,
                            use_decoupled_loss,
                        )


# 运行测试
if __name__ == "__main__":
    run_tests()
