# Copyright 2025 Ant Group Inc. All Rights Reserved.

import os
import shutil
from typing import *

import pytest

from realhf.api.core.data_api import MicroBatchSpec
from realhf.api.core.system_api import ExperimentSaveEvalControl
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.base import cluster, testing
from realhf.experiments.common.ppo_math_exp import (
    GenerationHyperparameters,
    PPOHyperparameters,
    PPOMATHConfig,
)
from tests.experiments.utils import run_test_exp
from tests.fixtures import *


@pytest.fixture(params=["llama"])
def model_class(request):
    return request.param


@pytest.fixture(params=[testing.TESTING_DATASET_SIZE])
def math_dataset(request, save_path):
    with open(os.getenv("REAL_MATH_METADATA_PATH"), "r") as f:
        query_ids = list(json.load(f).keys())
    size = request.param
    max_prompt_len = 8
    max_resp_len = 8
    dataset = []
    for i in range(size):
        prompt_len = random.randint(1, max_prompt_len)
        n_pairs = random.randint(1, 5)
        d = dict(
            query_id=query_ids[i],
            prompt=generate_random_sentence(prompt_len),
        )
        dataset.append(d)
    with open(str(save_path / "math_dataset.json"), "w") as f:
        json.dump(dataset, f)
    return dataset


@pytest.mark.parametrize(
    "dp,pp,mp",
    [
        (1, 1, 1),
        (2, 1, 2),
        (1, 2, 1),
        (1, 1, 2),
    ],
)
def test_ppo_symm(
    tmp_path_factory,
    tokenizer,
    math_dataset,
    save_path,
    cpu_hf_model,
    mconfig,
    dp,
    pp,
    mp,
):
    # Setup experiment env. Should be done before any other operations.
    log_root = tmp_path_factory.mktemp("ppo")
    cluster.spec.fileroot = str(log_root)
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    minbs = 32
    exp_cfg = PPOMATHConfig(
        experiment_name=testing._DEFAULT_EXPR_NAME,
        trial_name=testing._DEFAULT_TRIAL_NAME,
        mode="local",
        allocation_mode=f"m{mp}d{dp}p{pp}",
        n_nodes=1,
        n_gpus_per_node=mp * dp * pp,
        actor=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        ref=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
        ),
        critic=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            init_critic_from_actor=True,
            backend="mock_train",
        ),
        rew=ModelTrainEvalConfig(
            path=str(save_path),
            init_critic_from_actor=True,
            init_from_scratch=True,
        ),
        dataset=PromptOnlyDatasetConfig(
            path=str(save_path / "math_dataset.json"),
            max_prompt_len=mconfig.n_positions // 2,
            train_bs_n_seqs=minbs,
            fill_to_max_length=False,
        ),
        ppo=PPOHyperparameters(
            gen=GenerationHyperparameters(
                max_new_tokens=4,
                min_new_tokens=4,
                greedy=True,
                use_cuda_graph=False,
            ),
        ),
    )

    run_test_exp(exp_cfg)


# The global resharding strategy, where all MFCs
# occupy the same device mesh but with different
# parallelization strategies.
@pytest.mark.parametrize("actor_gen", [(1, 2, 1)])
@pytest.mark.parametrize("actor_train", [(1, 1, 2)])
@pytest.mark.parametrize("critic_inf", [(1, 1, 2)])
@pytest.mark.parametrize("critic_train", [(1, 2, 1)])
@pytest.mark.parametrize("ref_inf", [(1, 1, 2)])
@pytest.mark.parametrize("rew_inf", [(1, 2, 1)])
def test_ppo_global_reshard(
    tmp_path_factory,
    tokenizer,
    math_dataset,
    save_path,
    cpu_hf_model,
    mconfig,
    actor_gen,
    actor_train,
    critic_inf,
    critic_train,
    ref_inf,
    rew_inf,
):
    # Setup experiment env. Should be done before any other operations.
    log_root = tmp_path_factory.mktemp("ppo-global-reshard")
    cluster.spec.fileroot = str(log_root)
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    minbs = 32
    exp_cfg = PPOMATHConfig(
        experiment_name=testing._DEFAULT_EXPR_NAME,
        trial_name=testing._DEFAULT_TRIAL_NAME,
        mode="local",
        allocation_mode="manual",
        n_nodes=1,
        n_gpus_per_node=2,
        actor=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        ref=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
        ),
        critic=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            init_critic_from_actor=True,
            backend="mock_train",
        ),
        rew=ModelTrainEvalConfig(
            path=str(save_path),
            init_critic_from_actor=True,
            init_from_scratch=True,
        ),
        dataset=PromptOnlyDatasetConfig(
            path=str(save_path / "math_dataset.json"),
            max_prompt_len=mconfig.n_positions // 2,
            train_bs_n_seqs=minbs,
            fill_to_max_length=False,
        ),
        ppo=PPOHyperparameters(
            gen=GenerationHyperparameters(
                max_new_tokens=4,
                min_new_tokens=4,
                greedy=True,
                use_cuda_graph=False,
            ),
        ),
        actor_gen=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=actor_gen[0],
                model_parallel_size=actor_gen[1],
                pipeline_parallel_size=actor_gen[2],
            )
        ),
        actor_train=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=actor_train[0],
                model_parallel_size=actor_train[1],
                pipeline_parallel_size=actor_train[2],
            ),
        ),
        critic_inf=MFCConfig(
            mb_spec=MicroBatchSpec(max_tokens_per_mb=32),
            parallel=ParallelismConfig(
                data_parallel_size=critic_inf[0],
                model_parallel_size=critic_inf[1],
                pipeline_parallel_size=critic_inf[2],
            ),
        ),
        rew_inf=MFCConfig(
            mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
            parallel=ParallelismConfig(
                data_parallel_size=rew_inf[0],
                model_parallel_size=rew_inf[1],
                pipeline_parallel_size=rew_inf[2],
            ),
        ),
        ref_inf=MFCConfig(
            mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
            parallel=ParallelismConfig(
                data_parallel_size=ref_inf[0],
                model_parallel_size=ref_inf[1],
                pipeline_parallel_size=ref_inf[2],
            ),
        ),
        critic_train=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=critic_train[0],
                model_parallel_size=critic_train[1],
                pipeline_parallel_size=critic_train[2],
            ),
        ),
    )
    run_test_exp(exp_cfg)


# Actor/critic train and ref_inf/rew_inf are on disjoint
# device meshes and executed concurrently.
@pytest.mark.parametrize("actor_gen", [(2, 2, 1)])
@pytest.mark.parametrize("critic_inf", [(2, 1, 2)])
def test_ppo_param_realloc_sub_device_mesh(
    tmp_path_factory,
    tokenizer,
    math_dataset,
    save_path,
    cpu_hf_model,
    mconfig,
    actor_gen,
    critic_inf,
):
    # Setup experiment env. Should be done before any other operations.
    log_root = tmp_path_factory.mktemp("ppo-submesh")
    cluster.spec.fileroot = str(log_root)
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )

    minbs = 32
    exp_cfg = PPOMATHConfig(
        experiment_name=testing._DEFAULT_EXPR_NAME,
        trial_name=testing._DEFAULT_TRIAL_NAME,
        mode="local",
        allocation_mode="manual",
        n_nodes=1,
        n_gpus_per_node=8,
        actor=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        ref=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
        ),
        critic=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            init_critic_from_actor=True,
            backend="mock_train",
        ),
        rew=ModelTrainEvalConfig(
            path=str(save_path),
            init_critic_from_actor=True,
            init_from_scratch=True,
        ),
        dataset=PromptOnlyDatasetConfig(
            path=str(save_path / "math_dataset.json"),
            max_prompt_len=mconfig.n_positions // 2,
            train_bs_n_seqs=minbs,
            fill_to_max_length=False,
        ),
        ppo=PPOHyperparameters(
            gen=GenerationHyperparameters(
                max_new_tokens=4,
                min_new_tokens=4,
                greedy=True,
                use_cuda_graph=False,
            ),
        ),
        actor_gen=MFCConfig(
            device_mesh="NODE01:0,1,2,3",
            parallel=ParallelismConfig(
                data_parallel_size=actor_gen[0],
                model_parallel_size=actor_gen[1],
                pipeline_parallel_size=actor_gen[2],
            ),
        ),
        actor_train=MFCConfig(
            device_mesh="NODE01:4,5,6,7",
            parallel=ParallelismConfig(
                data_parallel_size=4,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            ),
        ),
        critic_inf=MFCConfig(
            device_mesh="NODE01:4,5,6,7",
            parallel=ParallelismConfig(
                data_parallel_size=critic_inf[0],
                model_parallel_size=critic_inf[1],
                pipeline_parallel_size=critic_inf[2],
            ),
        ),
        rew_inf=MFCConfig(
            device_mesh="NODE01:4,5,6,7",
            parallel=ParallelismConfig(
                data_parallel_size=4,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            ),
        ),
        ref_inf=MFCConfig(
            device_mesh="NODE01:4,5,6,7",
            parallel=ParallelismConfig(
                data_parallel_size=1,
                model_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        ),
        critic_train=MFCConfig(
            device_mesh="NODE01:4,5,6,7",
            parallel=ParallelismConfig(
                data_parallel_size=2,
                model_parallel_size=1,
                pipeline_parallel_size=2,
            ),
        ),
    )

    run_test_exp(exp_cfg)


@pytest.mark.parametrize("freq_step", [3, 4, 7])
@pytest.mark.parametrize("freq_epoch", [1, 2, 3])
@pytest.mark.parametrize("bs", [30, 80, 100])
def test_ppo_save(
    tmp_path_factory,
    tokenizer,
    save_path,
    cpu_hf_model,
    mconfig,
    freq_step,
    freq_epoch,
    bs,
):
    # Setup experiment env. Should be done before any other operations.
    log_root = tmp_path_factory.mktemp("ppo")
    cluster.spec.fileroot = str(log_root)
    constants.set_experiment_trial_names(
        testing._DEFAULT_EXPR_NAME, testing._DEFAULT_TRIAL_NAME
    )
    shutil.rmtree(constants.MODEL_SAVE_ROOT, ignore_errors=True)
    os.makedirs(constants.MODEL_SAVE_ROOT, exist_ok=True)

    total_train_epochs = 3

    exp_cfg = PPOMATHConfig(
        experiment_name=testing._DEFAULT_EXPR_NAME,
        trial_name=testing._DEFAULT_TRIAL_NAME,
        mode="local",
        allocation_mode="manual",
        n_nodes=1,
        n_gpus_per_node=2,
        actor=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            backend="mock_train",
        ),
        ref=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
        ),
        critic=ModelTrainEvalConfig(
            path=str(save_path),
            init_from_scratch=True,
            init_critic_from_actor=True,
            backend="mock_train",
        ),
        rew=ModelTrainEvalConfig(
            path=str(save_path),
            init_critic_from_actor=True,
            init_from_scratch=True,
        ),
        dataset=PromptOnlyDatasetConfig(
            path=str(save_path / "math_dataset.json"),
            max_prompt_len=mconfig.n_positions // 2,
            train_bs_n_seqs=bs,
            fill_to_max_length=False,
        ),
        exp_ctrl=ExperimentSaveEvalControl(
            total_train_epochs=total_train_epochs,
            save_freq_steps=freq_step,
            save_freq_epochs=freq_epoch,
        ),
        ppo=PPOHyperparameters(
            gen=GenerationHyperparameters(
                max_new_tokens=4,
                min_new_tokens=4,
                greedy=True,
                use_cuda_graph=False,
            ),
        ),
        actor_gen=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=1,
                model_parallel_size=2,
                pipeline_parallel_size=1,
            )
        ),
        actor_train=MFCConfig(
            device_mesh="NODE01:0",
            parallel=ParallelismConfig(
                data_parallel_size=1,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            ),
        ),
        critic_inf=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=2,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            )
        ),
        rew_inf=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=2,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            )
        ),
        ref_inf=MFCConfig(
            parallel=ParallelismConfig(
                data_parallel_size=2,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            )
        ),
        critic_train=MFCConfig(
            device_mesh="NODE01:1",
            parallel=ParallelismConfig(
                data_parallel_size=1,
                model_parallel_size=1,
                pipeline_parallel_size=1,
            ),
        ),
    )
    exp_cfg.actor.vllm.hybrid_train = True
    exp_cfg.actor.vllm.enforce_eager = True

    run_test_exp(exp_cfg)

    # Check that desired checkpoints have been saved.
    n_steps = (testing.TESTING_DATASET_SIZE * total_train_epochs + bs - 1) // bs
    for model_name in ["actor", "critic"]:
        desired_checkpoints = []
        for step in range(n_steps):
            if freq_step is not None and (step + 1) % freq_step == 0:
                desired_checkpoints.append(step + 1)
            epoch = step * bs // testing.TESTING_DATASET_SIZE
            is_last_epoch_step = (
                testing.TESTING_DATASET_SIZE - step * bs % testing.TESTING_DATASET_SIZE
                < bs
            )
            if (
                freq_epoch is not None
                and is_last_epoch_step
                and (epoch + 1) % freq_epoch == 0
            ):
                desired_checkpoints.append(step + 1)

        desired_checkpoints = set(desired_checkpoints)
        if not desired_checkpoints:
            break
        saved_checkpoints = set(
            int(os.path.basename(f).split("globalstep")[-1])
            for f in os.listdir(
                os.path.join(
                    constants.MODEL_SAVE_ROOT,
                    testing._DEFAULT_EXPR_NAME,
                    testing._DEFAULT_EXPR_NAME,
                    model_name,
                )
            )
            if int(os.path.basename(f).split("globalstep")[-1]) <= n_steps
        )
        assert desired_checkpoints.issubset(saved_checkpoints), (
            desired_checkpoints,
            saved_checkpoints,
        )
