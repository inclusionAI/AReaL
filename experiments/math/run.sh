#!/bin/bash

expr_name=mzy-math
trial_name=$(date +%Y%m%d%H%M%S)-flatten-tree
folder_name=${expr_name}_${trial_name}

rsync -av --delete --exclude='.git' --exclude='*.pyc' /home/admin/meizhiyu.mzy/inclusionAI/AReaL /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name

# Change enable_tree_training and pad_to_maximum to False in packed seq baseline
srun --mpi=pmi2 -J mzy-tau2-server --chdir /storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL --cpus-per-task=128 --gres=gpu:8 --mem-per-cpu=12G --pty \
    singularity exec --no-home --writable-tmpfs --nv --pid --bind /storage:/storage \
    --env PYTHONPATH=/storage/openpsi/users/meizhiyu.mzy/workspace/$folder_name/AReaL \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-8726acfd20a41b5ab8ac35fd51014b15169fa20a \
    --env HF_ENDPOINT=https://hf-mirror.com \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "python3 -m areal.launcher.local examples/multi_turn_math/gsm8k_rl_mt.py --config examples/multi_turn_math/gsm8k_grpo_mt.yaml \
        experiment_name=$expr_name trial_name=$trial_name \
        allocation_mode=sglang:d4+fsdp:d4 cluster.n_gpus_per_node=8 cluster.n_nodes=1 actor.path=/storage/openpsi/models/Qwen__Qwen2-1.5B-Instruct \
        cluster.fileroot=/storage/openpsi/users/meizhiyu.mzy/experiments \
        train_dataset.path=/storage/openpsi/data/gsm8k valid_dataset.path=/storage/openpsi/data/gsm8k \
        stats_logger.wandb.mode=online evaluator.freq_epochs=null \
        export_style=individual \
        actor.optimizer.lr=1.7e-5 \
        actor.optimizer.weight_decay=0.017 \
        actor.mb_spec.max_tokens_per_mb=16384 \
        ref.mb_spec.max_tokens_per_mb=16384 \
        train_dataset.batch_size=256 \
        agent_run_args.max_turns=8 \
        gconfig.n_samples=8 \
        rollout.enable_rollout_tracing=False \
        rollout.max_head_offpolicyness=2 \
        stats_logger.wandb.mode=online \
        actor.reward_norm.mean_level=batch \
        actor.reward_norm.std_level=batch \
        +total_train_steps=1000 \
        +actor.enable_tree_training=True \
        +actor.pad_to_maximum=True"