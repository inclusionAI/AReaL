#!/bin/sh

export WANDB_API_KEY=local-667d8d7f101dad4eb9597d718d0c68f40e3792f9
export WANDB_BASE_URL=http://8.150.1.98:8080

MODEL_FAMILY=qwen2
SFT_MODEL_PATH=/storage/openpsi/models/Qwen__Qwen2.5-7B-Instruct

DATA_PATH=/storage/openpsi/users/xmy/inclusionAI/AReaL/data/Werewolf_placeholder_1k.jsonl

MODE=slurm

EXP_NAME=xmy-werewolf-agent-test
TRIAL_NAME=trial-instruct-werewolf

unset CLUSTER_SPEC_PATH
CLUSTER_SPEC_PATH=/storage/openpsi/users/xmy/cluster_spec.json \
REAL_GPU_MEMORY_KILL_THRESHOLD=1 \
FUNCTIONCALL_SERVICE_DOMAIN="" \
REAL_ETCD_ADDR=etcd-client.openpsi-etcd.svc.sigma-na130-lingbo.na130.wl-robby.local:2379 \
 python3 -m realhf.apps.quickstart async-werewolf \
    mode=$MODE \
    cluster.fileroot=/storage/openpsi/experiments \
    n_nodes=2 n_gpus_per_node=8 \
    allocation_mode=sglang.d4m2p1+d1p2m4 \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    wandb.mode=online \
    exp_ctrl.total_train_epochs=5 \
    exp_ctrl.save_freq_steps=100 \
    exp_ctrl.ckpt_freq_secs=10800 \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$SFT_MODEL_PATH \
    critic.type._class=$MODEL_FAMILY \
    critic.type.is_critic=True \
    critic.init_critic_from_actor=True \
    critic.path=$SFT_MODEL_PATH \
    ref.type._class=$MODEL_FAMILY \
    ref.path=$SFT_MODEL_PATH \
    dataset.path=$DATA_PATH \
    dataset.max_prompt_len=2048 \
    dataset.train_bs_n_seqs=16 \
    group_size=1 \
    ppo.disable_value=True \
    ppo.gen.max_new_tokens=0 \
    ppo.gen.max_new_tokens=2048 \
    ppo.ppo_n_minibatches=1 \
    ppo.kl_ctl=0.0 \
    ppo.value_eps_clip=0.99 \
    ppo.max_reward_clip=40.0 \
    ppo.reward_output_scaling=1.0 \
    ppo.adv_norm=False ppo.value_norm=False \
    ppo.recompute_logprob=True \
    ppo.use_decoupled_loss=True \
    actor.optimizer.lr=2e-6 \
    actor.optimizer.gradient_clipping=1.0 \
    actor.optimizer.lr_scheduler_type=cosine \
    actor.optimizer.min_lr_ratio=0.01 \
    actor.optimizer.warmup_steps_proportion=0.001 \
    actor.sglang.mem_fraction_static=0.8 \
    ref_inf.mb_spec.max_tokens_per_mb=49152 \
    actor_train.mb_spec.max_tokens_per_mb=49152 \
    actor_inf.mb_spec.max_tokens_per_mb=49152 \
    cache_clear_freq=1 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4 \
    torch_cache_mysophobia=True \
    repeat_rules=true \
    role=both
