model_path=
train_dataset=

PYTHONPATH="modules/AReaL/" \
python3 -m areal.launcher.slurm tau2_train/workflow_megatron.py \
    --config tau2_train/train_merge_235B_megatron.yaml \
    experiment_name= \
    trial_name= \
    train_dataset.batch_size=16 \
    gconfig.max_new_tokens=8192 \
    max_context_length=32768 \
    rollout.max_concurrent_rollouts=128 \
    reward_type=db \
    dynamic_filtering=true \
    +user_llm_args.temperature=1 \
    recover.mode=auto \
    n_trajs=16 \
    cluster.n_nodes=10 \
    actor.optimizer.lr=1e-5 \
    actor.adv_norm=null \
    train_dataset.path=$train_dataset \
    valid_dataset.path=$train_dataset \
    actor.path=$model_path \
    +is_reasoning_model=true \
    +max_turns=200 \
    +reward_norm_type=grpo \
    +process_payment_history=True \