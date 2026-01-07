python3 -m areal.launcher.local multitask_agent/gem_train/gem_train.py \
    --config multitask_agent/gem_train/gem_train.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=minesweeper \
    trial_name=ray-2nodes-easy-1.7b \
    cluster.n_nodes=2 \
    allocation_mode=sglang:d8+megatron:d2p2t2 \
    saver.freq_steps=10 \
    recover.mode=auto \
    recover.retries=10 \
    actor.optimizer.lr=5e-6 \
    actor.mb_spec.max_tokens_per_mb=32768 \
    gconfig.max_new_tokens=8192 \
    rollout.max_concurrent_rollouts=8 \
    +env_name=game:Minesweeper-v0-easy-with-template \
    +max_traj_tokens=32768 \
    +reward_type=return \
    +minesweeper_solver_path=/path/to/solver \
    +minesweeper_solver_reward_coeff=0.1 \
    +invalid_action_reward=-0.05 \
    train_dataset.batch_size=64 \
    actor.path=Qwen/Qwen3-1.7B \
