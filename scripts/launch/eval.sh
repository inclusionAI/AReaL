python3 -m areal.launcher.ray multitask_agent/gem_train/gem_eval.py \
    --config multitask_agent/gem_train/gem_train.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=minesweeper \
    trial_name=eval \
    cluster.n_nodes=1 \
    allocation_mode=sglang:d2t4+cpu \
    saver.freq_steps=10 \
    recover.mode=auto \
    recover.retries=10 \
    actor.optimizer.lr=5e-6 \
    actor.mb_spec.max_tokens_per_mb=32768 \
    gconfig.max_new_tokens=8192 \
    rollout.max_concurrent_rollouts=32 \
    +env_name=game:Minesweeper-v0-easy \
    +n_trajs=4 \
    +max_traj_tokens=32768 \
    +minesweeper_solver_path=/path/to/solver \
    train_dataset.batch_size=64 \
    actor.path=/path/to/your/checkpoint/ \
