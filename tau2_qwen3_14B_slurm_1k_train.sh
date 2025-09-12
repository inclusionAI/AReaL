WANDB_API_KEY=local-e94a768686930cfc13051e562b807fc2d56bc4dd  \
WANDB_BASE_URL=http://8.150.1.98:8080 \
PYTHONPATH="modules/AReaL/" \
python3 -m areal.launcher.slurm tau2_train/workflow.py \
    --config tau2_train/train_debug.yaml \
    experiment_name=xss-tau2-qwen3_14B-train-1k-0909 \
    trial_name=trial-lr-2e-5 \
    cluster.fileroot=/storage/openpsi/experiments \
    cluster.name_resolve.type=nfs \
    cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve \
    launcher.slurm.trainer_image=/storage/openpsi/images/areal-25.02-v0.3.0.post2-v3.sif \
    launcher.slurm.inference_server_image=/storage/openpsi/images/areal-25.02-v0.3.0.post2-v3.sif  \
    allocation_mode=sglang:d8t4+fsdp:d4c8 \
    train_dataset.batch_size=32 \
    actor.optimizer.lr=2e-5 \
    cluster.n_nodes=8 \
    cluster.n_gpus_per_node=8 \
    gconfig.max_new_tokens=8192 \
    max_context_length=32768 \
    +reward_type=db \
    +n_trajs=4 \
    +user_base_url="http://33.180.164.182:30000/v1/" \
    train_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/tau2_airline_rl_dialogs_v2_0909_all_format_fixs_filter_error.jsonl \
    valid_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_eval/airline.jsonl \
    actor.path=/storage/openpsi/models/Qwen__Qwen3-14B/ \
