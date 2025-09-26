WANDB_API_KEY=local-e94a768686930cfc13051e562b807fc2d56bc4dd  \
WANDB_BASE_URL=http://8.150.1.98:8080 \
PYTHONPATH="modules/AReaL/" \
python3 -m areal.launcher.slurm tau2_train/workflow_megatron.py \
    --config tau2_train/train_debug_megatron.yaml \
    experiment_name=xss-tau2-qwen30-A3B-train-overfit-all-dbreward \
    trial_name=trial-1-user-qwen3-next-v4 \
    train_dataset.batch_size=64 \
    gconfig.max_new_tokens=8192 \
    max_context_length=32768 \
    +user_base_url=http://33.180.161.71:30000/v1 \
    +user_model=/storage/openpsi/models/Qwen3-Next-80B-A3B-Instruct_031825d6b716d454/ \
    +reward_type=db \
    +dynamic_filtering=True \
    +n_trajs=8 \
    train_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_eval/airline_repeate.jsonl \
    valid_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_eval/airline.jsonl \
    actor.path=/storage/openpsi/models/Qwen__Qwen3-30B-A3B \
