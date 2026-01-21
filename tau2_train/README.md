
1. launch a sglang server as user simulator
2. specify user_base_url and user_model

```
PYTHONPATH="modules/AReaL/" \
python3 -m areal.launcher.slurm tau2_train/workflow_megatron.py \
    --config tau2_train/train_debug_megatron.yaml \
    experiment_name=xss-tau2-qwen30-1227sft-train-1229-8x64-fixph-grpo-noadvnorm \
    trial_name=trial-1-user-qwen2.5-72B-v1 \
    train_dataset.batch_size=8 \
    gconfig.max_new_tokens=8192 \
    max_context_length=32768 \
    +user_base_url=http://xxxx:xxxx/v1/ \
    +user_model=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/ \
    reward_type=db \
    dynamic_filtering=True \
    +process_payment_history=True \
    +reward_norm_type=grpo \
    n_trajs=64 \
    cluster.n_nodes=8 \
    actor.adv_norm=null \
    train_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_eval/rl_training_data_chuyi_jiaxuan_merged_1229.jsonl \
    valid_dataset.path=/storage/openpsi/users/xushusheng.xss/data/agent_eval/airline.jsonl \
    actor.path=/storage/openpsi/experiments/checkpoints/admin/hcy-tau2-airline-qwen3-30B-2507-sft/data-1227/default/epoch9epochstep94globalstep949 \
```



