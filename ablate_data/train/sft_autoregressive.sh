
# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=1
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=2 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_autoregressive.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="/storage/openpsi/users/zzy/Multiverse_original/ar-1k-mixed" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/autoregressive-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --use-liger=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
