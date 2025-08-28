#!/bin/bash

# accelerate launch train_sft_trl.py \
#   --model_name_or_path "Qwen/Qwen2.5-0.5B-instruct" \
#   --output_dir "/home/ubuntu/victor-north-tx/tau2-bench-private/data/models/Qwen2.5-0.5B-instruct-sft-full-tau2-assistant-only-loss" \
#   --train_dataset_path "/home/ubuntu/victor-north-tx/tau2-bench-private/data/tau2/sft-v1-small/sft-v1-small-openai/train_sft-v1-small.jsonl" \
#   --test_dataset_path "/home/ubuntu/victor-north-tx/tau2-bench-private/data/tau2/sft-v1-small/sft-v1-small-openai/test_sft-v1-small.jsonl" \
#   --chat_template_path "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/prompt_templates/qwen2.5_prompt_template.jinja" \
#   --patience 2 \
#   --delete_existing_output_dir \
#   --assistant_only_loss \
#   --report_to "wandb" \
#   --run_name "Qwen2.5-0.5B-instruct-sft-v1-small-assistant-only-loss" \
#   --learning_rate 8e-5 \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --fp16_full_eval \
#   --eval_accumulation_steps 4 \
#   --per_device_eval_batch_size 2 \
#   --gradient_checkpointing \
#   --logging_steps 2 \
#   --eval_strategy "epoch" \
#   --save_strategy "epoch" \
#   --save_total_limit 1 \
#   --load_best_model_at_end \
#   --metric_for_best_model "eval_loss" \
#   --greater_is_better false \
#   --auto_find_batch_size \
#   --max_length 8192 \
#   --logging_strategy "steps" \
#   --use_accelerate


# Simple training with minimal memory usage for basic testing
# Uses the small dataset.
# batch size 1
# gradient accumulation steps 1
# per device train batch size 1
# per device eval batch size 1
# eval accumulation steps 1
# max length 8192
python train_sft_trl.py \
  --model_name_or_path "Qwen/Qwen2.5-0.5B-instruct" \
  --output_dir "/home/ubuntu/victor-north-tx/tau2-bench-private/data/models/Qwen2.5-0.5B-instruct-sft-full-tau2-assistant-only-loss" \
  --train_dataset_path "/home/ubuntu/victor-north-tx/tau2-bench-private/data/tau2/sft-v1-small/sft-v1-small-openai/train_sft-v1-small.jsonl" \
  --test_dataset_path "/home/ubuntu/victor-north-tx/tau2-bench-private/data/tau2/sft-v1-small/sft-v1-small-openai/test_sft-v1-small.jsonl" \
  --chat_template_path "/home/ubuntu/victor-north-tx/tau2-bench-private/src/experiments/model_training/prompt_templates/qwen2.5_prompt_template.jinja" \
  --patience 2 \
  --delete_existing_output_dir \
  --assistant_only_loss \
  --report_to "wandb" \
  --run_name "Qwen2.5-0.5B-instruct-sft-v1-small-assistant-only-loss" \
  --learning_rate 8e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --fp16_full_eval \
  --eval_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --gradient_checkpointing \
  --logging_steps 2 \
  --eval_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss" \
  --greater_is_better false \
  --auto_find_batch_size \
  --max_length 8192 \
  --logging_strategy "steps" 
    # --use_accelerate \

