uid="$(date +%Y%m%d_%H%M%S)"
base_model="${BASE_MODEL:-/storage/openpis/models/Qwen__Qwen3-8B}"
lr=1e-5
min_lr=0
epochs=8
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=2
max_steps=-1
push_to_hub=false
OUTPUT_DIR=${OUTPUT_DIR:-"ckpts/Q3-8B-131072-SFT-${uid}"}
TRAIN_DATA="${TRAIN_DATA:-./data/mult-10k-par}"
extra_args=()
resume_from_checkpoint=""
auto_resume=true

usage() {
    cat <<EOF
Usage: $0 [options] [extra_args_for_sft_threadweaver.py]

Options:
  --original_model_path <path>  Original/base model path
  --base_model <path>           Alias of --original_model_path
  --resume_from_checkpoint <path>
                                Resume from a specific checkpoint
  --no_resume                   Disable automatic resume from OUTPUT_DIR/checkpoint-*
  --output_dir <path>   Output directory (default: \$OUTPUT_DIR or ckpts/Q3-8B-131072-SFT-<timestamp>)
  --dataset_dir <path>  Dataset path (used exactly as provided, can be a parquet file path)
  --dataset_path <path> Alias of --dataset_dir
  --train_data <path>   Alias of --dataset_dir
  -h, --help            Show this help message

Examples:
  bash train.sh --original_model_path /models/Qwen3-8B --output_dir ckpts/run1 --dataset_dir /path/to/train.parquet
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --original_model_path|--base_model|--model_name)
            [ -n "$2" ] || { echo "Error: $1 requires a value."; exit 1; }
            base_model="$2"
            shift 2
            ;;
        --output_dir)
            [ -n "$2" ] || { echo "Error: --output_dir requires a value."; exit 1; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume_from_checkpoint)
            [ -n "$2" ] || { echo "Error: --resume_from_checkpoint requires a value."; exit 1; }
            resume_from_checkpoint="$2"
            auto_resume=false
            shift 2
            ;;
        --no_resume)
            auto_resume=false
            shift
            ;;
        --dataset_dir|--dataset_path|--train_data)
            [ -n "$2" ] || { echo "Error: $1 requires a value."; exit 1; }
            TRAIN_DATA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

export TRAIN_DATA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ./slurm_setup.sh

find_latest_checkpoint() {
    local output_dir="$1"
    local latest_checkpoint=""
    local last_index=0

    if [ -d "$output_dir" ]; then
        mapfile -t checkpoints < <(
            find "$output_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
        )
        if [ ${#checkpoints[@]} -gt 0 ]; then
            last_index=$((${#checkpoints[@]} - 1))
            latest_checkpoint="${output_dir}/${checkpoints[$last_index]}"
        fi
    fi

    printf '%s\n' "$latest_checkpoint"
}

is_resumable_checkpoint() {
    local checkpoint_dir="$1"

    compgen -G "${checkpoint_dir}/optimizer.pt" > /dev/null || \
        compgen -G "${checkpoint_dir}/scheduler.pt" > /dev/null || \
        compgen -G "${checkpoint_dir}/rng_state*.pth" > /dev/null || \
        compgen -G "${checkpoint_dir}/global_step*" > /dev/null
}

if [ -z "$resume_from_checkpoint" ] && [ "$auto_resume" = true ]; then
    resume_from_checkpoint="$(find_latest_checkpoint "$OUTPUT_DIR")"
fi

model_path="$base_model"
resume_args=()

if [ -n "$resume_from_checkpoint" ]; then
    model_path="$resume_from_checkpoint"
    if is_resumable_checkpoint "$resume_from_checkpoint"; then
        resume_args=(--resume_from_checkpoint "$resume_from_checkpoint")
        echo "Resuming training from checkpoint: $resume_from_checkpoint"
    else
        echo "Warm-starting from checkpoint weights: $resume_from_checkpoint"
        echo "Checkpoint does not include optimizer/scheduler state; this run restarts the trainer and will save full resumable checkpoints."
    fi
else
    echo "Starting training from base model: $base_model"
fi

torchrun_args=(
    --nnodes="$NNODES"
    --nproc_per_node="$GPUS_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
)

train_args=(
    src/sft_threadweaver.py
    --block_size=40960
    --per_device_train_batch_size="${micro_batch_size}"
    --per_device_eval_batch_size="${micro_batch_size}"
    --gradient_accumulation_steps="${gradient_accumulation_steps}"
    --num_train_epochs="${epochs}"
    --train_file_path="$TRAIN_DATA"
    --model_name="$model_path"
    --warmup_ratio=0.05
    --deepspeed configs/deepspeed_zero3.json
    --bf16=True
    --eval_strategy="no"
    --logging_steps=1
    --save_strategy="steps"
    --save_steps=20
    --save_total_limit=2
    --lr_scheduler_type="cosine"
    --learning_rate="${lr}"
    --weight_decay="${weight_decay}"
    --adam_beta1=0.9
    --adam_beta2=0.95
    --output_dir="$OUTPUT_DIR"
    --push_to_hub="${push_to_hub}"
    --save_only_model=False
    --gradient_checkpointing=True
    --use-liger=True
    --dataset_text_field="qwen_text"
    --attn_implementation="flex_attention"
    --template_name="qwen"
    --report_to="wandb"
)

torchrun "${torchrun_args[@]}" "${train_args[@]}" "${resume_args[@]}" "${extra_args[@]}"
