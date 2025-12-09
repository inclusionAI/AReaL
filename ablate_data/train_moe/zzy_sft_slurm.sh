#!/bin/bash

# Debug: Print PATH and check torchrun
echo "Initial PATH: $PATH"

# Try to find conda and python on compute nodes
CONDA_PYTHON=$(find /opt /home -name "conda" -type d 2>/dev/null | head -1)
if [ ! -z "$CONDA_PYTHON" ]; then
    export PATH="$CONDA_PYTHON/bin:$PATH"
fi

# Also check for anaconda
ANACONDA_PYTHON=$(find /opt /home -name "anaconda*" -type d 2>/dev/null | head -1)
if [ ! -z "$ANACONDA_PYTHON" ]; then
    export PATH="$ANACONDA_PYTHON/bin:$PATH"
fi

echo "Updated PATH: $PATH"
echo "Python location: $(which python 2>/dev/null || echo 'not found')"
echo "Python3 location: $(which python3 2>/dev/null || echo 'not found')"
echo "Torchrun location: $(which torchrun 2>/dev/null || echo 'not found')"

# Set training parameters
lr=1e-5
epochs=5
batch_size=16
weight_decay=1e-4
train_dataset_name="Multiverse-1K-mixed"
uid="$(date +%Y%m%d_%H%M%S)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Get node information
node_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nnodes=${#node_array[@]}
head_node=${node_array[0]}
# Get head node IP - try hostname first, fallback to NodeAddr
head_node_ip=$(getent hosts $head_node | awk '{print $1}')
if [ -z "$head_node_ip" ]; then
    head_node_ip=$(scontrol show node $head_node | grep -oP 'NodeAddr=\K[^ ]+')
fi
# If still empty, use hostname directly
if [ -z "$head_node_ip" ]; then
    head_node_ip=$head_node
fi
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600


# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((batch_size/(gpu_count * nnodes)))

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# Launch distributed training using torchrun with absolute path
run_name="Multiverse_${uid}"

# Try to find torchrun
TORCHRUN_CMD=$(which torchrun 2>/dev/null)
if [ -z "$TORCHRUN_CMD" ]; then
    # Fallback to common locations
    if [ -f "/opt/conda/bin/torchrun" ]; then
        TORCHRUN_CMD="/opt/conda/bin/torchrun"
    elif [ -f "$HOME/.local/bin/torchrun" ]; then
        TORCHRUN_CMD="$HOME/.local/bin/torchrun"
    else
        # Try using python -m torch.distributed.run as fallback
        TORCHRUN_CMD="python -m torch.distributed.run"
    fi
fi

echo "Using torchrun command: $TORCHRUN_CMD"

# Use srun to launch on all allocated nodes
srun bash << 'SRUN_EOF'
# Find and use the correct Python
PYTHON_BIN=""
for py_path in /opt/conda/bin/python /opt/anaconda3/bin/python /home/*/conda/bin/python /home/*/anaconda3/bin/python; do
    if [ -f "$py_path" ]; then
        PYTHON_BIN="$py_path"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Cannot find conda/anaconda python"
    exit 1
fi

echo "Found Python: $PYTHON_BIN"
PYTHON_DIR=$(dirname $PYTHON_BIN)
export PATH="$PYTHON_DIR:$PATH"

$PYTHON_BIN -m torch.distributed.run \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    train/sft_multiverse.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --train_file_path="Multiverse4FM/$train_dataset_name" \
    --model_name="Qwen/Qwen2.5-32B-Instruct" \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=$lr \
    --weight_decay=$weight_decay \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/$run_name" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
SRUN_EOF