conda init
conda activate llama

# Function to check for available GPUs
check_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk '$2 < 20000 {print $1}' | head -n 4 | paste -sd "," -
}

# Constantly poll GPU memory until 4 GPUs with < 20G usage are found
echo "Checking for available GPUs with memory usage < 20G..."
while true; do
    AVAILABLE_GPUS=$(check_gpus)
    
    if [ -n "$AVAILABLE_GPUS" ] && [ "$(echo $AVAILABLE_GPUS | awk -F',' '{print NF}')" -ge 4 ]; then
        break
    fi
    
    echo "Only $(echo $AVAILABLE_GPUS | awk -F',' '{print NF}') GPUs available. Waiting 30 seconds..."
    sleep 30
done

export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
echo "Found 4 available GPUs: $CUDA_VISIBLE_DEVICES"
while true; do
    llamafactory-cli train train_config.yaml 
done
export CUDA_VISIBLE_DEVICES=6