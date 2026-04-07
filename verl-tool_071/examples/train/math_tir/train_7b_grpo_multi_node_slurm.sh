#!/bin/bash
#SBATCH --job-name=verltool_mathtir_7b_grpo_2nodes
#SBATCH --output=logs/verltool_mathtir_7b_grpo_2nodes_%j/%a.out
#SBATCH --error=logs/verltool_mathtir_7b_grpo_2nodes_%j/%a.err
#SBATCH --time=60
#SBATCH --gpus-per-node=4
#SBATCH --partition=interactive
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2
#SBATCH --mem=0

set -eoux pipefail

########################################################
# Container and mount configuration
########################################################
CONTAINER=${1} # the global container path
MOUNTS=${2}
CONTAINER_WORKDIR=${3} # the workdir inside the container

echo "Using container: $CONTAINER"
echo "Container workdir: $CONTAINER_WORKDIR"
echo "Mounts: $MOUNTS"

########################################################
# Common srun arguments for container execution
########################################################
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-8}}
CPUS_PER_WORKER=$((GPUS_PER_NODE * CPUS_PER_TASK))
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "CPUS_PER_TASK=$CPUS_PER_TASK"
echo "CPUS_PER_WORKER=$CPUS_PER_WORKER"

COMMON_SRUN_ARGS="--container-image=$CONTAINER"
COMMON_SRUN_ARGS+=" --container-mounts=$MOUNTS"
COMMON_SRUN_ARGS+=" --container-workdir=$CONTAINER_WORKDIR"
COMMON_SRUN_ARGS+=" --export=ALL,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
COMMON_SRUN_ARGS+=" --mem=0"

echo "COMMON_SRUN_ARGS=$COMMON_SRUN_ARGS"

########################################################
# Ray cluster configuration
########################################################
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}
echo "RAY_TMPDIR=$RAY_TMPDIR"

# Ray ports configuration
NODE_MANAGER_PORT=${NODE_MANAGER_PORT:-53001}
OBJECT_MANAGER_PORT=${OBJECT_MANAGER_PORT:-53003}
RUNTIME_ENV_AGENT_PORT=${RUNTIME_ENV_AGENT_PORT:-53005}
DASHBOARD_AGENT_GRPC_PORT=${DASHBOARD_AGENT_GRPC_PORT:-53007}
METRICS_EXPORT_PORT=${METRICS_EXPORT_PORT:-53009}

# Head node ports
PORT=${PORT:-54514}
RAY_CLIENT_SERVER_PORT=${RAY_CLIENT_SERVER_PORT:-10001}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
DASHBOARD_AGENT_LISTEN_PORT=${DASHBOARD_AGENT_LISTEN_PORT:-52365}

MIN_WORKER_PORT=${MIN_WORKER_PORT:-54001}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-54513}

########################################################
# Create logs directory
########################################################
BASE_LOG_DIR=$CONTAINER_WORKDIR
LOG_DIR="$BASE_LOG_DIR/ray-logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR
echo "Logs will be stored in $LOG_DIR"

########################################################
# Track backgrounded srun PIDs
########################################################
declare -A SRUN_PIDS

check_srun_processes() {
  for name in "${!SRUN_PIDS[@]}"; do
    pid="${SRUN_PIDS[$name]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Background srun '$name' died (pid=$pid)" >&2
      touch "$LOG_DIR/ENDED"
      exit 1
    fi
  done
}

########################################################
# Get node information
########################################################
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
ip_addresses_array=()

for node in $nodes; do
    echo "[DEBUG] Resolving hostname: $node"
    ip_address=""
    
    # Try multiple methods to get IP address
    ip_address=$(host $node 2>/dev/null | awk '/has address/ { print $4 }' | head -1 || true)
    
    if [[ -z "$ip_address" ]]; then
        ip_address=$(getent hosts $node 2>/dev/null | awk '{ print $1 }' | head -1 || true)
    fi
    
    if [[ -z "$ip_address" ]]; then
        ip_address=$(nslookup $node 2>/dev/null | awk '/^Address: / { print $2 }' | head -1 || true)
    fi
    
    if [[ -z "$ip_address" ]]; then
        echo "[WARNING] Could not resolve IP for $node, using hostname as fallback"
        ip_address=$node
    fi
    
    echo "[INFO] Node: $node -> IP: $ip_address"
    ip_addresses_array+=("$ip_address")
done

head_node=${nodes_array[0]}
head_node_ip=${ip_addresses_array[0]}
ip_head=$head_node_ip:$PORT

NUM_ACTORS=$((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))

########################################################
# Start Ray head node
########################################################
head_cmd=$(cat <<'EOF'
set -x
source .venv/bin/activate
mkdir -p ${RAY_TMPDIR}

exit_dramatically() {
    pkill -P $$ || true
    kill -TERM 0 || true
    exit 1
}

monitor_sidecar() {
  set +x
  while true; do
    sleep 60
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Detected ENDED file, terminating..."
      exit_dramatically
    fi
  done
}
monitor_sidecar &

touch $LOG_DIR/STARTED_RAY_HEAD

ray start --head \
    --disable-usage-stats \
    --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
    --node-ip-address="$HEAD_NODE_IP" \
    --port=$PORT \
    --ray-client-server-port=$RAY_CLIENT_SERVER_PORT \
    --dashboard-port=$DASHBOARD_PORT \
    --node-manager-port=$((NODE_MANAGER_PORT + 1)) \
    --object-manager-port=$((OBJECT_MANAGER_PORT + 1)) \
    --runtime-env-agent-port=$((RUNTIME_ENV_AGENT_PORT + 1)) \
    --dashboard-agent-grpc-port=$((DASHBOARD_AGENT_GRPC_PORT + 1)) \
    --dashboard-agent-listen-port=$((DASHBOARD_AGENT_LISTEN_PORT + 1)) \
    --metrics-export-port=$((METRICS_EXPORT_PORT + 1)) \
    --temp-dir=${RAY_TMPDIR} \
    --block
EOF
)

srun $COMMON_SRUN_ARGS \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$CPUS_PER_WORKER \
    --gpus=$GPUS_PER_NODE \
    -w "$head_node" \
    -o $LOG_DIR/ray-head.log \
    bash -c "export HEAD_NODE_IP=$head_node_ip; \
    export PORT=$PORT; \
    export RAY_CLIENT_SERVER_PORT=$RAY_CLIENT_SERVER_PORT; \
    export DASHBOARD_PORT=$DASHBOARD_PORT; \
    export NODE_MANAGER_PORT=$NODE_MANAGER_PORT; \
    export OBJECT_MANAGER_PORT=$OBJECT_MANAGER_PORT; \
    export RUNTIME_ENV_AGENT_PORT=$RUNTIME_ENV_AGENT_PORT; \
    export DASHBOARD_AGENT_GRPC_PORT=$DASHBOARD_AGENT_GRPC_PORT; \
    export DASHBOARD_AGENT_LISTEN_PORT=$DASHBOARD_AGENT_LISTEN_PORT; \
    export METRICS_EXPORT_PORT=$METRICS_EXPORT_PORT; \
    export GPUS_PER_NODE=$GPUS_PER_NODE; \
    export LOG_DIR=$LOG_DIR; \
    export SLURM_JOB_ID=$SLURM_JOB_ID; \
    export RAY_TMPDIR=$RAY_TMPDIR; \
    $head_cmd" &
SRUN_PIDS["ray-head"]=$!

sleep 5

########################################################
# Start Ray worker nodes
########################################################
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
  node_i=${nodes_array[$i]}
  
  worker_cmd=$(cat <<'EOF'
set -x
source .venv/bin/activate
mkdir -p ${RAY_TMPDIR}

exit_dramatically() {
    pkill -P $$ || true
    kill -TERM 0 || true
    exit 1
}

monitor_sidecar() {
  set +x
  while true; do
    sleep 60
    if [[ -f "$LOG_DIR/ENDED" ]]; then
      echo "Detected ENDED file, terminating..."
      exit_dramatically
    fi
  done
}
monitor_sidecar &

ray start --address "$IP_HEAD" \
    --disable-usage-stats \
    --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
    --min-worker-port=$MIN_WORKER_PORT \
    --max-worker-port=$MAX_WORKER_PORT \
    --node-manager-port=$NODE_MANAGER_PORT \
    --object-manager-port=$OBJECT_MANAGER_PORT \
    --runtime-env-agent-port=$RUNTIME_ENV_AGENT_PORT \
    --dashboard-agent-grpc-port=$DASHBOARD_AGENT_GRPC_PORT \
    --dashboard-agent-listen-port=$DASHBOARD_AGENT_LISTEN_PORT \
    --metrics-export-port=$METRICS_EXPORT_PORT \
    --temp-dir=${RAY_TMPDIR} \
    --block
EOF
)

  srun $COMMON_SRUN_ARGS \
      --container-name=ray-worker-$i \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=$CPUS_PER_WORKER \
      --gpus=$GPUS_PER_NODE \
      -w "$node_i" \
      -o $LOG_DIR/ray-worker-$i.log \
      bash -c "export IP_HEAD=$ip_head; \
      export MIN_WORKER_PORT=$MIN_WORKER_PORT; \
      export MAX_WORKER_PORT=$MAX_WORKER_PORT; \
      export NODE_MANAGER_PORT=$NODE_MANAGER_PORT; \
      export OBJECT_MANAGER_PORT=$OBJECT_MANAGER_PORT; \
      export RUNTIME_ENV_AGENT_PORT=$RUNTIME_ENV_AGENT_PORT; \
      export DASHBOARD_AGENT_GRPC_PORT=$DASHBOARD_AGENT_GRPC_PORT; \
      export DASHBOARD_AGENT_LISTEN_PORT=$DASHBOARD_AGENT_LISTEN_PORT; \
      export METRICS_EXPORT_PORT=$METRICS_EXPORT_PORT; \
      export GPUS_PER_NODE=$GPUS_PER_NODE; \
      export LOG_DIR=$LOG_DIR; \
      export SLURM_JOB_ID=$SLURM_JOB_ID; \
      export RAY_TMPDIR=$RAY_TMPDIR; \
      $worker_cmd" &
  SRUN_PIDS["ray-worker-$i"]=$!
  sleep 3
done

########################################################
# Wait for Ray cluster to be ready
########################################################
while check_srun_processes && ! srun $COMMON_SRUN_ARGS --overlap --nodes=1 --ntasks=1 --gpus=1 -w $head_node test -f $LOG_DIR/STARTED_RAY_HEAD; do
  echo "[INFO][$(date)] Waiting for head node to start..."
  sleep 2
done

extract_worker_units() {
  status_output=$(srun $COMMON_SRUN_ARGS --overlap --container-name=ray-head --nodes=1 --ntasks=1 --gpus=1 -w "$head_node" bash -c "source .venv/bin/activate; ray status")
  if echo "$status_output" | grep -q "worker_units"; then
    worker_units=$(echo "$status_output" | grep "worker_units" | awk -F'[/. ]' '{print $4}')
    echo $worker_units
  else
    echo 0
  fi
}

while true; do
  worker_units=$(extract_worker_units)
  echo "[INFO] Number of actors online: $worker_units/$NUM_ACTORS"
  if [[ "$worker_units" -eq "$NUM_ACTORS" ]]; then
    break
  fi
  check_srun_processes
  sleep 2
done

echo "Ray cluster is ready with all $NUM_ACTORS workers connected!"

########################################################
# Training configuration
########################################################
dataset_name=deepmath_torl
work_dir=$CONTAINER_WORKDIR
train_data=$CONTAINER_WORKDIR/data/${dataset_name}/train.parquet
val_data=[$CONTAINER_WORKDIR/data/${dataset_name}/test.parquet,\
$CONTAINER_WORKDIR/data/${dataset_name}/math500_test.parquet,\
$CONTAINER_WORKDIR/data/${dataset_name}/aime24_test.parquet,\
$CONTAINER_WORKDIR/data/${dataset_name}/aime25_test.parquet]
model_name=Qwen/Qwen2.5-Math-7B
rl_alg=grpo
n_gpus_per_node=$GPUS_PER_NODE
n_nodes=$SLURM_JOB_NUM_NODES
n=16
batch_size=128
ppo_mini_batch_size=128
max_prompt_length=1024
max_response_length=3072
max_obs_length=512
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp"
action_stop_tokens='```output'
max_turns=1
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=torl
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.7
do_offload=False
use_dynamic_bsz=False
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=False
max_action_length=2048
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="debug"

if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-nodes${n_nodes}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}-nodes${n_nodes}${run_name_postfix}"
fi
checkpoint_dir=$CONTAINER_WORKDIR/checkpoints/${reward_manager}/${run_name}

export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'


########################################################
# Start tool server with better error handling
########################################################
host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

# Create action stop tokens file
action_stop_tokens_file="$CONTAINER_WORKDIR/tmp/verl_tool_action_stop_tokens_$SLURM_JOB_ID.txt"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# Start tool server in container
echo "[INFO] Starting tool server at $tool_server_url"
srun $COMMON_SRUN_ARGS \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --overlap \
    --gpus=$GPUS_PER_NODE \
    -w "$head_node" \
    -o $LOG_DIR/tool-server.log \
    -e $LOG_DIR/tool-server.err \
    bash -c "
        source .venv/bin/activate && \
        which python && \
        echo '[INFO] Python location verified' && \
        echo '[INFO] Starting tool server...' && \
        python -m verl_tool.servers.serve \
            --host $host \
            --port $port \
            --tool_type python_code \
            --workers_per_tool 8 \
            --use_ray=True 2>&1 | tee -a $LOG_DIR/tool-server-console.log
    " &

SRUN_PIDS["tool-server"]=$!
echo "[INFO] Tool server process started (pid=${SRUN_PIDS["tool-server"]})"

# Wait for tool server to be ready
echo "[INFO] Waiting for tool server to start..."
max_wait=60
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    # Check if the process is still alive
    if ! kill -0 ${SRUN_PIDS["tool-server"]} 2>/dev/null; then
        echo "[ERROR] Tool server process died! Check logs:"
        echo "  - $LOG_DIR/tool-server.log"
        echo "  - $LOG_DIR/tool-server.err"
        cat $LOG_DIR/tool-server.err
        exit 1
    fi
    
    # Try to connect to the server
    if srun $COMMON_SRUN_ARGS --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$head_node" \
        bash -c "curl -s --max-time 2 $tool_server_url/health > /dev/null 2>&1"; then
        echo "[INFO] Tool server is ready!"
        break
    fi
    
    wait_count=$((wait_count + 1))
    echo "[INFO] Still waiting for tool server... ($wait_count/$max_wait)"
    sleep 2
done

if [ $wait_count -eq $max_wait ]; then
    echo "[ERROR] Tool server failed to start within ${max_wait} seconds"
    echo "[ERROR] Check the logs at:"
    echo "  - $LOG_DIR/tool-server.log"
    echo "  - $LOG_DIR/tool-server.err"
    exit 1
fi

########################################################
# Submit Ray job
########################################################
job_submit_cmd=$(cat <<EOF
set -x
which ray
source .venv/bin/activate
which ray
ray status

RAY_ADDRESS="http://127.0.0.1:$DASHBOARD_PORT" \
ray job submit --runtime-env=verl_tool/trainer/runtime_env.yaml \
    -- \
    PYTHONUNBUFFERED=1 \
    python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1024 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    +reward_model.reward_kwargs={'record_dir':'$checkpoint_dir/step_records'} \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200



EOF
)

srun $COMMON_SRUN_ARGS \
    --overlap \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --gpus=$GPUS_PER_NODE \
    -w "$head_node" \
    -o $LOG_DIR/ray-job-submit.log \
    bash -c "$job_submit_cmd"

########################################################
# Cleanup
########################################################
echo "Training complete. Cleaning up..."

# Stop tool server
pkill -P $server_pid 2>/dev/null || true
kill -9 $server_pid 2>/dev/null || true

# Signal Ray nodes to stop
touch "$LOG_DIR/ENDED"

# Stop Ray cluster
srun $COMMON_SRUN_ARGS \
    --overlap \
    --container-name=ray-head \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    -w "$head_node" \
    bash -c "ray stop --force" 2>/dev/null || true

# Clean up temporary files
rm -f $action_stop_tokens_file

echo "Cleanup complete!"