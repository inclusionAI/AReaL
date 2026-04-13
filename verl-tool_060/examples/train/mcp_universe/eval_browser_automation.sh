#!/usr/bin/env bash
set -euo pipefail

cp examples/train/mcp_universe/env.example.sh examples/train/mcp_universe/env.local.sh
set -a; source examples/train/mcp_universe/env.local.sh; set +a
export WANDB_MODE=offline
export RAY_ADDRESS=${RAY_ADDRESS:-local}
export NVM_DIR="$HOME/.nvm"

[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

VAL_FILE=${VAL_FILE:-"$(pwd)/data/mcp_universe/browser_automation/test.parquet"}
RUN_NAME=${RUN_NAME:-"mcp_universe-browser_automation"}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
HOST=${HOST:-$(hostname -i | awk '{print $1}')}
PORT=${PORT:-5500}
max_prompt_length=${max_prompt_length:-2048}
max_response_length=${max_response_length:-30720}
max_action_length=${max_action_length:-16384}
max_obs_length=${max_obs_length:-4096}
MAX_TURNS=${MAX_TURNS:-20}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
N=${N:-4}
N_GPUS=${N_GPUS:-1}
N_NODES=${N_NODES:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
tensor_model_parallel_size=${tensor_model_parallel_size:-1}
# Gateway (auto-launch)
GW_CFG=${GW_CFG:-"benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json"}
GW_PORT=${GW_PORT:-8003}
SERVERS=${SERVERS:-"playwright,date"}

# Start MCP‑Universe gateway if servers available
if [[ -n "$SERVERS" ]]; then
  echo "Starting MCP‑Universe gateway on port $GW_PORT with servers: $SERVERS"
  nvm use 20
  NODE_BIN="$(dirname "$(which node)")"
  PATH="$NODE_BIN:$PATH"
  PYTHONPATH=benchmarks/MCP-Universe \
  python -m mcpuniverse.mcp.gateway \
    --port ${GW_PORT} \
    --config ${GW_CFG} \
    --mode stdio \
    --servers "${SERVERS}" \
    >/dev/null 2>&1 &
  GW_PID=$!
  export MCP_GATEWAY_ADDRESS="http://localhost:${GW_PORT}"
else
  GW_PID=""
fi

# Start tool server (mcp_interface)
python -m verl_tool.servers.serve \
  --host $HOST \
  --port $PORT \
  --tool_type mcp_interface \
  --workers_per_tool 4 \
  >/dev/null 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "[CLEANUP] Killing tool/gateway servers..."
  pkill -P $SERVER_PID >/dev/null 2>&1 || true
  kill -9 $SERVER_PID >/dev/null 2>&1 || true
  if [[ -n "${GW_PID}" ]]; then
    pkill -P $GW_PID >/dev/null 2>&1 || true
    kill -9 $GW_PID >/dev/null 2>&1 || true
  fi

  echo "[CLEANUP] Checking for leftover GPU processes..."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | \
  while IFS=',' read -r pid pname mem; do
    pid=$(echo $pid | xargs)     # 去掉空格
    pname=$(echo $pname | xargs)
    mem=$(echo $mem | xargs)

    if [[ "$pid" =~ ^[0-9]+$ ]]; then
      echo "  Killing leftover process $pid ($pname, ${mem}MiB)"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done

  echo "[CLEANUP] Checking for zombie processes..."
  for ppid in $(ps -eo pid,ppid,stat | awk '$3 ~ /Z/ {print $2}' | sort -u); do
    if [[ "$ppid" =~ ^[0-9]+$ ]]; then
      echo "  Killing parent of zombie processes: $ppid"
      kill -9 "$ppid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup EXIT


# Action stop tokens file
# Note: Only </tool_call> should signal an action boundary to tool server.
#       </answer> is the final answer tag and must NOT trigger a tool call.
action_stop_tokens='</tool_call>'
action_stop_tokens_file=$(mktemp)
echo -n "$action_stop_tokens" > "$action_stop_tokens_file"

# Run eval (val_only)
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
  data.train_files="[$VAL_FILE]" \
  data.val_files="[$VAL_FILE]" \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=$max_prompt_length  \
  data.max_response_length=$max_response_length \
  data.truncation='right' \
  reward_model.launch_reward_fn_async=False \
  reward_model.reward_manager=mcp_universe_eval \
  actor_rollout_ref.model.path="$MODEL_NAME" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.agent.enable_agent=True \
  actor_rollout_ref.agent.tool_server_url="http://$HOST:$PORT/get_observation" \
  actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
  actor_rollout_ref.agent.max_response_length=$max_response_length \
  actor_rollout_ref.agent.max_start_length=$max_prompt_length \
  actor_rollout_ref.agent.max_obs_length=$max_obs_length \
  actor_rollout_ref.agent.max_action_length=$max_action_length \
  actor_rollout_ref.agent.max_turns=$MAX_TURNS \
  actor_rollout_ref.agent.action_stop_tokens="$action_stop_tokens_file" \
  actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
  actor_rollout_ref.agent.mask_observations=True \
  actor_rollout_ref.agent.enable_mtrl=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.top_p=$TOP_P \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.mode='async' \
  critic.strategy=fsdp \
  critic.model.path="$MODEL_NAME" \
  critic.ppo_mini_batch_size=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=mcp_universe \
  trainer.experiment_name=$RUN_NAME \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=$N_NODES
