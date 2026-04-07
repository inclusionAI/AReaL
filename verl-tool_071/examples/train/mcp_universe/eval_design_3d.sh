#!/usr/bin/env bash
set -euo pipefail

VAL_FILE=${VAL_FILE:-"$(pwd)/data/mcp_universe/3d_design/test.parquet"}
RUN_NAME=${RUN_NAME:-"mcp_universe-3d_design"}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}
HOST=${HOST:-$(hostname -i | awk '{print $1}')}
PORT=${PORT:-5500}
MAX_TURNS=${MAX_TURNS:-6}
TEMP=${TEMP:-0.6}
TOP_P=${TOP_P:-0.95}
N_GPUS=${N_GPUS:-1}
N_NODES=${N_NODES:-1}

GW_CFG=${GW_CFG:-"verl-tool/benchmarks/MCP-Universe/mcpuniverse/mcp/configs/server_list.json"}
GW_PORT=${GW_PORT:-8010}
SERVERS=${SERVERS:-""}

# Blender MCP server requires a proper Blender environment on the host
# Ensure Blender is installed and the server in server_list.json is compatible with your setup
# No API keys required for blender

if [[ -z "$SERVERS" && -f "$VAL_FILE" ]]; then
  inferred=$(python - <<'PY'
import os
try:
    from datasets import load_dataset
except Exception:
    print("")
    raise SystemExit(0)
val=os.environ.get('VAL_FILE','')
if not val or not os.path.exists(val):
    print("")
    raise SystemExit(0)
ds = load_dataset('parquet', data_files={'data':[val]})['data']
servers=set()
for r in ds:
    extra = r.get('extra_info') or {}
    for s in (extra.get('mcp_servers') or []):
        if isinstance(s, dict):
            n=s.get('name');
            if n: servers.add(str(n))
        elif isinstance(s, str):
            servers.add(s)
print(",".join(sorted(servers)))
PY
  )
  if [[ -n "$inferred" ]]; then
    echo "Inferred servers from dataset: $inferred"
    SERVERS="$inferred"
  fi
fi

if [[ -n "$SERVERS" ]]; then
  echo "Starting MCP‑Universe gateway on port $GW_PORT with servers: $SERVERS"
  PYTHONPATH=verl-tool/benchmarks/MCP-Universe \
  python -m mcpuniverse.mcp.gateway \
    --port ${GW_PORT} \
    --config ${GW_CFG} \
    --mode sse \
    --servers "${SERVERS}" \
    >/dev/null 2>&1 &
  GW_PID=$!
  export MCP_GATEWAY_ADDRESS="http://localhost:${GW_PORT}"
else
  GW_PID=""
fi

python -m verl_tool.servers.serve \
  --host $HOST \
  --port $PORT \
  --tool_type mcp_interface \
  --workers_per_tool 4 \
  >/dev/null 2>&1 &
SERVER_PID=$!

cleanup() {
  pkill -P $SERVER_PID >/dev/null 2>&1 || true
  kill -9 $SERVER_PID >/dev/null 2>&1 || true
  if [[ -n "${GW_PID}" ]]; then
    pkill -P $GW_PID >/dev/null 2>&1 || true
    kill -9 $GW_PID >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

TMP_STOP=$(mktemp)
echo -n "</tool_call>" > "$TMP_STOP"

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
  data.train_files="[$VAL_FILE]" \
  data.val_files="[$VAL_FILE]" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
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
  actor_rollout_ref.agent.max_prompt_length=1024 \
  actor_rollout_ref.agent.max_response_length=2048 \
  actor_rollout_ref.agent.max_start_length=1024 \
  actor_rollout_ref.agent.max_obs_length=512 \
  actor_rollout_ref.agent.max_turns=$MAX_TURNS \
  actor_rollout_ref.agent.action_stop_tokens="$TMP_STOP" \
  actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
  actor_rollout_ref.agent.mask_observations=True \
  actor_rollout_ref.agent.enable_mtrl=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.top_p=$TOP_P \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.mode='async' \
  critic.strategy=fsdp \
  critic.model.path="$MODEL_NAME" \
  critic.ppo_mini_batch_size=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.logger=['console'] \
  trainer.project_name=mcp_universe \
  trainer.experiment_name=$RUN_NAME \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=$N_NODES
