#!/usr/bin/env bash
set -euo pipefail

# MCP Weather training harness (dynamic reward via validation_calls)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/mcp_weather}"
TRAIN_PATH="${TRAIN_PATH:-$DATA_DIR/train.parquet}"
VAL_PATH="${VAL_PATH:-$DATA_DIR/test.parquet}"
MAX_TRAIN="${MAX_TRAIN:-50}"
MAX_TEST="${MAX_TEST:-20}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
REWARD_MANAGER="${REWARD_MANAGER:-mcp_dynamic}"
RUN_TRAINING="${RUN_TRAINING:-0}"
MAX_TURNS="${MAX_TURNS:-5}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-512}"
ACTION_STOP_TOKENS="${ACTION_STOP_TOKENS:-['</tool_call>']}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

GATEWAY_PORT="${GATEWAY_PORT:-8002}"
TOOL_PORT="${TOOL_PORT:-5005}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

cleanup() {
  if [[ -n "${GATEWAY_PID:-}" ]]; then
    kill "$GATEWAY_PID" 2>/dev/null || true
  fi
  if [[ -n "${TOOL_PID:-}" ]]; then
    kill "$TOOL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

kill_by_pattern() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1 || return 0
  echo "[warn] killing existing processes: $pattern"
  pkill -9 -f "$pattern" || true
}
kill_by_pattern "verl_tool.servers.tools.utils.local_mcp.gateway .*${GATEWAY_PORT}"
kill_by_pattern "verl_tool.servers.tool_server .*${TOOL_PORT}"

TRAINING_STEPS_ARG=""
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TRAINING_STEPS_ARG="    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \\"
fi

# prepare data
if [[ ! -f "$TRAIN_PATH" || ! -f "$VAL_PATH" ]]; then
  echo "[data] generating parquet files at $DATA_DIR"
  python "$REPO_ROOT/examples/data_preprocess/mcp_weather.py" \
    --out_dir "$DATA_DIR" \
    --max_train "$MAX_TRAIN" \
    --max_test "$MAX_TEST"
fi

echo "[mcp] starting gateway on :$GATEWAY_PORT"
python -m verl_tool.servers.tools.utils.local_mcp.gateway \
  --port "$GATEWAY_PORT" \
  --config "$REPO_ROOT/verl_tool/servers/tools/utils/local_mcp/configs/server_list.json" \
  --mode stdio \
  --servers weather >/tmp/mcp_gateway.log 2>&1 &
GATEWAY_PID=$!
export MCP_GATEWAY_ADDRESS="http://127.0.0.1:${GATEWAY_PORT}"
echo "[mcp] MCP_GATEWAY_ADDRESS=$MCP_GATEWAY_ADDRESS (pid=$GATEWAY_PID)"
sleep 2

echo "[tool] starting tool server on :$TOOL_PORT"
python -m verl_tool.servers.tool_server \
  --tool_type "mcp_interface,finish" \
  --host "127.0.0.1" \
  --port "$TOOL_PORT" \
  --workers_per_tool 8 \
  --max_concurrent_requests 64 \
  --log_level "info" >/tmp/mcp_tool_server.log 2>&1 &
TOOL_PID=$!
TOOL_SERVER_URL="http://127.0.0.1:${TOOL_PORT}/get_observation"
export TOOL_SERVER_URL
echo "[tool] TOOL_SERVER_URL=$TOOL_SERVER_URL (pid=$TOOL_PID)"
sleep 2

echo "[smoke] invoking weather alerts through tool server"
python - <<'PY'
import json, requests, time, sys, os
tool_url = os.environ["TOOL_SERVER_URL"]
payload = {
    "trajectory_ids": ["smoke-1"],
    "actions": ['<tool_call>{"server":"weather","name":"get_alerts","arguments":{"state":"CA"}}</tool_call>'],
    "extra_fields": [{}],
}
for _ in range(10):
    try:
        resp = requests.post(tool_url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        obs = data["observations"][0]
        text = obs["obs"] if isinstance(obs, dict) else obs
        if "No active alerts" in text or "Event:" in text:
            print("[smoke] OK:", text.strip()[:160])
            sys.exit(0)
        print("[smoke] unexpected payload:", text.strip()[:200])
        sys.exit(1)
    except Exception:
        time.sleep(0.5)
print("[smoke] failed to get weather result")
sys.exit(1)
PY

if [[ "$RUN_TRAINING" == "1" ]]; then
  echo "[train] launching PPO training (requires GPUs and a HF token)"
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 python -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[$TRAIN_PATH]" \
    data.val_files="[$VAL_PATH]" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.truncation='right' \
    reward_model.reward_manager=$REWARD_MANAGER \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    +actor_rollout_ref.actor.rollout_n=${ROLLOUT_N} \
    actor_rollout_ref.agent.tool_server_url=$TOOL_SERVER_URL \
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL} \
    +actor_rollout_ref.ref.override_config.attn_implementation=${ATTN_IMPL} \
    actor_rollout_ref.agent.max_prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.agent.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.agent.max_start_length=${MAX_START_LENGTH} \
    actor_rollout_ref.agent.max_obs_length=${MAX_OBS_LENGTH} \
    actor_rollout_ref.agent.action_stop_tokens=${ACTION_STOP_TOKENS} \
    actor_rollout_ref.agent.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMP} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    +critic.model.override_config.attn_implementation=${ATTN_IMPL} \
    critic.optim.lr=${CRITIC_LR} \
    critic.strategy="fsdp" \
    critic.model.path=$MODEL_NAME \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ulysses_sequence_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger=['console'] \
    trainer.project_name="mcp_weather" \
    trainer.experiment_name="mcp_weather_demo" \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.log_val_generations=5 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
${TRAINING_STEPS_ARG}
fi
