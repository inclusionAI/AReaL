#!/usr/bin/env bash
set -euo pipefail

# Simple vLLM server launcher for tool-calling with Qwen3-VL-32B-Thinking.
# Override any setting by exporting the variable before running the script.

MODEL="${MODEL:-Qwen3-VL-32B-Thinking}"   # HF model ID or local path
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"                  # 8*A100
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"       # leave empty to use model default
EXTRA_ARGS="${EXTRA_ARGS:-}"

cmd=(
  vllm serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --trust-remote-code
)

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  cmd+=(--chat-template "${CHAT_TEMPLATE}")
fi

echo "Starting vLLM server:"
printf ' %q' "${cmd[@]}"
echo

# EXTRA_ARGS allows custom flags, e.g. --max-model-len 32768
# shellcheck disable=SC2086
exec "${cmd[@]}" ${EXTRA_ARGS}
