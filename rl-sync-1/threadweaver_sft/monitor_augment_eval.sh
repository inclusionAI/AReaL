#!/usr/bin/env bash
set -euo pipefail

MONITOR_DIR="/storage/openpsi/models/zzy/augment"
EVAL_SCRIPT="./eval_pause_sweep.sh"
DATA_PATH="/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet"
VENV_ACTIVATE="/storage/openpsi/users/zzy/.threadweaver/bin/activate"
STATE_FILE="/tmp/augment_eval_seen_dirs.txt"
LOG_FILE="/tmp/augment_eval_monitor.log"
POLL_SECONDS=20

cd "/storage/openpsi/users/zzy/zzy_tw_rollback/threadweaver_sft"

if [[ ! -x "$EVAL_SCRIPT" ]]; then
  echo "[$(date -u +'%F %T')] ERROR: $EVAL_SCRIPT not found or not executable" | tee -a "$LOG_FILE"
  exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[$(date -u +'%F %T')] ERROR: venv activate script not found at $VENV_ACTIVATE" | tee -a "$LOG_FILE"
  exit 1
fi

touch "$STATE_FILE" "$LOG_FILE"

seed_seen() {
  find "$MONITOR_DIR" -mindepth 1 -maxdepth 1 -type d | sort > "$STATE_FILE"
}

is_new_model_dir() {
  local d="$1"
  local b
  b="$(basename "$d")"
  [[ "$b" == "eval_pause_outputs" ]] && return 1
  [[ "$d" == */eval_pause_outputs/* ]] && return 1
  return 0
}

run_eval() {
  local model_dir="$1"
  echo "[$(date -u +'%F %T')] NEW checkpoint dir detected: $model_dir" | tee -a "$LOG_FILE"
  echo "[$(date -u +'%F %T')] Activating env: $VENV_ACTIVATE" | tee -a "$LOG_FILE"
  echo "[$(date -u +'%F %T')] Running: $EVAL_SCRIPT $model_dir --bfloat16 --verbose 2 -n 32 --data-type $DATA_PATH" | tee -a "$LOG_FILE"
  (
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
    "$EVAL_SCRIPT" "$model_dir" --bfloat16 --verbose 2 -n 32 --data-type "$DATA_PATH"
  ) >> "$LOG_FILE" 2>&1 || {
    echo "[$(date -u +'%F %T')] ERROR: eval failed for $model_dir" | tee -a "$LOG_FILE"
    return 1
  }
  echo "[$(date -u +'%F %T')] Eval finished for $model_dir" | tee -a "$LOG_FILE"
}

seed_seen
echo "[$(date -u +'%F %T')] Monitor started for $MONITOR_DIR" | tee -a "$LOG_FILE"

while true; do
  current="$(mktemp)"
  find "$MONITOR_DIR" -mindepth 1 -maxdepth 1 -type d | sort > "$current"

  while IFS= read -r d; do
    [[ -z "$d" ]] && continue
    if ! grep -Fxq "$d" "$STATE_FILE"; then
      if is_new_model_dir "$d"; then
        run_eval "$d"
      else
        echo "[$(date -u +'%F %T')] Skipping non-model dir: $d" | tee -a "$LOG_FILE"
      fi
      echo "$d" >> "$STATE_FILE"
    fi
  done < "$current"

  rm -f "$current"
  sleep "$POLL_SECONDS"
done
