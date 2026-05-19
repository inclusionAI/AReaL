#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <checkpoint_path> [--data-type|-d <parquet_path_or_dataset_key>] [-n|--n-samples <num>] [--bfloat16] [--verbose <level>] [extra args for simple_eval_pause_parallel_turn.py]"
  echo "Example: $0 ckpts/Q3-8B-131072-SFT --data-type data/mult-10k-par_pq/train.parquet --bfloat16 --verbose 2 -n 32"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CHECKPOINT_PATH="$1"
shift

DATA_TYPE_FROM_CLI=""
N_SAMPLES_FROM_CLI=""
VERBOSE_FROM_CLI=""
USE_BFLOAT16=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-type|-d)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value."
        usage
        exit 1
      fi
      DATA_TYPE_FROM_CLI="$2"
      shift 2
      ;;
    -n|--n-samples)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value."
        usage
        exit 1
      fi
      N_SAMPLES_FROM_CLI="$2"
      shift 2
      ;;
    --verbose)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value."
        usage
        exit 1
      fi
      VERBOSE_FROM_CLI="$2"
      shift 2
      ;;
    --bfloat16)
      USE_BFLOAT16=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/src/simple_eval_pause_parallel_turn.py"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Cannot find ${EVAL_SCRIPT}"
  exit 1
fi

TURN_LIMITS=(1 3 5 10 15 20 unlimit)
DATA_TYPE="${DATA_TYPE_FROM_CLI:-${DATA_TYPE:-}}"
if [[ -z "${DATA_TYPE}" ]]; then
  read -r -p "Enter --data-type parquet path (or dataset key): " DATA_TYPE
fi
if [[ -z "${DATA_TYPE}" ]]; then
  echo "Error: --data-type is required."
  exit 1
fi
N_SAMPLES="${N_SAMPLES_FROM_CLI:-${N_SAMPLES:-1}}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-40960}"
TEMPLATE_TYPE="${TEMPLATE_TYPE:-model}"
VERBOSE_LEVEL="${VERBOSE_FROM_CLI:-${VERBOSE_LEVEL:-}}"

echo "Running parallel-turn-limit sweep for checkpoint: ${CHECKPOINT_PATH}"
echo "Parallel turn limits: ${TURN_LIMITS[*]}"
echo "Data: ${DATA_TYPE}, n_samples: ${N_SAMPLES}, max_context_length: ${MAX_CONTEXT_LENGTH}, bfloat16: ${USE_BFLOAT16}"
if [[ -n "${VERBOSE_LEVEL}" ]]; then
  echo "Verbose: ${VERBOSE_LEVEL}"
fi
echo

for limit in "${TURN_LIMITS[@]}"; do
  echo "=============================="
  echo "Evaluating parallel turn limit: ${limit}"
  echo "=============================="
  CMD=(python "${EVAL_SCRIPT}" \
    --model_name "${CHECKPOINT_PATH}" \
    --data-type "${DATA_TYPE}" \
    --template-type "${TEMPLATE_TYPE}" \
    --launch_server \
    --branching-generate \
    --max-context-length "${MAX_CONTEXT_LENGTH}" \
    -n "${N_SAMPLES}")
  if [[ "${limit}" != "unlimit" ]]; then
    CMD+=(--parallel-turn-limit "${limit}")
  fi
  if [[ "${USE_BFLOAT16}" == "true" ]]; then
    CMD+=(--bfloat16)
  fi
  if [[ -n "${VERBOSE_LEVEL}" ]]; then
    CMD+=(--verbose "${VERBOSE_LEVEL}")
  fi
  CMD+=("${EXTRA_ARGS[@]}")
  "${CMD[@]}"
done

python - "${CHECKPOINT_PATH}" <<'PY'
import csv
import glob
import json
import os
import sys
from datetime import datetime

checkpoint_path = sys.argv[1]
turn_limits = ["1", "3", "5", "10", "15", "20", "unlimit"]
target_model_path = os.path.normpath(os.path.abspath(os.path.expanduser(checkpoint_path)))

abs_model = os.path.abspath(os.path.expanduser(checkpoint_path))
if os.path.isdir(abs_model):
    checkpoint_root = abs_model
elif os.path.exists(abs_model):
    checkpoint_root = os.path.dirname(abs_model)
else:
    checkpoint_root = os.path.abspath(".")

results_root = os.path.join(checkpoint_root, "eval_pause_outputs")
metrics_files = glob.glob(os.path.join(results_root, "**", "*_metrics.json"), recursive=True)

if not metrics_files:
    raise SystemExit(f"No metrics files found under: {results_root}")

latest_by_limit = {}
for path in metrics_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        continue
    payload_model_path = payload.get("model_path")
    if payload_model_path:
        payload_model_path = os.path.normpath(
            os.path.abspath(os.path.expanduser(payload_model_path))
        )
        if payload_model_path != target_model_path:
            continue
    if "parallel_turn_limit" not in payload:
        continue
    parallel_turn_limit = payload.get("parallel_turn_limit")
    if parallel_turn_limit is None:
        limit_key = "unlimit"
    else:
        try:
            limit_key = str(int(parallel_turn_limit))
        except (TypeError, ValueError):
            continue
    if limit_key not in turn_limits:
        continue
    prev = latest_by_limit.get(limit_key)
    if prev is None or os.path.getmtime(path) > os.path.getmtime(prev["path"]):
        latest_by_limit[limit_key] = {"path": path, "data": payload}

missing = [limit for limit in turn_limits if limit not in latest_by_limit]
if missing:
    raise SystemExit(f"Missing metrics files for parallel turn limits: {missing}")

rows = []
for limit in turn_limits:
    payload = latest_by_limit[limit]["data"]
    n_samples = int(payload.get("n_samples", 1))
    pass_at_1 = float(payload.get("pass@1", 0.0))
    pass_at_n = float(payload.get(f"pass@{n_samples}", 0.0))
    avg_longest = float(payload.get("avg_num_tokens_in_the_longest_thread", 0.0))
    rows.append(
        {
            "parallel_turn_limit": limit,
            "pass@1": pass_at_1,
            f"pass@{n_samples}": pass_at_n,
            "avg_num_tokens_in_the_longest_thread": avg_longest,
            "metrics_file": latest_by_limit[limit]["path"],
        }
    )

pass_n_key = [k for k in rows[0].keys() if k.startswith("pass@") and k != "pass@1"][0]

csv_path = os.path.join(results_root, "parallel_turn_limit_sweep_report.csv")
md_path = os.path.join(results_root, "parallel_turn_limit_sweep_report.md")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "parallel_turn_limit",
            "pass@1",
            pass_n_key,
            "avg_num_tokens_in_the_longest_thread",
            "metrics_file",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Parallel Turn Limit Sweep Report\n\n")
    f.write(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"- checkpoint: {checkpoint_path}\n")
    f.write(f"- results_root: {results_root}\n\n")
    f.write("| parallel_turn_limit | pass@1 | " + pass_n_key + " | avg_num_tokens_in_the_longest_thread |\n")
    f.write("|---:|---:|---:|---:|\n")
    for row in rows:
        f.write(
            f"| {row['parallel_turn_limit']} | {row['pass@1']:.6f} | "
            f"{row[pass_n_key]:.6f} | {row['avg_num_tokens_in_the_longest_thread']:.2f} |\n"
        )
    f.write("\n## Metrics Files\n\n")
    for row in rows:
        f.write(f"- {row['parallel_turn_limit']}: `{row['metrics_file']}`\n")

print("\nParallel turn limit sweep summary:")
print(f"{'turn_limit':>12}  {'pass@1':>10}  {pass_n_key:>10}  {'avg_longest':>12}")
for row in rows:
    print(
        f"{row['parallel_turn_limit']:>12}  {row['pass@1']:>10.6f}  {row[pass_n_key]:>10.6f}  "
        f"{row['avg_num_tokens_in_the_longest_thread']:>12.2f}"
    )

print(f"\nSaved CSV report: {csv_path}")
print(f"Saved Markdown report: {md_path}")
PY
