#!/usr/bin/env bash
set -x

HEAD_PORT=${HEAD_PORT:-6379}
HEAD_IP=$(hostname -I | awk '{print $1}')

RUNTIME_ENV='{"env_vars":{'
sep=""
for var in JUDGE_API_KEY JUDGE_API_BASE JUDGE_MODEL WANDB_API_KEY WANDB_BASE_URL; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        RUNTIME_ENV="${RUNTIME_ENV}${sep}\"${var}\":\"${val}\""
        sep=","
    fi
done
RUNTIME_ENV="${RUNTIME_ENV}}}"

ray stop --force 2>/dev/null
ray start --head \
    --port=$HEAD_PORT \
    --runtime-env-json="$RUNTIME_ENV" \
    --disable-usage-stats

echo "============================================"
echo "Ray head started at ${HEAD_IP}:${HEAD_PORT}"
echo "Run on each worker node:"
echo "  HEAD_IP=${HEAD_IP} bash ray_start_worker.sh"
echo "============================================"
