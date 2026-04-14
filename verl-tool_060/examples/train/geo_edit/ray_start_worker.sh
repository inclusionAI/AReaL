#!/usr/bin/env bash
set -x

if [ -z "$HEAD_IP" ]; then
    echo "ERROR: HEAD_IP not set. Usage: HEAD_IP=<head_ip> bash ray_start_worker.sh"
    exit 1
fi

HEAD_PORT=${HEAD_PORT:-6379}

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
ray start \
    --address=${HEAD_IP}:${HEAD_PORT} \
    --runtime-env-json="$RUNTIME_ENV" \
    --disable-usage-stats

echo "Worker joined Ray cluster at ${HEAD_IP}:${HEAD_PORT}"
