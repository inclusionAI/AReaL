#!/usr/bin/env bash
set -x

if [ -z "$HEAD_IP" ]; then
    echo "ERROR: HEAD_IP not set. Usage: HEAD_IP=<head_ip> bash ray_start_worker.sh"
    exit 1
fi

HEAD_PORT=${HEAD_PORT:-6379}
unset ROCR_VISIBLE_DEVICES

for var in JUDGE_API_KEY JUDGE_API_BASE JUDGE_MODEL WANDB_API_KEY WANDB_BASE_URL; do
    [ -n "${!var:-}" ] && export $var
done

ray stop --force 2>/dev/null
ray start \
    --address=${HEAD_IP}:${HEAD_PORT} \
    --disable-usage-stats

echo "Worker joined Ray cluster at ${HEAD_IP}:${HEAD_PORT}"
