#!/usr/bin/env bash
set -x

if [ -z "$HEAD_IP" ]; then
    echo "ERROR: HEAD_IP not set. Usage: HEAD_IP=<head_ip> bash ray_start_worker.sh"
    exit 1
fi

HEAD_PORT=${HEAD_PORT:-6379}

ray stop --force 2>/dev/null
ray start \
    --address=${HEAD_IP}:${HEAD_PORT} \
    --disable-usage-stats

echo "Worker joined Ray cluster at ${HEAD_IP}:${HEAD_PORT}"
