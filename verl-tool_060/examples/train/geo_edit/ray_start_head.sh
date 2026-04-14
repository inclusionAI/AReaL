#!/usr/bin/env bash
set -x

HEAD_PORT=${HEAD_PORT:-6379}
HEAD_IP=$(hostname -I | awk '{print $1}')

ray stop --force 2>/dev/null
ray start --head \
    --port=$HEAD_PORT \
    --disable-usage-stats

echo "============================================"
echo "Ray head started at ${HEAD_IP}:${HEAD_PORT}"
echo "Run on each worker node:"
echo "  HEAD_IP=${HEAD_IP} bash ray_start_worker.sh"
echo "============================================"
