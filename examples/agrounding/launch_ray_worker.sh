export HEAD_IP="33.180.163.80"
export RAY_PORT=26379



export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 每节点 8 卡
export WANDB_BASE_URL=""
export WANDB_API_KEY=""

# ③ 准备临时目录（避免 /tmp 限制）
ulimit -n 65536
ulimit -u unlimited
ray stop || true
export HOSTNAME_HASH=$(hostname | md5sum | cut -c1-4)
export RAY_TMPDIR="/tmp/ray/${HOSTNAME_HASH}"
mkdir -p "$RAY_TMPDIR"
# ④ 启动 worker 并加入 head
# （不同物理机可以复用同一端口区间；同一台机器上模拟多容器时才需要错开）
ray start --address="${HEAD_IP}:${RAY_PORT}" \
    --node-manager-port=52635 \
  --object-manager-port=8076 \
  --min-worker-port=27000 \
  --max-worker-port=31999 


# ⑤ 可选检查
ray status
