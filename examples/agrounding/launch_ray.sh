# 0) 基础环境（容器里）
export HOST_IP="33.180.163.80"
export RAY_PORT=26379                       
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_BASE_URL=""
export WANDB_API_KEY=""

# 1) 提高系统限制 & 清理残留
ulimit -n 65536
ulimit -u unlimited
ray stop || true
rm -rf /tmp/ray


export RAY_raylet_start_wait_time_s=60

# 3) 启动 head
ray start --head \
  --node-ip-address="$HOST_IP" \
  --port="$RAY_PORT" \
  --num-cpus=64 \
  --num-gpus=8 \
  --dashboard-port=8265 \
  --node-manager-port=52635 \
  --object-manager-port=8076 \
  --min-worker-port=27000 \
  --max-worker-port=31999 

