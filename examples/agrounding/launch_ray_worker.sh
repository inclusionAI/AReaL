export HEAD_IP=""
export RAY_PORT=26379



export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  
export WANDB_BASE_URL=""
export WANDB_API_KEY=""

ulimit -n 65536
ulimit -u unlimited
ray stop || true
export HOSTNAME_HASH=$(hostname | md5sum | cut -c1-4)
export RAY_TMPDIR="/tmp/ray/${HOSTNAME_HASH}"
mkdir -p "$RAY_TMPDIR"
ray start --address="${HEAD_IP}:${RAY_PORT}" \
    --node-manager-port=52635 \
  --object-manager-port=8076 \
  --min-worker-port=27000 \
  --max-worker-port=31999 


ray status
