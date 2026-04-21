mkdir -p /tmp/tiktoken-cache
cd /tmp/tiktoken-cache
curl -k -O https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-cache

python train.py --config config.yaml 2>&1 | tee llm_server.log