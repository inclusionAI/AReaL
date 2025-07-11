
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli login --token $HUGGINGFACE_TOKEN
# export CURL_CA_BUNDLE=""
# export REQUESTS_CA_BUNDLE=""
HF_ENDPOINT=http://hf-mirror.com huggingface-cli download --repo-type dataset --resume-download BUAADreamer/clevr_count_70k --local-dir /storage/openpsi/data/clevr_count_70k 
# huggingface-cli download htlou/obelics_obelics_100k_tokenized_2048 --local-dir ./obelics_obelics_100k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_10k_tokenized_2048 --local-dir ./obelics_obelics_10k_tokenized_2048 --repo-type dataset
# huggingface-cli download htlou/obelics_obelics_100k --local-dir ./obelics_obelics_100k --repo-type dataset
#!/usr/bin/env bash

# 你要下载的模型仓库列表（只需列出 "htlou/xxx" 的后半部分即可）
repos=(
  "MathLLMs/MM-MathInstruct"
)

# 你想把文件下载到的主目录
base_dir="/storage/openpsi/models"


for repo_name in "${repos[@]}"; do
  echo ">>> Downloading $repo_name ..."
  huggingface-cli download \
    "$repo_name" \
    --local-dir "${base_dir}/${repo_name}" \
    --repo-type dataset \dat
    --exclude "checkpoint*"
    # --include "text-image-to-text/*" \



  echo
done

echo "All downloads completed!"



