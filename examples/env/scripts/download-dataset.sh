#!/bin/sh
set -e

mkdir -p /storage/datasets/
cd /storage/datasets/
if [ $REGION == 'CN' ]; then
    wget https://www.modelscope.cn/datasets/inclusionAI/AReaL-RL-Data/resolve/master/data/prompts_for_r1_distilled.jsonl
    wget https://www.modelscope.cn/datasets/inclusionAI/AReaL-RL-Data/resolve/master/data/prompts_for_zero.jsonl
    wget https://www.modelscope.cn/datasets/inclusionAI/AReaL-RL-Data/resolve/master/data/id2info.json
else
    wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/prompts_for_r1_distilled.jsonl?download=true
    wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/prompts_for_zero.jsonl?download=true
    wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/id2info.json?download=true
fi