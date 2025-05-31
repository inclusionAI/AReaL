#!/bin/bash
python3 training/main_sft.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=d4p2m1 \
    cluster.fileroot=/storage/testing/experiments \
    model.type._class=qwen3 \
    exp_ctrl.eval_freq_epochs=1 \
    model.path=/storage/testing/models/Qwen__Qwen3-1.7B \
    dataset.train_path=/storage/testing/dataset/areal-sft-stage2-200.jsonl \
    dataset.valid_path=/storage/testing/dataset/areal-sft-stage2-200.jsonl \
    dataset.train_bs_n_seqs=64 \
    dataset.valid_bs_n_seqs=64 \
    allocation.mb_spec.max_tokens_per_mb=32768