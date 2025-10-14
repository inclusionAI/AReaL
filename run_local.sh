#!/bin/bash
python3 -m areal.launcher.local examples/math/gsm8k_grpo.py --config examples/math/gsm8k_grpo.yaml experiment_name=gsm8k-grpo trial_name=test0 \
    allocation_mode=sglang.d1+d1 cluster.n_gpus_per_node=2