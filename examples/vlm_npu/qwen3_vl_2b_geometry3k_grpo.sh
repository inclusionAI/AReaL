#!/bin/bash
export USE_OPTIMIZED_MODEL=0

python -m areal.launcher.local \
    examples/vlm_npu/geometry3k_grpo.py --config examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.yaml