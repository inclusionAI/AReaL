export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_NZ=0
export USE_OPTIMIZED_MODEL=0  
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.

allocation_mode="vllm:d1t8+fsdp:d1t8"

python -m areal.launcher.local \
    examples/vlm_npu/geometry3k_grpo.py --config examples/vlm_npu/qwen3_vl_32b_geometry3k_grpo.yaml \
    allocation_mode=${allocation_mode} 