expoprt USE_OPTIMIZED_MODEL=0  
# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training, 
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.

python -m areal.launcher.local \
    examples/vlm_npu/geometry3k_grpo.py --config examples/vlm_npu/geometry3k_grpo.yaml 