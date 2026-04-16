#!/usr/bin/env bash
# ================================================================
# Batch SFT: 8 experiments across 4 nodes (2 per node)
#
# Workspace: /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/
# Reference: qwen3vl8b-thinking-mixed-reasonmap-0413-lr1e-5
#
# Config: batch=4, DeepSpeed Z2, cutoff=32768, image_max_pixels=4M
#         lr=1e-5, epoch=1, cosine, bf16, gradient_checkpointing
#
# Experiments:
#   exp1: Qwen3-VL-8B-Thinking  + 4ds (mapqa+maptrace+reasonmap+reasonmap+)
#   exp2: Qwen3-VL-8B-Thinking  + 5ds (4ds + mini_o3 sampled 2000)
#   exp3: InternVL3-8B           + 2ds (reasonmap+reasonmap+)
#   exp4: InternVL3-8B           + 4ds
#   exp5: InternVL3-8B           + 5ds
#   exp6: InternVL3.5-8B         + 2ds
#   exp7: InternVL3.5-8B         + 4ds
#   exp8: InternVL3.5-8B         + 5ds
#
# Groups:
#   A (Node 1): exp1 + exp2  — Qwen3-VL-8B-Thinking
#   B (Node 2): exp3 + exp4  — InternVL3-8B
#   C (Node 3): exp5 + exp6  — InternVL3-8B + InternVL3.5-8B
#   D (Node 4): exp7 + exp8  — InternVL3.5-8B
#
# ================================================================
# PREREQUISITES (run on EACH node before launching):
#
#   pip install tensorboard -q
#   echo 'import PIL.Image; PIL.Image.MAX_IMAGE_PIXELS = None' \
#       > /opt/conda/lib/python3.11/site-packages/disable_pil_limit.pth

# ================================================================
# LAUNCH COMMANDS (one per node):
#
#   Node 1 (Group A):
#     setsid bash /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_group_A.sh \
#       > /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/logs/group_A_master.log 2>&1 &
#
#   Node 2 (Group B):
#     setsid bash /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_group_B.sh \
#       > /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/logs/group_B_master.log 2>&1 &
#
#   Node 3 (Group C):
#     setsid bash /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_group_C.sh \
#       > /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/logs/group_C_master.log 2>&1 &
#
#   Node 4 (Group D):
#     setsid bash /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/run_group_D.sh \
#       > /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/logs/group_D_master.log 2>&1 &
#
# ================================================================
# NOTES:
#
# 1. InternVL3/3.5 使用 intern_vl template，数据集原始为 Qwen3-VL 235B 增强生成。
#    LlamaFactory 会根据 template 自动处理 image token 格式差异，但建议先观察
#    Group B (exp3) 的 tokenize 阶段日志确认无报错后再放心运行。
#
# 2. mini_o3 数据集已做分层采样 (source x turns)，从 7267 条采样到 2000 条，
#    保证各来源和对话轮次的分布与原始一致。完整 trajectory 未被截断。
#
# 3. 所有超大图片 (>4096 edge 或 >4MB) 已预压缩为 JPEG (max_edge=4096, q=85)，
#    压缩副本存于 batch_0414/data/compressed_images/，train.json 中路径已更新。
#
# 4. 如果某个实验 OOM，可单独降低该实验的 per_device_train_batch_size 到 2 并重跑。
#
# ================================================================
# MONITOR:
#
#   tail -f /storage/openpsi/models/lcy_image_edit/sft_workspace/batch_0414/logs/exp{N}-*.log
#   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
#
# ================================================================
