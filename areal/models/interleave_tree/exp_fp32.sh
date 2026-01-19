# ===== common arguments =====
MODEL_FOLDER=/data/tree/models
MODEL=Qwen2.5-0.5B
DATA=data/tau2/roll.pt
DTYPE=fp32
ATTN_IMP=sdpa

# ===== dense training =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp dense \
  --throw-prefix 4096 \
  --grad-file grad/grad_dense_fp32.pt

# ===== tree training (block size = 128) =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp tree \
  --block-size 128 \
  --throw-prefix 4096 \
  --grad-file grad/grad_tree_fp32_B128.pt

# ===== tree training (block size = 2048) =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp tree \
  --block-size 2048 \
  --throw-prefix 4096 \
  --grad-file grad/grad_tree_fp32_B2048.pt

# ===== gradient comparison =====
python compare_grads.py \
  --baseline-grad grad/grad_dense_fp32.pt \
  --exp-grad grad/grad_tree_fp32_B128.pt \
  --out grad/grad_diff_tree_fp32_B128.txt

python compare_grads.py \
  --baseline-grad grad/grad_dense_fp32.pt \
  --exp-grad grad/grad_tree_fp32_B2048.pt \
  --out grad/grad_diff_tree_fp32_B2048.txt