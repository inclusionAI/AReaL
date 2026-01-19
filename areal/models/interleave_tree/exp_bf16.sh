# ===== common arguments =====
MODEL_FOLDER=/data/tree/models
MODEL=Qwen2.5-0.5B
DATA=data/tau2/roll.pt
DTYPE=bf16
ATTN_IMP=flash_attention_2

# ===== dense training =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp dense \
  --grad-file grad/grad_dense_bf16.pt

# ===== tree training (block size = 128) =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp tree \
  --block-size 128 \
  --grad-file grad/grad_tree_bf16_B128.pt

# ===== tree training (block size = 2048) =====
python run.py \
  --model-folder ${MODEL_FOLDER} \
  --model ${MODEL} \
  --data ${DATA} \
  --dtype ${DTYPE} \
  --attn-imp ${ATTN_IMP} \
  --train-imp tree \
  --block-size 2048 \
  --grad-file grad/grad_tree_bf16_B2048.pt

# ===== gradient comparison =====
python compare_grads.py \
  --baseline-grad grad/grad_dense_bf16.pt \
  --exp-grad grad/grad_tree_bf16_B128.pt \
  --out grad/grad_diff_tree_bf16_B128.txt

python compare_grads.py \
  --baseline-grad grad/grad_dense_bf16.pt \
  --exp-grad grad/grad_tree_bf16_B2048.pt \
  --out grad/grad_diff_tree_bf16_B2048.txt