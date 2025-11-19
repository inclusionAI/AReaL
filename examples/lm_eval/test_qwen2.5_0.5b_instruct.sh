#!/usr/bin/env bash
set -euo pipefail

OUTDIR=results
mkdir -p "${OUTDIR}"

# Common args
TASK=gsm8k
FEWSHOT=1
BATCH=1
TRUST="trust_remote_code=true"
MAX_TEST_SAMPLES=10
DEVICE=mps

# 1) Public Qwen2.5-0.5B-Instruct
lm_eval \
  --device ${DEVICE} \
  --model hf \
  --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,${TRUST},dtype=float16" \
  --tasks ${TASK} \
  --num_fewshot ${FEWSHOT} \
  --batch_size ${BATCH} \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path "${OUTDIR}/qwen2.5-0.5b-instruct_gsm8k_8shot" \
  --gen_kwargs "max_new_tokens=512,temperature=0.0,top_p=1.0" \
  --limit ${MAX_TEST_SAMPLES}
