# Load model from local directory (no internet required)
python inference.py \
  --model-name-or-path /storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct \
  --local-files-only \
  --prompt "Once upon a time" \
  --max-new-tokens 64