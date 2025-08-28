# Training using HF TRL (Deprecated)

## SFT

Install requirements:

```bash
pip install -r requirements.txt
```
or for gpu compatible version:
```bash
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```



Run the following command to train the model:
```bash
accelerate config
```
Choose:
- multi-GPU
- bf16 or fp16
DDP backend

Run the following command to train the model:
```bash
accelerate launch --config_file accelerate_config.yaml train_sft_lora.py
```

or set your CUDA_VISIBLE_DEVICES environment variable to the desired GPU IDs.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py
```


or provide args to accelerate launch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu \
  --mixed_precision fp16 \
  train.py
```

or use the accelerate config file:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```