# Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding

This repository releases the official [AReaL](https://github.com/inclusionAI/AReaL) implementation of **Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding**. Our approach enables a 8B parameter model to surpass the accuracy and efficiency of a 235B model on RefCOCO benchmarks. 

## RL Training

We provide the Grounding training and inference workflow for the **EGM-8B** model as the primary example below. Since AReaL adopts a decoupled loss loss to improve asynchronous training performance, the training hyperparameters in this repo are slightly different from those described in the paper.

### 1. Installation

```bash
git clone https://github.com/antoinegg1/EGM-AReaL
conda create -n EGM python=3.12
conda activate EGM

cd AReaL
pip install uv
uv pip install -e .[all]
```

### 2. Model 

| Training Phase | Model | 
| :--- | :--- | 
| Supervised Fine-Tuning (SFT) | [EGM-Qwen3-VL-8B-SFT](https://huggingface.co/JamesZGQ/EGM-8B-SFT) |
| Reinforcement Learning | [EGM-Qwen3-VL-8B-v1](https://huggingface.co/JamesZGQ/EGM-8B) |

### 3. Data Preparation

Please download the training and testing datasets before proceeding.

[Training annotations](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/tree/main/train_data) |
[Testing annotations](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/tree/main/eval_data) |
[Images tar1](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/blob/main/coco.tar) |
[Images tar2](https://huggingface.co/datasets/JamesZGQ/EGM_Datasets/blob/main/coco_flip.tar)

```bash
# Set your environment variables
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}
export OUTPUT_DIR=${YOUR_DATA_DIR}
export VAL_DIR=${YOUR_VAL_DIR}
export TRAIN_JSON=${QWEN3_8B_GROUNDING_TRAIN_JSON}

# Run preprocessing scripts
bash examples/agrounding/data_preprocess/grounding_val.sh
bash examples/agrounding/data_preprocess/grounding_all.sh
```

### 4. Training

Reinforcement Learning is conducted based on the [EGM-Qwen3-VL-8B-SFT](https://huggingface.co/JamesZGQ/EGM-8B-SFT). The default configuration utilizes 8 GPUs with ray framework. You may customize the distributed training settings via the `cluster.nnodes` and `cluster.n_gpus_per_node` arguments. **The data directory `(DATA_DIR)` should be the same as output directory `(OUTPUT_DIR)` in Data Preparation.**


```bash
export WANDB_BASE_URL=${YOUR_WANDB_BASE_URL}   
export WANDB_API_KEY=${YOUR_WANDB_API_KEY} 
DATA_DIR=${YOUR_DATA_DIR}
PROJECT_NAME=${YOUR_PROJECT_NAME}
MODEL_PATH=${YOUR_MODEL_PATH}

python3 -m areal.launcher.local \
  examples/agrounding/agrounding_grpo.py --config examples/agrounding/agrounding_grpo.yaml \
  actor.path="$MODEL_PATH" \
  train_dataset.path="$DATA_DIR/train_grounding.parquet" \
  valid_dataset.path="$DATA_DIR/val_grounding.parquet" \
  experiment_name="$PROJECT_NAME"

```

### 5. Inference and Evaluation


To evaluate the model, you can use the command provided below.

**Note:** The RefCOCO benchmark consists of eight distinct JSON files. Consequently, you must run the evaluation script sequentially for each of the 8 files to obtain the complete benchmark results.

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash examples/agrounding/evaluation/sglang_infer.sh
```

We also support evaluation with vLLM:

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash examples/agrounding/evaluation/vllm_infer.sh
```
---