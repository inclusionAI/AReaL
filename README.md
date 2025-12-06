# Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding

This repository releases the official [AReaL](https://github.com/inclusionAI/AReaL) implementation of **Bigger or Longer? Test-Time-Scaling is More Efficient than Model-Size-Scaling for Visual Grounding**. Our approach enables a 8B parameter model to surpass the accuracy and efficiency of a 235B model on RefCOCO benchmarks. 

## RL Training

We provide the Grounding training and inference workflow for the **EGM-8B** model as the primary example below.
由于AReaL中使用了decouple loss来提高异步训练表现，该repo中的训练参数设置与论文中的说明略有不同

### 1. Installation

```bash
git clone https://github.com/zgq1879/EGM.git
conda create -n EGM python=3.12
conda activate EGM

cd verl
pip install -e .[vllm]
pip install "pyzmq==26.4.0"
```

### 2. Model 

| Training Phase | Model | HuggingFace |
| :--- | :--- | :--- |
| Supervised Fine-Tuning (SFT) | EGM-Qwen3-VL-4B-SFT | [Link](https://huggingface.co/JamesZGQ/EGM-4B-SFT) |
| Reinforcement Learning | EGM-Qwen3-VL-4B-v1 | [Link](https://huggingface.co/JamesZGQ/EGM-4B) |
| Supervised Fine-Tuning (SFT) | EGM-Qwen3-VL-8B-SFT | [Link](https://huggingface.co/JamesZGQ/EGM-8B-SFT) |
| Reinforcement Learning | EGM-Qwen3-VL-8B-v1 | [Link](https://huggingface.co/JamesZGQ/EGM-8B) |

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
bash examples/data_preprocess/grounding_val.sh
bash examples/data_preprocess/grounding_all.sh
```

### 4. Training

Reinforcement Learning  is conducted based on the SFT checkpoint. The default configuration utilizes 8 GPUs. You may customize the distributed training settings via the `trainer.nnodes` and `trainer.n_gpus_per_node` arguments. The data directory `(DATA_DIR)` should be the same as output directory `(OUTPUT_DIR)` in Data Preparation.

```bash
export WANDB_BASE_URL=${YOUR_WANDB_BASE_URL}   
export WANDB_API_KEY=${YOUR_WANDB_API_KEY} 
export DATA_DIR=${YOUR_DATA_DIR}
export PROJECT_NAME=${YOUR_PROJECT_NAME}
export TASK_NAME=${YOUR_TASK_NAME}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export MODEL_PATH=${YOUR_MODEL_PATH}

bash scripts/grounding_qwen.sh
```

### 5. Inference and Evaluation


To evaluate the model, update sglang with `pip install sglang==0.5.5.post3` and use the command provided below.

**Note:** The RefCOCO benchmark consists of eight distinct JSON files. Consequently, you must run the evaluation script sequentially for each of the 8 files to obtain the complete benchmark results.

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash scripts/sglang_infer.sh
```

We also support evaluation with vLLM, update vLLM with `pip install vllm==0.11.0` and use the command provided below:

```bash
export MODEL_PATH=${YOUR_MODEL_PATH}
export DATA_JSON=${DATA_JSON}
export OUTPUT_DIR=${YOUR_OUTPUT_DIR}
export BASE_IMG_PATH=${YOUR_BASE_IMG_PATH}

bash scripts/vllm_infer.sh
```

### 6. Handling Environment Issues

1. If you encounter runtime errors related to FlashInfer, such as `GLIBC_2.32' not found (required by .../flashinfer/.../sampling.so)`, you can work around this by disabling the FlashInfer sampler: set the environment variable `VLLM_USE_FLASHINFER_SAMPLER=0` or `SGLANG_IS_FLASHINFER_AVAILABLE=false` before launching the training or inference command.

---