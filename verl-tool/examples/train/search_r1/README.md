# Search-R1 Port Implementation for Verl-Tool

This repository contains a complete port of the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework to the Verl-Tool ecosystem, enabling large language models to learn search-enhanced question answering through reinforcement learning.

## ⏩ Quick start

You may use the below command to fire up a quick training. Detailed commands for each step and their explanations are contained in the below sections. 

We assume you:
1. Will be executing the code at the root directory of verl-tool repo.
2. Will store both the training data and retriever index files at `./data/search_r1`.
3. Already have the conda environment: `verl-tool-env` properly configured.

(Note, `faiss-gpu` seems only able to install with conda, so if you encounter error using pip to install it, please use conda to install it.)

### Step 1: Download the retriever index and prepare the retrieval server

```bash
# download the index
save_path=./data/search_r1/retriever_index
python ./verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path

# Prepare index
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 2: Start the retrieval server
- create a separate environment for the retrieval server, e.g. `search-retriever`, and activate it.
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
uv pip install transformers datasets fastapi numpy torch uvicorn
```
- then run the retrieval server with the following command:

```bash
# activate sglang-retriever
file_path=./data/search_r1/retriever_index
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
python ./verl_tool/servers/tools/utils/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu &
```
need to wait for 5 mins

### Step 3: prepare the training dataset
```bash
python examples/data_preprocess/search_r1.py --local_dir ./data/search_r1/training_data --prefix_type search_r1
```

### Step 4: perform model training
```bash
bash ./examples/train/search_r1/train.sh
```

For model format conversation and tensorboard visualization, refer to the following sections. 

## 🎯 Overview

Search-R1 is a framework that trains language models to use search tools for enhanced question answering. This port adapts the original Search-R1 implementation to work seamlessly with Verl-Tool's training infrastructure, providing:

- **Local Dense Retriever**: A FastAPI-based retrieval server using FAISS for efficient document search
- **Search Tool Integration**: Seamless integration with Verl-Tool's tool system
- **Multi-turn Training**: Support for multi-turn conversations with search capabilities
- **Exact Match Reward**: Specialized reward function for question-answering tasks

## 🏗️ Architecture

We mainly refer to Search-R1's [official implementation](https://github.com/PeterGriffinJin/Search-R1/tree/main?tab=readme-ov-file) and adapt to [SGLang Team's port of Search-R1 on verl](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md). The data processing script and retriever are directly borrowed from SGLang's implementation which has already been integrated into verl officially.



### Core Components

1. **Local Dense Retriever** (`local_dense_retriever/`)
   - Location: `verl-tool/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py`
   - `retrieval_server.py`: FastAPI server providing document retrieval via FAISS
   - `download.py`: Script to download pre-built indices and corpus data

2. **Search Tool** (`search_retrieval.py`)
   - Location: `verl-tool/verl_tool/servers/tools/search_retrieval.py`
   - Supports batch query processing
   - Handles retry logic and error recovery

3. **Reward Function** (`search_r1_like_qa_em.py`)
   - Location: `verl-tool/verl_tool/workers/reward_manager/search_r1_qa_em.py`
   - Exact match scoring for question-answering tasks
   - Extracts answers from `<answer>` tags
   - Normalizes text for robust comparison

### Data Flow

```
User Query → LLM → Search Tool → Retrieval Server → FAISS Index → Corpus
                ↓
            <answer> tag → Reward Function → Training Signal
```

## 🔧 Implementation Details

### Retrieval Server Features

- **Dense Retrieval**: Uses E5 embeddings with FAISS for fast similarity search
- **BM25 Support**: Alternative sparse retrieval method
- **Batch Processing**: Efficient handling of multiple queries
- **GPU Acceleration**: Optional FAISS GPU support
- **Flexible Configuration**: Configurable top-k, pooling methods, and model paths

### Reward Function Logic

The reward function implements exact match scoring:

1. **Answer Extraction**: Extracts text between `<answer>` and `</answer>` tags
2. **Text Normalization**: Removes punctuation, articles, and normalizes whitespace
3. **Exact Match**: Compares normalized prediction with ground truth
4. **Format Penalty**: Penalizes excessive answer tags

For detailed reference, check Search-R1's original paper and official implementation.

## 🚀 Setup Instructions
### Environment Setup

Refer to verl-tool environment configuration.

### Data Preparation

1. **Download Index and Corpus**:
```bash
# Set up retriever environment
source <path_to_miniconda3>/bin/activate <path_to_miniconda3>/envs/verl-tool-env
export HF_ENDPOINT=https://hf-mirror.com    

# Download data
save_path=<path_to_save_data>
python verl-tool/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path

# Prepare index
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

2. **Prepare Training Dataset**:
```bash
python verl-tool/verl/scripts/data_preprocess/preprocess_search_r1_dataset.py\
   --local_dir <path_to_target_directory>
```

### Configuration

1. **Update Timeout Settings**:
```bash
# Edit verl-tool/verl/verl/tools/utils/search_r1_like_utils.py
# Change DEFAULT_TIMEOUT from 30 to 120 to avoid HTTP500 errors
```

2. **Set Environment Variables**:
```bash
export RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export RETRIEVER_TOPK=3
export RETRIEVER_TIMEOUT=120
```

## 📖 Usage Guide

### Starting the Retrieval Server

```bash
# Launch the dense retriever server
cd verl-tool/verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever
python retrieval_server.py \
    --index_path /path/to/e5_Flat.index \
    --corpus_path /path/to/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

if encounter error: "/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found", check solution at https://github.com/pybind/pybind11/discussions/3453. Each time the retriever might need ~5min to load.


### Training a Model

```bash
# Basic training command
cd verl-tool
bash examples/train/search_r1/train_search_r1_reproduce.sh
```

Note: by default wandb recording is disabled. To activate it, modify this line in `train_search_r1_reproduce.sh`:

```bash
trainer.logger=['console','tensorboard', 'wandb'] \
```

and set your wandb API key as an environment variable:

```bash
export WANDB_API_KEY="<your_key>"
```

### Model Checkpoint Merging

After training, merge FSDP checkpoints to HuggingFace format:

```bash
cd verl-tool/verl
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/checkpoint/actor \
    --target_dir /path/to/merged/checkpoint
```

## 📊 Performance Results

Our implementation achieves competitive results compared to the one reported by SGLang's team.

Their wandb training report is [here](https://wandb.ai/lingchang-ustc/search_async_rl/runs/21rubwvs?nw=nwuserlingchang)

📊 Our reimplementation's Performance
| Training Steps | `popqa` | `triviaqa` | `wikimultihopqa` | `nq`  | `hotpotqa` | `bamboogle` | `musique` |
| -------------- | ------- | ---------- | ---------------- | ----- | ---------- | ----------- | --------- |
| 40             | 0.434   | 0.573      | 0.261            | 0.390 | 0.272      | 0.104       | 0.058     |
| 80             | 0.438   | 0.603      | 0.280            | 0.430 | 0.301      | 0.128       | 0.068     |
| 120            | 0.437   | 0.604      | 0.362            | 0.442 | 0.358      | 0.288       | 0.114     |
| 160            | 0.463   | 0.617      | 0.402            | 0.448 | 0.390      | 0.368       | 0.145     |
| 200            | 0.476   | 0.619      | 0.401            | 0.462 | 0.401      | 0.360       | 0.151     |

📊 Original Search-R1 Performance
| Training Steps | `popqa` | `triviaqa` | `wikimultihopqa` | `nq`  | `hotpotqa` | `bamboogle` | `musique` |
| -------------- | ------- | ---------- | ---------------- | ----- | ---------- | ----------- | --------- |
| 50             | 0.358   | 0.510      | 0.189            | 0.349 | 0.233      | 0.104       | 0.051     |
| 100            | 0.372   | 0.524      | 0.208            | 0.365 | 0.244      | 0.136       | 0.056     |
| 150            | 0.378   | 0.528      | 0.221            | 0.377 | 0.250      | 0.104       | 0.061     |
| 200            | 0.388   | 0.540      | 0.253            | 0.383 | 0.267      | 0.136       | 0.057     |

Our implementation's training record is provided as `verl-tool/examples/train/search_r1/reimplementation_tensorboard_records.0`. To view the results on tensorboard, run:

```bash
tensorboard --logdir <folder_to_tensorboard_report>
```

**Key Improvements:**
- **WikimultihopQA**: Significant improvement (0.362 vs 0.208)
- **HotpotQA**: Strong performance gains across all training steps
- **Bamboogle**: Excellent performance at 120/100 steps (0.288 vs 0.136)


### Performance Optimization

1. **Memory Management**:
   ```bash
   # For large models, use these settings
   gpu_memory_utilization=0.5
   do_offload=True
   use_dynamic_bsz=True
   ```

2. **Batch Size Tuning**:
   ```bash
   # Adjust based on your GPU memory
   ppo_micro_batch_size_per_gpu=8
   log_prob_micro_batch_size_per_gpu=16
   ```

3. **Retrieval Performance**:
   ```bash
   # Optimize retrieval server
   retrieval_batch_size=512
   faiss_gpu=True
   retrieval_use_fp16=True
   ```
