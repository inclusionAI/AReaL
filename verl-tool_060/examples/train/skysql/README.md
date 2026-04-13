# skysql Training Guide

This guide covers data preprocessing and training setup for the skysql model using the verl-tool framework.

## Prerequisites

⚠️ **Important**: All operations below assume you are in the parent directory of the `verl-tool` repository.

```bash
uv pip install -e .[sql_tool]
```

## Data Preparation
```bash
# prepare the the training databases
huggingface-cli download --local-dir "data/skysql" --repo-type dataset VerlTool/SkyRL-SQL-Reproduction databases.zip
unzip data/skysql/databases.zip -d data/skysql
python examples/data_preprocess/skysql/sql.py --dataset_path VerlTool/SkyRL-SQL-Reproduction --train_databases_dir ./data/skysql/databases --test_databases_dir ./data/skysql/databases --save_dir ./data/skysql
```

## Test Sql server
```bash
# Start the tool server
host=localhost
port=5000
tool_type=ipython_code # separate by comma if you want to start multiple tool servers
workers_per_tool=512 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool --use_ray=True --log_level debug &
```

```bash
python -m verl_tool/servers/tests/test_sql_tool.py python --url=http://localhost:$port/get_observation

## Training
```bash
bash examples/train/skysql/train_7b.sh
```

## 📝 Important Notes

- **Note: please set `enable_prefix_caching=False` when running to reproduce the validation results!!!**



## Reference:

- https://github.com/RUCKBReasoning/OmniSQL/tree/main/train_and_evaluate
- https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/eval_open_source_models.py
- https://skyrl.readthedocs.io/en/latest/examples/multi_turn_text2sql.html
- https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train
- https://novasky-ai.notion.site/skyrl-sql



## Others
<details>
  <summary>Other Dataset Preparation logs (Optional, might be outdated)</summary>
The skysql training requires two main datasets:
- **Omni-SQL dataset** (~50GB): Contains evaluation data for Spider-series and BIRD datasets, plus the SynSQL-2.5M training dataset
- **Preprocessed training/evaluation datasets**: Ready-to-use parquet files for training

### Step 1: Download Omni-SQL Dataset

**Option A: Manual Download**
1. Download from: https://huggingface.co/datasets/seeklhy/OmniSQL-datasets
2. Decompress to `verl-tool/data/`

**Option B: Automated Script**
```bash
cd <verl-tool-parent-path>
bash ./examples/data_preprocess/skysql/download_skysql.sh
```

### Step 2: Verify Data Structure

After downloading, your folder structure should look like:

```bash
📁 verl-tool/data/synsql/data
├── 📁 bird/
├── 📁 spider/
├── 📁 Spider-DK/
├── 📁 spider-realistic/
├── 📁 Spider-Syn/
├── 📁 ... (other dataset subfolders)
├── 📄 dev_bird.json
├── 📄 dev_spider_dk.json
└── 📄 ... (other evaluation json files)
```

## 🚀 Quick Start: Using Preprocessed Datasets

### Download Ready-to-Use Datasets

Clone the preprocessed datasets directly:

```bash
huggingface-cli download --local-dir "data/skysql" --repo-type dataset VerlTool/SkyRL-SQL-Reproduction train.parquet test.parquet
```

Split test.parquet by data_sources:

```python
import datasets
import json
dataset = datasets.load_dataset('parquet', data_files="./data/skysql/test.parquet", split="train")
all_data_sources = dataset.unique('data_source')
for data_source in all_data_sources:
    subset_dataset = dataset.filter(lambda x: x['data_source'] == data_source)
    subset_dataset.to_parquet(f"./data/skysql/{data_source}.parquet")
    print(f"Saved {data_source} dataset with {len(subset_dataset)} records to ./data/skysql/{data_source}.parquet")
```

### Configure Training Script

Update the dataset paths in `examples/train/skysql/train_7b.sh`:

```bash
# Set these paths according to your downloaded data
train_data=$(pwd)/data/${dataset_name}/train.parquet
val_data=$(pwd)/data/${dataset_name}/test.parquet
model_name=/path/to/your/model/weights  # e.g., Qwen2.5-Coder-7B-Instruct
```

### Optional: Enable Weights & Biases Logging

If you want to track training metrics:

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```

## 🔧 Manual Dataset Preprocessing

If you need to reprocess the datasets from scratch:

### Training Dataset Preprocessing

The training dataset is converted from SynSQL-2.5M format to verl-tool format.

**Script Location**: `verl-tool/examples/data_preprocess/skysql/prepare_train.py`

**Configuration**:
- Modify `DEFAULT_DATABASE_PATH` to point to the SynSQL-2.5M dataset's `databases` subfolder
- If you downloaded Omni-SQL correctly, SynSQL-2.5M should be included as a subfolder

### Evaluation Dataset Preprocessing

The evaluation dataset merges 6 subsets from the Omni-SQL dataset.

**Script Location**: `verl-tool/examples/data_preprocess/skysql/prepare_test.py`

**Before Running**:
1. Check the demo instructions at the bottom of the script (commented out)
2. **Critical**: Update all database paths for `DEV_PROMPT` and `DEV_SCHEMA` (lines 25-41 in the script)

</details>