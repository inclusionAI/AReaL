
# Training Results

This document summarizes the training results of our models enhanced with Verl-Tool across various agentic RL tasks. 

## Model Checkpoints 

All these models are available in our [Huggingface Collection](https://huggingface.co/VerlTool).

### Math TIR Models
|Model|Link| Wandb |
|-|-|-|
|Qwen-2.5-Math-1.5B-Verl-tool|[🤗](https://huggingface.co/VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step)|[📈](https://wandb.ai/1004271927-SHU/Verl-Tool-Math?nw=nwuser1004271927)|
|Qwen-2.5-Math-7B-Verl-tool|[🤗](https://huggingface.co/VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-7b-grpo-n16-b128-t1.0-lr1e-6-310-step)|[📈](https://wandb.ai/1004271927-SHU/Verl-Tool-Math?nw=nwuser1004271927)|

### Search-R1 Models
|Model|Link| Wandb |
|-|-|-|
|VT-Search-zero-3B (GRPO)|[🤗](https://huggingface.co/VerlTool/search_r1_qa_em-qwen_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr1e-6debug_global_step_100)|[📈](https://wandb.ai/dongfu/search_r1_qa_em/runs/6g2t20cv)|
|VT-Search-zero-3B (DAPO)|[🤗](https://huggingface.co/VerlTool/search_r1_qa_em-qwen_qwen2.5-3b-grpo-n16-b512-64-t1.0-lr1e-6-dapo_global_step_160)|[📈](https://wandb.ai/dongfu/search_r1_qa_em/runs/4bwgd2fj)|
|VT-Search-zero-7B (GRPO)|[🤗](https://huggingface.co/VerlTool/search_r1_qa_em-qwen_qwen2.5-7b-grpo-n16-b512-64-t1.0-lr1e-6_global_step_100)|[📈](https://wandb.ai/dongfu/search_r1_qa_em/runs/6g2t20cv)|
|VT-Search-zero-7B (DAPO)|[🤗](https://huggingface.co/VerlTool/search_r1_qa_em-qwen_qwen2.5-7b-grpo-n16-b512-64-t1.0-lr1e-6-dapo_global_step_140)|[📈](https://wandb.ai/dongfu/search_r1_qa_em/reports/VT-Search-zero-7B-DAPO---VmlldzoxNDc1Nzk0Nw)|

### NL2SQL Models
|Model|Link| Wandb |
|-|-|-|
VT-SQL-7B|[🤗](https://huggingface.co/VerlTool/sqlcoder-qwen2.5-coder-7b-instruct-grpo-n5-b256-t0.6-lr1e-6_global_step_60)|[📈](https://wandb.ai/dongfu/sqlcoder/reports/VT-SQL-7B--VmlldzoxNDc1Nzk5OA)|

### SWE Models
|Model|Link| Wandb |
|-|-|-|
|VT-SWE-8B|[🤗](https://huggingface.co/VerlTool/SWE-Qwen3-8B-VT-grpo-n32-b256-t1.0-lr2e-6)|[📈](https://wandb.ai/zhihenglyu-cs/qwen3_r2e/reports/Qwen3-8B-SWE-on-VerlTool--VmlldzoxNDMyNTA1Mg)|

### Visual Reasoner Models
|Model|Link| Wandb |
|-|-|-|
|VT-VisualReasoner-7B|[🤗](https://huggingface.co/VerlTool/pixel_reasoner-7b-grpo-n8-b128-t1.0-lr1e-6-complex-reward-new_global_step_50)|[📈](https://wandb.ai/dongfu/pixel_reasoner/runs/0v0few4l)|

### DeepSearch Models
|Model|Link| Wandb |
|-|-|-|
|VT-DeepSearch-8B|[🤗](VerlTool/deepsearch-qwen_qwen3-8b-grpo-n16-b128-t1.0-lr1e-6-new_global_step_70)|[📈](https://wandb.ai/dongfu/deepsearch/reports/VT-DeepSearch-8B--VmlldzoxNDc1ODA0NA)|


### Math Benchmark Results

**1.5B Model Performance across challenging mathematical benchmarks:**
| Model Name                                 | Tool | GSM8K | MATH 500 | Minerva Math | Olympiad Bench | AIME24 | AMC23 | Avg   |
|--------------------------------------------|-----------|--------|-----------|---------------|------------------|------------------|--------|--------|
| Qwen2.5-Math-1.5B                           | ❌        | 39.50  | 34.80     | 8.10          | 23.00            | 13.30            | 35.00  | 25.62 |
| Qwen2.5-Math-1.5B-Instruct                  | ❌        | 84.90  | 74.20     | 26.80         | 39.00            | 10.00            | 57.50  | 48.70 |
| Qwen2.5-Math-1.5B-Instruct + SimpleRL-Zoo   | ❌        | 81.90  | 70.20     | 20.60         | 33.90            | 20.00            | 55.00  | 46.90 |
| Qwen-2.5-Math-1.5B-Instruct-TIR             | ✅        | 83.70  | 76.20     | 24.30         | 41.30            | 26.70            | 55.00  | 51.20 |
| ToRL-1.5B                                   | ✅        | 85.60  | 77.80     | 29.80         | 44.00            | 26.70            | 67.50  | 55.23 |
| **Qwen-2.5-Math-1.5B + Verl-Tool**          | ✅        | **85.10** | **77.40** | **28.30**     | **44.00**        | **33.30**        | **65.00** | **55.52** |


**7B Model Performance across challenging mathematical benchmarks:**
|Model Name                                 |Tool|GSM8K|MATH 500|Minerva  Math|Olympiad  Bench|AIME24 |AMC23|Avg  |
|-------------------------------------------|---------|-----|--------|-------------|---------------|----------------|-----|-----|
|Qwen-2.5-Math-7B                           |❌        |65.50|63.60   |12.50        |25.80          |13.30           |42.50|37.20|
|Qwen2.5-Math-7B-Instruct                   |❌        |95.20|83.00   |37.10        |41.60          |16.70           |70.00|57.27|
|Qwen-2.5-Math-7B + SimpleRL-Zoo            |❌        |88.80|80.20   |26.80        |41.60          |30.00           |52.50|53.30|
|Qwen-2.5-Math-7B-Instruct-TIR              |✅        |94.60|82.40   |29.00        |50.50          |30.00           |62.50|58.17|
|TORL-7B    |✅        |92.70|82.20   |33.50        |49.90          |43.30           |65.00|61.10|
|**Qwen-2.5-Math-7B + Verl-Tool**           |✅        |**91.40**|**83.40**|**29.80**    |**50.20**      |**40.00**       |**72.50**|**61.22**|

