# ThreadWeaver Data Generation Pipeline
A data generation pipeline for creating high-quality mathematical reasoning trajectories with parallel branching generation.

## Overview
This pipeline transforms model-generated reasoning chains into structured, multi-threaded trajectories through five refinement stages. Each stage uses LLM-based analysis and rewriting to identify parallel reasoning paths, generate outlines, and produce high-quality training data.

## Setup
### Install the required dependencies
```bash
pip install numpy==1.26.4 pandas==2.2.3 transformers==4.51.1 datasets==3.6.0 torch==2.6.0 openai==1.75.0 sympy==1.13.1 pylatexenc==2.10 matplotlib==3.10.3 tqdm==4.67.1 termcolor==3.1.0 requests==2.32.3 sglang==0.4.6.post1 tokenizers==0.21.1 safetensors==0.5.3 huggingface-hub==0.31.4
```

### Download polaris-53K dataset
```bash
mkdir -p data
wget -O data/polaris-data-53K.parquet https://github.com/ChenxinAn-fdu/POLARIS/raw/refs/heads/main/parquet/stage1/polaris-data-53K.parquet
[ "$(md5sum data/polaris-data-53K.parquet | awk '{print $1}')" = "58e1e523f9946956f68055a374d43a46" ] && echo "md5 matches the reference" || echo "md5 does not match the reference"
```

### Set up your OpenAI API key
Set `OPENAI_KEY_PATH` environment variable to point to a file with the key (default `~/.openai_api_key`). If the file is missing, `OPENAI_API_KEY` env variable will be used.

```bash
export OPENAI_KEY_PATH=~/.openai_api_key
```

## Quick Start

### 1. Generate Initial Responses
Run inference on your model to generate initial sequential reasoning chains for mathematical problems.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/generate_trajectories.py \
  --model_name Qwen/Qwen3-8B \
  --launch_server \
  --verbose 1 \
  --template-type model \
  --suffix bfloat16_model_template \
  --bfloat16 \
  --autoregressive \
  -n 1 \
  --max-context-length 40960 \
  --data-type polaris-data-53K \
  --no-conclusion \
  --total-splits 1 \
  --current-split 0
```

### 2. Process to Training Data Format
Filter correct solutions using the reward model, formats them with proper chat templates (DeepSeek/Qwen), and creates training-ready datasets.
```bash
python src/generated_data_to_training_data_polaris.py \
  --dataset-json data/Qwen3-8B_bfloat16_model_template/polaris-data-53K_1.json \
  --polaris-parquet data/polaris-data-53K.parquet \
  --output-dir data/processed \
  --sample-size 1000 \
  --workers 32
```

### 3. Run Multi-Stage Refinement Pipeline
Run the complete 5-stage pipeline to transform reasoning chains into structured parallel trajectories. See "Pipeline Stages Explained" below for details.
```bash
NUM_SAMPLES=1000 ./run.sh
```

**💡 Tip:** Start with a small sample to test the pipeline:
```bash
NUM_SAMPLES=10 ./run.sh  # Annotate just 10 samples first to verify everything works
```

This script will also save the huggingface dataset to `dataset` for training.

### 4. Visualize Results (Optional)
Launch an interactive web UI to browse, search, and analyze the final trajectories with statistics and syntax highlighting.
```bash
python src/visualize_trajectories.py \
  --root data/output_step5 \
  --port 8899
```

Open http://127.0.0.1:8899 in your browser.

![Visualization Example](../assets/visualization_example.png)

## Pipeline Stages Explained

The multi-stage pipeline (`run.sh`) processes trajectories through five sequential stages. Each stage builds upon the previous one, progressively refining the data.

### Stage 0: Data Collection (`collect_trajectories.py`)
**Purpose:** Sample and prepare trajectories from the source dataset.

**What happens:**
- Loads the processed training data
- Samples N trajectories (configurable via `NUM_SAMPLES`)
- Exports to individual `.txt` files and `collected.jsonl`

**Output:** `data/<name>_<N>samples/`
- `collected.jsonl` - All samples in JSONL format
- `*.txt` - Individual trajectory files

---

### Stage 1: Step Extraction (`step1.sh`)
**Script:** `src/gpt.py` with `prompt/step1-prompt_v1.txt`

**Purpose:** Analyze reasoning chains and extract hierarchical step structure.

**What happens:**
- **Task 1:** Identifies all main reasoning steps (S1, S2, S3, ...)
- **Task 2:** Breaks down complex steps into substeps (S1.1, S1.2, ...)
- Assigns line number ranges to each step (e.g., `S3: Check edge cases (L45-L67)`)
- Preserves both successful and abandoned reasoning paths

**Why it matters:** This creates a semantic map of the reasoning process, identifying where different approaches or parallel explorations occur.

**Output:** `data/<name>_step1_v1/`
- `*_reasoning.txt` - Step-by-step breakdown with line ranges

**Example output:**
```
S1: Problem understanding and setup (L1-L15)
S2: Explore algebraic approach (L16-L34)
  S2.1: Try factorization (L16-L22)
  S2.2: Apply quadratic formula (L23-L34)
S3: Switch to geometric interpretation (L35-L52)
S4: Verify solution with test cases (L53-L67)
```

---

### Stage 2: Parallel Path Extraction (`step2.sh`)
**Script:** `src/extract_v1.py`

**Purpose:** Identify and mark parallel reasoning threads that explore different approaches.

**What happens:**
- Analyzes the step structure from Stage 1
- Identifies reasoning segments that could be explored in parallel
- Inserts `<Parallel>` and `<Thread>` markers around these segments
- Applies filtering to remove low-information parallel blocks (via `--filter-diff-threshold`)

**Why it matters:** This structures the trajectory for multi-threaded reasoning, identifying where the model explored multiple approaches simultaneously.

**Output:** `data/<name>_step2_v1/`
- `*.txt` - Trajectories with `<Parallel>` annotations
- `trace/*.txt` - Processing logs for debugging

**Example transformation:**
```
Original: [Linear reasoning chain exploring 3 approaches sequentially]

After Step 2:
<Parallel>
Reason: Multiple solution approaches attempted
<Thread>
Thread 1: Algebraic approach using substitution...
</Thread>
<Thread>
Thread 2: Geometric interpretation using diagrams...
</Thread>
<Thread>
Thread 3: Numerical verification with examples...
</Thread>
</Parallel>
```

---

### Stage 3: Context Rewriting (`step3.sh`)
**Script:** `src/rewrite-context_v1.py` with `prompt/step3-prompt_v1_v1.txt`

**Purpose:** Improve and refine the content within each parallel thread.

**What happens:**
- Rewrites thread content for clarity and coherence
- Ensures each thread is self-contained and readable
- Maintains mathematical accuracy while improving presentation
- Preserves the parallel structure markers

**Why it matters:** Raw model outputs can be verbose or unclear. This stage polishes the content while preserving the logical structure.

**Output:** `data/<name>_step3_v1_v1/`
- `*.txt` - Rewritten trajectories with cleaner content
- `trace/*.txt` - Rewriting logs

---

### Stage 4: Outline Generation (`step4.sh`)
**Script:** `src/generate-outline_v1.py` with `prompt/step4-prompt_v1_v1_v1.txt`

**Purpose:** Generate structured outlines for parallel reasoning blocks.

**What happens:**
- Creates hierarchical outlines for each `<Parallel>` section
- Wraps outlines in `<Outlines>` tags with numbered `<Outline>` entries
- Provides a roadmap of what each thread will explore
- Prepares the trajectory for the final ThreadWeaver format

**Why it matters:** Outlines help the model (and humans) understand the high-level strategy before diving into detailed reasoning.

**Output:** `data/<name>_step4_v1_v1_v1/`
- `*.txt` - Trajectories with `<Outlines>` and `<Thread>` blocks

**Example structure:**
```
<Parallel>
<Outlines>
<Outline> 1: Algebraic substitution method
<Outline> 2: Geometric visualization approach
<Outline> 3: Numerical verification strategy
</Outlines>

<Thread>
1: [Detailed algebraic reasoning...]
</Thread>
<Thread>
2: [Detailed geometric reasoning...]
</Thread>
<Thread>
3: [Detailed numerical verification...]
</Thread>
<Conclusion>
[Final synthesis of all approaches...]
</Conclusion>
</Parallel>
```

---

### Stage 5: Quality Filtering (`step5.sh`)
**Script:** `src/filter-format-correct-and-obtain-stats.py`

**Purpose:** Validate and filter trajectories for quality and correctness.

**What happens:**
- Checks formatting: ensures proper `<Parallel>`, `<Thread>`, `<Outlines>` structure
- Validates completeness: verifies all threads are properly closed
- Filters malformed trajectories
- Generates statistics on pass rates and quality metrics

**Why it matters:** Only well-formed, complete trajectories should be used for training. This stage ensures data quality.

**Output:** `data/<name>_step5_v1_v1_v1/`
- `*.txt` - Final validated trajectories (ready for training)
- Statistics on filtering results

## Understanding the Output

After running the full pipeline, your final trajectories will have this structure:

```
<think>
[Initial problem analysis and setup]

<Parallel>
<Outlines>
<Outline> 1: [First approach description]
<Outline> 2: [Second approach description]
<Outline> 3: [Third approach description]
</Outlines>

<Thread>
1: [Detailed reasoning for first approach...]
</Thread>
<Thread>
2: [Detailed reasoning for second approach...]
</Thread>
<Thread>
3: [Detailed reasoning for third approach...]
</Thread>
</Parallel>
</think>

...
[Final answer in \boxed{} format]
```

# Acknowledgements
We thank the Polaris team for providing [the Polaris-53K dataset](https://github.com/ChenxinAn-fdu/POLARIS/blob/main/parquet/stage1/polaris-data-53K.parquet). Our prompts for the first stage (reasoning step extraction stage) of data generation are adapted from the [prompts of Multiverse](https://github.com/Multiverse4FM/Multiverse/blob/main/data/prompt/step1-prompt.txt).
