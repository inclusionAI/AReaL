# geo_edit

A Vision-Language Model (VLM) tool-calling and evaluation framework supporting multiple backends (Google Gemini, OpenAI, vLLM, SGLang).

## Overview

geo_edit provides:
- **Multi-backend support**: Google Gemini, OpenAI, vLLM, SGLang
- **Tool system**: Image editing tools + VLM analysis agents (Ray-based)
- **Task abstraction**: Unified VisionQA task interface
- **Evaluation**: Multiple evaluation methods (Judge model, exact match, numerical)
- **Extensibility**: Plugin-style tool definitions and dataset adapters

## Architecture

```text
geo_edit/
├── agents/                 # API Agent implementations
│   ├── base.py            # BaseAgent ABC and AgentConfig
│   └── api_agent.py       # APIBasedAgent (Google/OpenAI/vLLM/SGLang)
├── config.py              # Configuration builders
├── constants.py           # Global constants (MAX_TOOL_CALLS=4)
├── datasets/              # Dataset registration and prompt templates
│   ├── task_registry.py   # DATASET_SPECS definitions
│   └── input_template.py  # Prompt templates
├── environment/
│   ├── task/              # VisionQA task implementations
│   │   ├── base.py        # AbstractVLMTask
│   │   ├── vision_qa_task.py
│   │   ├── google_vision_qa_task.py
│   │   └── openai_compatible_vision_qa_task.py
│   └── tool_agents/       # Ray Actor tool agents
│       ├── actor.py       # BaseToolModelActor
│       └── manager.py     # ToolAgentManager
├── evaluation/            # Evaluation scripts
│   ├── openai_as_judge.py # LLM-based evaluation
│   ├── eval_shortest_path.py
│   └── eval_stmf_counting.py
├── prompts/               # System prompt management
│   ├── system_prompts.py  # Main agent prompts
│   ├── eval_prompts.py    # Evaluation prompts
│   └── tool_agent_prompts.py
├── scripts/               # Entry point scripts
│   └── async_generate_with_tool_call_api.py
├── tool_definitions/      # Tool definitions
│   ├── functions/         # Image editing tools (crop, label, draw_line, bbox, highlight)
│   ├── agents/            # VLM analysis agents (multimath, gllava, chartmoe)
│   ├── router.py          # ToolRouter
│   └── config.yaml        # Tool enable/disable config
├── utils/                 # Utility functions
└── tests/                 # Test files
```

---

## Tool System

geo_edit provides two types of tools:

| Type | Description | Execution |
|------|-------------|-----------|
| **Function Tools** | Image editing tools | Local Python functions |
| **Agent Tools** | VLM analysis agents | Ray Actor (GPU-resident) |

### Function Tools (Image Editing)

| Tool | Description | Return Type |
|------|-------------|-------------|
| `image_crop` | Crop image by bounding box | image |
| `image_label` | Add text label to image | image |
| `draw_line` | Draw line on image | image |
| `bounding_box` | Draw bounding box on image | image |
| `image_highlight` | Highlight area with yellow overlay | image |

### Agent Tools (VLM Analysis)

| Tool | Description | Return Type |
|------|-------------|-------------|
| `multimath` | Mathematical vision analysis | text |
| `gllava` | General vision-language analysis | text |
| `chartmoe` | Chart analysis | text |

---

## Tool Configuration

### 1. Enable/Disable Tools

Edit `tool_definitions/config.yaml`:

```yaml
# Image editing tools (functions/)
image_crop: true
image_label: true
draw_line: true
bounding_box: true
image_highlight: true

# VLM analysis tools (agents/)
multimath: true
gllava: false      # Set to false to disable
chartmoe: true
```

### 2. Runtime Tool Mode

Control via `--use_tools` parameter:

| Mode | Description |
|------|-------------|
| `auto` | Optional tool use (model decides) |
| `force` | Require tool call in each response |
| `direct` | Disable all tools (direct answer mode) |

---

## Adding New Tools

### Adding a Function Tool (Image Editing)

**Step 1**: Create tool file `tool_definitions/functions/my_tool.py`

```python
"""My custom image tool."""

from typing import List
from PIL import Image

DECLARATION = {
    "name": "my_tool",
    "description": "Description of what this tool does.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "Observation image index, such as 0 for Observation 0.",
            },
            "param1": {
                "type": "string",
                "description": "Description of param1.",
            },
        },
        "required": ["image_index", "param1"],
    },
}

RETURN_TYPE = "image"  # "image" or "text"


def execute(image_list: List[Image.Image], image_index: int, param1: str) -> Image.Image:
    """Execute the tool and return modified image."""
    image = image_list[image_index].copy()
    # ... process image ...
    return image
```

**Step 2**: Register in `tool_definitions/functions/__init__.py`

```python
from geo_edit.tool_definitions.functions import my_tool

FUNCTION_TOOLS: Dict[str, tuple] = {
    # ... existing tools ...
    "my_tool": (my_tool.DECLARATION, my_tool.execute, "function", my_tool.RETURN_TYPE),
}
```

**Step 3**: Enable in `tool_definitions/config.yaml`

```yaml
my_tool: true
```

### Adding an Agent Tool (VLM Analysis)

**Step 1**: Create agent file `tool_definitions/agents/my_agent.py`

```python
"""My VLM Agent Tool."""

from typing import List
from PIL import Image

from geo_edit.environment.tool_agents import call_agent

# Model configuration for Ray Actor
agent_config = {
    "model_name_or_path": "/path/to/your/vlm/model",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 1024,
    "num_gpus": 1,  # Number of GPUs to use
}

DECLARATION = {
    "name": "my_agent",
    "description": "Use a specialized VLM agent for specific analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "Observation image index, such as 0 for Observation 0.",
            },
            "question": {
                "type": "string",
                "description": "What you want to ask about the selected image.",
            },
        },
        "required": ["image_index", "question"],
    },
}

RETURN_TYPE = "text"


def execute(image_list: List[Image.Image], image_index: int, question: str) -> str:
    """Call the Ray Actor to analyze the image."""
    return call_agent("my_agent", image_list, image_index, question)
```

**Step 2**: Register in `tool_definitions/agents/__init__.py`

```python
from geo_edit.tool_definitions.agents import my_agent

AGENT_TOOLS: Dict[str, tuple] = {
    # ... existing agents ...
    "my_agent": (my_agent.DECLARATION, my_agent.execute, "agent", my_agent.RETURN_TYPE),
}

AGENT_CONFIGS: Dict[str, dict] = {
    # ... existing configs ...
    "my_agent": my_agent.agent_config,
}
```

**Step 3**: Add system prompt in `prompts/tool_agent_prompts.py`

```python
TOOL_AGENT_PROMPTS = {
    # ... existing prompts ...
    "my_agent": "You are a specialized VLM for [specific task]. Analyze the image and answer the question.",
}
```

**Step 4**: Enable in `tool_definitions/config.yaml`

```yaml
my_agent: true
```

---

## Deployment Architecture

Typical deployment: Head node runs vLLM 8-TP inference service, Worker node runs Tool Agents (each Agent uses 1 GPU).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster                                │
├─────────────────────────────────┬───────────────────────────────────────┤
│         Head Node (8 GPU)       │        Worker Node (8 GPU)            │
│                                 │                                       │
│  ┌───────────────────────────┐  │  ┌─────────────┐  ┌─────────────┐    │
│  │   vLLM Server (8-TP)      │  │  │  chartmoe   │  │  multimath  │    │
│  │   Qwen3-VL-235B           │  │  │  (GPU 0)    │  │  (GPU 1)    │    │
│  │   GPU 0-7                 │  │  └─────────────┘  └─────────────┘    │
│  └───────────────────────────┘  │  ┌─────────────┐  ┌─────────────┐    │
│                                 │  │   gllava    │  │   (spare)   │    │
│  ┌───────────────────────────┐  │  │  (GPU 2)    │  │  GPU 3-7    │    │
│  │  async_generate script    │  │  └─────────────┘  └─────────────┘    │
│  │  (calls vLLM + Agents)    │  │                                       │
│  └───────────────────────────┘  │                                       │
└─────────────────────────────────┴───────────────────────────────────────┘
```

### Step 1: Start Head Node (vLLM Inference Service)

```bash
# Start Ray on Head node
ray start --head --port=6379 --resources='{"tool_agent": 8}'

# Start vLLM 8-TP inference service
bash geo_edit/scripts/launch_vllm_generate.sh
```

`launch_vllm_generate.sh` contents:

```bash
#!/usr/bin/env bash
set -euo pipefail
model_path="/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking"
echo "model: $model_path"
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
mkdir -p /tmp/log/

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 65536 \
  --dtype auto \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  > /tmp/log/vllm_api.log 2>&1 &

echo "waiting endpoint..."
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do
  tail -n 10 /tmp/log/vllm_api.log
  sleep 2
done
echo "endpoint ready"
```

### Step 2: Start Worker Node (Tool Agents)

```bash
# Join Ray cluster on Worker node, mark as tool_agent node
ray start --address='33.180.172.15:6379' --resources='{"tool_agent": 8}'
```

> `"tool_agent": 8` indicates this node has 8 GPUs available for Tool Agents

### Step 3: Run Inference Script

Run the inference script on Head node. Tool Agents will be automatically created on Worker node:

```bash
python -m geo_edit.scripts.async_generate_with_tool_call_api \
  --model_type vLLM \
  --model_name_or_path /storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking \
  --api_base http://127.0.0.1:8000 \
  --dataset_path /path/to/data.parquet \
  --dataset_name mathvisionqa \
  --output_dir output \
  --use_tools auto \
  --node_resource tool_agent \
  --max_concurrent_requests 8 \
  --sample_rate 1.0
```

### Tool Agent Configuration

Each Tool Agent uses 1 GPU by default. Configuration in `tool_definitions/agents/`:

```python
# chartmoe.py
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/ChartMoE",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 1024,
    "num_gpus": 1,  # Each agent uses 1 GPU
}
```

### Scheduling Tool Agents on Worker Node

Use `--node_resource` parameter to ensure agents are scheduled on Worker node:

```bash
# Via command line
python -m geo_edit.scripts.async_generate_with_tool_call_api \
  --use_tools auto \
  --node_resource tool_agent \
  ...
```

Or programmatically with `ToolRouter`:

```python
from geo_edit.tool_definitions import ToolRouter

# ToolRouter automatically initializes agents on nodes with "tool_agent" resource
router = ToolRouter(
    tool_mode="auto",
    node_resource="tool_agent"
)
# 3 agents created, each using 1 GPU on Worker node
```

### Verify Deployment Status

```python
import ray
ray.init(address="auto")

# View cluster nodes and resources
for node in ray.nodes():
    print(f"Node: {node['NodeName']}")
    print(f"  Resources: {node['Resources']}")
    print(f"  Alive: {node['Alive']}")

# View Tool Agent status
from geo_edit.environment.tool_agents import get_manager
status = get_manager().status()
for name, info in status.items():
    print(f"Agent {name}: {info['status']}")
```

### Manual Agent Lifecycle Management

```python
from geo_edit.tool_definitions import ToolRouter
from geo_edit.environment.tool_agents import call_agent
from PIL import Image

# 1. Create router (automatically initializes agents)
router = ToolRouter(tool_mode="auto", node_resource="tool_agent")

# 2. Call agent
image = Image.open("chart.png")
result = call_agent("chartmoe", [image], 0, "What does this chart show?")
print(result)

# 3. Check status
from geo_edit.environment.tool_agents import get_manager
status = get_manager().status()
print(status)

# 4. Shutdown agents
router.shutdown_agents()
```

---

## Running Inference

### Script: `async_generate_with_tool_call_api.py`

| Parameter | Description |
|-----------|-------------|
| `--api_key` | API key for Google/OpenAI |
| `--dataset_path` | Path to parquet dataset file |
| `--dataset_name` | Dataset adapter name (see DATASET_SPECS) |
| `--output_dir` | Output directory for results |
| `--model_name_or_path` | Model name or path |
| `--model_type` | Provider: `Google`, `OpenAI`, `vLLM`, `SGLang` |
| `--use_tools` | Tool mode: `auto`, `force`, `direct` |
| `--api_base` | Base URL for OpenAI-compatible server |
| `--port` | Port for vLLM server |
| `--max_concurrent_requests` | Number of worker processes (default: 8) |
| `--sample_rate` | Sampling rate for dataset (default: 0.1) |
| `--n_trajectories` | Number of trajectories per task (default: 1) |

### Examples

**Google Gemini - With Tools**:

```bash
python -m geo_edit.scripts.async_generate_with_tool_call_api \
  --api_key "your-api-key" \
  --dataset_path /path/to/data.parquet \
  --dataset_name mathvisionqa \
  --output_dir output_gemini_tools \
  --model_name_or_path "gemini-3-flash-preview" \
  --model_type Google \
  --use_tools auto \
  --sample_rate 1.0
```

**OpenAI - No Tools Mode**:

```bash
python -m geo_edit.scripts.async_generate_with_tool_call_api \
  --api_key "your-api-key" \
  --dataset_path /path/to/data.parquet \
  --dataset_name cartomapqa_stmf_presence \
  --output_dir output_openai_direct \
  --model_name_or_path "o3-2025-04-16" \
  --model_type OpenAI \
  --use_tools direct \
  --sample_rate 1.0
```

**vLLM Local Server**:

```bash
python -m geo_edit.scripts.async_generate_with_tool_call_api \
  --model_type vLLM \
  --model_name_or_path /storage/openpsi/models/Qwen3-VL-32B-Thinking/ \
  --api_base http://127.0.0.1:8000 \
  --dataset_path /path/to/data.parquet \
  --dataset_name mathvisionqa \
  --output_dir output_vllm \
  --use_tools direct \
  --max_concurrent_requests 8
```

---

## Evaluation

### OpenAI as Judge

```bash
python -m geo_edit.evaluation.openai_as_judge \
  --api_key "your-api-key" \
  --result_path output_dir \
  --output_path eval_result
```

### STMF Counting Evaluation

```bash
python -m geo_edit.evaluation.eval_stmf_counting \
  --result_path output_dir \
  --output_path eval_result
```

### Shortest Path Evaluation

```bash
python -m geo_edit.evaluation.eval_shortest_path \
  --result_path output_dir \
  --output_path eval_result
```

---

## Dataset Registration

### Registered Datasets

- `mathvisionqa` - Mathematical visual QA
- `cartomapqa_srn` - Map navigation routes
- `cartomapqa_stmf_presence` - Map feature presence
- `cartomapqa_stmf_counting` - Map feature counting
- `cartomapqa_stmf_name_listing` - Map feature listing
- `shortest_path_text` - Shortest path (text weights)
- `shortest_path_image` - Shortest path (image weights)
- `shortest_path_image_text` - Shortest path (mixed)

### Adding New Datasets

```python
from geo_edit.datasets.task_registry import DatasetSpec, DATASET_SPECS

DATASET_SPECS["my_dataset"] = DatasetSpec(
    name="my_dataset",
    id_key="id",
    answer_key="answer",
    image_key="image",  # None for text-only
    prompt_template="Question: {question}",
    notool_prompt_template="Question: {question} (no tools)",
    template_fields={"question": "question_field"},
    task_kwargs_fields={"extra": lambda item: item.get("metadata", {})}
)
```

---

## Output Format

```text
output_dir/
├── {task_id}/
│   ├── meta_info.jsonl    # Task metadata and results
│   ├── input_image.png    # Original input image
│   └── step_*.png         # Tool output images (if any)
└── global_meta_info.jsonl # Aggregated statistics
```

---

## Testing

```bash
pytest geo_edit/tests/ -v
```

---

## License

[Add license information here]
