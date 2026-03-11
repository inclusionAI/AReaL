# register-tool-agent

注册新的工具 Agent 到 geo_edit 项目中。

## 使用方法

```
/register-tool-agent <agent_name> [model_source]
```

例如：
```
/register-tool-agent chartr1 DocTron/Chart-R1
```

## 需要修改的文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `geo_edit/tool_definitions/agents/<name>.py` | 新建 | Agent 实现 |
| `geo_edit/tool_definitions/agents/__init__.py` | 修改 | 注册 Agent |
| `geo_edit/tool_definitions/router.py` | 修改 | 更新 TOOL_CATEGORIES |
| `geo_edit/tool_definitions/config.yaml` | 修改 | 添加工具开关 |
| `geo_edit/evaluation/trajectory_judge.py` | 修改 | 更新 KNOWN_TOOL_NAMES |
| `geo_edit/scripts/run_*_batch.sh` | 修改 | 更新测试脚本（如适用） |

## 执行步骤

### 1. 调研模型

获取模型信息：
- HuggingFace 仓库地址
- 模型加载方式（Transformers / vLLM）
- 输入输出格式
- 推荐的推理参数（temperature, max_tokens 等）
- 特殊的 prompt 格式或标签（如 `<think>`, `<answer>`）

### 2. 创建 Agent 文件

文件: `geo_edit/tool_definitions/agents/<name>.py`

参考模板：
- vLLM 模式: `ovr.py`
- Transformers 模式: `chartmoe.py`

必须导出的内容：

```python
# 系统提示
SYSTEM_PROMPT = "..."

# 模型配置
agent_config = {
    "model_name_or_path": "/storage/openpsi/models/<ModelName>",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "temperature": 0.0,
    "max_tokens": 4096,
    "num_gpus": 1,
}

# Actor 类
class <Name>Actor(BaseToolModelActor):
    def __init__(self, model_name, max_model_len, gpu_memory_utilization, system_prompt):
        ...

    def analyze(self, image_b64, temperature, max_tokens, **kwargs) -> str:
        ...

    def _parse_output(self, text: str) -> str:
        ...

    def health_check(self) -> dict:
        ...

ACTOR_CLASS = <Name>Actor

# 单工具声明（向后兼容）
DECLARATION = {
    "name": "<name>",
    "description": "...",
    "parameters": {...},
}

RETURN_TYPE = "text"

# 多工具声明（细粒度工具）
DECLARATIONS = {
    "<tool_name_1>": {
        "name": "<tool_name_1>",
        "description": "...",
        "parameters": {...},
        "fixed_prompt": "...",  # 可选，固定 prompt
        "return_type": "text",
    },
    ...
}
```

### 3. 注册 Agent

文件: `geo_edit/tool_definitions/agents/__init__.py`

```python
# 1. 添加导入
from geo_edit.tool_definitions.agents import <name>

# 2. 添加到各字典
AGENT_DECLARATIONS["<name>"] = <name>.DECLARATION
AGENT_RETURN_TYPES["<name>"] = <name>.RETURN_TYPE
AGENT_CONFIGS["<name>"] = <name>.agent_config
AGENT_SYSTEM_PROMPTS["<name>"] = <name>.SYSTEM_PROMPT
AGENT_ACTOR_CLASSES["<name>"] = <name>.ACTOR_CLASS

# 3. 注册多工具声明（如果有 DECLARATIONS）
if hasattr(<name>, 'DECLARATIONS'):
    for tool_name, decl in <name>.DECLARATIONS.items():
        MULTI_TOOL_DECLARATIONS[tool_name] = {
            "declaration": decl,
            "base_agent": "<name>",
            "actor_class": <name>.ACTOR_CLASS,
            "agent_config": <name>.agent_config,
            "system_prompt": <name>.SYSTEM_PROMPT,
        }
```

### 4. 更新 Router

文件: `geo_edit/tool_definitions/router.py`

在 `TOOL_CATEGORIES` 中添加新工具：

```python
TOOL_CATEGORIES = {
    ...
    "<category>": [..., "<new_tool_1>", "<new_tool_2>"],
    ...
}
```

### 5. 更新配置

文件: `geo_edit/tool_definitions/config.yaml`

```yaml
# =============================================================================
# <CATEGORY> TOOLS
# =============================================================================
<new_tool_1>: false   # [<AgentName>] 工具描述
<new_tool_2>: false   # [<AgentName>] 工具描述
```

### 6. 更新评估工具列表

文件: `geo_edit/evaluation/trajectory_judge.py`

在 `KNOWN_TOOL_NAMES` 列表中添加新工具：

```python
KNOWN_TOOL_NAMES = [
    ...
    # <Category> tools (<AgentName>)
    "<new_tool_1>", "<new_tool_2>",
    ...
]
```

### 7. 更新测试脚本（如适用）

文件: `geo_edit/scripts/run_*_batch.sh`

更新实验配置中的工具名称。

### 8. 验证注册

```bash
# 验证语法
python -c "import ast; ast.parse(open('geo_edit/tool_definitions/agents/<name>.py').read()); print('OK')"

# 验证注册（需要完整环境）
python -c "from geo_edit.tool_definitions.agents import AGENT_CONFIGS; print('<name>' in AGENT_CONFIGS)"
```

## 工具声明设计指南

### Description 风格

与其他工具保持一致，使用 "for downstream reasoning" 结尾：

```python
"description": "<Tool purpose>. <What it does>. <Output format> for downstream reasoning."
```

示例：
- "Chart reasoning tool with chain-of-thought capabilities. Analyzes charts to extract insights, understand data relationships, and identify key observations for downstream reasoning."
- "Chart data extraction tool. Extracts data points, values, axis information, legend items, and structure from charts into structured format for downstream reasoning."

### 固定 Prompt 设计

对于有固定任务的工具，使用 `fixed_prompt` 字段：

```python
"<tool_name>": {
    ...
    "fixed_prompt": "Extract data from this chart. Include chart type, data points with values, axis information, and legend items.",
    ...
}
```

### 输出解析

如果模型有特殊输出格式（如 `<think>/<answer>` 标签），在 `_parse_output()` 中处理：

```python
def _parse_output(self, text: str) -> str:
    # 提取 <answer> 内容
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return text.strip()
```

## 常见问题

### vLLM vs Transformers

- **vLLM**: 高吞吐推理，适合大规模部署（参考 `ovr.py`）
- **Transformers**: 更灵活，支持特殊模型结构（参考 `chartmoe.py`）

### 多工具 vs 单工具

- **单工具**: 用户需要传递所有参数（`DECLARATION`）
- **多工具**: 不同工具有固定 prompt，简化使用（`DECLARATIONS`）

### 废弃旧工具

如果新 Agent 替换旧 Agent：
1. 在 `__init__.py` 注释掉旧 Agent 的注册
2. 在 `config.yaml` 注释掉旧工具并标记 `[Deprecated]`
3. 保留旧文件但不再使用

## 关键文件

- `geo_edit/tool_definitions/agents/*.py` - Agent 实现
- `geo_edit/tool_definitions/agents/__init__.py` - Agent 注册
- `geo_edit/tool_definitions/router.py` - 工具路由和分类
- `geo_edit/tool_definitions/config.yaml` - 工具开关配置
- `geo_edit/evaluation/trajectory_judge.py` - 评估工具列表
- `geo_edit/environment/tool_agents/actor.py` - BaseToolModelActor 基类
