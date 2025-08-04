# vLLM Server Management

This directory contains a comprehensive vLLM server management script that provides easy deployment and management of vLLM inference servers with support for tensor, pipeline, and data parallelism.

## Features

- **Service Management**: Start, stop, restart, and check server status
- **Parallelism Support**: Full control over tensor, pipeline, and data parallelism
- **Tool Support**: Auto tool choice and configurable tool call parsers
- **GPU Detection**: Automatic detection of available GPUs
- **Health Monitoring**: Built-in health checks and status monitoring
- **Model Status**: Real-time information about currently loaded models
- **Logging**: Comprehensive logging and error reporting
- **Dependency Management**: Automatic environment setup and dependency installation

## Prerequisites

- **NVIDIA GPU(s)** with CUDA support
- **NVIDIA drivers** installed
- **uv** package manager (for dependency management)
- **Python 3.12** (recommended)

## Quick Start

### 1. Make the script executable
```bash
chmod +x vllm_start_server.sh
```

### 2. Start the server with default settings
```bash
./vllm_start_server.sh start
```

### 3. Check server status
```bash
./vllm_start_server.sh status
```

### 4. Stop the server
```bash
./vllm_start_server.sh stop
```

## Usage

### Basic Commands

```bash
# Start server with default model (Qwen/Qwen2.5-7B)
./vllm_start_server.sh start

# Start with custom model
./vllm_start_server.sh start meta-llama/Llama-2-7b-chat-hf

# Start with tool support (auto tool choice enabled, hermes parser)
./vllm_start_server.sh start Qwen/Qwen3-8B

# Start with custom tool settings
./vllm_start_server.sh start Qwen/Qwen3-8B 2 1 1 false json

# Check server status and GPU info
./vllm_start_server.sh status

# Show GPU information only
./vllm_start_server.sh gpu-info

# Stop server
./vllm_start_server.sh stop

# Restart server
./vllm_start_server.sh restart

# Install dependencies only
./vllm_start_server.sh install

# Show help
./vllm_start_server.sh help
```

### Parallelism Configuration

The script supports three types of parallelism:

1. **Tensor Parallelism**: Splits model layers across GPUs
2. **Pipeline Parallelism**: Splits model stages across GPUs  
3. **Data Parallelism**: Replicates model across GPUs for batch processing

```bash
# Usage: ./vllm_start_server.sh [COMMAND] [MODEL_NAME] [TENSOR_PARALLEL_SIZE] [PIPELINE_PARALLEL_SIZE] [DATA_PARALLEL_SIZE]

# Example: 2-way tensor parallelism, 1-way pipeline, 2-way data parallelism
./vllm_start_server.sh start Qwen/Qwen2.5-7B 2 1 2

# Example: 4-way tensor parallelism, 2-way pipeline, 1-way data parallelism
./vllm_start_server.sh start meta-llama/Llama-2-7b-chat-hf 4 2 1

# Restart with different parallelism settings
./vllm_start_server.sh restart Qwen/Qwen2.5-7B 1 1 4
```

### Tool Configuration

The script supports vLLM's tool calling features with configurable auto tool choice and tool call parsers:

```bash
# Usage: ./vllm_start_server.sh [COMMAND] [MODEL_NAME] [TENSOR_PARALLEL_SIZE] [PIPELINE_PARALLEL_SIZE] [DATA_PARALLEL_SIZE] [ENABLE_AUTO_TOOL_CHOICE] [TOOL_CALL_PARSER]

# Start with default tool settings (auto tool choice enabled, hermes parser)
./vllm_start_server.sh start Qwen/Qwen3-8B

# Start with auto tool choice disabled
./vllm_start_server.sh start Qwen/Qwen3-8B 2 1 1 false

# Start with custom tool parser
./vllm_start_server.sh start Qwen/Qwen3-8B 2 1 1 true json

# Start with both custom settings
./vllm_start_server.sh start Qwen/Qwen3-8B 2 1 1 false json

# Restart with tool settings
./vllm_start_server.sh restart Qwen/Qwen3-8B 2 1 1 true hermes
```

#### Tool Settings

- **Auto Tool Choice**: When enabled, the model can automatically choose which tools to use based on the conversation context
- **Tool Call Parser**: Determines how the model's tool calls are parsed and executed

#### Supported Tool Call Parsers

- **hermes** (default): Optimized parser for Hermes-style tool calling
- **json**: Standard JSON-based tool calling
- **regex**: Regular expression-based parsing
- **auto**: Automatic parser selection based on model capabilities

### Default Settings

- **Default Model**: `Qwen/Qwen2.5-7B`
- **Default Tensor Parallel Size**: Number of available GPUs
- **Default Pipeline Parallel Size**: 1
- **Default Data Parallel Size**: 1
- **Default Auto Tool Choice**: `true`
- **Default Tool Call Parser**: `hermes`
- **Host**: 127.0.0.1
- **Port**: 8000

## GPU Requirements

The script automatically validates that your parallelism configuration doesn't exceed available GPUs:

```
Total GPUs Required = Tensor Parallel Size × Pipeline Parallel Size × Data Parallel Size
```

### Examples

| Configuration | Tensor | Pipeline | Data | Total GPUs | Use Case |
|---------------|--------|----------|------|------------|----------|
| Single GPU | 1 | 1 | 1 | 1 | Development, small models |
| Tensor Parallel | 4 | 1 | 1 | 4 | Large models, single requests |
| Pipeline Parallel | 1 | 2 | 1 | 2 | Very large models |
| Data Parallel | 1 | 1 | 4 | 4 | High throughput, batch processing |
| Mixed | 2 | 2 | 2 | 8 | Large-scale deployment |

## Server Information

### Health Endpoint
Once started, the server provides a health check endpoint:
```bash
curl http://127.0.0.1:8000/health
```

### API Endpoints
The vLLM server exposes standard vLLM API endpoints:
- `/v1/completions` - Text completion
- `/v1/chat/completions` - Chat completion (with tool calling support)
- `/v1/models` - List currently loaded models
- `/health` - Health check

### Logs
Server logs are written to `vllm_server.log` in the same directory.

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Stop the existing process or use a different port
   ```

2. **GPU memory insufficient**
   ```bash
   # Check GPU memory
   ./vllm_start_server.sh gpu-info
   
   # Reduce tensor parallel size or use a smaller model
   ```

3. **CUDA not available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Install CUDA if needed
   ```

4. **Dependencies missing**
   ```bash
   # Install dependencies
   ./vllm_start_server.sh install
   ```

5. **Model status not showing**
   ```bash
   # Check if jq is installed for JSON parsing
   which jq
   
   # Install jq if needed (optional - script has fallback)
   sudo apt-get install jq  # Ubuntu/Debian
   brew install jq          # macOS
   ```

6. **Tool calling not working**
   ```bash
   # Check if auto tool choice is enabled
   ./vllm_start_server.sh status
   
   # Restart with explicit tool settings
   ./vllm_start_server.sh restart Qwen/Qwen3-8B 2 1 1 true hermes
   
   # Verify tool parser compatibility with your model
   # Some models work better with specific parsers
   ```

### Validation Errors

The script validates your configuration and will show helpful error messages:

```
[ERROR] Total parallel size (8) exceeds available GPUs (4)
[ERROR] Tensor parallel: 2, Pipeline parallel: 2, Data parallel: 2
```

## Advanced Configuration

### Environment Variables
You can modify the script to change default settings:
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)
- `DEFAULT_MODEL`: Default model to load
- `DEFAULT_TENSOR_PARALLEL_SIZE`: Default tensor parallelism
- `DEFAULT_PIPELINE_PARALLEL_SIZE`: Default pipeline parallelism
- `DEFAULT_DATA_PARALLEL_SIZE`: Default data parallelism
- `DEFAULT_ENABLE_AUTO_TOOL_CHOICE`: Default auto tool choice setting (default: true)
- `DEFAULT_TOOL_CALL_PARSER`: Default tool call parser (default: hermes)

### Custom Models
The script supports any model that vLLM supports:
- Hugging Face models: `Qwen/Qwen2.5-7B`
- Local models: `/path/to/local/model`
- Custom models with specific configurations

### Tool Calling Use Cases

Tool calling is particularly useful for:

1. **Function Calling**: Models can call predefined functions based on user requests
2. **API Integration**: Connect to external APIs and services
3. **Data Processing**: Perform calculations, data transformations, or analysis
4. **Multi-step Reasoning**: Break down complex tasks into tool-assisted steps

#### Example Tool Calling Workflow

```bash
# Start server with tool support
./vllm_start_server.sh start Qwen/Qwen3-8B 2 1 1 true hermes

# The server will now support tool calling in chat completions
# Example API call with tools:
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is the weather like in New York?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather information for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

## Performance Tips

1. **Tensor Parallelism**: Best for large models that don't fit in single GPU memory
2. **Pipeline Parallelism**: Best for very large models with many layers
3. **Data Parallelism**: Best for high-throughput scenarios with multiple requests
4. **Mixed Strategies**: Combine different parallelism types for optimal performance

## Monitoring

Use the status command to monitor your server:
```bash
./vllm_start_server.sh status
```

This will show:
- Server process status
- GPU utilization and memory information
- Currently loaded models
- Tool configuration (auto tool choice and parser settings)
- Recent logs
- Health check results

### Model Status Example

When you run the status command, you'll see output like:
```bash
[INFO] Checking running models...
[INFO] Running models:
[INFO]   - Qwen/Qwen2.5-7B-instruct
[INFO]   - meta-llama/Llama-2-7b-chat-hf
```

This feature automatically queries the `/v1/models` endpoint to show which models are currently loaded and available for inference.

## Testing

### Test Script

A comprehensive test script `test_vllm.py` is provided to verify that your vLLM server is working correctly with litellm integration.

#### Usage

```bash
# Test with default model
python test_vllm.py

# Test with specific model
python test_vllm.py Qwen/Qwen2.5-14B

# Enable debug mode with default model
python test_vllm.py --debug

# Enable debug mode with specific model
python test_vllm.py --debug Qwen/Qwen2.5-14B

# Show help
python test_vllm.py --help
```

#### Test Coverage

The test script performs the following checks:

1. **Health Check**: Verifies the server is running and responding
2. **Basic Completion**: Tests the `/v1/completions` endpoint
3. **Chat Completion**: Tests the `/v1/chat/completions` endpoint with system/user messages
4. **Streaming Completion**: Tests streaming responses from the server
5. **Tool Calling**: Tests tool calling functionality (when enabled)

#### Debug Mode

When using the `--debug` flag, the script:
- Enables litellm debug logging
- Shows detailed request/response information
- Helps troubleshoot connection or configuration issues

#### Example Output

```bash
╭─ vLLM Server Test Suite ─╮
│ Testing model: Qwen/Qwen2.5-7B at http://127.0.0.1:8000 │
╰─────────────────────────────────────────────────────────╯

=== Test 1: Health Check ===
✅ Server is healthy

=== Test 2: Basic Completion ===
✅ Completion test successful
╭─ Response ─╮
│ Hello! I'm doing well, thank you for asking. How are you today? │
╰────────────╯

=== Test 3: Chat Completion ===
✅ Chat completion test successful
╭─ Response ─╮
│ The capital of France is Paris. │
╰────────────╯

=== Test 4: Streaming Completion ===
✅ Streaming completion test successful

╭─ Test Summary ─╮
│ 🎉 All tests completed! │
╰─────────────────╯
```

### Bash Test Script

For quick testing without Python dependencies, a bash script `test_vllm.sh` is also provided.

#### Usage

```bash
# Test with default model
./test_vllm.sh

# Test with specific model
./test_vllm.sh Qwen/Qwen2.5-14B

# Show help
./test_vllm.sh --help
```

#### Test Coverage

The bash script performs:
1. **Health Check**: Verifies the server is running and responding
2. **Completion Test**: Tests the `/v1/completions` endpoint with a simple prompt

#### Example Output

```bash
[HEADER] vLLM Server Test Suite
[INFO] Testing model: Qwen/Qwen2.5-7B
[INFO] Server endpoint: http://127.0.0.1:8000

[INFO] === Test 1: Health Check ===
[INFO] Testing health endpoint...
[INFO] ✅ Server is healthy

[INFO] === Test 2: Completion Endpoint ===
[INFO] Testing completion endpoint...
[INFO] Sending request with prompt: "Hello, how are you today?"
[INFO] ✅ Request successful!

[INFO] Response:
----------------------------------------
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-7B",
  "choices": [
    {
      "text": "Hello! I'm doing well, thank you for asking. How are you today?",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 15,
    "total_tokens": 23
  }
}
----------------------------------------

[INFO] 🎉 Test completed successfully!
[INFO] Model 'Qwen/Qwen2.5-7B' is working correctly with the vLLM server.
```

### Choosing Between Test Scripts

- **Python Script (`test_vllm.py`)**: More comprehensive testing with litellm integration, streaming support, and debug capabilities
- **Bash Script (`test_vllm.sh`)**: Lightweight testing with minimal dependencies, good for quick health checks

## Files Created

The script creates several files in the same directory:
- `.vllm_server.pid` - Process ID file
- `.vllm_server.log` - Server logs
- `.venv/` - Virtual environment directory

## Contributing

To extend the script:
1. Add new commands to the case statement
2. Implement corresponding functions
3. Update the help text
4. Test with different GPU configurations 