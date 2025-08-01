# vLLM Server Management

This directory contains a comprehensive vLLM server management script that provides easy deployment and management of vLLM inference servers with support for tensor, pipeline, and data parallelism.

## Features

- **Service Management**: Start, stop, restart, and check server status
- **Parallelism Support**: Full control over tensor, pipeline, and data parallelism
- **GPU Detection**: Automatic detection of available GPUs
- **Health Monitoring**: Built-in health checks and status monitoring
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

### Default Settings

- **Default Model**: `Qwen/Qwen2.5-7B`
- **Default Tensor Parallel Size**: Number of available GPUs
- **Default Pipeline Parallel Size**: 1
- **Default Data Parallel Size**: 1
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
- `/v1/chat/completions` - Chat completion
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

### Custom Models
The script supports any model that vLLM supports:
- Hugging Face models: `Qwen/Qwen2.5-7B`
- Local models: `/path/to/local/model`
- Custom models with specific configurations

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
- GPU utilization
- Recent logs
- Health check results

## Files Created

The script creates several files in the same directory:
- `vllm_server.pid` - Process ID file
- `vllm_server.log` - Server logs
- `.venv/` - Virtual environment directory

## Contributing

To extend the script:
1. Add new commands to the case statement
2. Implement corresponding functions
3. Update the help text
4. Test with different GPU configurations 