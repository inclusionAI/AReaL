# Ollama Cluster with Load Balancing

A production-ready setup for running multiple Ollama instances with nginx load balancing to distribute AI model inference across multiple GPUs.

## 🚀 Overview

This system provides:
- **Drop-in Replacement**: Load balancer runs on port 11434 (standard Ollama port)
- **Multi-GPU Support**: Run separate Ollama instances on different GPUs
- **Load Balancing**: Nginx distributes requests across instances using round-robin
- **High Availability**: If one instance fails, others continue serving
- **Easy Management**: Simple scripts to start, stop, and monitor the cluster
- **Configurable Context Length**: Set custom context lengths for all instances

> **Note**: The load balancer runs on port 11434, making it a seamless replacement for single-instance Ollama. Tools like LiteLLM, Continue.dev, and others will work without configuration changes!

## 📁 Scripts

### `setup_ollama_cluster.sh`
The main orchestration script that manages the entire cluster.

```bash
./setup_ollama_cluster.sh {start|stop|restart|status|test} [port1 port2 ...]
```

**Commands:**
- `start` - Start all Ollama instances and nginx load balancer
- `stop` - Stop all components
- `restart` - Restart the entire cluster
- `status` - Check cluster health
- `test` - Run diagnostic tests

**Examples:**
```bash
# Start with default ports (11435, 11436, 11437, 11438) and default context length (32768)
./setup_ollama_cluster.sh start

# Start with custom ports
./setup_ollama_cluster.sh start 11435 11436

# Start with custom context length
OLLAMA_CONTEXT_LENGTH=16384 ./setup_ollama_cluster.sh start

# Check status (shows context length configuration)
./setup_ollama_cluster.sh status
```

### `manage_ollama_instance.sh`
Manages individual Ollama instances on different ports/GPUs.

```bash
./manage_ollama_instance.sh {start|stop|status} [port1 port2 ...]
```

**Features:**
- Automatically assigns GPU devices based on port order
- Sets `OLLAMA_ORIGINS="*"` for nginx compatibility
- Configurable context length via environment variable
- Manages process lifecycle

**GPU Assignment:**
- Port 1 → GPU 0
- Port 2 → GPU 1
- Port 3 → GPU 2
- etc.

**Examples:**
```bash
# Start with default context length (32768)
./manage_ollama_instance.sh start

# Start with custom context length
OLLAMA_CONTEXT_LENGTH=65536 ./manage_ollama_instance.sh start

# Check status (shows context length for each instance)
./manage_ollama_instance.sh status
```

### `manage_nginx_lb.sh`
Configures and manages the nginx load balancer.

```bash
./manage_nginx_lb.sh {start|stop|restart|status|config} [port1 port2 ...]
```

**Commands:**
- `start` - Generate config and start nginx
- `stop` - Stop load balancer
- `restart` - Restart with new configuration
- `status` - Check nginx and backend status
- `config` - Show generated nginx configuration

**Load Balancer Details:**
- Listens on port 11434 (standard Ollama port)
- Round-robin distribution
- WebSocket support for streaming
- Custom timeouts for long-running inference

### `test_ollama_generate.sh`
Quick test script to verify cluster functionality.

```bash
./test_ollama_generate.sh [MODEL] [PROMPT]
```

**Features:**
- Health checks
- Model availability verification
- Performance metrics
- Colored output for easy reading

**Examples:**
```bash
# Test with defaults
./test_ollama_generate.sh

# Test specific model
./test_ollama_generate.sh qwen2.5:7b

# Custom prompt
./test_ollama_generate.sh qwen2.5:1.5b "Write a haiku"
```

## 🛠️ Prerequisites

1. **Ollama** must be installed:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Nginx** (automatically installed if missing)

3. **Multiple GPUs** (recommended for multi-instance setup)

4. **Models** pulled in Ollama:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull qwen2.5:7b
   ```

## 🚦 Quick Start

1. **Start the cluster:**
   ```bash
   ./setup_ollama_cluster.sh start
   ```

2. **Verify it's working:**
   ```bash
   ./test_ollama_generate.sh
   ```

3. **Use the load balancer:**
   ```bash
   # API endpoint (using standard Ollama port)
   curl http://localhost:11434/api/tags
   
   # Generate request
   curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model": "qwen2.5:1.5b", "prompt": "Hello"}'
   ```

## ⚙️ Configuration

### Environment Variables

#### `OLLAMA_CONTEXT_LENGTH`
Sets the context length for all Ollama instances in the cluster.

- **Default**: `32768` (32k tokens)
- **Usage**: Set before running start commands
- **Examples**:
  ```bash
  # Use 16k context length
  OLLAMA_CONTEXT_LENGTH=16384 ./setup_ollama_cluster.sh start
  
  # Use 64k context length
  OLLAMA_CONTEXT_LENGTH=65536 ./manage_ollama_instance.sh start
  
  # Use 128k context length
  OLLAMA_CONTEXT_LENGTH=131072 ./setup_ollama_cluster.sh start
  ```

**Context Length Guidelines:**
- **16k (16384)**: Good for most conversational tasks, lower memory usage
- **32k (32768)**: Default, balanced performance and memory
- **64k (65536)**: For longer conversations, higher memory usage
- **128k+ (131072+)**: For very long contexts, requires significant GPU memory

### Default Ports
- Load Balancer: `11434` (standard Ollama port)
- Ollama Instances: `11435`, `11436`, `11437`, `11438`

### Nginx Configuration
The load balancer configuration includes:
- WebSocket support for streaming responses
- 5-minute timeouts for long inference
- Health check endpoints
- Proper Host header handling for Ollama

### Instance Environment Variables
Each Ollama instance runs with:
- `OLLAMA_HOST=127.0.0.1:PORT`
- `OLLAMA_ORIGINS=*`
- `OLLAMA_CONTEXT_LENGTH=<value>` (configurable)
- `CUDA_VISIBLE_DEVICES=GPU_INDEX`

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Client Applications           │
└─────────────────┬───────────────────────┘
                  │
                  ▼ Port 11434 (Standard Ollama Port)
         ┌────────────────┐
         │  Nginx Load    │
         │   Balancer     │
         └───┬──┬──┬──┬───┘
             │  │  │  │
    ┌────────┴──┼──┼──┴────────┐
    │           │  │           │
    ▼           ▼  ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Ollama   │ │ Ollama   │ │ Ollama   │ │ Ollama   │
│ :11435   │ │ :11436   │ │ :11437   │ │ :11438   │
│ (GPU 0)  │ │ (GPU 1)  │ │ (GPU 2)  │ │ (GPU 3)  │
│ Context: │ │ Context: │ │ Context: │ │ Context: │
│ 32k      │ │ 32k      │ │ 32k      │ │ 32k      │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## 🔧 Troubleshooting

### 403 Forbidden Error
If you get a 403 error through the load balancer:
1. Check that Ollama instances were started with the scripts (not manually)
2. Verify `OLLAMA_ORIGINS="*"` is set
3. Restart the cluster: `./setup_ollama_cluster.sh restart`

### Model Not Found
```bash
# List available models
curl http://localhost:11434/api/tags

# Pull missing model
ollama pull model_name
```

### Port Already in Use
```bash
# Check what's using a port
sudo lsof -i :11434

# Force stop all components
./setup_ollama_cluster.sh stop
pkill -f "ollama serve"
sudo pkill -f nginx
```

### Context Length Issues
```bash
# Check current context length configuration
./manage_ollama_instance.sh status

# Restart with different context length
OLLAMA_CONTEXT_LENGTH=16384 ./setup_ollama_cluster.sh restart

# Verify context length is applied
./setup_ollama_cluster.sh status
```

### Checking Logs
```bash
# Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Check Ollama instance status (includes context length)
./manage_ollama_instance.sh status
```

## 🔌 Integration Examples

### With LiteLLM
```python
import litellm

# No need to specify api_base - uses default Ollama port!
response = litellm.completion(
    model="ollama/qwen2.5:1.5b",
    messages=[{"role": "user", "content": "Hello"}]
)

# Or explicitly specify if needed
response = litellm.completion(
    model="ollama/qwen2.5:1.5b",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:11434"  # Standard Ollama port
)
```

### With curl
```bash
# List models
curl http://localhost:11434/api/tags

# Generate (streaming)
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:1.5b",
    "prompt": "Why is the sky blue?",
    "stream": true
  }'

# Chat completion
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:1.5b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### With Python requests
```python
import requests

# Non-streaming
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:1.5b",
        "prompt": "Hello world",
        "stream": False
    }
)
print(response.json()["response"])
```

## 📊 Performance Tips

1. **Model Loading**: First request to each instance loads the model into GPU memory
2. **Warm-up**: Run a few test requests to warm up all instances
3. **Monitor GPU Usage**: Use `nvidia-smi` to check GPU utilization
4. **Scaling**: Add more instances/ports based on your GPU availability
5. **Context Length**: Choose appropriate context length based on your use case:
   - Lower context lengths use less GPU memory
   - Higher context lengths enable longer conversations
   - Monitor memory usage with `nvidia-smi` when changing context length

## 🤝 Contributing

Feel free to submit issues or pull requests to improve this setup!

## 📄 License

This project is provided as-is for educational and production use. 