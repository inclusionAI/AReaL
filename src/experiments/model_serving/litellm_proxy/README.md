# LiteLLM Proxy Setup

This directory contains a LiteLLM proxy configuration and management scripts for serving AI models through a unified API interface.

## Overview

LiteLLM is a library that provides a unified interface to various LLM providers. This setup includes:
- A management script (`litellm_proxy.sh`) for easy proxy control
- Configuration file (`litellm_config.yaml`) for model definitions
- Test script (`test_query.sh`) for validating the setup

## Files

- **`litellm_proxy.sh`** - Main management script for the LiteLLM proxy
- **`litellm_config.yaml`** - Configuration file defining available models
- **`test_query.sh`** - Test script to validate proxy functionality
- **`test_litellm_sdk.py`** - Python test script using OpenAI SDK to call LiteLLM proxy

## Prerequisites

- Python 3.7+
- pip package manager
- Bash shell
- API keys (stored in `.env` file)

## Quick Start

1. **Setup environment:**
   ```bash
   # Copy the environment template
   cp env_example.txt .env
   
   # Edit .env and add your API keys
   nano .env
   ```

2. **Start the proxy:**
   ```bash
   ./litellm_proxy.sh start
   ```

3. **Test the connection:**
   ```bash
   ./test_query.sh
   ```

4. **Stop the proxy:**
   ```bash
   ./litellm_proxy.sh stop
   ```

## Using `litellm_proxy.sh`

The `litellm_proxy.sh` script provides a comprehensive interface for managing the LiteLLM proxy server.

### Available Commands

| Command | Description |
|---------|-------------|
| `start` | Start the LiteLLM proxy (installs LiteLLM if needed) |
| `stop` | Stop the running proxy |
| `restart` | Restart the proxy |
| `models` | List available models |
| `status` | Show current proxy status |
| `help` | Display help information |

### Examples

```bash
# Start the proxy
./litellm_proxy.sh start

# Check if proxy is running
./litellm_proxy.sh status

# List available models
./litellm_proxy.sh models

# Restart the proxy
./litellm_proxy.sh restart

# Stop the proxy
./litellm_proxy.sh stop
```

### Features

- **Automatic Installation**: Automatically installs LiteLLM with proxy support if not present
- **Process Management**: Handles PID tracking and graceful shutdown
- **Logging**: Redirects output to `/tmp/litellm_proxy.log`
- **Configuration**: Uses the local `litellm_config.yaml` file
- **Port Management**: Runs on port 4000 by default

## Environment Setup

### API Keys Configuration

The scripts use `python-dotenv` to load environment variables from a `.env` file. This keeps your API keys secure and separate from your code.

1. **Copy the environment template:**
   ```bash
   cp env_example.txt .env
   ```

2. **Edit the `.env` file and add your API keys:**
   ```bash
   # Required for GPT-4o
   OPENAI_API_KEY=your-actual-openai-api-key-here
   
   # Optional: Other API keys
   ANTHROPIC_API_KEY=your-anthropic-key-here
   GOOGLE_API_KEY=your-google-key-here
   ```

3. **Keep your `.env` file secure:**
   - Never commit it to version control
   - Add `.env` to your `.gitignore` file
   - Use different keys for development and production

## Configuration

The `litellm_config.yaml` file defines the models available through the proxy:

```yaml
model_list:
  - model_name: qwen2.5-7b
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-7b
      api_base: http://localhost:8000/v1
```

### Adding New Models

To add additional models, edit `litellm_config.yaml` and add new entries to the `model_list`:

```yaml
model_list:
  - model_name: qwen2.5-7b
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-7b
      api_base: http://localhost:8000/v1
  - model_name: your-new-model
    litellm_params:
      model: your-model-provider/model-name
      api_base: http://your-api-endpoint/v1
```

After modifying the configuration, restart the proxy:
```bash
./litellm_proxy.sh restart
```

## Testing

### Basic Testing

Use the provided test script to validate your setup:

```bash
./test_query.sh
```

This script sends a test request to the proxy and should return a response from the configured model.

### Python Testing for GPT-4o

For comprehensive testing of the OpenAI GPT-4o model, use the Python test script:

```bash
# Make sure you have the required libraries installed
pip install requests python-dotenv

# Run the comprehensive test suite (loads keys from .env)
python3 test_gpt4o.py
```

The Python test script includes:
- Environment validation (API key check)
- Proxy status verification
- Multiple test scenarios:
  - Basic conversation
  - Code generation
  - Multi-turn conversation
  - Creative writing
- Detailed error reporting and success indicators

### OpenAI SDK Testing

For testing using the OpenAI Python SDK to call the LiteLLM proxy:

```bash
# Install required dependencies
pip install openai loguru python-dotenv

# Run the OpenAI SDK test suite (loads keys from .env)
python3 test_litellm_sdk.py
```

The OpenAI SDK test script includes:
- OpenAI SDK usage to call the proxy's OpenAI-compatible API
- Tests for both GPT-4o and local Qwen models
- Model listing functionality
- Verbose logging for debugging
- Tests the proxy's OpenAI-compatible interface

### Manual Testing

You can also test manually using curl:

```bash
curl --location 'http://localhost:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "qwen2.5-7b",
    "messages": [
        {
        "role": "user",
        "content": "Hello, how are you?"
        }
    ]
}'
```

## Troubleshooting

### Common Issues

1. **Port already in use**: The proxy runs on port 4000 by default. If this port is occupied, modify the `LITELLM_PORT` variable in `litellm_proxy.sh`.

2. **Model not responding**: Ensure your vLLM server is running on `http://localhost:8000` or update the `api_base` in the configuration.

3. **Permission denied**: Make sure the script is executable:
   ```bash
   chmod +x litellm_proxy.sh
   chmod +x test_query.sh
   ```

4. **API key not found**: Make sure your `.env` file exists and contains the required API keys:
   ```bash
   # Check if .env file exists
   ls -la .env
   
   # Verify API key is loaded
   python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
   ```

5. **Proxy timeout**: If you encounter timeout errors, the scripts now use 15-second timeouts. If issues persist:
   ```bash
   # Check proxy status
   ./litellm_proxy.sh status
   
   # Restart proxy if needed
   ./litellm_proxy.sh restart
   ```

### Logs

Check the proxy logs for detailed information:
```bash
tail -f /tmp/litellm_proxy.log
```

### Status Check

Verify the proxy is running:
```bash
./litellm_proxy.sh status
```

## API Endpoints

Once running, the proxy provides these endpoints:

- **Chat Completions**: `POST http://localhost:4000/chat/completions`
- **Models List**: `GET http://localhost:4000/models`
- **Health Check**: `GET http://localhost:4000/health`

## Integration

The proxy can be integrated with applications that expect OpenAI-compatible APIs. Simply point your application to `http://localhost:4000` instead of the OpenAI API endpoint.

## Security Notes

- The proxy runs on `0.0.0.0:4000` by default (accessible from any IP)
- For production use, consider adding authentication and restricting access
- The current setup is intended for development/testing purposes 