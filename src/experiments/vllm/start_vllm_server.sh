#!/bin/bash

# Default model if not specified
DEFAULT_MODEL="Qwen/Qwen-7B"
MODEL=${1:-$DEFAULT_MODEL}

# Check if VLLM is already running
if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "VLLM server is already running!"
    echo "Available models:"
    curl -s http://localhost:8000/v1/models | jq -r '.data[].id' 2>/dev/null || echo "Could not retrieve model list"
    exit 0
fi

# Check if VLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ VLLM is not installed!"
    echo "Please install VLLM first:"
    echo "  pip install vllm"
    exit 1
fi

# Start VLLM server in the background
echo "Starting VLLM server with model: $MODEL"
echo "This may take a few minutes to download and load the model..."

vllm serve $MODEL --host localhost --port 8000 &
VLLM_PID=$!

# Wait for the server to be ready
echo "Waiting for VLLM server to start..."
attempts=0
until curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
    sleep 2
    attempts=$((attempts + 1))
    if [ $attempts -gt 90 ]; then
        echo "❌ Failed to start VLLM server after 3 minutes"
        echo "The model may be too large or there might be insufficient GPU memory"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "Attempt $attempts/90..."
done

echo "✅ VLLM server is ready!"
echo "Server PID: $VLLM_PID"
echo "API endpoint: http://localhost:8000"
echo ""
echo "Available models:"
curl -s http://localhost:8000/v1/models | jq -r '.data[].id' 2>/dev/null || echo "Could not retrieve model list"
echo ""
echo "To stop the server later, run:"
echo "  kill $VLLM_PID"
echo ""
echo "To test the server, run:"
echo "  curl http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'" 