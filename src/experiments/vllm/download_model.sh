#!/bin/bash

# Default model if not specified
DEFAULT_MODEL="qwen2.5-1.5b"
MODEL=${1:-$DEFAULT_MODEL}

echo "Setting up VLLM with model: $MODEL"

# Check if VLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "❌ VLLM is not installed!"
    echo "Please install VLLM first:"
    echo "  pip install vllm"
    exit 1
fi

# Check if we have GPU support
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "⚠️  Warning: CUDA not available. VLLM will run on CPU (much slower)"
fi

# Start VLLM server to download and test the model
echo "Starting VLLM server to download and test model..."
echo "This may take several minutes for the first run..."

# Start server in background
vllm serve $MODEL --host localhost --port 8000 &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for VLLM server to start and model to load..."
attempts=0
until curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
    sleep 5
    attempts=$((attempts + 1))
    if [ $attempts -gt 72 ]; then
        echo "❌ Failed to start VLLM server after 6 minutes"
        echo "The model may be too large or there might be insufficient memory"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    echo "Attempt $attempts/72..."
done

echo "✅ VLLM server is ready!"

# Test that the model is working
echo ""
echo "Testing model..."
TEST_RESPONSE=$(curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello, please respond with: I am working!\"}],
    \"max_tokens\": 50,
    \"temperature\": 0.1
  }" | jq -r '.choices[0].message.content' 2>/dev/null)

if [ -n "$TEST_RESPONSE" ] && [ "$TEST_RESPONSE" != "null" ]; then
    echo "✅ Model test successful!"
    echo "Response: $TEST_RESPONSE"
else
    echo "❌ Model test failed! The model may not be loaded correctly."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Stop the server
echo "Stopping test server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

echo ""
echo "Setup complete! Model $MODEL is ready to use with VLLM."
echo ""
echo "To start the server, run:"
echo "  ./start_vllm_server.sh $MODEL"
echo ""
echo "Or manually:"
echo "  vllm serve $MODEL --host localhost --port 8000"
echo ""
echo "To test the integration, run:"
echo "  python vllm_test.py" 