#!/bin/bash

# Simple vLLM test script using curl
# Usage: ./test_vllm.sh [MODEL_NAME]

set -e

# Configuration
HOST=127.0.0.1
PORT=8000
DEFAULT_MODEL="Qwen/Qwen2.5-7B"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Get model name from argument or use default
MODEL_NAME=${1:-$DEFAULT_MODEL}

print_status "Testing vLLM server with model: $MODEL_NAME"
print_status "Server endpoint: http://$HOST:$PORT"

# Test health endpoint first
print_status "Testing health endpoint..."
if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    print_status "Server is healthy"
else
    print_error "Server health check failed. Make sure vLLM server is running."
    exit 1
fi

# Test completion endpoint
print_status "Testing completion endpoint..."

# Create a simple test prompt
TEST_PROMPT="Hello, how are you today?"

# Send request to vLLM
RESPONSE=$(curl -s -X POST "http://$HOST:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"$TEST_PROMPT\",
        \"max_tokens\": 50,
        \"temperature\": 0.7
    }")

# Check if request was successful
if [ $? -eq 0 ]; then
    print_status "Request successful!"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    print_error "Request failed"
    exit 1
fi

print_status "Test completed successfully!" 