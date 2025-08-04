#!/bin/bash

# Simple vLLM test script using curl
# Usage: ./test_vllm.sh [MODEL_NAME]

set -e

# Configuration
HOST=127.0.0.1
PORT=8000
DEFAULT_MODEL="Qwen/Qwen2.5-7B-instruct"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

show_help() {
    echo "vLLM Server Test Script"
    echo ""
    echo "Usage: $0 [OPTIONS] [MODEL_NAME]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  MODEL_NAME     Name of the model to test (default: $DEFAULT_MODEL)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Test with default model"
    echo "  $0 Qwen/Qwen2.5-14B         # Test with specific model"
    echo "  $0 --help                    # Show this help"
    echo ""
    echo "The script will test:"
    echo "  1. Server health endpoint"
    echo "  2. Completion endpoint with a simple prompt"
}

# Parse command line arguments
MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_NAME" ]]; then
                MODEL_NAME="$1"
            else
                print_error "Multiple model names provided. Use only one model name."
                exit 1
            fi
            ;;
    esac
    shift
done

# Use default model if none provided
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$DEFAULT_MODEL"
fi

print_header "vLLM Server Test Suite"
print_status "Testing model: $MODEL_NAME"
print_status "Server endpoint: http://$HOST:$PORT"
echo ""

# Test health endpoint first
print_status "=== Test 1: Health Check ==="
print_status "Testing health endpoint..."
if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    print_status "✅ Server is healthy"
else
    print_error "❌ Server health check failed. Make sure vLLM server is running."
    print_warning "Try running: ./vllm_start_server.sh status"
    exit 1
fi
echo ""

# Test completion endpoint
print_status "=== Test 2: Completion Endpoint ==="
print_status "Testing completion endpoint..."

# Create a simple test prompt
TEST_PROMPT="Hello, how are you today?"

print_status "Sending request with prompt: \"$TEST_PROMPT\""

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
    print_status "✅ Request successful!"
    echo ""
    print_status "Response:"
    echo "----------------------------------------"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    echo "----------------------------------------"
else
    print_error "❌ Request failed"
    print_warning "Check if the model '$MODEL_NAME' is loaded on the server"
    print_warning "Try running: ./vllm_start_server.sh status"
    exit 1
fi

echo ""
print_status "🎉 Test completed successfully!"
print_status "Model '$MODEL_NAME' is working correctly with the vLLM server." 