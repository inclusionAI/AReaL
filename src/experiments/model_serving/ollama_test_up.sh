# Test that the model is working
# Usage: ./ollama_test_up.sh [model_name]
# If no model name is provided, defaults to "qwen2.5:1.5b"

# Set default model if no argument provided
MODEL_NAME=${1:-"qwen2.5:1.5b"}

echo ""
echo "Testing $MODEL_NAME model..."
TEST_RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "'$MODEL_NAME'",
  "prompt": "Hello, please respond with: I am working!",
  "stream": false
}' | grep -o '"response":"[^"]*"' | sed 's/"response":"//;s/"$//')

if [ -n "$TEST_RESPONSE" ]; then
    echo "✅ Model test successful!"
    echo "Response: $TEST_RESPONSE"
else
    echo "❌ Model test failed! The model may not be loaded correctly."
    exit 1
fi
