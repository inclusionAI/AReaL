#!/bin/bash

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done
echo "Ollama server is ready!"

# Pull the Qwen model
echo "Pulling Qwen 2.5 1.5B model..."
ollama pull qwen2.5:1.5b

# Test that the model is working
echo ""
echo "Testing Qwen model..."
TEST_RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:1.5b",
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

echo ""
echo "Setup complete! Qwen 2.5 1.5B is ready to use."
echo "Ollama server is running with PID: $OLLAMA_PID"
echo ""
echo "To use the model, run:"
echo "  ollama run qwen2.5:1.5b"
echo ""
echo "To stop the server later, run:"
echo "  kill $OLLAMA_PID"