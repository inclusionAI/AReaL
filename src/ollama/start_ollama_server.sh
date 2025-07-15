#!/bin/bash

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama server is already running!"
    exit 0
fi

# Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
attempts=0
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
    attempts=$((attempts + 1))
    if [ $attempts -gt 30 ]; then
        echo "❌ Failed to start Ollama server after 30 seconds"
        exit 1
    fi
done

echo "✅ Ollama server is ready!"
echo "Server PID: $OLLAMA_PID"
echo ""
echo "To stop the server later, run:"
echo "  kill $OLLAMA_PID" 