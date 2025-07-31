#!/bin/bash

# Test script for LiteLLM proxy
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "qwen2.5-7b",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}' 