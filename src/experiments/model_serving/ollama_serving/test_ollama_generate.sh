#!/bin/bash
# test_ollama_generate.sh - Quick test of Ollama cluster using generate API

# Show usage if help is requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [MODEL] [PROMPT]"
    echo ""
    echo "Test the Ollama cluster by making a generate request."
    echo ""
    echo "Arguments:"
    echo "  MODEL   The model to use (default: qwen2.5:1.5b)"
    echo "  PROMPT  The prompt to send (default: 'What is the capital of France? Answer in one word.')"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use defaults"
    echo "  $0 qwen2.5:7b                        # Use different model"
    echo "  $0 qwen2.5:1.5b \"Write a haiku\"      # Custom prompt"
    exit 0
fi

# Configuration
LOAD_BALANCER_URL="http://localhost:11434"
MODEL="${1:-qwen2.5:1.5b}"
PROMPT="${2:-What is the capital of France? Answer in one word.}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 Testing Ollama cluster with generate request"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Load Balancer: $LOAD_BALANCER_URL"
echo "🤖 Model: $MODEL"
echo "💬 Prompt: \"$PROMPT\""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if load balancer is responding
echo -n "🔍 Checking load balancer health... "
if curl -s "$LOAD_BALANCER_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
    echo "   Please ensure the cluster is running: ./setup_ollama_cluster.sh start"
    exit 1
fi

# Check if model exists
echo -n "🔍 Checking if model exists... "
if curl -s "$LOAD_BALANCER_URL/api/tags" | grep -q "\"$MODEL\""; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${YELLOW}⚠ Model not found${NC}"
    echo "   Available models:"
    curl -s "$LOAD_BALANCER_URL/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sed 's/^/   - /'
    echo ""
    echo "   To pull the model, run: ollama pull $MODEL"
    exit 1
fi

echo ""
echo "🚀 Sending generate request..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Record start time
START_TIME=$(date +%s.%N)

# Make the generate request
RESPONSE=$(curl -s -X POST "$LOAD_BALANCER_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "'"$PROMPT"'",
        "stream": false
    }' 2>&1)

# Record end time
END_TIME=$(date +%s.%N)

# Calculate duration
DURATION=$(awk "BEGIN {print $END_TIME - $START_TIME}")

# Check if request was successful
if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q '"response"'; then
    # Extract the response text
    ANSWER=$(echo "$RESPONSE" | grep -o '"response":"[^"]*"' | sed 's/"response":"//' | sed 's/"$//')
    
    echo -e "${GREEN}✅ Success!${NC}"
    echo ""
    echo "📝 Response: $ANSWER"
    echo ""
    echo "⏱️  Time taken: ${DURATION}s"
    
    # Show which backend might have handled it (optional)
    if echo "$RESPONSE" | grep -q '"total_duration"'; then
        TOTAL_DURATION=$(echo "$RESPONSE" | grep -o '"total_duration":[0-9]*' | cut -d':' -f2)
        TOTAL_DURATION_S=$(awk "BEGIN {printf \"%.3f\", $TOTAL_DURATION / 1000000000}")
        echo "📊 Model inference time: ${TOTAL_DURATION_S}s"
    fi
else
    echo -e "${RED}❌ Request failed${NC}"
    echo ""
    echo "Error response:"
    echo "$RESPONSE" | head -5
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✨ Ollama cluster is working correctly!${NC}" 