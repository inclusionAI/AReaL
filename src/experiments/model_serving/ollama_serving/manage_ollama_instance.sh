#!/bin/bash
# manage_ollama_instances.sh

# Default ports if none provided
DEFAULT_PORTS=(11435 11436 11437 11438)
# Default context length
DEFAULT_CONTEXT_LENGTH=32768

# Function to get ports from arguments or use defaults
get_ports() {
    if [ $# -eq 0 ]; then
        echo "${DEFAULT_PORTS[@]}"
    else
        echo "$@"
    fi
}

# Function to get GPU device for a given port index
get_gpu_device() {
    local port_index=$1
    echo $port_index
}

# Function to get context length from environment or use default
get_context_length() {
    if [ -n "$OLLAMA_CONTEXT_LENGTH" ]; then
        echo "$OLLAMA_CONTEXT_LENGTH"
    else
        echo "$DEFAULT_CONTEXT_LENGTH"
    fi
}

case "$1" in
    start)
        shift  # Remove 'start' from arguments
        PORTS=($(get_ports "$@"))
        CONTEXT_LENGTH=$(get_context_length)
        
        echo "Starting Ollama instances on ports: ${PORTS[*]}"
        echo "Using context length: $CONTEXT_LENGTH"
        echo ""
        
        for i in "${!PORTS[@]}"; do
            PORT=${PORTS[$i]}
            GPU_DEVICE=$(get_gpu_device $i)
            
            echo "Starting instance on port $PORT using GPU $GPU_DEVICE (context length: $CONTEXT_LENGTH)"
            # OLLAMA_ORIGINS="*" allows all origins (needed for nginx proxy)
            OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_ORIGINS="*" OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH CUDA_VISIBLE_DEVICES=$GPU_DEVICE ollama serve &
        done
        ;;
    stop)
        shift  # Remove 'stop' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "Stopping Ollama instances on ports: ${PORTS[*]}"
        
        for PORT in "${PORTS[@]}"; do
            echo "Stopping instance on port $PORT"
            # Find and kill the specific ollama process on this port
            PIDS=$(lsof -ti:$PORT 2>/dev/null)
            if [ -n "$PIDS" ]; then
                echo "Killing processes on port $PORT: $PIDS"
                kill $PIDS
            fi
        done
        
        # Also kill any remaining ollama serve processes
        pkill -f "ollama serve"
        ;;
    status)
        shift  # Remove 'status' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "Checking Ollama instances on ports: ${PORTS[*]}"
        echo ""
        
        for PORT in "${PORTS[@]}"; do
            if curl -s http://localhost:$PORT/api/tags >/dev/null 2>&1; then
                # Try to get context length from environment of running process
                CONTEXT_LENGTH=$(ps aux | grep "ollama serve" | grep "OLLAMA_HOST=127.0.0.1:$PORT" | grep -o "OLLAMA_CONTEXT_LENGTH=[0-9]*" | head -1 | cut -d'=' -f2)
                if [ -n "$CONTEXT_LENGTH" ]; then
                    echo "✅ Instance on port $PORT is running (context length: $CONTEXT_LENGTH)"
                else
                    echo "✅ Instance on port $PORT is running (context length: unknown)"
                fi
            else
                echo "❌ Instance on port $PORT is not running"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status} [port1 port2 port3 ...]"
        echo ""
        echo "Environment variables:"
        echo "  OLLAMA_CONTEXT_LENGTH    Context length for Ollama instances (default: $DEFAULT_CONTEXT_LENGTH)"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start on default ports with default context length"
        echo "  $0 start 11434 11435        # Start on ports 11434 and 11435 only"
        echo "  $0 start 11440 11441 11442  # Start on custom ports"
        echo "  $0 stop                     # Stop all ollama instances"
        echo "  $0 stop 11434 11435         # Stop instances on specific ports"
        echo "  $0 status                   # Check status of all instances"
        echo "  $0 status 11434 11435       # Check status of specific instances"
        echo ""
        echo "Examples with custom context length:"
        echo "  OLLAMA_CONTEXT_LENGTH=16384 $0 start  # Start with 16k context length"
        echo "  OLLAMA_CONTEXT_LENGTH=65536 $0 start  # Start with 64k context length"
        echo ""
        echo "Note: GPU assignment is automatic based on port order (GPU 0, 1, 2, etc.)"
        exit 1
        ;;
esac