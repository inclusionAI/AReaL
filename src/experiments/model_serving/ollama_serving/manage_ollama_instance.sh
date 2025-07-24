#!/bin/bash
# manage_ollama_instances.sh

# Default ports if none provided
DEFAULT_PORTS=(11435 11436 11437 11438)

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

case "$1" in
    start)
        shift  # Remove 'start' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "Starting Ollama instances on ports: ${PORTS[*]}"
        
        for i in "${!PORTS[@]}"; do
            PORT=${PORTS[$i]}
            GPU_DEVICE=$(get_gpu_device $i)
            
            echo "Starting instance on port $PORT using GPU $GPU_DEVICE"
            # OLLAMA_ORIGINS="*" allows all origins (needed for nginx proxy)
            OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_ORIGINS="*" CUDA_VISIBLE_DEVICES=$GPU_DEVICE ollama serve &
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
        
        for PORT in "${PORTS[@]}"; do
            if curl -s http://localhost:$PORT/api/tags >/dev/null 2>&1; then
                echo "✅ Instance on port $PORT is running"
            else
                echo "❌ Instance on port $PORT is not running"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status} [port1 port2 port3 ...]"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start on default ports: 11434, 11435, 11436, 11437"
        echo "  $0 start 11434 11435        # Start on ports 11434 and 11435 only"
        echo "  $0 start 11440 11441 11442  # Start on custom ports"
        echo "  $0 stop                     # Stop all ollama instances"
        echo "  $0 stop 11434 11435         # Stop instances on specific ports"
        echo "  $0 status                   # Check status of all instances"
        echo "  $0 status 11434 11435       # Check status of specific instances"
        echo ""
        echo "Note: GPU assignment is automatic based on port order (GPU 0, 1, 2, etc.)"
        exit 1
        ;;
esac