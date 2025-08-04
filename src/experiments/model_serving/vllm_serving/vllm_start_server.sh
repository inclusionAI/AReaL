#!/bin/bash

# vLLM Server Management Script
# Usage: ./vllm_start_server.sh [start|stop|restart|status] [model_name]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.vllm_server.pid"
LOG_FILE="$SCRIPT_DIR/.vllm_server.log"
VENV_DIR="$SCRIPT_DIR/.venv"
HOST=127.0.0.1
PORT=8000
DEFAULT_MODEL="Qwen/Qwen2.5-7B-instruct"

# Get number of available GPUs
get_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | wc -l
    else
        echo 1  # Default to 1 if nvidia-smi not available
    fi
}

# Default parallelism settings
DEFAULT_TENSOR_PARALLEL_SIZE=$(get_gpu_count)
DEFAULT_PIPELINE_PARALLEL_SIZE=1
DEFAULT_DATA_PARALLEL_SIZE=1

# Default tool settings
DEFAULT_ENABLE_AUTO_TOOL_CHOICE=true
DEFAULT_TOOL_CALL_PARSER="hermes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[vLLM Server]${NC} $1"
}

# Function to check if server is running
is_server_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Function to check if port is in use
is_port_in_use() {
    lsof -i :$PORT > /dev/null 2>&1
}

# Function to setup environment
setup_environment() {
    print_status "Setting up vLLM environment..."
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_status "Creating virtual environment..."
        uv venv --python 3.12 --seed "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install vLLM if not already installed
    if ! python -c "import vllm" 2>/dev/null; then
        print_status "Installing vLLM..."
        uv pip install vllm --torch-backend=auto
    else
        print_status "vLLM is already installed."
    fi
}

# Function to start the server
start_server() {
    local model_name=${1:-$DEFAULT_MODEL}
    local tensor_parallel_size=${2:-$DEFAULT_TENSOR_PARALLEL_SIZE}
    local pipeline_parallel_size=${3:-$DEFAULT_PIPELINE_PARALLEL_SIZE}
    local data_parallel_size=${4:-$DEFAULT_DATA_PARALLEL_SIZE}
    local enable_auto_tool_choice=${5:-$DEFAULT_ENABLE_AUTO_TOOL_CHOICE}
    local tool_call_parser=${6:-$DEFAULT_TOOL_CALL_PARSER}
    
    print_header "Starting vLLM server..."
    
    if is_server_running; then
        print_warning "Server is already running (PID: $(cat $PID_FILE))"
        return 0
    fi
    
    if is_port_in_use; then
        print_error "Port $PORT is already in use. Please stop the existing service or use a different port."
        exit 1
    fi
    
    # Setup environment
    setup_environment
    
    # Validate parallelism settings
    local total_gpus=$(get_gpu_count)
    local total_parallel_size=$((tensor_parallel_size * pipeline_parallel_size * data_parallel_size))
    
    if [ "$total_parallel_size" -gt "$total_gpus" ]; then
        print_error "Total parallel size ($total_parallel_size) exceeds available GPUs ($total_gpus)"
        print_error "Tensor parallel: $tensor_parallel_size, Pipeline parallel: $pipeline_parallel_size, Data parallel: $data_parallel_size"
        exit 1
    fi
    
    # Start the server in background
    print_status "Starting server with model: $model_name"
    print_status "Tensor parallel size: $tensor_parallel_size"
    print_status "Pipeline parallel size: $pipeline_parallel_size"
    print_status "Data parallel size: $data_parallel_size"
    print_status "Auto tool choice: $enable_auto_tool_choice"
    print_status "Tool call parser: $tool_call_parser"
    print_status "Server will be available at: http://$HOST:$PORT"
    
    # Build vllm command with optional data parallelism
    local vllm_cmd="vllm serve $model_name --host $HOST --port $PORT --dtype auto --tensor-parallel-size $tensor_parallel_size --pipeline-parallel-size $pipeline_parallel_size"
    
    # Add tool choice options
    if [ "$enable_auto_tool_choice" = "true" ]; then
        vllm_cmd="$vllm_cmd --enable-auto-tool-choice"
    fi
    
    # Add tool call parser
    if [ -n "$tool_call_parser" ]; then
        vllm_cmd="$vllm_cmd --tool-call-parser $tool_call_parser"
    fi
    
    # Add data parallelism if specified and greater than 1
    if [ "$data_parallel_size" -gt 1 ]; then
        vllm_cmd="$vllm_cmd --data-parallel-size $data_parallel_size"
    fi
    
    nohup $vllm_cmd > "$LOG_FILE" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "$PID_FILE"
    
    # Wait a moment for server to start
    sleep 3
    
    if is_server_running; then
        print_status "Server started successfully (PID: $server_pid)"
        print_status "Logs are available at: $LOG_FILE"
        
        # Test the server
        print_status "Testing server connection..."
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            print_status "Server is responding to health checks"
        else
            print_warning "Server started but health check failed. Check logs at: $LOG_FILE"
        fi
    else
        print_error "Failed to start server. Check logs at: $LOG_FILE"
        exit 1
    fi
}

# Function to stop the server
stop_server() {
    print_header "Stopping vLLM server..."
    
    if ! is_server_running; then
        print_warning "Server is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "Stopping server (PID: $pid)..."
    
    # Try graceful shutdown first
    kill "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local count=0
    while is_server_running && [ $count -lt 10 ]; do
        sleep 1
        ((count++))
    done
    
    # Force kill if still running
    if is_server_running; then
        print_warning "Graceful shutdown failed, forcing termination..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    if ! is_server_running; then
        print_status "Server stopped successfully"
    else
        print_error "Failed to stop server"
        exit 1
    fi
}

# Function to restart the server
restart_server() {
    local model_name=${1:-$DEFAULT_MODEL}
    local tensor_parallel_size=${2:-$DEFAULT_TENSOR_PARALLEL_SIZE}
    local pipeline_parallel_size=${3:-$DEFAULT_PIPELINE_PARALLEL_SIZE}
    local data_parallel_size=${4:-$DEFAULT_DATA_PARALLEL_SIZE}
    local enable_auto_tool_choice=${5:-$DEFAULT_ENABLE_AUTO_TOOL_CHOICE}
    local tool_call_parser=${6:-$DEFAULT_TOOL_CALL_PARSER}
    
    print_header "Restarting vLLM server..."
    stop_server
    sleep 2
    start_server "$model_name" "$tensor_parallel_size" "$pipeline_parallel_size" "$data_parallel_size" "$enable_auto_tool_choice" "$tool_call_parser"
}

# Function to show GPU information
show_gpu_info() {
    print_status "GPU Information:"
    local gpu_count=$(get_gpu_count)
    print_status "  Available GPUs: $gpu_count"
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "  GPU Details:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while IFS=, read -r name total free; do
            print_status "    $name - Total: ${total}MB, Free: ${free}MB"
        done
    fi
}

# Function to check server status
check_status() {
    print_header "Checking vLLM server status..."
    
    # Show GPU information
    show_gpu_info
    
    if is_server_running; then
        local pid=$(cat "$PID_FILE")
        print_status "Server is running (PID: $pid)"
        
        # Check if port is accessible
        if is_port_in_use; then
            print_status "Port $PORT is in use"
            
            # Test health endpoint
            if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
                print_status "Server is responding to health checks"
                
                # Get running models
                print_status "Checking running models..."
                local models_response=$(curl -s "http://$HOST:$PORT/v1/models" 2>/dev/null)
                if [ $? -eq 0 ] && [ -n "$models_response" ]; then
                    # Extract model IDs using jq if available, otherwise use grep/sed
                    if command -v jq &> /dev/null; then
                        local model_ids=$(echo "$models_response" | jq -r '.data[].id' 2>/dev/null)
                        if [ $? -eq 0 ] && [ -n "$model_ids" ]; then
                            print_status "Running models:"
                            echo "$model_ids" | while read -r model_id; do
                                print_status "  - $model_id"
                            done
                        else
                            print_warning "Failed to parse models response or no models found"
                        fi
                    else
                        # Fallback to grep/sed if jq is not available
                        local model_ids=$(echo "$models_response" | grep -o '"id":"[^"]*"' | sed 's/"id":"//g' | sed 's/"//g')
                        if [ -n "$model_ids" ]; then
                            print_status "Running models:"
                            echo "$model_ids" | while read -r model_id; do
                                print_status "  - $model_id"
                            done
                        else
                            print_warning "Failed to parse models response or no models found"
                        fi
                    fi
                else
                    print_warning "Failed to retrieve models information"
                fi
            else
                print_warning "Server is running but not responding to health checks"
            fi
        else
            print_warning "Server process exists but port $PORT is not in use"
        fi
        
        # Show recent logs
        if [ -f "$LOG_FILE" ]; then
            print_status "Recent logs (last 10 lines):"
            tail -n 10 "$LOG_FILE" | sed 's/^/  /'
        fi
    else
        print_status "Server is not running"
        
        # Check if port is still in use by another process
        if is_port_in_use; then
            print_warning "Port $PORT is in use by another process"
            lsof -i :$PORT
        fi
    fi
}

# Function to show usage
show_usage() {
    echo "vLLM Server Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [MODEL_NAME] [TENSOR_PARALLEL_SIZE] [PIPELINE_PARALLEL_SIZE] [DATA_PARALLEL_SIZE] [ENABLE_AUTO_TOOL_CHOICE] [TOOL_CALL_PARSER]"
    echo ""
    echo "Commands:"
    echo "  start     Start the vLLM server"
    echo "  stop      Stop the vLLM server"
    echo "  restart   Restart the vLLM server"
    echo "  status    Check server status"
    echo "  gpu-info  Show GPU information"
    echo "  install   Install dependencies only"
    echo ""
    echo "Arguments:"
    echo "  MODEL_NAME              Model to load (default: $DEFAULT_MODEL)"
    echo "  TENSOR_PARALLEL_SIZE    Number of GPUs for tensor parallelism (default: $DEFAULT_TENSOR_PARALLEL_SIZE)"
    echo "  PIPELINE_PARALLEL_SIZE  Number of GPUs for pipeline parallelism (default: $DEFAULT_PIPELINE_PARALLEL_SIZE)"
    echo "  DATA_PARALLEL_SIZE      Number of GPUs for data parallelism (default: $DEFAULT_DATA_PARALLEL_SIZE)"
    echo "  ENABLE_AUTO_TOOL_CHOICE Enable auto tool choice (true/false, default: $DEFAULT_ENABLE_AUTO_TOOL_CHOICE)"
    echo "  TOOL_CALL_PARSER        Tool call parser to use (default: $DEFAULT_TOOL_CALL_PARSER)"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 start Qwen/Qwen2.5-7B"
    echo "  $0 start Qwen/Qwen2.5-7B 2 1 1"
    echo "  $0 start Qwen/Qwen2.5-7B 2 1 2"
    echo "  $0 start Qwen/Qwen3-8B 2 1 1 true hermes"
    echo "  $0 start Qwen/Qwen3-8B 2 1 1 false json"
    echo "  $0 restart meta-llama/Llama-2-7b-chat-hf 4 2 1 true hermes"
    echo "  $0 status"
    echo "  $0 gpu-info"
    echo "  $0 stop"
    echo ""
    echo "Tool Settings:"
    echo "  Default auto tool choice: $DEFAULT_ENABLE_AUTO_TOOL_CHOICE"
    echo "  Default tool call parser: $DEFAULT_TOOL_CALL_PARSER"
    echo ""
    echo "GPU Information:"
    echo "  Available GPUs: $(get_gpu_count)"
    echo "  Default tensor parallel size: $DEFAULT_TENSOR_PARALLEL_SIZE"
    echo "  Default pipeline parallel size: $DEFAULT_PIPELINE_PARALLEL_SIZE"
    echo "  Default data parallel size: $DEFAULT_DATA_PARALLEL_SIZE"
}

# Function to install dependencies only
install_deps() {
    print_header "Installing dependencies..."
    setup_environment
    print_status "Dependencies installed successfully"
}

# Main script logic
case "${1:-}" in
    start)
        start_server "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server "$2" "$3" "$4" "$5" "$6" "$7"
        ;;
    status)
        check_status
        ;;
    gpu-info)
        show_gpu_info
        ;;
    install)
        install_deps
        ;;
    help|--help|-h)
        show_usage
        ;;
    "")
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac