#!/bin/bash

# LiteLLM Proxy Management Script
# Usage: ./start_proxy.sh [start|stop|restart|models]

# Configuration
LITELLM_PID_FILE="/tmp/litellm_proxy.pid"
LITELLM_LOG_FILE="/tmp/litellm_proxy.log"
LITELLM_PORT=4000

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

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if LiteLLM is installed
check_litellm_installed() {
    if command -v litellm &> /dev/null; then
        print_status "LiteLLM is already installed"
        return 0
    else
        print_warning "LiteLLM is not installed"
        return 1
    fi
}

# Function to install LiteLLM
install_litellm() {
    print_info "Installing LiteLLM with proxy support..."
    pip install litellm[proxy]
    if [ $? -eq 0 ]; then
        print_status "LiteLLM installed successfully"
        return 0
    else
        print_error "Failed to install LiteLLM"
        return 1
    fi
}

# Function to check if LiteLLM proxy is running
is_litellm_running() {
    if [ -f "$LITELLM_PID_FILE" ]; then
        local pid=$(cat "$LITELLM_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm -f "$LITELLM_PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start LiteLLM proxy
start_litellm() {
    print_info "Starting LiteLLM proxy..."
    
    # Check if already running
    if is_litellm_running; then
        print_warning "LiteLLM proxy is already running"
        return 0
    fi
    
    # Check if LiteLLM is installed
    if ! check_litellm_installed; then
        if ! install_litellm; then
            print_error "Failed to install LiteLLM. Cannot start proxy."
            return 1
        fi
    fi
    
    # Get the directory where this script is located
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local config_file="$script_dir/litellm_config.yaml"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        return 1
    fi
    
    print_info "Using configuration file: $config_file"
    
    # Start LiteLLM proxy in background with config file
    nohup litellm --port $LITELLM_PORT --config "$config_file" > "$LITELLM_LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID to file
    echo $pid > "$LITELLM_PID_FILE"
    
    # Wait a moment to check if it started successfully
    sleep 2
    if is_litellm_running; then
        print_status "LiteLLM proxy started successfully on port $LITELLM_PORT (PID: $pid)"
        print_info "Configuration file: $config_file"
        print_info "Logs are available at: $LITELLM_LOG_FILE"
        return 0
    else
        print_error "Failed to start LiteLLM proxy"
        rm -f "$LITELLM_PID_FILE"
        return 1
    fi
}

# Function to stop LiteLLM proxy
stop_litellm() {
    print_info "Stopping LiteLLM proxy..."
    
    if ! is_litellm_running; then
        print_warning "LiteLLM proxy is not running"
        return 0
    fi
    
    local pid=$(cat "$LITELLM_PID_FILE")
    kill "$pid" 2>/dev/null
    
    # Wait for process to terminate
    local count=0
    while [ $count -lt 10 ] && ps -p "$pid" > /dev/null 2>&1; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "Force killing LiteLLM proxy process"
        kill -9 "$pid" 2>/dev/null
    fi
    
    rm -f "$LITELLM_PID_FILE"
    print_status "LiteLLM proxy stopped"
    return 0
}

# Function to restart LiteLLM proxy
restart_litellm() {
    print_info "Restarting LiteLLM proxy..."
    
    # Stop if running
    if is_litellm_running; then
        stop_litellm
    fi
    
    # Start
    start_litellm
}

# Function to list models
list_models() {
    print_info "Listing available models..."
    
    # Use litellm CLI to list models
    litellm-proxy models info 
    }

# Function to show status
show_status() {
    if is_litellm_running; then
        local pid=$(cat "$LITELLM_PID_FILE")
        print_status "LiteLLM proxy is running (PID: $pid, Port: $LITELLM_PORT)"
    else
        print_warning "LiteLLM proxy is not running"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start LiteLLM proxy (install if needed)"
    echo "  stop      - Stop LiteLLM proxy"
    echo "  restart   - Restart LiteLLM proxy"
    echo "  models    - List available models"
    echo "  status    - Show current status"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 stop"
    echo "  $0 models"
}

# Main script logic
case "${1:-help}" in
    start)
        start_litellm
        ;;
    stop)
        stop_litellm
        ;;
    restart)
        restart_litellm
        ;;
    models)
        list_models
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac