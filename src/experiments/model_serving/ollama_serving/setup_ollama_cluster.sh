#!/bin/bash
# setup_ollama_cluster.sh - Complete setup for Ollama cluster with load balancer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGE_OLLAMA="$SCRIPT_DIR/manage_ollama_instance.sh"
MANAGE_NGINX="$SCRIPT_DIR/manage_nginx_lb.sh"

# Default ports if none provided
DEFAULT_PORTS=(11435 11436 11437 11438)
LOAD_BALANCER_PORT=11434
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

# Function to get context length from environment or use default
get_context_length() {
    if [ -n "$OLLAMA_CONTEXT_LENGTH" ]; then
        echo "$OLLAMA_CONTEXT_LENGTH"
    else
        echo "$DEFAULT_CONTEXT_LENGTH"
    fi
}

# Function to check if nginx is installed
check_nginx() {
    if ! command -v nginx &> /dev/null; then
        echo "❌ nginx is not installed. Installing nginx..."
        sudo apt update && sudo apt install -y nginx
        if [ $? -eq 0 ]; then
            echo "✅ nginx installed successfully"
        else
            echo "❌ Failed to install nginx"
            exit 1
        fi
    fi
}

# Function to check if ollama is installed
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        echo "❌ ollama is not installed. Please install ollama first:"
        echo "   curl -fsSL https://ollama.ai/install.sh | sh"
        exit 1
    fi
}

case "$1" in
    start)
        shift  # Remove 'start' from arguments
        PORTS=($(get_ports "$@"))
        CONTEXT_LENGTH=$(get_context_length)
        
        echo "🚀 Starting Ollama cluster with load balancer"
        echo "Backend ports: ${PORTS[*]}"
        echo "Load balancer port: $LOAD_BALANCER_PORT"
        echo "Context length: $CONTEXT_LENGTH"
        echo ""
        
        # Check prerequisites
        check_ollama
        check_nginx
        
        # Start Ollama instances with context length
        echo "📦 Starting Ollama instances..."
        OLLAMA_CONTEXT_LENGTH=$CONTEXT_LENGTH $MANAGE_OLLAMA start "${PORTS[@]}"
        
        # Wait a moment for instances to start
        echo "⏳ Waiting for Ollama instances to initialize..."
        sleep 5
        
        # Check if instances are running
        echo "🔍 Checking Ollama instances status..."
        $MANAGE_OLLAMA status "${PORTS[@]}"
        
        # Start nginx load balancer
        echo ""
        echo "⚖️  Starting nginx load balancer..."
        $MANAGE_NGINX start "${PORTS[@]}"
        
        echo ""
        echo "✅ Ollama cluster setup complete!"
        echo ""
        echo "🌐 Access your load-balanced Ollama at:"
        echo "   http://localhost:$LOAD_BALANCER_PORT"
        echo ""
        echo "🔧 Useful endpoints:"
        echo "   Health check: http://localhost:$LOAD_BALANCER_PORT/health"
        echo "   Status check: http://localhost:$LOAD_BALANCER_PORT/status"
        echo ""
        echo "📝 For LiteLLM, use this base URL:"
        echo "   http://localhost:$LOAD_BALANCER_PORT"
        echo ""
        echo "⚙️  Cluster configuration:"
        echo "   Context length: $CONTEXT_LENGTH"
        echo "   Backend instances: ${#PORTS[@]}"
        ;;
    stop)
        shift  # Remove 'stop' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "🛑 Stopping Ollama cluster and load balancer"
        
        # Stop nginx load balancer
        echo "⚖️  Stopping nginx load balancer..."
        $MANAGE_NGINX stop
        
        # Stop Ollama instances
        echo "📦 Stopping Ollama instances..."
        $MANAGE_OLLAMA stop "${PORTS[@]}"
        
        echo "✅ Ollama cluster stopped"
        ;;
    restart)
        shift  # Remove 'restart' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "🔄 Restarting Ollama cluster"
        $0 stop
        sleep 3
        $0 start "${PORTS[@]}"
        ;;
    status)
        echo "📊 Ollama cluster status"
        echo ""
        
        # Check Ollama instances
        echo "📦 Ollama instances:"
        $MANAGE_OLLAMA status
        echo ""
        
        # Check nginx load balancer
        echo "⚖️  Nginx load balancer:"
        $MANAGE_NGINX status
        echo ""
        
        # Show cluster configuration
        CONTEXT_LENGTH=$(get_context_length)
        PORTS=($(get_ports))
        echo "⚙️  Cluster configuration:"
        echo "   Context length: $CONTEXT_LENGTH"
        echo "   Backend instances: ${#PORTS[@]}"
        echo "   Load balancer port: $LOAD_BALANCER_PORT"
        ;;
    test)
        echo "🧪 Testing Ollama cluster"
        
        # Test load balancer health
        echo "Testing load balancer health..."
        if curl -s http://localhost:$LOAD_BALANCER_PORT/health >/dev/null 2>&1; then
            echo "✅ Load balancer is healthy"
        else
            echo "❌ Load balancer is not responding"
        fi
        
        # Test Ollama API through load balancer
        echo "Testing Ollama API through load balancer..."
        if curl -s http://localhost:$LOAD_BALANCER_PORT/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama API is accessible through load balancer"
        else
            echo "❌ Ollama API is not accessible through load balancer"
        fi
        
        # Test direct Ollama instances
        PORTS=($(get_ports))
        echo "Testing direct Ollama instances..."
        for PORT in "${PORTS[@]}"; do
            if curl -s http://localhost:$PORT/api/tags >/dev/null 2>&1; then
                echo "✅ Port $PORT is responding"
            else
                echo "❌ Port $PORT is not responding"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test} [port1 port2 port3 ...]"
        echo ""
        echo "Environment variables:"
        echo "  OLLAMA_CONTEXT_LENGTH    Context length for Ollama instances (default: $DEFAULT_CONTEXT_LENGTH)"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start cluster with default ports and context length"
        echo "  $0 start 11434 11435        # Start cluster with specific ports"
        echo "  $0 stop                     # Stop the entire cluster"
        echo "  $0 restart                  # Restart the entire cluster"
        echo "  $0 status                   # Check cluster status"
        echo "  $0 test                     # Test cluster functionality"
        echo ""
        echo "Examples with custom context length:"
        echo "  OLLAMA_CONTEXT_LENGTH=16384 $0 start  # Start with 16k context length"
        echo "  OLLAMA_CONTEXT_LENGTH=65536 $0 start  # Start with 64k context length"
        echo ""
        echo "Load balancer will be available at: http://localhost:$LOAD_BALANCER_PORT"
        echo ""
        echo "For LiteLLM configuration, use:"
        echo "  base_url: http://localhost:$LOAD_BALANCER_PORT"
        echo ""
        echo "Note: The load balancer runs on the standard Ollama port (11434)"
        echo "      making it a drop-in replacement for single-instance Ollama"
        echo "  model: ollama/your-model-name"
        exit 1
        ;;
esac 