#!/bin/bash
# manage_nginx_lb.sh - Manage nginx load balancer for Ollama instances

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/nginx_ollama_lb.conf"
NGINX_CONF_DIR="/etc/nginx/sites-available"
NGINX_ENABLED_DIR="/etc/nginx/sites-enabled"
SITE_NAME="ollama-lb"
LOAD_BALANCER_PORT=11434

# Function to check if nginx is installed
check_nginx() {
    if ! command -v nginx &> /dev/null; then
        echo "❌ nginx is not installed. Please install nginx first:"
        echo "   sudo apt update && sudo apt install nginx"
        exit 1
    fi
}

# Function to generate nginx config with custom ports
generate_config() {
    local ports=("$@")
    local config_content="# Nginx configuration for Ollama load balancer\n"
    config_content+="# This configuration provides round-robin load balancing across multiple Ollama instances\n\n"
    
    config_content+="upstream ollama_backend {\n"
    config_content+="    # Round-robin load balancing (default)\n"
    
    for port in "${ports[@]}"; do
        config_content+="    server 127.0.0.1:$port;\n"
    done
    
    config_content+="}\n\n"
    
    config_content+="server {\n"
    config_content+="    listen $LOAD_BALANCER_PORT;\n"
    config_content+="    server_name localhost;\n\n"
    
    config_content+="    # Increase timeouts for long-running model inference\n"
    config_content+="    proxy_connect_timeout 300s;\n"
    config_content+="    proxy_send_timeout 300s;\n"
    config_content+="    proxy_read_timeout 300s;\n\n"
    
    config_content+="    # Buffer settings for large responses\n"
    config_content+="    proxy_buffering on;\n"
    config_content+="    proxy_buffer_size 4k;\n"
    config_content+="    proxy_buffers 8 4k;\n"
    config_content+="    proxy_busy_buffers_size 8k;\n\n"
    
    config_content+="    # Headers\n"
    
    config_content+="    # Handle all Ollama API endpoints\n"
    config_content+="    location / {\n"
    config_content+="        proxy_pass http://ollama_backend;\n\n"
        
    config_content+="        # Headers (must be in location block to ensure they're applied)\n"
    config_content+="        proxy_set_header Host localhost;\n"
    config_content+="        proxy_set_header X-Real-IP \$remote_addr;\n"
    config_content+="        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;\n"
    config_content+="        proxy_set_header X-Forwarded-Proto \$scheme;\n\n"
        
    config_content+="        # Handle WebSocket connections for streaming responses\n"
    config_content+="        proxy_http_version 1.1;\n"
    config_content+="        proxy_set_header Upgrade \$http_upgrade;\n"
    config_content+="        proxy_set_header Connection \"upgrade\";\n\n"
        
    config_content+="        # Disable buffering for streaming responses\n"
    config_content+="        proxy_buffering off;\n"
    config_content+="    }\n\n"
    
    config_content+="    # Health check endpoint\n"
    config_content+="    location /health {\n"
    config_content+="        access_log off;\n"
    config_content+="        return 200 \"healthy\\n\";\n"
    config_content+="        add_header Content-Type text/plain;\n"
    config_content+="    }\n\n"
    
    config_content+="    # Status endpoint to check backend health\n"
    config_content+="    location /status {\n"
    config_content+="        access_log off;\n"
    config_content+="        proxy_pass http://ollama_backend/api/tags;\n"
    config_content+="        proxy_buffering off;\n"
    config_content+="    }\n"
    config_content+="}\n"
    
    echo -e "$config_content"
}

# Function to get ports from arguments or use defaults
get_ports() {
    if [ $# -eq 0 ]; then
        echo "11434 11435 11436 11437"
    else
        echo "$@"
    fi
}

case "$1" in
    start)
        shift  # Remove 'start' from arguments
        PORTS=($(get_ports "$@"))
        
        check_nginx
        
        echo "Starting nginx load balancer on port $LOAD_BALANCER_PORT"
        echo "Backend ports: ${PORTS[*]}"
        
        # Generate config with the specified ports
        CONFIG_CONTENT=$(generate_config "${PORTS[@]}")
        
        # Write config to nginx sites-available
        echo "$CONFIG_CONTENT" | sudo tee "$NGINX_CONF_DIR/$SITE_NAME" > /dev/null
        
        # Create symlink to enable the site
        if [ ! -L "$NGINX_ENABLED_DIR/$SITE_NAME" ]; then
            sudo ln -s "$NGINX_CONF_DIR/$SITE_NAME" "$NGINX_ENABLED_DIR/$SITE_NAME"
        fi
        
        # Test nginx configuration
        if sudo nginx -t; then
            # Start or reload nginx
            if systemctl is-active --quiet nginx; then
                sudo systemctl reload nginx
            else
                echo "📌 nginx is not running, starting it..."
                sudo systemctl start nginx
            fi
            echo "✅ Load balancer started successfully"
            echo "   Access your load-balanced Ollama at: http://localhost:$LOAD_BALANCER_PORT"
            echo "   Health check: http://localhost:$LOAD_BALANCER_PORT/health"
            echo "   Status check: http://localhost:$LOAD_BALANCER_PORT/status"
        else
            echo "❌ nginx configuration test failed"
            exit 1
        fi
        ;;
    stop)
        echo "Stopping nginx load balancer"
        
        # Remove symlink to disable the site
        if [ -L "$NGINX_ENABLED_DIR/$SITE_NAME" ]; then
            sudo rm "$NGINX_ENABLED_DIR/$SITE_NAME"
        fi
        
        # Reload nginx if it's running
        if systemctl is-active --quiet nginx; then
            sudo systemctl reload nginx
        fi
        echo "✅ Load balancer stopped"
        ;;
    restart)
        shift  # Remove 'restart' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "Restarting nginx load balancer"
        $0 stop
        sleep 2
        $0 start "${PORTS[@]}"
        ;;
    status)
        echo "Checking nginx load balancer status"
        
        # Check if nginx is running
        if systemctl is-active --quiet nginx; then
            echo "✅ nginx service is running"
        else
            echo "❌ nginx service is not running"
        fi
        
        # Check if our site is enabled
        if [ -L "$NGINX_ENABLED_DIR/$SITE_NAME" ]; then
            echo "✅ Load balancer site is enabled"
        else
            echo "❌ Load balancer site is not enabled"
        fi
        
        # Test load balancer endpoint
        if curl -s http://localhost:$LOAD_BALANCER_PORT/health >/dev/null 2>&1; then
            echo "✅ Load balancer is responding on port $LOAD_BALANCER_PORT"
        else
            echo "❌ Load balancer is not responding on port $LOAD_BALANCER_PORT"
        fi
        
        # Show current configuration
        if [ -f "$NGINX_CONF_DIR/$SITE_NAME" ]; then
            echo ""
            echo "Current backend configuration:"
            grep "server 127.0.0.1:" "$NGINX_CONF_DIR/$SITE_NAME" | sed 's/^[[:space:]]*//'
        fi
        ;;
    config)
        shift  # Remove 'config' from arguments
        PORTS=($(get_ports "$@"))
        
        echo "Generating nginx configuration for ports: ${PORTS[*]}"
        echo ""
        generate_config "${PORTS[@]}"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|config} [port1 port2 port3 ...]"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start load balancer with default ports: 11434, 11435, 11436, 11437"
        echo "  $0 start 11434 11435        # Start load balancer with specific ports"
        echo "  $0 stop                     # Stop the load balancer"
        echo "  $0 restart                  # Restart the load balancer"
        echo "  $0 status                   # Check load balancer status"
        echo "  $0 config                   # Show generated nginx configuration"
        echo ""
        echo "Load balancer will be available at: http://localhost:$LOAD_BALANCER_PORT"
        echo "Health check: http://localhost:$LOAD_BALANCER_PORT/health"
        echo "Status check: http://localhost:$LOAD_BALANCER_PORT/status"
        exit 1
        ;;
esac 