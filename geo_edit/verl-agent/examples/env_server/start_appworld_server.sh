#!/bin/bash

# Configuration parameters
train_batch_size=16
val_batch_size=168
group_size=8

# Start services
current_port=7000

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate appworld

# Stop existing appworld services
ps aux | grep "appworld serve" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 5

# Calculate total number of instances needed
total_instances=$((train_batch_size * group_size + val_batch_size))

# Port file (.ports extension for easy gitignore)
port_file="appworld_ports.ports"
> "$port_file"  # Clear port file

# Function to check if port is in use
check_port() {
    local port=$1
    # Use netstat or ss to check if port is in use
    if command -v ss >/dev/null 2>&1; then
        ss -tuln | grep -q ":$port "
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep -q ":$port "
    else
        # If neither is available, try connecting to port
        timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null
    fi
}

echo "Starting $total_instances AppWorld service instances..."
echo "Port usage will be saved to: $port_file"

started_count=0

while [ $started_count -lt $total_instances ]; do
    if check_port $current_port; then
        echo "Port $current_port is already in use, skipping..."
        current_port=$((current_port + 1))
        continue
    fi
    
    echo "Starting service on port $current_port..."
    if appworld serve environment --port $current_port & then
        # Write successfully started port to file
        echo "$current_port" >> "$port_file"
        echo "Service successfully started on port: $current_port"
        started_count=$((started_count + 1))
    else
        echo "Failed to start on port $current_port, trying next port..."
    fi
    
    current_port=$((current_port + 1))
    
    # Prevent infinite loop by setting maximum port range
    if [ $current_port -gt 9000 ]; then
        echo "Error: Reached maximum port range (9000), cannot start more services"
        break
    fi
done