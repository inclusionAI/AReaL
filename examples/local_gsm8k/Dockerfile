# AReaL GRPO Docker Environment
# Based on AReaL's official runtime image
# Supports both GPU (CUDA) and CPU-only modes

# Use AReaL's official base image (includes SGLang, CUDA, dependencies)
FROM ghcr.io/inclusionai/areal-runtime:v0.3.4

WORKDIR /workspace

# Install additional utilities
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy local files (if building from this directory)
# The main code will be mounted as a volume at runtime
COPY requirements-docker.txt /tmp/requirements-docker.txt

# Install any additional Python dependencies if needed
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt || echo "No additional requirements"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_MODE=disabled

# Default command
CMD ["/bin/bash"]

