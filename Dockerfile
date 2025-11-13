FROM lmsysorg/sglang:v0.5.5.post1-cu129-amd64 AS base

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    net-tools \
    kmod \
    ccache \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-utils \
    rdmacm-utils \
    python3-pyverbs \
    opensm \
    ibutils \
    perftest \
    python3-venv \
    tmux \
    lsof \
    nvtop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install uv
RUN pip install -U pip "setuptools<80,>=77.0.3" uv

# Environment variables for build configuration
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 9.0a"
ENV MAX_JOBS=32

##############################################################
# The following block is adapted from slime's Dockerfile
# https://github.com/THUDM/slime/blob/ebf16c57c223d6f1f66ef89177d5e27938c6caaf/docker/Dockerfile

# Install torch memory saver
RUN uv pip install --no-build-isolation --system --no-cache-dir --force-reinstall \
    git+https://github.com/fzyzcjy/torch_memory_saver.git

# Install grouped_gemm
RUN uv pip install --no-build-isolation --system \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# Install apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
    pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git

# Install transformer engine (with --no-deps to avoid installing torch and torch-extensions)
RUN pip install nvidia-mathdx pybind11 \
    && pip -v install --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Install flash attention (v2.8.1 is the latest version that megatron supports)
RUN uv pip -v install flash-attn==2.8.1 --no-build-isolation --system
##############################################################

# Install flash-attn3
RUN git clone https://github.com/Dao-AILab/flash-attention -b v2.8.1 \
    && uv pip install -v /flash-attention/hopper/ --no-build-isolation --system \
    && mkdir -p /usr/local/lib/python3.12/dist-packages/flash_attn_3/ \
    && cp /flash-attention/hopper/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/

# Misc fixes
RUN uv pip uninstall pynvml --system
# Update setuptools to fix a wandb bug
# Install nvidia-ml-py to replace pynvml
RUN uv pip install -U setuptools nvidia-ml-py --system

# Remove libcudnn9 to avoid conflicts with torch
RUN apt-get --purge remove -y --allow-change-held-packages libcudnn9* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy AReaL source code from build context (checked out by CI)
COPY . /AReaL

# Install AReaL from local source
RUN cd /AReaL \
    && uv pip install -e ".[dev,docs]" --system