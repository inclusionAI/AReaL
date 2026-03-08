# Supports sglang and vllm variants via the VARIANT build argument.
# VARIANT is declared early so each variant gets the correct torch version
# and C++ extensions compiled against it.
#
# Usage:
#   docker build -t areal-runtime:dev-sglang .                          # default (sglang)
#   docker build --build-arg VARIANT=vllm -t areal-runtime:dev-vllm .    # vllm variant
#   docker build --build-arg SKIP_CPP=true -t areal-runtime:dev-quick .  # skip C++ builds

FROM lmsysorg/sglang:v0.5.9-cu129-amd64-runtime

# Inference backend selector: sglang (default) or vllm
# Declared early so torch version and C++ builds match the chosen backend.
ARG VARIANT=sglang

# Set to "true" to skip all C++ compilations (STAGE 2) for quick validation.
# The resulting image will NOT have GPU-accelerated extensions but builds much faster.
ARG SKIP_CPP=false

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    net-tools \
    unzip \
    kmod \
    ccache \
    cmake \
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
    rsync \
    dnsutils \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install uv
RUN pip install -U pip uv

WORKDIR /AReaL

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Environment variables for build configuration
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 9.0a"
ENV MAX_JOBS=32

# Set VIRTUAL_ENV so uv pip install targets the venv created below
ENV VIRTUAL_ENV=/AReaL/.venv

##############################################################
# STAGE 1: Install base torch FIRST
# Torch version depends on VARIANT (sglang=2.9.1, vllm=2.10.0)
##############################################################

# Create venv and install torch with CUDA support
# Version is variant-specific: sglang pins 2.9.1, vllm pins 2.10.0
RUN uv venv $VIRTUAL_ENV \
    && if [ "$VARIANT" = "vllm" ]; then TORCH_VER="2.10.0"; else TORCH_VER="2.9.1"; fi \
    && uv pip install --index-url https://download.pytorch.org/whl/cu129 \
    "torch==${TORCH_VER}+cu129" "torchaudio" "torchvision"

RUN uv pip install "setuptools>=77.0.3,<80" pybind11 nvidia-mathdx

##############################################################
# STAGE 2: Install heavy C++ dependencies BEFORE uv sync
# These require only torch and rarely change.
# Moving these BEFORE uv sync prevents recompilation when
# pyproject.toml/uv.lock changes (C++ packages stay cached).
#
# Set SKIP_CPP=true to skip all compilations for quick builds.
##############################################################

# Install torch memory saver
RUN if [ "$SKIP_CPP" != "true" ]; then \
    uv pip install --no-build-isolation --no-cache-dir --force-reinstall \
    git+https://github.com/fzyzcjy/torch_memory_saver.git \
    ; fi

# Install grouped_gemm (for MoE models)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    uv pip install --no-build-isolation \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 \
    ; fi

# Install apex (NVIDIA apex for mixed precision training)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git \
    ; fi

# Install transformer engine (for FP8 training)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    uv pip -v install --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@stable \
    ; fi

# Install flash-attn-3
ENV CUDA_HOME=/usr/local/cuda
RUN if [ "$SKIP_CPP" != "true" ]; then \
    git clone https://github.com/Dao-AILab/flash-attention -b v2.8.3 /flash-attention \
    && cd /flash-attention/hopper/ && python3 setup.py sdist bdist_wheel \
    && uv pip install --no-build-isolation --no-cache-dir /flash-attention/hopper/dist/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl \
    && mkdir -p $VIRTUAL_ENV/lib/python3.12/site-packages/flash_attn_3/ \
    && cp /flash-attention/hopper/flash_attn_interface.py $VIRTUAL_ENV/lib/python3.12/site-packages/flash_attn_3/ \
    && touch $VIRTUAL_ENV/lib/python3.12/site-packages/flash_attn_3/__init__.py \
    && rm -rf /flash-attention \
    ; fi

# FlashMLA (Multi-head Latent Attention for DeepSeek-V3)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    git clone https://github.com/deepseek-ai/FlashMLA.git /flash-mla \
    && cd /flash-mla \
    && git submodule update --init --recursive \
    && uv pip install -v . --no-build-isolation \
    && rm -rf /flash-mla \
    ; fi

# DeepGEMM (FP8 GEMM library for DeepSeek-V3)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    git clone https://github.com/deepseek-ai/DeepGEMM /DeepGEMM \
    && cd /DeepGEMM \
    && git submodule update --init --recursive \
    && uv pip install . --no-build-isolation \
    && rm -rf /DeepGEMM \
    ; fi

# DeepEP (Expert Parallelism communication library for MoE)
# Note: TORCH_CUDA_ARCH_LIST="9.0" enables SM90 features and aggressive PTX instructions
# The NVSHMEM path is auto-detected from nvidia.nvshmem module installed above
RUN if [ "$SKIP_CPP" != "true" ]; then \
    git clone https://github.com/deepseek-ai/DeepEP /DeepEP \
    && cd /DeepEP \
    && TORCH_CUDA_ARCH_LIST="9.0 9.0a" uv pip install -v . --no-build-isolation \
    && rm -rf /DeepEP \
    ; fi

# flash-linear-attention (pure Triton kernels, no CUDA extensions)
RUN if [ "$SKIP_CPP" != "true" ]; then \
    git clone https://github.com/fla-org/flash-linear-attention /flash-linear-attention \
    && cd /flash-linear-attention \
    && uv pip install -v . \
    && rm -rf /flash-linear-attention \
    ; fi

##############################################################
# STAGE 2.5: Install Node.js and npm-based tools
##############################################################

# Install Node.js via fnm and Claude Code
ENV FNM_DIR=/root/.fnm
ENV NODE_VERSION=24.13.0
ENV PATH="$FNM_DIR/aliases/default/bin:/root/.local/bin:$PATH"
RUN curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$FNM_DIR" --skip-shell \
    && eval "$($FNM_DIR/fnm env --shell bash)" \
    && $FNM_DIR/fnm install $NODE_VERSION \
    && $FNM_DIR/fnm default $NODE_VERSION \
    && npm install -g npm@latest \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && curl -fsSL https://opencode.ai/install | bash \
    && npm install -g @openai/codex \
    && npm install -g @google/gemini-cli

##############################################################
# STAGE 3: Install project dependencies from pyproject.toml
# Changes to pyproject.toml/uv.lock will invalidate from here
# but C++ packages above remain cached.
# Using `uv pip install` instead of `uv sync` to avoid removing
# C++ packages that aren't in uv.lock.
##############################################################

# Install the project's dependencies (not the project itself)
# This adds packages without removing unlisted ones (like our C++ packages)
# VARIANT selects the inference backend (sglang or vllm)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --no-build-isolation -r pyproject.toml --extra ${VARIANT} --extra cuda-train --group dev

##############################################################
# STAGE 4: Misc fixes and final setup
##############################################################

# Misc fixes
RUN uv pip uninstall pynvml
# Update setuptools to fix a wandb bug
# Install nvidia-ml-py to replace pynvml
RUN uv pip install -U setuptools nvidia-ml-py

# Remove libcudnn9 to avoid conflicts with torch
RUN apt-get --purge remove -y --allow-change-held-packages libcudnn9* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

##############################################################
# STAGE 5: Install AReaL from local source
# This is last so code changes don't invalidate C++ builds
##############################################################

# Copy AReaL source code from build context (checked out by CI)
COPY . /AReaL

# Install areal package in editable mode without dependencies
# Using pip install instead of uv sync to avoid overwriting C++ packages
RUN uv pip install --no-deps -e /AReaL

# Place executables in the environment at the front of the path
ENV PATH="/AReaL/.venv/bin:$PATH"

# Reset entrypoint (some base images set custom entrypoints; this ensures /bin/bash)
ENTRYPOINT []
CMD ["/bin/bash"]
