FROM lmsysorg/sglang:v0.5.2 as base

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update

RUN apt install -y net-tools kmod ccache \
    libibverbs-dev librdmacm-dev ibverbs-utils \
    rdmacm-utils python3-pyverbs opensm ibutils perftest python3-venv tmux lsof nvtop

RUN pip config set global.index-url https://pypi.antfin-inc.com/simple && pip config set global.extra-index-url "" && pip install -U pip setuptools uv

ENV NVTE_WITH_USERBUFFERS=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 9.0a" MAX_JOBS=64

##############################################################
# The following block is adapted from slime's Dockerfile
# https://github.com/THUDM/slime/blob/ebf16c57c223d6f1f66ef89177d5e27938c6caaf/docker/Dockerfile

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git --no-cache-dir --force-reinstall
RUN pip install ray[default]
RUN pip install httpx[http2] wandb pylatexenc blobfile accelerate "mcp[cli]"

# mbridge
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps
RUN pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# apex
RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git

# transformer engine, we install with --no-deps to avoid installing torch and torch-extensions
RUN pip install pybind11
RUN pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

# flash attn
# the newest version megatron supports is v2.8.1
RUN pip -v install flash-attn==2.8.1
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && \
    git checkout core_v0.13.1 && \
    pip install -e .
##############################################################

# cugae
RUN git clone https://github.com/garrett4wade/cugae && pip install -e /cugae --no-build-isolation --verbose

# AReaL and dependencies
RUN pip install "deepspeed>=0.17.2" accelerate peft sentence_transformers tensordict torchdata

# This is a AReaL with customized pyproject.toml, such that it doesn't require the re-installation of pytorch
# TODO: fix the dependency issue
COPY ./AReaL /AReaL
RUN  cd /AReaL &&  pip install -e evaluation/latex2sympy && pip install --ignore-installed -e .

# misc fix
RUN pip uninstall pynvml -y
RUN pip install -U setuptools nvidia-ml-py
RUN pip install vllm==0.10.2
RUN rm /root/.tmux.conf

# Remove libcudnn9 to avoid conflicts with torch
RUN apt-get --purge remove -y --allow-change-held-packages libcudnn9* libcudnn9-dev* libcudnn9-samples* && \
    apt-get autoremove -y

# flash-attn3
RUN git clone https://github.com/Dao-AILab/flash-attention -b v2.8.1
RUN pip install -v /flash-attention/hopper/
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/flash_attn_3/ && \
    cp /flash-attention/hopper/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/
