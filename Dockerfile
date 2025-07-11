FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list.d/ubuntu.sources
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list.d/ubuntu.sources
RUN apt update
RUN apt install -y net-tools kmod ccache \
    libibverbs-dev librdmacm-dev ibverbs-utils \
    rdmacm-utils python3-pyverbs opensm ibutils perftest python3-venv

RUN pip3 install -U pip

ENV NVTE_WITH_USERBUFFERS=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

RUN pip uninstall cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml \
    cugraph-pyg lightning_thunder opt_einsum nvfuser  looseversion lightning_utilities -y
RUN pip3 install -U uv nvidia-ml-py pipdeptree importlib_metadata packaging platformdirs typing_extensions wheel zipp

# Things that should be compiled
ENV MAX_JOBS=64
RUN pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.2
RUN git clone https://github.com/garrett4wade/cugae && pip install -e /cugae --no-build-isolation --verbose
RUN git clone -b v0.11.0 https://github.com/NVIDIA/Megatron-LM.git && \
        pip install ./Megatron-LM && rm -rf /Megatron-LM

# flash-attn 2
RUN git clone -b v2.6.3 https://github.com/Dao-AILab/flash-attention && \
    cd /flash-attention && \
    git submodule update --init --recursive && \
    pip uninstall -y flash-attn && \
    pip install . --no-build-isolation
# flash-attn 3
RUN mkdir /flash-attn3 && cd /flash-attn3 && git clone -b v2.7.2 https://github.com/Dao-AILab/flash-attention && \
    pip install -v /flash-attn3/flash-attention/hopper/ && \
    mkdir -p /usr/local/lib/python3.12/dist-packages/flashattn_hopper && \
    wget -P /usr/local/lib/python3.12/dist-packages/flashattn_hopper \
         https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py && \
    python -c "import flashattn_hopper; import torch; print(torch.__version__)"

# sglang depends on flash-infer
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 9.0a"
RUN pip install -U setuptools
RUN git clone --recursive -b v0.2.5 https://github.com/flashinfer-ai/flashinfer && \
    FLASHINFER_ENABLE_AOT=1 pip install --no-build-isolation --verbose /flashinfer && \
        rm -rf /flashinfer

# sglang
ENV SGL_KERNEL_ENABLE_BF16=1 SGL_KERNEL_ENABLE_FP8=1 SGL_KERNEL_ENABLE_FP4=0
ENV SGL_KERNEL_ENABLE_SM100A=0 SGL_KERNEL_ENABLE_SM90A=1 UV_CONCURRENT_BUILDS=16
RUN git clone -b v0.4.6.post4 https://github.com/sgl-project/sglang.git && \
    cd /sglang/sgl-kernel && make build && \
    pip install /sglang/sgl-kernel/ --force-reinstall --no-build-isolation && \
    cd /sglang && pip3 install -e "python[all]" --no-deps
# sglang dependencies
RUN pip install aiohttp requests tqdm numpy IPython setproctitle \
    compressed-tensors datasets decord fastapi hf_transfer huggingface_hub \
    interegular "llguidance>=0.7.11,<0.8.0" modelscope ninja orjson packaging \
    pillow "prometheus-client>=0.20.0" psutil pydantic pynvml python-multipart \
    "pyzmq>=25.1.2" "soundfile==0.13.1" "torchao>=0.7.0" "transformers==4.51.1" \
    uvicorn uvloop "xgrammar==0.1.17" cuda-python "outlines>=0.0.44,<=0.1.11" \
    partial_json_parser einops jsonlines matplotlib pandas sentence_transformers \
    accelerate peft

# vllm, some quantization dependencies required by sglang
RUN git clone -b v0.8.4 --depth=1 https://github.com/vllm-project/vllm.git /vllm && \
        git config --global http.version HTTP/1.1 && cd /vllm && \
    python3 use_existing_torch.py && \
    pip3 install -r requirements/build.txt  && \
    MAX_JOBS=64 pip install -v . --no-build-isolation  && \
    rm -rf /vllm

# AReaL and dependencies
RUN git clone https://code.alipay.com/inclusionAI/AReaL && \
    pip install -e /AReaL
