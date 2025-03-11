FROM nvcr.io/nvidia/pytorch:24.07-py3 AS gpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update
RUN apt install -y net-tools \
    libibverbs-dev librdmacm-dev ibverbs-utils \
    rdmacm-utils python3-pyverbs opensm ibutils perftest

RUN pip3 install -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# set environment variables for building transformer engine
ENV NVTE_WITH_USERBUFFERS=1 NVTE_FRAMEWORK=pytorch MAX_JOBS=8 MPI_HOME=/usr/local/mpi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

RUN pip3 install deepspeed==0.14.0 megatron==0.6.0

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && rm /requirements.txt

# We don't use TransformerEngine's flash-attn integration, so it's okay to disrespect dependencies
RUN pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.8 --no-deps --no-build-isolation
RUN pip3 install flash-attn==2.4.2 --no-build-isolation
# Install grouped_gemm for MoE acceleration
RUN pip3 install git+https://github.com/tgale96/grouped_gemm.git@v0.1.4 --no-build-isolation --no-deps

COPY . /AReaL
RUN REAL_CUDA=1 pip3 install -e /AReaL --no-build-isolation
WORKDIR /AReaL

RUN git clone --depth=1 -b v0.6.3.post1 https://github.com/vllm-project/vllm.git /vllm
RUN apt install kmod ccache -y
RUN cd /vllm && \
    python3 use_existing_torch.py && \
    pip3 install -r requirements-build.txt && \
    MAX_JOBS=64 pip3 install -e . --no-build-isolation
RUN pip3 install opencv-python-headless==4.5.4.58

RUN apt-get update && apt-get install -y python3.10-venv

RUN git clone --depth=1 https://github.com/QwenLM/Qwen2.5-Math /qwen2_5-math && mv /qwen2_5-math/evaluation/latex2sympy /latex2sympy
RUN python3 -m venv /sympy
RUN /sympy/bin/pip install /latex2sympy
RUN /sympy/bin/pip install regex numpy tqdm datasets python_dateutil sympy==1.12 antlr4-python3-runtime==4.11.1 word2number Pebble timeout-decorator prettytable
