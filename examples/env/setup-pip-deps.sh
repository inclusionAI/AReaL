#!/bin/bash
# basic dependencies
pip install uv
uv pip install -U pip
uv pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg
uv pip install -U transformers==4.56.1
python3 -c "from transformers import AutoProcessor"
uv pip install -U torch==2.8.0 torchaudio torchvision "deepspeed>=0.17.2" pynvml
python3 -c "from transformers import AutoProcessor"
uv pip install -U flashinfer-python==0.3.1 --no-build-isolation
python3 -c "from transformers import AutoProcessor"
uv pip install -U "sglang[all]==0.5.2"
python3 -c "from transformers import AutoProcessor"
uv pip install -U megatron-core==0.13.1 nvidia-ml-py
python3 -c "from transformers import AutoProcessor"
# NOTE: To use megatron training backend with transformers engine, 
# you need to install flash-attn<=2.8.1, which requires compilation with torch==2.8.0.
uv pip install -U "flash-attn==2.8.3" --no-build-isolation
python3 -c "from transformers import AutoProcessor"
uv pip install -U vllm==0.10.2
python3 -c "from transformers import AutoProcessor"

# Package used for calculating math reward
uv pip install -U -e evaluation/latex2sympy
python3 -c "from transformers import AutoProcessor"
# Install AReaL in upgrade mode to ensure package version dependencies are met. 
uv pip install -U -e .[dev]
python3 -c "from transformers import AutoProcessor"

# uv pip install -U sphinx-nefertiti
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U sphinx
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "build>=1.2.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "wheel>=0.43.0"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "distro-info>=1.0"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "python-debian>=0.1.49"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U huggingface_hub
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U datasets
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U accelerate
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U ninja
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U matplotlib
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U ipython
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U h5py
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U nltk
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U sentencepiece
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U wandb
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U tensorboardx
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U blosc
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U colorama
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U colorlog
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U einops
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "hydra-core==1.3.2"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U matplotlib
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U numba
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U packaging
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pandas
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "pybind11>=2.10.0"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U psutil
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pynvml
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pytest
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U PyYAML
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pyzmq
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "ray[default]"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U redis
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U scipy
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U seaborn
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U tqdm
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "networkx==3.3"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U matplotlib
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U tabulate
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U aiofiles
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pydantic
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "clang-format==19.1.7"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U ninja
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U paramiko
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "megatron-core==0.13.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "mbridge==0.13.0"
# python3 -c "from transformers import AutoProcessor"
# # To eliminate security risks
# uv pip install -U torch==2.8.0
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "ruff==0.14.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "cookiecutter>2.1.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U asyncio
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "aiohttp>=3.11.10"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "httpx>=0.28.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U etcd3
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U rich
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "orjson>=3.10.16"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U flask
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "setuptools<80,>=77.0.3"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U func_timeout
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U jupyter-book
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U lark
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "uvloop>=0.21.0"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "uvicorn>=0.34.2"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "fastapi>=0.115.12"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U regex
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U python_dateutil
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U word2number
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U Pebble
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U timeout-decorator
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U prettytable
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "gymnasium>=1.1.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "swanboard==0.1.9b1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "swanlab[dashboard]==0.6.12"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U torchdata
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "autoflake==2.3.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U tensordict
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "deepspeed>=0.17.2"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U pybase64
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U msgspec
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "transformers==4.56.1"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "openai==1.99.6"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U sh
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U peft
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U "mdformat==0.7.17"
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U mdformat-gfm
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U mdformat-tables
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U mdformat-frontmatter
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U setproctitle
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U qwen_agent
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U dotenv
# python3 -c "from transformers import AutoProcessor"
# uv pip install -U json5
# python3 -c "from transformers import AutoProcessor"

# uv pip install -e .[dev] --no-deps --no-build-isolation
# python3 -c "from transformers import AutoProcessor"