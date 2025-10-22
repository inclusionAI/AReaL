#!/bin/bash
# basic dependencies
pip install uv
uv pip install -U pip
uv pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg
uv pip install transformers==4.56.1
python -c "from transformers import AutoProcessor"
uv pip install torch==2.8.0 torchaudio torchvision "deepspeed>=0.17.2" pynvml
python -c "from transformers import AutoProcessor"
uv pip install flashinfer-python==0.3.1 --no-build-isolation
python -c "from transformers import AutoProcessor"
uv pip install "sglang[all]==0.5.2" --prerelease=allow
python -c "from transformers import AutoProcessor"
uv pip install megatron-core==0.13.1 nvidia-ml-py
python -c "from transformers import AutoProcessor"
# NOTE: To use megatron training backend with transformers engine, 
# you need to install flash-attn<=2.8.1, which requires compilation with torch==2.8.0.
uv pip install "flash-attn==2.8.3" --no-build-isolation
python -c "from transformers import AutoProcessor"
uv pip install vllm==0.10.2
python -c "from transformers import AutoProcessor"

# Package used for calculating math reward
uv pip install -e evaluation/latex2sympy
python -c "from transformers import AutoProcessor"
# Install AReaL in upgrade mode to ensure package version dependencies are met. 
# uv pip install --prerelease=allow -U -e .[dev]

# pip uninstall transformers -y
# pip install transformers==4.56.1

uv pip install sphinx-nefertiti
python -c "from transformers import AutoProcessor"
uv pip install sphinx
python -c "from transformers import AutoProcessor"
uv pip install "build>=1.2.1"
python -c "from transformers import AutoProcessor"
uv pip install "wheel>=0.43.0"
python -c "from transformers import AutoProcessor"
uv pip install "distro-info>=1.0"
python -c "from transformers import AutoProcessor"
uv pip install "python-debian>=0.1.49"
python -c "from transformers import AutoProcessor"
uv pip install huggingface_hub
python -c "from transformers import AutoProcessor"
uv pip install datasets
python -c "from transformers import AutoProcessor"
uv pip install accelerate
python -c "from transformers import AutoProcessor"
uv pip install ninja
python -c "from transformers import AutoProcessor"
uv pip install matplotlib
python -c "from transformers import AutoProcessor"
uv pip install ipython
python -c "from transformers import AutoProcessor"
uv pip install h5py
python -c "from transformers import AutoProcessor"
uv pip install nltk
python -c "from transformers import AutoProcessor"
uv pip install sentencepiece
python -c "from transformers import AutoProcessor"
uv pip install wandb
python -c "from transformers import AutoProcessor"
uv pip install tensorboardx
python -c "from transformers import AutoProcessor"
uv pip install blosc
python -c "from transformers import AutoProcessor"
uv pip install colorama
python -c "from transformers import AutoProcessor"
uv pip install colorlog
python -c "from transformers import AutoProcessor"
uv pip install einops
python -c "from transformers import AutoProcessor"
uv pip install "hydra-core==1.4.0.dev1"
python -c "from transformers import AutoProcessor"
uv pip install matplotlib
python -c "from transformers import AutoProcessor"
uv pip install numba
python -c "from transformers import AutoProcessor"
uv pip install packaging
python -c "from transformers import AutoProcessor"
uv pip install pandas
python -c "from transformers import AutoProcessor"
uv pip install "pybind11>=2.10.0"
python -c "from transformers import AutoProcessor"
uv pip install psutil
python -c "from transformers import AutoProcessor"
uv pip install pynvml
python -c "from transformers import AutoProcessor"
uv pip install pytest
python -c "from transformers import AutoProcessor"
uv pip install PyYAML
python -c "from transformers import AutoProcessor"
uv pip install pyzmq
python -c "from transformers import AutoProcessor"
uv pip install "ray[default]"
python -c "from transformers import AutoProcessor"
uv pip install redis
python -c "from transformers import AutoProcessor"
uv pip install scipy
python -c "from transformers import AutoProcessor"
uv pip install seaborn
python -c "from transformers import AutoProcessor"
uv pip install tqdm
python -c "from transformers import AutoProcessor"
uv pip install "networkx==3.3"
python -c "from transformers import AutoProcessor"
uv pip install matplotlib
python -c "from transformers import AutoProcessor"
uv pip install tabulate
python -c "from transformers import AutoProcessor"
uv pip install aiofiles
python -c "from transformers import AutoProcessor"
uv pip install pydantic
python -c "from transformers import AutoProcessor"
uv pip install "clang-format==19.1.7"
python -c "from transformers import AutoProcessor"
uv pip install ninja
python -c "from transformers import AutoProcessor"
uv pip install paramiko
python -c "from transformers import AutoProcessor"
uv pip install "megatron-core==0.13.1"
python -c "from transformers import AutoProcessor"
uv pip install "mbridge==0.13.0"
python -c "from transformers import AutoProcessor"
# To eliminate security risks
uv pip install "torch>2.0.0"
python -c "from transformers import AutoProcessor"
uv pip install "ruff==0.14.1"
python -c "from transformers import AutoProcessor"
uv pip install "cookiecutter>2.1.1"
python -c "from transformers import AutoProcessor"
uv pip install asyncio
python -c "from transformers import AutoProcessor"
uv pip install "aiohttp>=3.11.10"
python -c "from transformers import AutoProcessor"
uv pip install "httpx>=0.28.1"
python -c "from transformers import AutoProcessor"
uv pip install etcd3
python -c "from transformers import AutoProcessor"
uv pip install rich
python -c "from transformers import AutoProcessor"
uv pip install "orjson>=3.10.16"
python -c "from transformers import AutoProcessor"
uv pip install flask
python -c "from transformers import AutoProcessor"
uv pip install "setuptools<80,>=77.0.3"
python -c "from transformers import AutoProcessor"
uv pip install func_timeout
python -c "from transformers import AutoProcessor"
uv pip install jupyter-book
python -c "from transformers import AutoProcessor"
uv pip install lark
python -c "from transformers import AutoProcessor"
uv pip install "uvloop>=0.21.0"
python -c "from transformers import AutoProcessor"
uv pip install "uvicorn>=0.34.2"
python -c "from transformers import AutoProcessor"
uv pip install "fastapi>=0.115.12"
python -c "from transformers import AutoProcessor"
uv pip install regex
python -c "from transformers import AutoProcessor"
uv pip install python_dateutil
python -c "from transformers import AutoProcessor"
uv pip install word2number
python -c "from transformers import AutoProcessor"
uv pip install Pebble
python -c "from transformers import AutoProcessor"
uv pip install timeout-decorator
python -c "from transformers import AutoProcessor"
uv pip install prettytable
python -c "from transformers import AutoProcessor"
uv pip install "gymnasium>=1.1.1"
python -c "from transformers import AutoProcessor"
uv pip install "swanlab[dashboard]" --prerelease=allow
python -c "from transformers import AutoProcessor"
uv pip install torchdata
python -c "from transformers import AutoProcessor"
uv pip install "autoflake==2.3.1"
python -c "from transformers import AutoProcessor"
uv pip install tensordict
python -c "from transformers import AutoProcessor"
uv pip install "deepspeed>=0.17.2"
python -c "from transformers import AutoProcessor"
uv pip install pybase64
python -c "from transformers import AutoProcessor"
uv pip install msgspec
python -c "from transformers import AutoProcessor"
uv pip install "transformers==4.56.1"
python -c "from transformers import AutoProcessor"
uv pip install "openai==1.99.6"
python -c "from transformers import AutoProcessor"
uv pip install sh
python -c "from transformers import AutoProcessor"
uv pip install peft
python -c "from transformers import AutoProcessor"
uv pip install "mdformat==0.7.17"
python -c "from transformers import AutoProcessor"
uv pip install mdformat-gfm
python -c "from transformers import AutoProcessor"
uv pip install mdformat-tables
python -c "from transformers import AutoProcessor"
uv pip install mdformat-frontmatter
python -c "from transformers import AutoProcessor"
uv pip install setproctitle
python -c "from transformers import AutoProcessor"
uv pip install qwen_agent
python -c "from transformers import AutoProcessor"
uv pip install dotenv
python -c "from transformers import AutoProcessor"
uv pip install json5
python -c "from transformers import AutoProcessor"