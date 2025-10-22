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
uv pip install sphinx
uv pip install "build>=1.2.1"
uv pip install "wheel>=0.43.0"
uv pip install "distro-info>=1.0"
uv pip install "python-debian>=0.1.49"
uv pip install huggingface_hub
uv pip install datasets
uv pip install accelerate
uv pip install ninja
uv pip install matplotlib
uv pip install ipython
uv pip install h5py
uv pip install nltk
uv pip install sentencepiece
uv pip install wandb
uv pip install tensorboardx
uv pip install blosc
uv pip install colorama
uv pip install colorlog
uv pip install einops
uv pip install "hydra-core==1.4.0.dev1"
uv pip install matplotlib
uv pip install numba
uv pip install packaging
uv pip install pandas
uv pip install "pybind11>=2.10.0"
uv pip install psutil
uv pip install pynvml
uv pip install pytest
uv pip install PyYAML
uv pip install pyzmq
uv pip install "ray[default]"
python -c "from transformers import AutoProcessor"
uv pip install redis
uv pip install scipy
uv pip install seaborn
uv pip install tqdm
uv pip install "networkx==3.3"
uv pip install matplotlib
uv pip install tabulate
uv pip install aiofiles
uv pip install pydantic
uv pip install "clang-format==19.1.7"
uv pip install ninja
uv pip install paramiko
uv pip install "megatron-core==0.13.1"
uv pip install "mbridge==0.13.0"
# To eliminate security risks
uv pip install "torch>2.0.0"
uv pip install "ruff==0.14.1"
uv pip install "cookiecutter>2.1.1"
uv pip install asyncio
uv pip install "aiohttp>=3.11.10"
uv pip install "httpx>=0.28.1"
uv pip install etcd3
uv pip install rich
uv pip install "orjson>=3.10.16"
uv pip install flask
uv pip install "setuptools<80,>=77.0.3"
uv pip install func_timeout
uv pip install jupyter-book
uv pip install lark
uv pip install "uvloop>=0.21.0"
uv pip install "uvicorn>=0.34.2"
uv pip install "fastapi>=0.115.12"
uv pip install regex
uv pip install python_dateutil
uv pip install word2number
uv pip install Pebble
uv pip install timeout-decorator
uv pip install prettytable
uv pip install "gymnasium>=1.1.1"
uv pip install "swanlab[dashboard]"
uv pip install torchdata
uv pip install "autoflake==2.3.1"
uv pip install tensordict
uv pip install "deepspeed>=0.17.2"
uv pip install pybase64
uv pip install msgspec
uv pip install "transformers==4.56.1"
uv pip install "openai==1.99.6"
uv pip install sh
uv pip install peft
uv pip install "mdformat==0.7.17"
uv pip install mdformat-gfm
uv pip install mdformat-tables
uv pip install mdformat-frontmatter
uv pip install setproctitle
uv pip install qwen_agent
uv pip install dotenv
uv pip install json5