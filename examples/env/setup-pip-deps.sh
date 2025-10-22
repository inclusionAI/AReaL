#!/bin/bash
# basic dependencies
pip install uv
uv pip install -U pip
uv pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg
uv pip install torch==2.8.0 torchaudio torchvision "deepspeed>=0.17.2" pynvml
uv pip install flashinfer-python==0.3.1 --no-build-isolation
uv pip install "sglang[all]==0.5.2" --prerelease=allow
uv pip install megatron-core==0.13.1 nvidia-ml-py
# NOTE: To use megatron training backend with transformers engine, 
# you need to install flash-attn<=2.8.1, which requires compilation with torch==2.8.0.
uv pip install "flash-attn==2.8.3" --no-build-isolation
uv pip install vllm==0.10.2

# Package used for calculating math reward
uv pip install -e evaluation/latex2sympy
# Install AReaL in upgrade mode to ensure package version dependencies are met. 
uv pip install --prerelease=allow -U -e .[dev]

pip uninstall transformers -y
uv pip install transformers==4.56.1
