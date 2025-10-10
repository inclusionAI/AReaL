#!/bin/bash
# basic dependencies
pip install -U pip
pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 "deepspeed>=0.17.2" pynvml
pip install flashinfer-python==0.2.7.post1 --no-build-isolation
pip install "sglang[all]==0.4.9.post2" 
pip install megatron-core==0.11.0 nvidia-ml-py
pip install git+https://github.com/garrett4wade/cugae --no-build-isolation --verbose
pip install "flash-attn<=2.8.2" --no-build-isolation
pip install vllm==0.10.1

# Package used for calculating math reward
pip install -e evaluation/latex2sympy
# Install AReaL
pip install -e .[dev]
