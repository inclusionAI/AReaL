#!/bin/bash
# basic dependencies
pip install -U pip
pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install torch==2.8.0 torchaudio torchvision "deepspeed>=0.17.2" pynvml
pip install flashinfer-python==0.3.1 --no-build-isolation
pip install "sglang[all]>=0.5.2" 
pip install megatron-core==0.13.1 nvidia-ml-py
pip install "flash-attn<=2.8.1" --no-build-isolation
pip install vllm==0.10.2

# Package used for calculating math reward
pip install -e evaluation/latex2sympy
# Install AReaL
pip install -e .[dev]
