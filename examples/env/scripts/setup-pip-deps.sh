#!/bin/bash
# basic dependencies
pip install -U pip
pip uninstall torch deepspeed flash-attn pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install "sglang[all]==0.4.6.post4" 
pip install megatron-core==0.11.0 nvidia-ml-py
pip install git+https://github.com/garrett4wade/cugae --no-build-isolation --verbose
pip install "flash-attn<=2.7.3" --no-build-isolation

# Package used for calculating math reward
pip install -e evaluation/latex2sympy

# Install an editable sglang
rm -rf ./sglang
git clone -b v0.4.6.post4 https://github.com/sgl-project/sglang
AREAL_PATH=$PWD
cd sglang
git apply ../patch/sglang/v0.4.6.post4.patch
pip install -e "python[all]" --no-deps
cd $AREAL_PATH

# Install AReaL
pip install -e .
