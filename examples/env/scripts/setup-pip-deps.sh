#!/bin/bash
# basic dependencies
pip install -U pip
pip uninstall flash-attn pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install "sglang[all]==0.4.6.post4" 
pip install megatron-core==0.11.0 nvidia-ml-py
pip install git+https://github.com/garrett4wade/cugae --no-build-isolation --verbose
pip install flash-attn --no-build-isolation

# the sympy virtual env for reward computation
pip install virtualenv
python3 -m venv ./sympy
# equivalent to install `./evaluation/latex2sympy` in the sympy virtual env
./sympy/bin/pip install git+https://github.com/QwenLM/Qwen2.5-Math.git#subdirectory=evaluation/latex2sympy
./sympy/bin/pip install regex numpy tqdm datasets python_dateutil sympy==1.12 antlr4-python3-runtime==4.11.1 word2number Pebble timeout-decorator prettytable

# Install an editable sglang
git clone -b v0.4.6.post4 https://github.com/sgl-project/sglang
AREAL_PATH=$PWD
cd sglang
git apply ../patch/sglang/v0.4.6.post4.patch
pip install -e "python[all]" --no-deps
cd $AREAL_PATH

# Install AReaL
pip install -e .
