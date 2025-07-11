#/bin/bash
# basic dependencies
pip install -U pip
pip uninstall deepspeed flash-attn pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install nvidia-ml-py
pip install -e evaluation/latex2sympy
pip install vllm==0.8.5 --no-build-isolation
pip install flash_attn --no-build-isolation
pip install -r evaluation/requirements.txt