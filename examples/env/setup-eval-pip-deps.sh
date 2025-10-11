#/bin/bash
# basic dependencies
pip install -U pip
pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg -y
pip install pynvml nvidia-ml-py
pip install -e evaluation/latex2sympy
pip install vllm==0.10.2 --no-build-isolation
pip install "flash-attn<=2.8.1" --no-build-isolation
pip install -r evaluation/requirements.txt