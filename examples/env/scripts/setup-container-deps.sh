#!/bin/sh
AREAL_PATH=$PWD
cd /sglang
git apply $AREAL_PATH/patch/sglang/v0.4.6.post4.patch
cd $AREAL_PATH

# Package used for calculating math reward
pip install -e evaluation/latex2sympy

# Install AReaL
pip install -e .