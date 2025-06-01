#!/bin/sh
AREAL_PATH=$PWD
cd /sglang
git apply $AREAL_PATH/patch/sglang/v0.4.6.post4.patch
cd $AREAL_PATH

# Install AReaL
pip install -e .