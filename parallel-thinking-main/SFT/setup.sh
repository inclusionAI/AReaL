#!/bin/bash

# Quick setup script for parallel thinking fine-tuning

echo "=== Setting up Parallel Thinking Fine-tuning Environment ==="

# Create data directory
mkdir -p data

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Check if output.jsonl exists
if [ ! -f "../data_processing/output.jsonl" ]; then
    echo "Warning: ../data_processing/output.jsonl not found!"
    echo "Please ensure your training data is available before running training."
    exit 1
fi

# Convert data
echo "Converting training data..."
python data_converter.py

# Check if LLaMA Factory should be installed
if [ ! -d "LLaMA-Factory" ]; then
    echo "Cloning LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]"
    cd ..
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python train_parallel_thinking.py"
echo "2. After training, test with: python test_model.py"
