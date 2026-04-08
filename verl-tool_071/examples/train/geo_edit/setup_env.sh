#!/usr/bin/env bash
# Run this script on EACH node to set up the environment.
# Usage: source setup_env.sh
#   or:  . setup_env.sh
set -e
cd verl-tool_071
pip install -e verl/ --no-deps
pip install -e . --no-deps
# Auto-import verl_tool in every Python process to register reward managers
python3 -c "import site; open(site.getsitepackages()[0]+'/verl_tool_init.pth','w').write('import verl_tool\n')"
pip install vllm==0.17.0
# Install dependencies
pip install timm fire iopath ftfy tensordict codetiming qwen_omni_utils nvitop httptools colorlog
pip install --upgrade numpy
