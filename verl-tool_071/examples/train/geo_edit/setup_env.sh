sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
apt update
pip3 config set global.index-url https://pypi.antfin-inc.com/simple && pip3 config set global.extra-index-url "" && pip install -U pip
cd verl-tool_071/verl
pip3 install -e ".[vllm]"
pip install fire iopath ftfy tensordict codetiming qwen_omni_utils nvitop httptools colorlog