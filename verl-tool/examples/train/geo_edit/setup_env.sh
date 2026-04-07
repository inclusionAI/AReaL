#!/usr/bin/env bash
set -e

echo "=== ChartQA RL Environment Setup ==="
echo "Run this on BOTH head and worker nodes."

pip install "vllm<=0.11.0" fire colorlog iopath ftfy "pyzmq==26.2.0" \
    qwen_omni_utils timeout-decorator paddlepaddle-gpu paddlex

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
VERL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/verl-tool"
AREAL_ROOT="/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL"

cat > "$SITE/verl_tool_paths.pth" << PTH
$VERL_ROOT
$VERL_ROOT/verl
$AREAL_ROOT
PTH
echo "Created $SITE/verl_tool_paths.pth"

python3 -c "
import json
for m in ['qwen3vl8b-instruct-chartqa-1third', 'qwen3vl8b-thinking-chartqa-1third']:
    p = f'/storage/openpsi/models/lcy_image_edit/sft_workspace/{m}/tokenizer_config.json'
    try:
        with open(p) as f: c = json.load(f)
        ets = c.get('extra_special_tokens', [])
        if isinstance(ets, list):
            c['extra_special_tokens'] = {t: t for t in ets}
            with open(p, 'w') as f: json.dump(c, f, indent=2, ensure_ascii=False)
            print(f'{m}: tokenizer_config fixed')
    except FileNotFoundError:
        print(f'{m}: not found, skipping')

ref_path = '/storage/openpsi/models/Qwen3-VL-8B-Instruct/config.json'
try:
    ref = json.load(open(ref_path))
    rope = ref.get('text_config', {}).get('rope_scaling')
    for m in ['qwen3vl8b-instruct-chartqa-1third', 'qwen3vl8b-thinking-chartqa-1third']:
        p = f'/storage/openpsi/models/lcy_image_edit/sft_workspace/{m}/config.json'
        try:
            c = json.load(open(p))
            if c.get('rope_scaling') is None:
                c['rope_scaling'] = rope
                if 'text_config' in c and isinstance(c['text_config'], dict):
                    c['text_config']['rope_scaling'] = rope
                json.dump(c, open(p, 'w'), indent=2, ensure_ascii=False)
                print(f'{m}: rope_scaling fixed')
        except FileNotFoundError:
            print(f'{m}: not found, skipping')
except FileNotFoundError:
    print('Reference Qwen3-VL-8B-Instruct not found, skipping rope_scaling fix')
"

echo ""
echo "=== Setup complete ==="
echo "After running on both nodes, kill idle Ray workers:"
echo "  ps aux | grep 'ray::IDLE' | grep -v grep | awk '{print \$2}' | xargs kill -9"
