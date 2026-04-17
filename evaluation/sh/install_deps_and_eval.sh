#!/bin/bash
# Users should run this script under AReaL directory

/usr/bin/python3 -m pip install -e evaluation/latex2sympy/
/usr/bin/python3 -m pip install -r evaluation/requirements.txt

cd evaluation && /usr/bin/python3 eval_and_aggregate.py --model_path $1 --output_path $2 --data_names $3 --max_gen_tokens $4 --prompt_type $5
