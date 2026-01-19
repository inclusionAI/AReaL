#/bin/bash

model_list=(
    claude-3-7-sonnet-20250219
    claude-3-5-haiku-20241022
    claude-sonnet-4-20250514
    claude-opus-4-1-20250805
    claude-sonnet-4-5-20250929
    claude-haiku-4-5-20251001
    claude-opus-4-5-20251101
)

for model in ${model_list[@]}; do 
    echo $model 
    sleep 5
    MODEL=$model python reasoning_analyze/solve_problem_claude.py
done
