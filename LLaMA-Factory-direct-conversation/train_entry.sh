#
set -x
# parallel
eai-run -J ag-Qwen3-4b-instruct-parallel \
    llamafactory-cli train \
    train_parallel.yaml \
    model_name_or_path=Qwen/Qwen3-4b-instruct-2507 \
    dataset=parallel_sft \
    output_dir=./models/Qwen3-4b-instruct-parallel &

eai-run -J ag-Qwen3-4b-instruct-sequential \
    llamafactory-cli train \
    train_parallel.yaml \
    model_name_or_path=Qwen/Qwen3-4b-instruct-2507 \
    dataset=sequential_sft \
    output_dir=./models/Qwen3-4b-instruct-sequential &

eai-run -J ag-train_parallel \
    llamafactory-cli train \
    train_parallel.yaml &

eai-run -J ag-train_seq \
    llamafactory-cli train \
    train_seq.yaml &

wait