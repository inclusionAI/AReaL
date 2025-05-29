set -ex


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_LOGGING_LEVEL=DEBUG

MODEL_NAME_OR_PATH=$1
# OUTPUT_DIR=$1
MAX_GEN_TOKENS=${2:-4096}
DATA_NAME=${3:-"aime24,aime23"}
PROMPT_TYPE=${4:-"qwen-boxed"}

SPLIT="test"
NUM_TEST_SAMPLE=-1
OUTPUT_DIR=${5:-$MODEL_NAME_OR_PATH}
TASK=${6:-"math"}

# English open datasets
# DATA_NAME="math_500,math,gsm8k,train_amc_aime,aime24,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u ${TASK}_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --max_tokens_per_call=$MAX_GEN_TOKENS \
    --tensor_parallel_size 2 \
    --save_outputs \
    # --overwrite \

