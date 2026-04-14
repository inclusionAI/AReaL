bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh  > logs/qwen_1.5B_math_deep_math_debug_10_turns.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh  > logs/qwen_1.5B_math_deep_math_debug_1_turn_512_64.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh  > logs/qwen_1.5B_math_deep_math_debug_2.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh  > logs/qwen_1.5B_math_deep_math_with_penalty.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math_tdgrpo.sh > logs/qwen_1.5B_math_deep_math_tdgrpo_debug_10_turns.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math_tdgrpo.sh > logs/qwen_1.5B_math_deep_math_tdgspo_debug_10_turns.log 2>&1 &
bash examples/train/search_r1/train.sh > logs/search_r1_debug.log 2>&1 &
bash examples/train/search_r1/train_3b_dapo.sh > logs/search_r1_3b_dapo_debug.log 2>&1 &
bash examples/train/search_r1/train_7b_dapo.sh > logs/search_r1_7b_dapo_debug.log 2>&1 &
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh  > logs/qwen_1.5B_math_deep_math_debug_512_64_gapo.log 2>&1 &
bash examples/train/pixel_reasoner/train_qwen25vl.sh > logs/qwen25vl_pixel_reasoner_debug_4gpu_64mb_8n.log 2>&1 &
bash examples/train/torl/train_drgrpo.sh > logs/qwen_1.5B_math_deep_math_drgrpo_debug.log 2>&1 &
bash examples/train/torl/train_drgrpo.sh > logs/qwen_1.5B_math_deep_math_drgrpo_debug_with_tool_penalty.log 2>&1 &
bash examples/train/search_r1/train_drgrpo.sh > logs/search_r1_drgrpo_debug.log 2>&1 &
bash examples/train/deepsearch/train_4b.sh > logs/deepsearch_4b_debug.log 2>&1 &
bash examples/train/deepsearch/train_8b.sh > logs/deepsearch_8b_debug.log 2>&1 &

bash examples/train/pixel_reasoner/train_qwen25vl.sh > logs/pixel_reasoner_qwen25vl_debug.log 2>&1 &
bash examples/train/pixel_reasoner/train_3b.sh > logs/pixel_reasoner_3b_debug.log 2>&1 &

bash examples/train/skysql/train_7b.sh > logs/skysql_7b_debug.log 2>&1 &

bash examples/train/pixel_reasoner/eval.sh > logs/pixel_reasoner_eval.log 2>&1 &

# add-apt-repository ppa:deki/firejail
# apt-get update
# DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
# apt-get -y install build-essential


sailctl job create verltooldebug -g 2 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --image asia-docker.pkg.dev/sail-tpu-02/images/language/miqaenv:latest --debug
sailctl job create verltool -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug
sailctl job create verltoolmath -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug
sailctl job create verltool80g -g 8 -r 1 -p high --high-vram -f ~/sailctl_high_shm_config.yaml --debug 
sailctl job create verltooldebug -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug 

sailctl job create vtmath15dapo -g 4 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtmath7dapo -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtmath15 -g 4 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtmath7 -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args

sailctl job create vtsrdapo3b -g 4 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtsrdapo7b -g 8 -r 1 -p high --high-vram -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtsr3b -g 4 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args
sailctl job create vtsr7b -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args

sailctl job create vtds8b80g -g 8 -r 1 -p high --high-vram -f ~/sailctl_high_shm_config.yaml --debug --args

sailctl job create vtpr8b -g 8 -r 1 -p high --high-vram -f ~/sailctl_high_shm_config.yaml --debug --args

sailctl job create vtsql7b -g 8 -r 1 -p high -f ~/sailctl_high_shm_config.yaml --debug --args


export HF_TOKEN=
export WANDB_API_KEY=""
export HF_HOME="/home/aiops/jiangdf/.cache/huggingface"
source /home/aiops/jiangdf/Workspace/verl-tool/.venv/bin/activate
cd /home/aiops/jiangdf/Workspace/verl-tool 

bash examples/train/math_tir/train_1.5b_grpo.sh
bash examples/train/math_tir/train_7b_grpo.sh


bash examples/train/torl/train_qwen_7B_math_deep_math_dapo.sh
bash examples/train/torl/train_qwen_1.5B_math_deep_math.sh
bash examples/train/torl/train_qwen_7B_math_deep_math.sh

bash examples/train/search_r1/train_3b_dapo.sh
bash examples/train/search_r1/train_7b_dapo.sh
bash examples/train/search_r1/train_3b.sh
bash examples/train/search_r1/train_7b.sh

bash examples/train/deepsearch/train_8b.sh

bash examples/train/pixel_reasoner/train_qwen25vl.sh

bash examples/train/skysql/train_7b.sh



unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
cd ~/WorkSpace/verl-tool
source .venv/bin/activate
bash examples/train/pixel_reasoner/train_3b.sh > logs/pixel_reasoner_3b_debug.log 2>&1 &
bash examples/train/pixel_reasoner/train_3b.sh > logs/pixel_reasoner_3b_debug_complex_reward.log 2>&1 &
bash examples/train/pixel_reasoner/train_qwen25vl.sh > logs/pixel_reasoner_qwen25vl_debug.log 2>&1 &
bash examples/train/pixel_reasoner/train_qwen25vl.sh > logs/pixel_reasoner_qwen25vl_debug_complex_reward.log 2>&1 &
bash examples/train/pixel_reasoner/train_qwen25vl.sh > logs/pixel_reasoner_reproduce.log 2>&1 &

bash examples/train/skysql/train_7b.sh > logs/skysql_7b_debug.log 2>&1 &