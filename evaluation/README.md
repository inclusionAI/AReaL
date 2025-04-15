## Evaluate

This evaluation package was modified from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main).

Install the following packages:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm --no-build-isolation
pip install transformers==4.47.0
pip install prettytable timeout_decorator
```

Run evaluation:
```bash
python eval_and_aggregate.py \
--model_path ${MODEL_PATH} \
--output_path ${OUTPUT_PATH} \
--data_names aime24 \
--max_gen_tokens 32768 \ # max number of tokens to generate, defaults to 32768
```

The results are saved in `{OUTPUT_PATH}/math_eval_32768`.

Evaluate AReaL-boba-RL-7B:
```bash
python eval_and_aggregate.py \
--model_path ${MODEL_PATH} \
--output_path ${OUTPUT_PATH} \
--data_names aime24,aime25,gpqa_diamond \
--prompt_type AReaL-boba \
--output_path outputs --temperature 1.0
```

Evaluate AReaL-boba-SFT-32B:
```bash
python eval_and_aggregate.py \
--model_path ${MODEL_PATH} \
--output_path ${OUTPUT_PATH} \
--data_names aime24,aime25,gpqa_diamond \
--prompt_type AReaL-boba-SFT \
--samples_per_node 2 --num_sample_nodes 16 \
--output_path outputs --temperature 0.6
```
