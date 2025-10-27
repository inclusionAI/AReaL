# Parallel reasoner
## Data curation


1. Fill in the api key in data_paroceeing/.env (also fill )

2. Change the model to use: in `api.py`. deepseek v3 0324 is recommended

3. Change the 
```
with open("output_1001_to_1500_batched.jsonl", "w", encoding='utf-8') as output_file:
```
line 84 in batch_processor.py


To your file, this file can be transformed from parquet via `parquet_to_jsonl.py` you can download the parquet from huggingface using dataset "open-math-reasoning"

https://huggingface.co/datasets/nvidia/OpenMathReasoning


4. Go to `data_processing` directory and run `batch_processing.py`


5. A jsonl file will be generated.

6. Load this jsonl and run `add_planning_conclusion.py`

7. Load the jsonl file after adding planning and conclusion and then put it into `data_converter.py` it will generate 2 files

8. Shuffle these 2 files and we get the dataset

## SFT

0. Install llama factory

1. goes to SFT/LLaMa-Factory

2. Modify the `train_config.yaml`, the hyper-paras has been tuned, but the input path and the output path should be changed

3. Modify the `dataset_info.json`, change the "mixed_new_data_clean_large" to the generated dataset path

4. Run 
```
llamafactory-cli train train_config.yaml
```

## RL

0. Install the areal env

1. Move to AReaL directory

2. Run 
```
 python3 -m areal.launcher.local examples/math/parallel_grpo_different_reward.py --config examples/math/openr1.yaml experiment_name=zzy_test trial_name=fix_sigmoid_thread_token
```







