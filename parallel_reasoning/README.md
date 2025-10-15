# Parallel reasoner

## Data curation

Fill your API key

Then run 
```
cd data_processing
python3 batch_processor.py
```

Caveat: deepseek-v3-0324 is suggested for data curation

Then follow the instruction according to API rate limit

It will output a jsonl file

If API key failed half way, you can get your partial result and recover them by running `batch_recovery.py`

After the jsonl is generated, run `add_planning_conclusion.py` to add planning and conclusion

Then run `data_converter.py` to convert to sharegpt format.

OPTIONAL: 

run `extract_objective.py` to add headers for thread conversation

## SFT

Find the jsonl file in the final and fill the path to `dataset_info.json` in `SFT/LLaMa-factory`

Change the `train_config.yaml`

And then run the llamafactory for training

## RL

Unzip AReaL.zip if there are not AReaL files.

Run

```
 python3 -m areal.launcher.local examples/math/parallel_grpo_different_reward.py --config examples/math/openr1.yaml experiment_name=zzy_test trial_name=fix_sigmoid_thread_token
```
For grpo

Hardward env: 8 * nvidia H200

I will give the dataset to you via wechat

## Evaluation

1. Start a sglang server on the port you prefer
2. For the code after SFT, run 
```
python3 independent_sglang_after_sft.py --use-existing-server
```
For the code after RL, run
```
python3 independent_sglang_after_rl.py --use-existing-server
```

If you wanted to test 8 times simultaneously, run

```
./Test/test8times.sh
```

For statistics: find the directory of generated answer and then run Full_statistics.py

