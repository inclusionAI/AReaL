# Code for dependency analysis

## Data preparation

Get an AIME24 files and convert to this format.

```
{"data_source": "", "prompt": [{"content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.", "role": "user"}], "difficulty": "", "ability": "math", "reward_model": {"ground_truth": "\\boxed{204}", "style": "rule"}, "extra_info": {"index": 0, "split": "test"}}
```

## Start sglang server

```
python3 -m sglang.launch_server --model-path path-to-your-model --tp 8 &
```
Add & to run in backend
## Generate original trajectory
```
python3 run_experiment_8_times.py --input <input_file.jsonl> --num-workers 32 --max-tokens 40960 --temperature 0.6 --top-p 0.95
```

## Clean the trajectory and remove failed cases

Be sure to fill the api_key and url in api.py before running the cleaning script.
```
python3 clean_failed_trials.py --input <input_file.jsonl> --output <output_file.jsonl>
``` 

## Continue to finish the trajectory with the cleaned file
Find the path to the cleaned file and run the following command to continue the trajectory generation.
```
python3 continue_unfinished_trials.py --input <cleaned_file.jsonl> --num-workers 32 --max-tokens 40960 --temperature 0.6 --top-p 0.95
```

## Grading

Find the original generated file and the continue generated file, and run the following command to grade the trajectory.
```python3 grade_trajectory.py <jsonl path>
```
The jsonl file can be either the original generated file or the continue generated file, as long as it contains the complete trajectory.