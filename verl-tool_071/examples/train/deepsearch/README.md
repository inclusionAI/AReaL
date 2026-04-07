### Data Preparation
```bash
python examples/data_preprocess/deepsearch.py --dataset_path=VerlTool/deepsearch
```

You can download our previous google serper cache to save your cost during the training:
```bash
huggingface-cli download VerlTool/deepsearch google_search_cache.jsonl --local-dir ~/.verl_cache --repo-type dataset
```

### Test the Tool Server
```bash
# Start the tool server
export SERPER_API_KEY="..."
host=localhost
port=5000
tool_type=google_search,ipython_code # separate by comma if you want to start multiple tool servers
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool --use_ray True & # run in background
```
```bash
python -m verl_tool.servers.tests.test_google_search_tool test_google_search --url=http://localhost:5000/get_observation
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5000/get_observation
```
### Training
```bash
bash examples/train/deepsearch/train_8b.sh > logs/deepsearch_8b_debug.log 2>&1 &
```

Note: Serper API's return snippet is limited to be the first 128 chars. If want full text need to set `process_snippets: bool = True` in the `GoogleSearchTool` tool class. But this can involve large-scale url scraping and slow down the RL training. Therefore, we set `process_snippets: bool = False` during the RL trainng and only bring it on during the evaluation. Therefore, it's common to see the validation score during the training is lower than the actual evalution score.

### Evaluation
Deepsearch tasks can have a summarization model to do page summarization during the evaluation. This can be slow, and thus not used in the RL, but can be used in the evaluation stage.

1. server summarization model
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/QwQ-32B \
  --dtype auto \
  --api-key "EMPTY" \
  --enforce-eager \
  --no-enable-prefix-caching \
  -tp 4 > logs/server_summarization_model.log 2>&1 &
```

2. change the [`google_search.py`](../../../verl_tool/servers/tools/google_search.py) and set the following in the `GoogleSearchTool` class:

- If only further url processing, set 
```python
process_snippets: bool = True
```
- If also with a summarization model:
```python
process_snippets: bool = True,
summ_model_url: str = "http://0.0.0.0:8000/v1",
summ_model_path: str = "Qwen/QwQ-32B"
```
3. Start the evaluation via the training script, set `val_before_train=True` and kill it once it finished the first step evaluation.
```bash
bash examples/train/deepsearch/train.sh > logs/deepsearch_3b_evaluation.log 2>&1 &
```



