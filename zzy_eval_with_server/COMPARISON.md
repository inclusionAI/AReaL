# Comparison: zzy_eval vs zzy_eval_with_server

## Quick Comparison Table

| Feature | zzy_eval (Original) | zzy_eval_with_server (New) |
|---------|---------------------|----------------------------|
| Server Management | Manual (bash script) | Automatic (Python) |
| Startup Complexity | High (bash + Python) | Low (just Python) |
| Code Lines (server mgmt) | ~150 (bash) | ~80 (Python class) |
| Platform Support | Unix-like only | Cross-platform |
| Error Handling | Bash exit codes | Python exceptions |
| Server Lifecycle | Separate process | Tied to Python object |
| Parallel Runs | Requires bash coordination | Each Python process manages its own server |
| Cleanup | Manual/scripted | Automatic (atexit + __del__) |

## File Comparison

### Original (`zzy_eval`)

```
zzy_eval/
├── test_32_times_auto.sh       # Bash script to launch server + 32 parallel inference runs
├── generation.py                # Core generation logic (expects existing server)
├── new_batch_inference_new.py  # Batch inference (connects to existing server)
└── aggregate_accuracy.py        # Aggregates results from multiple runs
```

**Workflow:**
1. Edit `test_32_times_auto.sh` to set MODEL path
2. Run `./test_32_times_auto.sh -m /path/to/model`
3. Script launches server via `python3 -m sglang.launch_server`
4. Script waits 100 seconds for server to start
5. Script spawns 32 parallel `new_batch_inference_new.py` processes
6. Each process connects to port 30000
7. After all complete, runs `aggregate_results.py`

### New (`zzy_eval_with_server`)

```
zzy_eval_with_server/
├── generation.py                # Core generation + SGLangServerManager class
├── new_batch_inference_new.py  # Batch inference (launches its own server)
├── run_inference.py             # Simple wrapper for ease of use
├── aggregate_accuracy.py        # Same as original
├── example.py                   # Simple example script
└── README.md                    # Documentation
```

**Workflow:**
1. Run `python run_inference.py -m /path/to/model -i data.jsonl`
2. Script launches server automatically
3. Server becomes ready (health check polling)
4. Runs inference
5. Aggregates results
6. Server shuts down automatically

## Code Examples

### Starting a Server

**Original (bash):**
```bash
# In test_32_times_auto.sh
python3 -m sglang.launch_server --model-path $MODEL --tp-size 8 &
sleep 5
sleep 100  # Hope server is ready by now
```

**New (Python):**
```python
# In generation.py
generator = MultiverseGeneratorNew(
    model_path=model_path,
    launch_server=True  # That's it!
)
# Server launches, waits for health check, returns when ready
```

### Running Inference

**Original:**
```bash
# Must start server separately first, then:
python new_batch_inference_new.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --port 30000 \
    --model "$MODEL"
```

**New:**
```bash
# Server starts automatically:
python run_inference.py \
    -m /path/to/model \
    -i input.jsonl
```

### Cleanup

**Original:**
```bash
# Manual process:
# 1. Ctrl+C or wait for bash script to finish
# 2. Kill server process manually if needed
# 3. Or rely on bash script cleanup (if any)
```

**New:**
```python
# Automatic:
# - Server shuts down when generator object is destroyed
# - atexit handler ensures cleanup on normal exit
# - SIGTERM/SIGINT handlers for graceful shutdown
```

## Use Case Recommendations

### Use Original (`zzy_eval`) when:
- You need to run exactly 32 parallel inference runs on same server
- You have existing bash infrastructure
- You want to manually manage server lifecycle
- You're on a Unix-like system

### Use New (`zzy_eval_with_server`) when:
- You want simple, clean Python-only workflow
- You need flexible number of runs (not just 32)
- You want automatic server lifecycle management
- You need cross-platform compatibility
- You're integrating into a larger Python system
- You want better error handling and debugging

## Migration Guide

If you have a script using the original:

```bash
# OLD
./test_32_times_auto.sh -m /path/to/model
```

Replace with:

```bash
# NEW - Single run
python run_inference.py -m /path/to/model -i data.jsonl

# NEW - Multiple runs (for statistics)
for i in {1..32}; do
    python run_inference.py -m /path/to/model -i data.jsonl -o results/run$i
done
python aggregate_accuracy.py --test-dir results
```

Or use the new approach with Python multiprocessing:

```python
from multiprocessing import Pool
from run_inference import run_single

def run_trial(trial_num):
    # Each trial gets its own server on different port
    run_single(
        model="/path/to/model",
        input_file="data.jsonl",
        output_dir=f"results/run{trial_num}",
        port=30000 + trial_num  # Different port per trial
    )

# Run 32 trials in parallel (if you have resources)
with Pool(processes=4) as pool:
    pool.map(run_trial, range(32))
```

## Performance Considerations

**Original:**
- One server, 32 parallel clients
- Efficient use of GPU (server handles batching)
- All clients share same port/resources
- Risk: if server crashes, all 32 runs fail

**New:**
- One server per run
- Less efficient if running multiple times (each starts its own server)
- Each run is isolated (one fails, others continue)
- Better for sequential runs or different models

**Recommendation:**
- For 32 parallel runs on same model: Use **original** (more efficient)
- For sequential runs or different models: Use **new** (simpler, more reliable)
- For automated CI/testing: Use **new** (better error handling)

## Summary

The new `zzy_eval_with_server` directory provides a **cleaner, more Pythonic** approach that's **easier to use and maintain**. The original `zzy_eval` is still useful for specific use cases (like 32 parallel runs on one server), but for most scenarios, the new version is recommended.

**Key Innovation:** Server lifecycle management moved from bash to Python, making the code more portable, maintainable, and easier to integrate into larger systems.
