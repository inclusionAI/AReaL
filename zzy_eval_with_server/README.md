# Evaluation with Automatic Server Launch

This directory contains evaluation scripts that automatically launch the SGLang server from within Python code, eliminating the need for separate bash scripts to manage the server lifecycle.

## Key Differences from `zzy_eval`

### Original (`zzy_eval`)
- Requires manually starting SGLang server via bash script
- Uses `test_32_times_auto.sh` to launch server and coordinate multiple inference runs
- Server must be running before inference starts
- Requires port management across multiple parallel runs

### New (`zzy_eval_with_server`)
- **Automatically launches SGLang server** from Python code
- Server lifecycle managed within the generator class
- Cleaner, more Pythonic approach
- Easier to use for single runs or scripted evaluations

## Files

### `generation.py`
Modified version that includes `SGLangServerManager` class to:
- Launch SGLang server using `subprocess`
- Wait for server to be ready
- Automatically shutdown server when done

Key changes:
```python
class SGLangServerManager:
    """Manages SGLang server lifecycle - launching and shutdown."""
    
    def launch(self, wait_for_ready: bool = True) -> bool:
        """Launch the SGLang server using subprocess."""
        # Launches: python3 -m sglang.launch_server --model-path ... --tp-size ...
        
    def shutdown(self):
        """Shutdown the SGLang server."""
```

The `MultiverseGeneratorNew` class now accepts a `launch_server` parameter:
```python
generator = MultiverseGeneratorNew(
    model_path="/path/to/model",
    host="127.0.0.1",
    port=30000,
    tp_size=8,
    launch_server=True  # <-- Launches server automatically
)
```

### `new_batch_inference_new.py`
Batch inference script that uses the auto-launching generator.

Usage:
```bash
python new_batch_inference_new.py \
    --input data.jsonl \
    --output-dir results \
    --model /path/to/model \
    --tp-size 8 \
    --port 30000
```

### `run_inference.py`
Simplified wrapper script that:
1. Runs batch inference (which auto-launches server)
2. Aggregates results automatically

Usage:
```bash
python run_inference.py \
    -m /path/to/model \
    -i S1-parallel/AIME2425.jsonl \
    -o results \
    --tp-size 8
```

### `aggregate_accuracy.py`
Same as original - aggregates results from multiple test runs.

## Usage Examples

### Single Inference Run

```bash
python run_inference.py \
    -m /storage/openpsi/models/zzy/Multiverse-20251030_154726 \
    -i S1-parallel/AIME2425.jsonl \
    --tp-size 8 \
    --port 30000
```

### Multiple Runs (for statistical analysis)

You can run the same evaluation multiple times to gather statistics:

```bash
# Run 1
python run_inference.py -m /path/to/model -i data.jsonl -o results/run1

# Run 2
python run_inference.py -m /path/to/model -i data.jsonl -o results/run2

# Run 3
python run_inference.py -m /path/to/model -i data.jsonl -o results/run3

# Aggregate all runs
python aggregate_accuracy.py --test-dir results
```

### Direct Python Usage

```python
from generation import MultiverseGeneratorNew

# Initialize and launch server
generator = MultiverseGeneratorNew(
    model_path="/path/to/model",
    tp_size=8,
    port=30000,
    launch_server=True  # Server launches automatically
)

# Run inference
result = generator.run_generation(
    problem="Your problem here",
    max_normal_tokens=10000,
    max_path_tokens=10000,
    max_conclusion_tokens=10000,
    max_total_tokens=32768
)

# Server shuts down automatically when generator is destroyed
```

## Advantages

1. **Simpler**: No need for complex bash scripts
2. **More Reliable**: Server lifecycle tied to Python process
3. **Better Error Handling**: Python exceptions vs bash exit codes
4. **Easier to Extend**: Add features in Python rather than bash
5. **Cross-platform**: Works on any platform with Python (not just Unix-like systems)
6. **Cleaner Code**: Server management encapsulated in a class

## Technical Details

### Server Launch Process

1. `SGLangServerManager.launch()` spawns subprocess:
   ```
   python3 -m sglang.launch_server --model-path <path> --tp-size <size> --host <host> --port <port>
   ```

2. Waits for server health check at `http://host:port/health`

3. Returns control when server is ready (or timeout after 300s)

### Server Shutdown

- Automatic via `atexit` registration
- Also triggered when `MultiverseGeneratorNew` object is destroyed
- Graceful termination with 10s timeout, then force kill if needed

### Port Management

- Each run should use a unique port to avoid conflicts
- Default: 30000
- Can specify via `--port` argument

## Migration from Original

If you have existing scripts using `test_32_times_auto.sh`, you can migrate to:

**Old approach:**
```bash
./test_32_times_auto.sh -m /path/to/model
```

**New approach:**
```bash
python run_inference.py -m /path/to/model -i S1-parallel/AIME2425.jsonl
```

The functionality is the same, but the new approach is:
- Simpler (one command vs bash script + Python)
- More maintainable (all Python)
- Easier to customize (Python args vs bash variables)

## Requirements

Same as original:
- sglang
- transformers
- requests (for health checks)
- Standard Python libraries (subprocess, json, argparse, etc.)

## Notes

- Server logs are captured but not displayed by default
- Server process is tied to the Python generator object
- Multiple generators on different ports can run simultaneously
- Clean shutdown on SIGTERM/SIGINT/normal exit
