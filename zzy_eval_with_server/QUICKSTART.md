# Quick Start Guide

## Installation

Ensure you have the required dependencies:

```bash
pip install sglang transformers requests
```

## Basic Usage

### 1. Single Inference Run

Run inference on a dataset with automatic server management:

```bash
python run_inference.py \
    -m /path/to/your/model \
    -i /path/to/input.jsonl \
    -o ./results \
    --tp-size 8
```

**What happens:**
1. ✅ SGLang server launches automatically
2. ✅ Waits for server to be ready (health check)
3. ✅ Runs inference on all problems
4. ✅ Generates grading report
5. ✅ Aggregates results
6. ✅ Server shuts down automatically

### 2. Simple Example

Try the example script to see how it works:

```bash
# Edit example.py to set your model path
# Then run:
python example.py
```

### 3. Direct Python Usage

```python
from generation import MultiverseGeneratorNew

# Initialize with auto server launch
generator = MultiverseGeneratorNew(
    model_path="/path/to/model",
    tp_size=8,
    launch_server=True
)

# Run inference
result = generator.run_generation(
    problem="Your problem here...",
    max_total_tokens=32768
)

# Print results
generator.print_result(result)

# Server automatically shuts down when script exits
```

## Command Line Options

### `run_inference.py`

```
Required:
  -m, --model PATH          Path to the model
  -i, --input PATH          Path to input JSONL file

Optional:
  -o, --output-dir PATH     Output directory (default: <model>/Test_<timestamp>)
  --tp-size N               Tensor parallelism size (default: 8)
  --port N                  Port for SGLang server (default: 30000)
  --start-idx N             Start index for problems (default: 0)
  --end-idx N               End index for problems (default: all)
  --temperature F           Sampling temperature (default: 0.6)
  --max-*-tokens N          Token limits (default: 10000/32768)
```

### Example Commands

**Process all problems:**
```bash
python run_inference.py -m /model/path -i data.jsonl
```

**Process subset (problems 0-30):**
```bash
python run_inference.py -m /model/path -i data.jsonl --start-idx 0 --end-idx 30
```

**Use different port:**
```bash
python run_inference.py -m /model/path -i data.jsonl --port 30001
```

**Adjust temperature:**
```bash
python run_inference.py -m /model/path -i data.jsonl --temperature 0.8
```

## Input Format

Your JSONL file should have this format:

```json
{"problem": "What is 2+2?", "answer": "4"}
{"problem": "What is 3+3?", "answer": "6"}
```

## Output Format

The script creates a timestamped directory with:

```
results/
└── 20240209_091500/
    ├── answer0.txt           # Generated answer for problem 0
    ├── answer1.txt           # Generated answer for problem 1
    ├── ...
    ├── grading_report.txt    # Detailed grading report
    ├── metadata.json         # Machine-readable results
    └── summary_report.txt    # Aggregated statistics (if multiple runs)
```

## Running Multiple Times (for statistics)

```bash
# Create output directory
mkdir -p my_experiment

# Run 10 times
for i in {1..10}; do
    python run_inference.py \
        -m /model/path \
        -i data.jsonl \
        -o my_experiment \
        --port $((30000 + i))  # Different port for each run
done

# Aggregate results
python aggregate_accuracy.py --test-dir my_experiment
```

## Troubleshooting

### Server fails to start

**Error:** Server did not become ready within 300s

**Solutions:**
- Check if port is already in use: `lsof -i :30000`
- Increase timeout in `generation.py` (edit `is_server_ready` method)
- Check GPU availability: `nvidia-smi`
- Verify model path exists
- Check sglang installation: `python -m sglang.launch_server --help`

### Out of memory

**Error:** CUDA out of memory

**Solutions:**
- Reduce `--tp-size` (e.g., from 8 to 4)
- Reduce `--max-total-tokens`
- Use a smaller model
- Close other GPU processes

### Port already in use

**Error:** Address already in use

**Solutions:**
- Use different port: `--port 30001`
- Kill existing process: `lsof -ti:30000 | xargs kill -9`
- Wait for previous server to shut down

### Import errors

**Error:** No module named 'sglang'

**Solution:**
```bash
pip install sglang transformers requests
```

## Advanced Usage

### Custom Generation Parameters

```python
from generation import MultiverseGeneratorNew

generator = MultiverseGeneratorNew(
    model_path="/path/to/model",
    host="127.0.0.1",
    port=30000,
    tp_size=8,
    launch_server=True
)

result = generator.run_generation(
    problem="Complex problem...",
    max_normal_tokens=15000,      # More tokens for main generation
    max_path_tokens=12000,        # More tokens per parallel path
    max_conclusion_tokens=8000,   # More tokens for conclusions
    max_total_tokens=50000,       # Higher total limit
    temperature=0.8               # More creative sampling
)
```

### Connect to Existing Server

If you want to use an existing server instead of launching a new one:

```python
generator = MultiverseGeneratorNew(
    model_path="/path/to/model",  # Still needed for tokenizer
    host="127.0.0.1",
    port=30000,
    launch_server=False  # Don't launch new server
)
```

### Manual Server Management

```python
from generation import SGLangServerManager

# Launch server
manager = SGLangServerManager(
    model_path="/path/to/model",
    port=30000,
    tp_size=8
)
manager.launch(wait_for_ready=True)

# ... do your work ...

# Shutdown
manager.shutdown()
```

## Next Steps

- Read `README.md` for detailed documentation
- See `COMPARISON.md` for differences from original
- Check `example.py` for a working example
- Explore `generation.py` for customization options

## Support

For issues or questions:
1. Check the logs in server output
2. Review error messages carefully
3. Consult the SGLang documentation: https://docs.sglang.io/
4. Open an issue on the repository
