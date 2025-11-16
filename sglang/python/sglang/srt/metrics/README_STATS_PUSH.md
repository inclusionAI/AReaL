# Stats Push Feature

Push SGLang metrics to an external gRPC server.

## Usage

### Start SGLang with Stats Push

```bash
python -m sglang.launch_server \
    --model-path <model-path> \
    --enable-stats-push \
    --stats-push-address localhost:50051 \
    --enable-metrics \
    --enable-prefill-completion-stats \
    --prefill-completion-stats-batch-size 1
```

### Command-Line Flags

- `--enable-stats-push`: Enable pushing stats to gRPC server
- `--stats-push-address`: gRPC server address (e.g., 'localhost:50051')
- `--enable-prefill-completion-stats`: Enable per-request prefill completion stats
- `--prefill-completion-stats-batch-size`: Batch size for prefill stats (default: 32,
  set to 1 for per-request)

## Testing

### Run Test Server

```bash
cd python/sglang/srt/metrics
python test_stats_server.py
```

The test server will print received stats to console.

### Example Output

```
================================================================================
PushStats received from 127.0.0.1:30000
Timestamp: 1234567890123
Stats type: scheduler

Scheduler Stats:
  Running requests: 5
  Queue requests: 2
  Token usage: 45.32%
  Gen throughput: 123.45 tok/s
  Cache hit rate: 78.90%
================================================================================
```

## Stats Types

### Scheduler Stats (pushed on each log interval)

- `num_running_reqs` - Number of running requests
- `num_queue_reqs` - Number of queued requests
- `token_usage` - Token usage ratio (0.0-1.0)
- `gen_throughput` - Generation throughput (tokens/sec)
- `cache_hit_rate` - Prefix cache hit rate
- PD disaggregation metrics (if enabled)
- Speculative decoding metrics (if enabled)

### Prefill Completion Stats (pushed when enabled)

Per-request metrics sent when a request completes its prefill phase:

- `request_id` - Unique request identifier
- `prefill_latency_ms` - Time spent in prefill (milliseconds)
- `ttft_ms` - Time to first token from queue entry (milliseconds)
- `prompt_tokens` - Number of tokens in the prompt
- `cached_tokens` - Number of tokens served from prefix cache
- `queue_time_ms` - Time spent waiting in queue (milliseconds)
- `priority` - Request priority
- `batch_size` - Number of requests in the prefill batch

**Batching**: Stats are batched according to `--prefill-completion-stats-batch-size`
(default: 32). Set to 1 for immediate per-request pushing.

### Server Info (included with every push)

- `server_host` - SGLang server host
- `server_port` - SGLang server port
- `timestamp` - Unix timestamp in milliseconds

## Error Handling

Stats pushing uses silent failure mode:

- If gRPC server is unavailable, errors are logged at DEBUG level
- SGLang operations continue normally without interruption
- No exceptions are raised on push failures

## Implementation Details

- Stats are pushed asynchronously using `asyncio.create_task()`
- Pushes occur on internal log intervals (same as Prometheus metrics)
- Each push has a 1-second timeout
- gRPC channel is lazily initialized on first push
