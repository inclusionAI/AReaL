# Stats Push Feature

Push SGLang metrics to an external gRPC server.

## Setup

### 1. Install gRPC Tools

```bash
pip install "grpcio==1.75.1" "grpcio-tools==1.75.1"
```

### 2. Compile Protocol Buffers

```bash
cd python/sglang/srt/metrics
python compile_router_proto.py
```

This generates:

- `router_pb2.py` - Protocol buffer message classes
- `router_pb2_grpc.py` - gRPC service classes
- `router_pb2.pyi` - Type stubs

## Usage

### Start SGLang with Stats Push

```bash
python -m sglang.launch_server \
    --model-path <model-path> \
    --enable-stats-push \
    --stats-push-address localhost:50051 \
    --enable-metrics
```

### Command-Line Flags

- `--enable-stats-push`: Enable pushing stats to gRPC server
- `--stats-push-address`: gRPC server address (e.g., 'localhost:50051')

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
