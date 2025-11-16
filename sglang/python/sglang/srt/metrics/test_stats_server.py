#!/usr/bin/env python3
"""
Test gRPC server for receiving stats from SGLang.

This is a simple test server that implements the RouterManagement service
and prints received stats to the console.

Usage:
    1. First compile the proto:
       cd python/sglang/srt/metrics
       python compile_router_proto.py

    2. Run this test server:
       python test_stats_server.py

    3. Start SGLang with stats push enabled:
       python -m sglang.launch_server \
           --model-path <model> \
           --enable-stats-push \
           --stats-push-address localhost:50051
"""

import logging
from concurrent import futures

import grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_server():
    """Run the test gRPC server."""
    try:
        from sglang.srt.metrics import router_pb2, router_pb2_grpc
    except ImportError:
        print("Error: router_pb2 modules not found. Please compile the proto first:")
        print("  cd python/sglang/srt/metrics")
        print("  python compile_router_proto.py")
        return

    class RouterManagementServicer(router_pb2_grpc.RouterManagementServicer):
        """Test implementation of RouterManagement service."""

        def __init__(self):
            self.backends = []

        def AddBackend(self, request, context):
            logger.info(f"AddBackend called: {request.address}")
            self.backends.append(request.address)
            return router_pb2.AddBackendResponse(
                success=True, message=f"Added backend: {request.address}"
            )

        def DeleteBackend(self, request, context):
            logger.info(f"DeleteBackend called: {request.address}")
            if request.address in self.backends:
                self.backends.remove(request.address)
                return router_pb2.DeleteBackendResponse(
                    success=True, message=f"Deleted backend: {request.address}"
                )
            return router_pb2.DeleteBackendResponse(
                success=False, message=f"Backend not found: {request.address}"
            )

        def ListBackends(self, request, context):
            logger.info(f"ListBackends called")
            return router_pb2.ListBackendsResponse(backends=self.backends)

        def PushStats(self, request, context):
            """Handle stats push from SGLang."""
            logger.info("\n" + "=" * 80)
            logger.info(
                f"PushStats received from {request.server_host}:{request.server_port}"
            )
            logger.info(f"Timestamp: {request.timestamp}")
            logger.info(f"Stats type: {request.stats_type}")

            if request.HasField("scheduler_stats"):
                stats = request.scheduler_stats
                logger.info("\nScheduler Stats:")
                logger.info(f"  Running requests: {stats.num_running_reqs}")
                logger.info(f"  Queue requests: {stats.num_queue_reqs}")
                logger.info(f"  Token usage: {stats.token_usage:.2%}")
                logger.info(f"  Gen throughput: {stats.gen_throughput:.2f} tok/s")
                logger.info(f"  Cache hit rate: {stats.cache_hit_rate:.2%}")

            if request.HasField("tokenizer_stats"):
                stats = request.tokenizer_stats
                logger.info("\nTokenizer Stats:")
                logger.info(f"  Total requests: {stats.num_requests}")
                logger.info(f"  Prompt tokens: {stats.prompt_tokens}")
                logger.info(f"  Generation tokens: {stats.generation_tokens}")
                logger.info(f"  Avg TTFT: {stats.avg_ttft:.3f}s")
                logger.info(f"  Avg ITL: {stats.avg_itl:.3f}s")

            if request.HasField("storage_stats"):
                stats = request.storage_stats
                logger.info("\nStorage Stats:")
                logger.info(f"  Prefetch pgs: {list(stats.prefetch_pgs)}")
                logger.info(f"  Backup pgs: {list(stats.backup_pgs)}")

            logger.info("=" * 80 + "\n")

            return router_pb2.PushStatsResponse(
                success=True, message="Stats received successfully"
            )

    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    router_pb2_grpc.add_RouterManagementServicer_to_server(
        RouterManagementServicer(), server
    )

    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(f"Test gRPC server started on port {port}")
    logger.info("Waiting for stats from SGLang...")
    logger.info("Press Ctrl+C to stop")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        server.stop(0)


if __name__ == "__main__":
    run_server()
