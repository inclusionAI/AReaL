"""Stats pusher for sending metrics to external gRPC server."""

import logging
import threading
import time
from typing import Optional

import grpc

logger = logging.getLogger(__name__)


class StatsPusher:
    """Synchronous gRPC client for pushing stats to external server."""

    def __init__(self, address: str, server_host: str, server_port: int):
        self.address = address
        self.server_host = server_host
        self.server_port = server_port
        self.channel: Optional[grpc.Channel] = None
        self.stub = None
        self._pb2 = None
        self._initialized = False
        self._init_lock = threading.Lock()

    def _ensure_initialized(self):
        """Lazy initialization of gRPC channel and stub."""
        if self._initialized:
            return True

        with self._init_lock:
            if self._initialized:
                return True

            try:
                from sglang.srt.metrics import router_pb2, router_pb2_grpc

                self.channel = grpc.insecure_channel(self.address)
                self.stub = router_pb2_grpc.RouterManagementStub(self.channel)
                self._pb2 = router_pb2
                self._initialized = True
                logger.info(f"Stats pusher initialized for {self.address}")
                return True
            except ImportError:
                logger.error(
                    "Failed to import router_pb2. Run: python -m grpc_tools.protoc "
                    "-I python/sglang/srt/metrics --python_out=python/sglang/srt/metrics "
                    "--grpc_python_out=python/sglang/srt/metrics "
                    "python/sglang/srt/metrics/router.proto"
                )
                return False
            except Exception as e:
                logger.info(f"Stats pusher init failed: {e}")
                return False

    def push_scheduler_stats(self, stats, stats_type: str = "scheduler"):
        """Push scheduler stats to gRPC server."""
        if not self._ensure_initialized():
            return

        try:
            scheduler_stats_proto = self._pb2.SchedulerStatsProto(
                num_running_reqs=stats.num_running_reqs,
                num_used_tokens=stats.num_used_tokens,
                token_usage=stats.token_usage,
                pending_prealloc_token_usage=stats.pending_prealloc_token_usage,
                swa_token_usage=stats.swa_token_usage,
                mamba_usage=stats.mamba_usage,
                gen_throughput=stats.gen_throughput,
                num_queue_reqs=stats.num_queue_reqs,
                num_grammar_queue_reqs=stats.num_grammar_queue_reqs,
                num_running_reqs_offline_batch=stats.num_running_reqs_offline_batch,
                cache_hit_rate=stats.cache_hit_rate,
                spec_accept_length=stats.spec_accept_length,
                spec_accept_rate=stats.spec_accept_rate,
                num_retracted_reqs=stats.num_retracted_reqs,
                num_paused_reqs=stats.num_paused_reqs,
                num_prefill_prealloc_queue_reqs=stats.num_prefill_prealloc_queue_reqs,
                num_prefill_inflight_queue_reqs=stats.num_prefill_inflight_queue_reqs,
                num_decode_prealloc_queue_reqs=stats.num_decode_prealloc_queue_reqs,
                num_decode_transfer_queue_reqs=stats.num_decode_transfer_queue_reqs,
                kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
                kv_transfer_bootstrap_ms=stats.kv_transfer_bootstrap_ms,
                kv_transfer_alloc_ms=stats.kv_transfer_alloc_ms,
                utilization=stats.utilization,
                max_running_requests_under_SLO=(
                    stats.max_running_requests_under_SLO
                    if stats.max_running_requests_under_SLO is not None
                    else -1
                ),
                engine_startup_time=stats.engine_startup_time,
                engine_load_weights_time=stats.engine_load_weights_time,
                new_token_ratio=stats.new_token_ratio,
                is_cuda_graph=stats.is_cuda_graph,
            )

            request = self._pb2.PushStatsRequest(
                server_host=self.server_host,
                server_port=self.server_port,
                timestamp=int(time.time() * 1000),
                stats_type=stats_type,
                scheduler_stats=scheduler_stats_proto,
            )

            self.stub.PushStats(request, timeout=1.0)

        except grpc.RpcError as e:
            logger.info(f"Stats push failed (gRPC error): {e.code()}")
        except Exception as e:
            logger.info(f"Stats push failed: {e}")

    def push_tokenizer_stats(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        num_requests: int,
        num_aborted: int,
        avg_ttft: float,
        avg_itl: float,
        avg_e2e: float,
    ):
        """Push tokenizer stats to gRPC server."""
        if not self._ensure_initialized():
            return

        try:
            tokenizer_stats_proto = self._pb2.TokenizerStatsProto(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                cached_tokens=cached_tokens,
                num_requests=num_requests,
                num_aborted_requests=num_aborted,
                avg_ttft=avg_ttft,
                avg_itl=avg_itl,
                avg_e2e_latency=avg_e2e,
            )

            request = self._pb2.PushStatsRequest(
                server_host=self.server_host,
                server_port=self.server_port,
                timestamp=int(time.time() * 1000),
                stats_type="tokenizer",
                tokenizer_stats=tokenizer_stats_proto,
            )

            self.stub.PushStats(request, timeout=1.0)

        except grpc.RpcError as e:
            logger.info(f"Stats push failed (gRPC error): {e.code()}")
        except Exception as e:
            logger.info(f"Stats push failed: {e}")

    def push_storage_stats(self, storage_metrics):
        """Push storage stats to gRPC server."""
        if not self._ensure_initialized():
            return

        try:
            storage_stats_proto = self._pb2.StorageStatsProto(
                prefetch_pgs=storage_metrics.prefetch_pgs,
                backup_pgs=storage_metrics.backup_pgs,
                prefetch_bandwidth=storage_metrics.prefetch_bandwidth,
                backup_bandwidth=storage_metrics.backup_bandwidth,
            )

            request = self._pb2.PushStatsRequest(
                server_host=self.server_host,
                server_port=self.server_port,
                timestamp=int(time.time() * 1000),
                stats_type="storage",
                storage_stats=storage_stats_proto,
            )

            self.stub.PushStats(request, timeout=1.0)

        except grpc.RpcError as e:
            logger.info(f"Stats push failed (gRPC error): {e.code()}")
        except Exception as e:
            logger.info(f"Stats push failed: {e}")

    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            self._initialized = False
            logger.info("Stats pusher closed")
