import sys
import time

import grpc

import areal.router.proto.router_pb2 as router_pb2
import areal.router.proto.router_pb2_grpc as router_pb2_grpc


def test_router_management():
    """Test basic backend management"""
    channel = grpc.insecure_channel("localhost:50051")
    stub = router_pb2_grpc.RouterManagementStub(channel)

    print("=== Testing Backend Management ===")

    # Add backends
    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://localhost:8001")
    )
    assert response.success, f"Failed to add backend: {response.message}"
    print("✓ Added backend: http://localhost:8001")

    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://localhost:8002")
    )
    assert response.success, f"Failed to add backend: {response.message}"
    print("✓ Added backend: http://localhost:8002")

    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://localhost:8003")
    )
    assert response.success, f"Failed to add backend: {response.message}"
    print("✓ Added backend: http://localhost:8003")

    # List backends
    response = stub.ListBackends(router_pb2.ListBackendsRequest())
    assert len(response.backends) == 3
    print(f"✓ Listed {len(response.backends)} backends: {response.backends}")

    # Try adding duplicate
    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://localhost:8001")
    )
    assert not response.success
    print("✓ Duplicate backend rejected")

    # Delete backend
    response = stub.DeleteBackend(
        router_pb2.DeleteBackendRequest(address="http://localhost:8003")
    )
    assert response.success, f"Failed to delete backend: {response.message}"
    print("✓ Deleted backend: http://localhost:8003")

    # List again
    response = stub.ListBackends(router_pb2.ListBackendsRequest())
    assert len(response.backends) == 2
    print(f"✓ Listed {len(response.backends)} backends after deletion")

    channel.close()


def test_push_stats():
    """Test PushStats endpoint and intelligent routing"""
    channel = grpc.insecure_channel("localhost:50051")
    stub = router_pb2_grpc.RouterManagementStub(channel)

    print("\n=== Testing PushStats ===")

    # Push stats for backend 1 (high usage - 0.9)
    scheduler_stats = router_pb2.SchedulerStatsProto(
        num_running_reqs=10,
        num_used_tokens=9000,
        token_usage=0.9,
        num_queue_reqs=5,
        cache_hit_rate=0.8,
        gen_throughput=100.0,
    )

    response = stub.PushStats(
        router_pb2.PushStatsRequest(
            server_host="localhost",
            server_port=8001,
            timestamp=int(time.time() * 1000),
            stats_type="scheduler",
            scheduler_stats=scheduler_stats,
        )
    )
    assert response.success, f"Failed to push stats: {response.message}"
    print("✓ Pushed stats for localhost:8001 (token_usage=0.9)")

    # Push stats for backend 2 (low usage - 0.2)
    scheduler_stats = router_pb2.SchedulerStatsProto(
        num_running_reqs=2,
        num_used_tokens=2000,
        token_usage=0.2,
        num_queue_reqs=0,
        cache_hit_rate=0.9,
        gen_throughput=150.0,
    )

    response = stub.PushStats(
        router_pb2.PushStatsRequest(
            server_host="localhost",
            server_port=8002,
            timestamp=int(time.time() * 1000),
            stats_type="scheduler",
            scheduler_stats=scheduler_stats,
        )
    )
    assert response.success, f"Failed to push stats: {response.message}"
    print("✓ Pushed stats for localhost:8002 (token_usage=0.2)")

    # Test without scheduler stats
    response = stub.PushStats(
        router_pb2.PushStatsRequest(
            server_host="localhost",
            server_port=8003,
            timestamp=int(time.time() * 1000),
            stats_type="other",
        )
    )
    assert not response.success
    assert "No scheduler stats" in response.message
    print("✓ Correctly rejected PushStats without scheduler_stats")

    channel.close()


def test_empty_address():
    """Test validation of empty addresses"""
    channel = grpc.insecure_channel("localhost:50051")
    stub = router_pb2_grpc.RouterManagementStub(channel)

    print("\n=== Testing Validation ===")

    # Empty address for AddBackend
    response = stub.AddBackend(router_pb2.AddBackendRequest(address=""))
    assert not response.success
    assert "cannot be empty" in response.message
    print("✓ Empty address rejected in AddBackend")

    # Empty address for DeleteBackend
    response = stub.DeleteBackend(router_pb2.DeleteBackendRequest(address=""))
    assert not response.success
    assert "cannot be empty" in response.message
    print("✓ Empty address rejected in DeleteBackend")

    channel.close()


def main():
    print("Starting Router gRPC Tests\n")
    print("Make sure the router is running on localhost:50051")
    print("Start it with: cd rust/bin/router && cargo run\n")

    time.sleep(1)

    try:
        test_router_management()
        test_push_stats()
        test_empty_address()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)

    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()} - {e.details()}")
        print("Make sure the router is running!")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
