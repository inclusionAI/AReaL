import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import grpc
import requests

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


# Mock backend server for testing HTTP endpoints
class MockBackendHandler(BaseHTTPRequestHandler):
    request_log = []  # Track all requests received

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        # Log the request
        self.__class__.request_log.append(
            {
                "port": self.server.server_port,
                "path": self.path,
                "method": "POST",
                "body": body,
            }
        )

        # Simulate successful response
        response = {
            "success": True,
            "message": f"Mock backend on port {self.server.server_port}",
            "port": self.server.server_port,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_PUT(self):
        # Same as POST for our tests
        self.do_POST()


def start_mock_backend(port):
    """Start a mock backend server on the given port"""
    server = HTTPServer(("127.0.0.1", port), MockBackendHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_generate_endpoint():
    """Test /generate endpoint - should route to single backend"""
    print("\n=== Testing /generate Endpoint (Single Backend Routing) ===")

    # Clear request log
    MockBackendHandler.request_log = []

    # Test POST /generate
    test_data = {
        "text": "Hello, world!",
        "sampling_params": {"temperature": 0.7, "max_new_tokens": 100},
    }

    # Make multiple requests to test round-robin
    for i in range(6):
        response = requests.post(
            "http://localhost:3000/generate", json=test_data, timeout=5
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["success"]
        print(f"  Request {i + 1}: Routed to backend on port {data['port']}")

    # Check that requests were distributed (round-robin)
    ports_used = [
        req["port"]
        for req in MockBackendHandler.request_log
        if req["path"] == "/generate"
    ]
    print(f"✓ Routed {len(ports_used)} requests across backends: {set(ports_used)}")

    # Test PUT /generate
    MockBackendHandler.request_log = []
    response = requests.put("http://localhost:3000/generate", json=test_data, timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("✓ PUT /generate works correctly")


def test_broadcast_endpoints():
    """Test broadcast endpoints - should send to all backends"""
    print("\n=== Testing Broadcast Endpoints (Weight Updates) ===")

    endpoints_to_test = [
        "/init_weights_update_group",
        "/update_weights_from_distributed",
        "/update_weights_from_disk",
    ]

    for endpoint in endpoints_to_test:
        print(f"\n  Testing {endpoint}:")

        # Clear request log
        MockBackendHandler.request_log = []

        test_data = {"model_path": "/path/to/model", "load_format": "safetensors"}

        # Test POST
        response = requests.post(
            f"http://localhost:3000{endpoint}", json=test_data, timeout=5
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["success"]
        assert "backends" in data["message"].lower()

        # Check that ALL backends received the request
        requests_for_endpoint = [
            req for req in MockBackendHandler.request_log if req["path"] == endpoint
        ]
        ports_hit = sorted([req["port"] for req in requests_for_endpoint])

        # We should have 2 backends (8001 and 8002)
        assert len(ports_hit) == 2, f"Expected 2 backends, got {len(ports_hit)}"
        assert 8001 in ports_hit and 8002 in ports_hit, (
            f"Expected ports 8001 and 8002, got {ports_hit}"
        )

        print(f"    POST: ✓ Broadcast to {len(ports_hit)} backends: {ports_hit}")

        # Test PUT
        MockBackendHandler.request_log = []
        response = requests.put(
            f"http://localhost:3000{endpoint}", json=test_data, timeout=5
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        requests_for_endpoint = [
            req for req in MockBackendHandler.request_log if req["path"] == endpoint
        ]
        assert len(requests_for_endpoint) == 2, (
            f"Expected 2 backends for PUT, got {len(requests_for_endpoint)}"
        )

        print(f"    PUT:  ✓ Broadcast to {len(requests_for_endpoint)} backends")


def test_broadcast_failure_handling():
    """Test that broadcast endpoints handle backend failures correctly"""
    print("\n=== Testing Broadcast Failure Handling ===")

    # Add a backend that will fail (non-existent port)
    channel = grpc.insecure_channel("localhost:50051")
    stub = router_pb2_grpc.RouterManagementStub(channel)

    # Add a failing backend
    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://127.0.0.1:9999")
    )
    assert response.success

    # Try to broadcast - should fail because one backend is unreachable
    test_data = {"model_path": "/path/to/model"}

    response = requests.post(
        "http://localhost:3000/update_weights_from_disk", json=test_data, timeout=5
    )

    # Should return error (BAD_GATEWAY) when any backend fails
    assert response.status_code == 502, f"Expected 502, got {response.status_code}"
    data = response.json()
    assert "error" in data
    assert "failures" in data["error"]
    assert len(data["error"]["failures"]) >= 1  # At least the failing backend

    print(
        f"✓ Broadcast correctly failed with {len(data['error']['failures'])} backend(s) down"
    )
    print(f"  Error message: {data['error']['message'][:80]}...")

    # Remove the failing backend
    stub.DeleteBackend(router_pb2.DeleteBackendRequest(address="http://127.0.0.1:9999"))
    channel.close()


def test_http_endpoints_not_found():
    """Test that unknown endpoints return 404"""
    print("\n=== Testing 404 Handling ===")

    response = requests.get("http://localhost:3000/unknown_endpoint", timeout=5)
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
    print("✓ Unknown endpoints return 404")


def test_health_endpoint():
    """Test /health endpoint"""
    print("\n=== Testing /health Endpoint ===")

    response = requests.get("http://localhost:3000/health", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.text == "OK"
    print("✓ /health endpoint returns OK")


def setup_http_test_environment():
    """Setup mock backends and configure router"""
    print("\n=== Setting up HTTP Test Environment ===")

    # Start mock backends on ports 8001 and 8002
    backend1 = start_mock_backend(8001)
    backend2 = start_mock_backend(8002)
    print("✓ Started mock backends on ports 8001 and 8002")

    # Wait for servers to be ready
    time.sleep(0.5)

    # Configure router with backends via gRPC
    channel = grpc.insecure_channel("localhost:50051")
    stub = router_pb2_grpc.RouterManagementStub(channel)

    # Clear any existing backends first
    try:
        response = stub.ListBackends(router_pb2.ListBackendsRequest())
        for backend in response.backends:
            stub.DeleteBackend(router_pb2.DeleteBackendRequest(address=backend))
    except Exception:
        pass

    # Add our test backends
    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://127.0.0.1:8001")
    )
    assert response.success, f"Failed to add backend 8001: {response.message}"

    response = stub.AddBackend(
        router_pb2.AddBackendRequest(address="http://127.0.0.1:8002")
    )
    assert response.success, f"Failed to add backend 8002: {response.message}"

    print("✓ Configured router with 2 backends")
    channel.close()

    return backend1, backend2


def main():
    print("Starting Router Tests\n")
    print("Make sure the router is running on:")
    print("  - gRPC: localhost:50051")
    print("  - HTTP: localhost:3000")
    print("\nStart it with: cd rust/bin/router && cargo run\n")

    time.sleep(1)

    try:
        # Run gRPC tests
        print("=" * 60)
        print("PART 1: gRPC Management API Tests")
        print("=" * 60)
        test_router_management()
        test_push_stats()
        test_empty_address()

        # Setup and run HTTP endpoint tests
        print("\n" + "=" * 60)
        print("PART 2: HTTP Endpoint Tests")
        print("=" * 60)
        backend1, backend2 = setup_http_test_environment()

        test_health_endpoint()
        test_generate_endpoint()
        test_broadcast_endpoints()
        test_broadcast_failure_handling()
        test_http_endpoints_not_found()

        # Cleanup
        print("\n=== Cleaning Up ===")
        backend1.shutdown()
        backend2.shutdown()
        print("✓ Shut down mock backends")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()} - {e.details()}")
        print("Make sure the router is running!")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"\n✗ HTTP Request Error: {e}")
        print("Make sure the router HTTP server is running on port 3000!")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
