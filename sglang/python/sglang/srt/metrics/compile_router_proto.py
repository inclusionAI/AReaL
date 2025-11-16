#!/usr/bin/env python3
"""
Compile router.proto to Python gRPC code.

Usage:
    cd python/sglang/srt/metrics
    python compile_router_proto.py

Requirements:
    pip install "grpcio==1.75.1" "grpcio-tools==1.75.1"
"""

import subprocess
import sys
from pathlib import Path


def compile_router_proto():
    """Compile router.proto to Python gRPC code."""
    # Get script directory
    script_dir = Path(__file__).parent
    proto_file = script_dir / "router.proto"

    if not proto_file.exists():
        print(f"Error: router.proto not found at {proto_file}")
        return False

    # Check if grpc_tools is available
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        print("Error: grpcio-tools not installed")
        print('Install with: pip install "grpcio-tools==1.75.1" "grpcio==1.75.1"')
        return False

    # Compile command
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{script_dir}",
        f"--python_out={script_dir}",
        f"--grpc_python_out={script_dir}",
        f"--pyi_out={script_dir}",
        "router.proto",
    ]

    print(f"Compiling router.proto...")
    print(f"Command: {' '.join(cmd)}")

    # Run protoc
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

    if result.returncode != 0:
        print(f"Error compiling proto:")
        print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False

    # Fix imports
    grpc_file = script_dir / "router_pb2_grpc.py"
    if grpc_file.exists():
        content = grpc_file.read_text()
        content = content.replace("import router_pb2", "from . import router_pb2")
        grpc_file.write_text(content)
        print("Fixed imports in router_pb2_grpc.py")

    print("\n✅ Successfully compiled router.proto")
    print("Generated files:")
    print(f"  - router_pb2.py")
    print(f"  - router_pb2_grpc.py")
    print(f"  - router_pb2.pyi")
    return True


if __name__ == "__main__":
    success = compile_router_proto()
    sys.exit(0 if success else 1)
