#!/usr/bin/env python3
"""
Test script to verify GRPO setup in Docker environment.
This script checks:
1. All AReaL dependencies are installed
2. SGLang can be imported
3. GRPO components are available
4. Configuration can be loaded
5. Basic GPU/CPU detection
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")

def print_warning(text):
    print(f"⚠ {text}")

def test_imports():
    """Test that all required modules can be imported."""
    print_header("Testing Python Imports")
    
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("areal", "AReaL framework"),
        ("areal.api.cli_args", "AReaL CLI args"),
        ("areal.engine.ppo.actor", "PPO Actor"),
        ("areal.workflow.rlvr", "RLVR Workflow"),
        ("areal.engine.sglang_remote", "SGLang Remote Engine"),
    ]
    
    all_passed = True
    for module_name, description in tests:
        try:
            __import__(module_name)
            print_success(f"{description} ({module_name})")
        except ImportError as e:
            print_error(f"{description} ({module_name}): {e}")
            all_passed = False
    
    return all_passed

def test_sglang():
    """Test SGLang availability."""
    print_header("Testing SGLang")
    
    try:
        import sglang
        print_success(f"SGLang version: {sglang.__version__}")
        
        # Try to import SGLang runtime
        try:
            import sglang.runtime_endpoint
            print_success("SGLang runtime endpoint available")
        except ImportError:
            print_warning("SGLang runtime endpoint not available (may need server running)")
        
        return True
    except ImportError as e:
        print_error(f"SGLang not available: {e}")
        print_warning("SGLang is required for AReaL GRPO training")
        return False

def test_device():
    """Test device availability (GPU/CPU)."""
    print_header("Testing Device Availability")
    
    import torch
    
    # Check CUDA
    if torch.cuda.is_available():
        print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print_success(f"CUDA version: {torch.version.cuda}")
        print_success(f"Number of GPUs: {torch.cuda.device_count()}")
        return "cuda"
    else:
        print_warning("CUDA not available - will use CPU")
        print_warning("CPU training will be much slower than GPU")
        return "cpu"

def test_config():
    """Test that GRPO config can be loaded."""
    print_header("Testing GRPO Configuration")
    
    try:
        from areal.api.cli_args import load_expr_config, GRPOConfig
        
        # Try to load a minimal config
        config_path = Path(__file__).parent / "train_grpo.yaml"
        
        if not config_path.exists():
            print_warning(f"Config file not found: {config_path}")
            print_warning("Skipping config test")
            return False
        
        # Set dummy WANDB key for testing
        if "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_API_KEY"] = "test-key-for-validation"
        
        config, _ = load_expr_config(
            ["--config", str(config_path)],
            GRPOConfig
        )
        
        print_success(f"Config loaded: {config_path}")
        print_success(f"  Model: {config.actor.path}")
        print_success(f"  Group size: {config.actor.group_size}")
        print_success(f"  Epochs: {config.total_train_epochs}")
        
        return True
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grpo_components():
    """Test that GRPO components can be instantiated."""
    print_header("Testing GRPO Components")
    
    try:
        from areal.api.cli_args import PPOActorConfig
        
        # Try to create a minimal PPOActorConfig
        config = PPOActorConfig(
            path="Qwen/Qwen2.5-0.5B-Instruct",
            group_size=4,
        )
        
        print_success("PPOActorConfig created successfully")
        print_success(f"  Group size: {config.group_size}")
        
        return True
    except Exception as e:
        print_error(f"Failed to create GRPO components: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that required files exist."""
    print_header("Testing File Structure")
    
    base_dir = Path(__file__).parent
    required_files = [
        "train_grpo.py",
        "train_grpo.yaml",
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print_success(f"{file_name} exists")
        else:
            print_error(f"{file_name} missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print_header("AReaL GRPO Docker Environment Test")
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    results = {}
    
    results["imports"] = test_imports()
    results["sglang"] = test_sglang()
    results["device"] = test_device() is not None
    results["config"] = test_config()
    results["components"] = test_grpo_components()
    results["files"] = test_file_structure()
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! GRPO environment is ready.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

