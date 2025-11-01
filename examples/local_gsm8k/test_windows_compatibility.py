#!/usr/bin/env python3
"""
Quick compatibility test for Windows/CUDA setup.
Run this to verify your environment is ready for training.
"""

import sys
import torch

def test_cuda_availability():
    """Test CUDA availability and device info."""
    print("=" * 60)
    print("CUDA Availability Test")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available!")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("[X] CUDA is not available")
        print("   Make sure you have:")
        print("   1. NVIDIA GPU with CUDA support")
        print("   2. CUDA toolkit installed")
        print("   3. PyTorch with CUDA support (install from pytorch.org)")
        return False


def test_mps_availability():
    """Test MPS availability (should fail gracefully on Windows)."""
    print("\n" + "=" * 60)
    print("MPS (macOS) Availability Test")
    print("=" * 60)
    
    try:
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print("[OK] MPS is available (macOS detected)")
                return True
            else:
                print("[INFO] MPS backend exists but is not available")
                return False
        else:
            print("[INFO] MPS backend not available (expected on Windows/Linux)")
            return False
    except (AttributeError, RuntimeError) as e:
        print(f"[INFO] MPS check failed gracefully: {type(e).__name__}")
        return False


def test_device_selection():
    """Test device selection logic."""
    print("\n" + "=" * 60)
    print("Device Selection Test")
    print("=" * 60)
    
    # Test MPS detection (should not crash)
    try:
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    except (AttributeError, RuntimeError):
        mps_available = False
    
    if mps_available:
        device = torch.device("mps")
        print(f"[OK] Selected device: {device} (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Selected device: {device} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"[WARNING] Selected device: {device} (CPU - no GPU available)")
    
    return device


def test_simple_tensor_operation():
    """Test basic tensor operations on detected device."""
    print("\n" + "=" * 60)
    print("Tensor Operation Test")
    print("=" * 60)
    
    device = test_device_selection()
    
    try:
        # Create a simple tensor
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print(f"[OK] Successfully performed tensor operations on {device}")
        
        if device.type == "cuda":
            # Clear cache to test the cache clearing code
            torch.cuda.empty_cache()
            print("[OK] Successfully cleared CUDA cache")
        
        return True
    except Exception as e:
        print(f"[X] Failed to perform tensor operations: {e}")
        return False


def main():
    """Run all compatibility tests."""
    print("\n" + "=" * 60)
    print("Windows/CUDA Compatibility Test")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    results = []
    
    # Test CUDA
    results.append(("CUDA", test_cuda_availability()))
    
    # Test MPS (should fail gracefully, not a failure on Windows)
    mps_result = test_mps_availability()
    # MPS not available on Windows is expected, not a failure
    import platform
    if platform.system() != "Darwin":  # Not macOS
        mps_result = True  # Treat as pass on non-macOS systems
    results.append(("MPS Check", mps_result))
    
    # Test device selection
    device = test_device_selection()
    results.append(("Device Selection", device.type in ["cuda", "cpu", "mps"]))
    
    # Test tensor operations
    results.append(("Tensor Operations", test_simple_tensor_operation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[SUCCESS] All tests passed! Your environment is ready for training.")
        print(f"\nTo start training, run:")
        print(f"  python examples/local_gsm8k/train_local_simple.py --device cuda")
        print(f"  or")
        print(f"  python examples/local_gsm8k/train_hf_trainer.py --device cuda")
    else:
        print("[WARNING] Some tests failed. Please fix the issues above before training.")
        if not torch.cuda.is_available():
            print("\nTo install PyTorch with CUDA support:")
            print("  Visit: https://pytorch.org/get-started/locally/")
            print("  Select: Windows, CUDA, and your CUDA version")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

