#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simple test to verify the module imports correctly

"""
Simple test script to verify thinkmorph_vllm module.
This doesn't require the actual model, just tests imports.
"""

import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add models directory to path
models_dir = Path(__file__).parent
sys.path.insert(0, str(models_dir.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from thinkmorph_vllm import VLLMInterleavedInference
        print("[OK] VLLMInterleavedInference imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import VLLMInterleavedInference: {e}")
        return False

    try:
        from thinkmorph_vllm.configs import (
            DEFAULT_CONFIG,
            FAST_CONFIG,
            HIGH_QUALITY_CONFIG,
            REASONING_CONFIG,
            EDITING_CONFIG,
        )
        print("[OK] All config presets imported successfully")
        print(f"  - DEFAULT_CONFIG keys: {list(DEFAULT_CONFIG.keys())[:3]}...")
    except ImportError as e:
        print(f"[FAIL] Failed to import configs: {e}")
        return False

    return True


def test_config_validation():
    """Test that config presets have expected structure."""
    print("\nValidating config presets...")

    from thinkmorph_vllm.configs import DEFAULT_CONFIG, FAST_CONFIG, HIGH_QUALITY_CONFIG

    required_keys = [
        "max_think_tokens",
        "text_temperature",
        "cfg_text_scale",
        "cfg_img_scale",
        "num_timesteps",
    ]

    for config_name, config in [
        ("DEFAULT_CONFIG", DEFAULT_CONFIG),
        ("FAST_CONFIG", FAST_CONFIG),
        ("HIGH_QUALITY_CONFIG", HIGH_QUALITY_CONFIG),
    ]:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"[FAIL] {config_name} missing keys: {missing_keys}")
            return False
        print(f"[OK] {config_name} is valid (timesteps={config['num_timesteps']})")

    return True


def test_module_structure():
    """Test module structure and attributes."""
    print("\nChecking module structure...")

    import thinkmorph_vllm

    if not hasattr(thinkmorph_vllm, "__version__"):
        print("[FAIL] Module missing __version__ attribute")
        return False
    print(f"[OK] Module version: {thinkmorph_vllm.__version__}")

    expected_exports = [
        "VLLMInterleavedInference",
        "DEFAULT_CONFIG",
        "FAST_CONFIG",
        "HIGH_QUALITY_CONFIG",
    ]

    for export in expected_exports:
        if not hasattr(thinkmorph_vllm, export):
            print(f"[FAIL] Module missing export: {export}")
            return False

    print(f"[OK] All expected exports present")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("ThinkMorph vLLM Module Test")
    print("="*60)
    print()

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n[WARNING] Import test failed - check dependencies")
        print("Make sure to install: pip install torch torchvision vllm transformers")
        return 1

    # Test config validation
    if not test_config_validation():
        all_passed = False
        print("\n[WARNING] Config validation failed")
        return 1

    # Test module structure
    if not test_module_structure():
        all_passed = False
        print("\n[WARNING] Module structure test failed")
        return 1

    print("\n" + "="*60)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("1. Install vLLM: pip install vllm")
        print("2. Download ThinkMorph model from HuggingFace")
        print("3. Run example_usage.py with actual images")
    else:
        print("[FAIL] Some tests failed")
        return 1

    print("="*60)
    return 0


if __name__ == "__main__":
    exit(main())
