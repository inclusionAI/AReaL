#!/usr/bin/env python3
"""
Test script to verify the setup works correctly.
This performs basic checks without requiring a full model.
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import sglang
        print("  ✅ sglang imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import sglang: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("  ✅ transformers imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import transformers: {e}")
        return False
    
    try:
        import requests
        print("  ✅ requests imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import requests: {e}")
        return False
    
    return True


def test_generation_module():
    """Test that generation.py can be imported."""
    print("\nTesting generation module...")
    
    try:
        # Import without launching server
        import generation
        print("  ✅ generation.py imported successfully")
        
        # Check classes exist
        assert hasattr(generation, 'SGLangServerManager')
        print("  ✅ SGLangServerManager class found")
        
        assert hasattr(generation, 'MultiverseGeneratorNew')
        print("  ✅ MultiverseGeneratorNew class found")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_batch_inference_module():
    """Test that batch inference module can be imported."""
    print("\nTesting batch inference module...")
    
    try:
        import new_batch_inference_new
        print("  ✅ new_batch_inference_new.py imported successfully")
        
        # Check functions exist
        assert hasattr(new_batch_inference_new, 'extract_boxed_answer')
        print("  ✅ extract_boxed_answer function found")
        
        assert hasattr(new_batch_inference_new, 'compare_answers')
        print("  ✅ compare_answers function found")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_utility_functions():
    """Test utility functions without server."""
    print("\nTesting utility functions...")
    
    try:
        from new_batch_inference_new import extract_boxed_answer, compare_answers
        
        # Test answer extraction
        text = r"The answer is \boxed{42}"
        answer = extract_boxed_answer(text)
        assert answer == "42", f"Expected '42', got '{answer}'"
        print("  ✅ extract_boxed_answer works correctly")
        
        # Test answer comparison
        assert compare_answers("42", "42") == True
        assert compare_answers("42", "43") == False
        print("  ✅ compare_answers works correctly")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_existence():
    """Test that all required files exist."""
    print("\nTesting file existence...")
    
    required_files = [
        'generation.py',
        'new_batch_inference_new.py',
        'run_inference.py',
        'aggregate_accuracy.py',
        'example.py',
        'README.md',
        'QUICKSTART.md',
        'COMPARISON.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} not found")
            all_exist = False
    
    return all_exist


def test_executability():
    """Test that scripts are executable."""
    print("\nTesting executability...")
    
    executable_files = [
        'generation.py',
        'new_batch_inference_new.py',
        'run_inference.py',
        'aggregate_accuracy.py',
        'example.py'
    ]
    
    all_executable = True
    for filename in executable_files:
        if os.path.exists(filename):
            if os.access(filename, os.X_OK):
                print(f"  ✅ {filename} is executable")
            else:
                print(f"  ⚠️  {filename} is not executable (run: chmod +x {filename})")
                all_executable = False
        else:
            print(f"  ❌ {filename} not found")
            all_executable = False
    
    return all_executable


def main():
    """Run all tests."""
    print("="*70)
    print("Testing zzy_eval_with_server Setup")
    print("="*70)
    
    results = []
    
    # Test file existence first
    results.append(("File Existence", test_file_existence()))
    
    # Test executability
    results.append(("Executability", test_executability()))
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test modules
    results.append(("Generation Module", test_generation_module()))
    results.append(("Batch Inference Module", test_batch_inference_module()))
    
    # Test utility functions
    results.append(("Utility Functions", test_utility_functions()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*70)
    if all_passed:
        print("✅ All tests passed! Setup is ready to use.")
        print("\nNext steps:")
        print("1. Read QUICKSTART.md for usage instructions")
        print("2. Try running example.py (edit model path first)")
        print("3. Run your first inference with run_inference.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install sglang transformers requests")
        print("2. Make scripts executable: chmod +x *.py")
        print("3. Check that all files were created correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
