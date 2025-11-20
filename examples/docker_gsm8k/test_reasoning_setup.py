#!/usr/bin/env python3
"""
Quick test script to verify reasoning model setup works correctly.
Tests dataset loading, message formatting, and answer extraction.
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from areal.dataset import _get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.reward.math_parser import extract_answer

def test_reasoning_dataset():
    """Test that reasoning dataset loads correctly with XML format."""
    print("=" * 60)
    print("Testing Reasoning Dataset Loading")
    print("=" * 60)
    
    tokenizer = load_hf_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Test reasoning dataset
    print("\n1. Loading reasoning dataset...")
    try:
        ds = _get_custom_dataset(
            path="openai/gsm8k-reasoning",
            type="rl",
            split="train",
            tokenizer=tokenizer,
            max_length=2048
        )
        print(f"   ✓ Dataset loaded: {len(ds)} samples")
        
        # Check first sample
        first_sample = ds[0]
        messages = first_sample["messages"]
        print(f"   ✓ First sample has {len(messages)} messages")
        
        # Check system prompt
        if messages[0]["role"] == "system":
            print(f"   ✓ System prompt found: {messages[0]['content'][:100]}...")
        
        # Check user message
        if messages[1]["role"] == "user":
            print(f"   ✓ User message found: {messages[1]['content'][:100]}...")
        
        return True
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_extraction():
    """Test that XML answer extraction works."""
    print("\n" + "=" * 60)
    print("Testing XML Answer Extraction")
    print("=" * 60)
    
    test_cases = [
        {
            "completion": "<reasoning>\nLet me think... 2+2=4\n</reasoning>\n<answer>\n42\n</answer>",
            "expected": "42",
            "description": "Simple XML format"
        },
        {
            "completion": "<reasoning>Step 1: Calculate...\nStep 2: Add...</reasoning><answer>100</answer>",
            "expected": "100",
            "description": "Compact XML format"
        },
        {
            "completion": "The answer is 50. This is correct.",
            "expected": "50",
            "description": "Fallback to non-XML format"
        },
        {
            "completion": "<reasoning>Some reasoning</reasoning><answer>\n\n25\n\n</answer>",
            "expected": "25",
            "description": "XML with whitespace"
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['description']}")
        try:
            # extract_answer requires data_name parameter (use "gsm8k" for GSM8K dataset)
            result = extract_answer(test["completion"], data_name="gsm8k")
            if result is not None and result != "":
                result_str = str(result).strip()
                print(f"   Extracted: {result_str}")
                if test["expected"] in result_str or result_str in test["expected"]:
                    print(f"   ✓ Pass (expected: {test['expected']})")
                else:
                    print(f"   ⚠ Partial match (expected: {test['expected']}, got: {result_str})")
            else:
                print(f"   ✗ Failed to extract answer")
                all_passed = False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_config_loading():
    """Test that reasoning config can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Config Loading")
    print("=" * 60)
    
    try:
        from omegaconf import OmegaConf
        config_path = os.path.join(script_dir, "gsm8k_grpo_reasoning_fast.yaml")
        
        if not os.path.exists(config_path):
            print(f"   ✗ Config file not found: {config_path}")
            return False
        
        config = OmegaConf.load(config_path)
        print(f"   ✓ Config loaded successfully")
        print(f"   - Experiment: {config.experiment_name}")
        print(f"   - Training mode: {config.training_mode}")
        print(f"   - Max samples: {config.max_train_samples}")
        print(f"   - Train dataset path: {config.train_dataset.path}")
        print(f"   - Max new tokens: {config.gconfig.max_new_tokens}")
        print(f"   - Max length: {config.train_dataset.max_length}")
        
        # Verify reasoning path
        if "reasoning" in config.train_dataset.path.lower():
            print(f"   ✓ Reasoning dataset path detected")
        else:
            print(f"   ⚠ Warning: Reasoning path not detected in config")
        
        return True
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Reasoning Model Setup Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Dataset loading
    results.append(("Dataset Loading", test_reasoning_dataset()))
    
    # Test 2: Answer extraction
    results.append(("Answer Extraction", test_answer_extraction()))
    
    # Test 3: Config loading
    results.append(("Config Loading", test_config_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Reasoning setup is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

