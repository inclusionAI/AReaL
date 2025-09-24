import asyncio
import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import TIR modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tir_workflow import TIRWorkflow
from tool_manager import ToolManager
from train_tir import math_reward_fn


async def test_tool_manager():
    """Test tool manager"""
    print("Testing ToolManager...")

    tool_manager = ToolManager(timeout=10)

    # Test Python execution
    python_code = "print(2 + 3)"
    result = await tool_manager.execute_python(python_code)
    print(f"Python execution result: {result}")
    assert "5" in result, f"Expected '5' in result, got: {result}"

    # Test calculator
    calc_expr = "2 * 3 + 4"
    result = await tool_manager.execute_calculator(calc_expr)
    print(f"Calculator result: {result}")
    assert result == "10", f"Expected '10', got: {result}"

    # Test unsafe code
    unsafe_code = "import os; os.system('ls')"
    result = await tool_manager.execute_python(unsafe_code)
    print(f"Unsafe code result: {result}")
    assert "Error" in result, f"Expected error for unsafe code, got: {result}"

    tool_manager.cleanup()
    print("ToolManager tests passed!")


async def test_tir_workflow():
    """Test TIR workflow (requires mock engine)"""
    print("Testing TIRWorkflow...")

    # Here we only test workflow initialization, actual inference requires a real engine
    from transformers import AutoTokenizer

    # Use a simple tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tool_manager = ToolManager()

    # Create TIR workflow
    workflow = TIRWorkflow(
        reward_fn=math_reward_fn,
        gconfig=None,  # Simplified here, actual needs GenerationHyperparameters
        tokenizer=tokenizer,
        tool_manager=tool_manager,
        max_turns=3,
        max_length=2000,  # Use smaller length for testing
    )

    tool_manager.cleanup()
    print("TIRWorkflow tests passed!")


def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")

    data_file = Path(__file__).parent / "data" / "sample_math.jsonl"
    assert data_file.exists(), f"Data file not found: {data_file}"

    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f]

    assert len(data) > 0, "No data loaded"
    assert "messages" in data[0], "Missing 'messages' field"
    assert "answer" in data[0], "Missing 'answer' field"

    print(f"Loaded {len(data)} samples")
    print("Data loading tests passed!")


async def main():
    """Run all tests"""
    print("Running TIR tests...")

    try:
        await test_tool_manager()
        await test_tir_workflow()
        test_data_loading()

        print("\n All tests passed!")

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
