"""
Test script to verify remainder content is properly saved.
"""

from process_jsonl import extract_think_content_and_remainder
from one_problem import analyze_one_step_problem

# Test the extraction function
test_cot = """<think>
Step 1: This is the first step.

Step 2: This is the second step.
</think>
Therefore, the final answer is 42."""

print("Testing extraction function...")
print("="*80)
think_content, remainder_content = extract_think_content_and_remainder(test_cot)

print("Think content extracted:")
print(think_content)
print("\n" + "="*80)
print("Remainder content extracted:")
print(remainder_content)
print("="*80 + "\n")

# Verify it's extracted correctly
assert "Step 1" in think_content
assert "Step 2" in think_content
assert "final answer is 42" in remainder_content

print("✓ Extraction works correctly!")
print("\nNow testing with analyze_one_step_problem...")
print("This will call the API and save a file with the remainder content.\n")

# Test with a simple example
simple_test = """This is step 1 content.

This is step 2 content."""

try:
    result = analyze_one_step_problem(simple_test, "This is the remainder content that should be saved!")
    print("\n✓ Function executed successfully!")
    print("Check the result/ directory for the latest merged_steps_*.txt file")
    print("It should contain a section called 'Content after </think> tag' at the end.")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("This is expected if API is not available in test environment.")
