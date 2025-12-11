"""
Example script demonstrating the ParallelGenerationWorkflow.

This shows how to set up and use the workflow for structured parallel generation.
"""

import asyncio
from areal.api.cli_args import GenerationHyperparameters
from areal.workflow.parallel_generation import ParallelGenerationWorkflow


def simple_math_reward(prompt: str, completion: str, input_ids: list[int], 
                       output_ids: list[int], **kwargs) -> float:
    """
    Simple reward function for math problems.
    Awards 1.0 if the final answer matches the ground truth, 0.0 otherwise.
    """
    import re
    
    # Get ground truth from kwargs
    ground_truth = str(kwargs.get("answer", "")).strip()
    
    # Try to extract the final answer from the completion
    # Look for patterns like "Final Answer: X" or "the answer is X"
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$|</)",
        r"[Tt]he answer is:?\s*(.+?)(?:\n|$|</)",
        r"</[Cc]onclusion>.*?(\d+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            model_answer = match.group(1).strip()
            # Clean up the answer
            model_answer = re.sub(r'[^\d.]', '', model_answer)
            ground_truth_clean = re.sub(r'[^\d.]', '', ground_truth)
            
            if model_answer == ground_truth_clean:
                return 1.0
            break
    
    return 0.0


async def main():
    """Example usage of ParallelGenerationWorkflow."""
    
    # Configure generation parameters
    gconfig = GenerationHyperparameters(
        n_samples=1,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=8192,
    )
    
    # Create the workflow
    workflow = ParallelGenerationWorkflow(
        reward_fn=simple_math_reward,
        gconfig=gconfig,
        tokenizer="Qwen/Qwen2.5-7B-Instruct",  # Replace with your model path
        enable_thinking=False,
        max_goal_tokens=2048,
        max_path_tokens=4096,
        max_conclusion_tokens=2048,
        rollout_stat_scope="rollout",
        dump_dir="./parallel_gen_dumps",
    )
    
    # Example data
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math assistant. Structure your reasoning with <Goal>, <Path>, and <Conclusion> tags."
            },
            {
                "role": "user",
                "content": """A list of positive integers has the following properties:
• The sum of the items in the list is 30.
• The unique mode of the list is 9.
• The median of the list is a positive integer that does not appear in the list itself.
Find the sum of the squares of all the items in the list."""
            }
        ],
        "answer": "364",  # Ground truth answer
        "query_id": "math_problem_001",
    }
    
    print("="*80)
    print("Parallel Generation Workflow Example")
    print("="*80)
    print(f"\nProblem: {data['messages'][-1]['content'][:100]}...")
    print(f"Expected Answer: {data['answer']}")
    print("\n" + "="*80)
    print("Starting generation...")
    print("="*80 + "\n")
    
    # Note: In actual usage, this would be called by the inference engine
    # Here we're just showing the structure
    # You would need a running InferenceEngine instance to actually execute this
    
    # Simulate what the engine would do (pseudo-code):
    # engine = InferenceEngine(...)
    # trajectory = await workflow.arun_episode(engine, data)
    
    print("✓ Workflow created successfully!")
    print("\nWorkflow configuration:")
    print(f"  - Max goal tokens: {workflow.max_goal_tokens}")
    print(f"  - Max path tokens: {workflow.max_path_tokens}")
    print(f"  - Max conclusion tokens: {workflow.max_conclusion_tokens}")
    print(f"  - Enable thinking: {workflow.enable_thinking}")
    print(f"  - Dump directory: {workflow.dump_dir}")
    
    print("\n" + "="*80)
    print("Expected Generation Structure:")
    print("="*80)
    print("""
<Goal>
  <Outline>1. Analyze the constraints on the list</Outline>
  <Outline>2. Find possible list configurations</Outline>
  <Outline>3. Calculate sum of squares</Outline>
</Goal>

<Path>
1. Analyzing constraints:
   - Sum = 30
   - Mode = 9 (appears most frequently, unique)
   - Median is an integer not in the list
   [detailed reasoning...]
</Path>

<Path>
2. Finding configurations:
   - Need at least 2 nines for mode
   - Try different list sizes
   [detailed reasoning...]
</Path>

<Path>
3. Calculating sum of squares:
   - Once we have the list: [x, y, z, ...]
   - Compute x² + y² + z² + ...
   [detailed reasoning...]
</Path>

<Conclusion>
The list is [1, 3, 9, 9, 8] with median 8.
Sum of squares: 1² + 3² + 9² + 9² + 8² = 1 + 9 + 81 + 81 + 64 = 236
Wait, let me recalculate... [verification]
Final Answer: 364
</Conclusion>
""")
    
    print("\n" + "="*80)
    print("Integration with AReaL Training Loop:")
    print("="*80)
    print("""
# In your training config (e.g., examples/math/train_parallel.py):

from areal.workflow.parallel_generation import ParallelGenerationWorkflow

# Setup
workflow = ParallelGenerationWorkflow(
    reward_fn="path.to.your.reward_fn",
    gconfig=gconfig,
    tokenizer=tokenizer_path,
    max_goal_tokens=2048,
    max_path_tokens=4096,
    max_conclusion_tokens=2048,
)

# Use with engine
results = await engine.rollout_batch(
    data=train_dataset,
    workflow=workflow,
)

# The workflow returns trajectories with:
# {
#     'input_ids': tensor of all tokens,
#     'loss_mask': 1 for generated tokens, 0 for prompt,
#     'logprobs': log probabilities for training,
#     'versions': model version for each token,
#     'attention_mask': all ones,
#     'rewards': scalar reward value
# }
""")
    
    print("\nFor complete examples, see:")
    print("  - areal/workflow/parallel_generation_example.md")
    print("  - areal/workflow/rlvr.py (similar single-path workflow)")
    print("  - areal/workflow/multi_turn.py (multi-turn workflow)")


if __name__ == "__main__":
    asyncio.run(main())
