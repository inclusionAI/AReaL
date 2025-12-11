"""
Multiverse Structured Generation with SGLang - New Version
Handles parallel path generation without attention masking.

Generation structure:
<Parallel>
  <Goal>
    [Parallel Goal]
    <Outline>[Goal 1]</Outline>
    <Outline>[Goal 2]</Outline>
    ...
  </Goal>
  <Path>[Path 1]</Path>
  <Path>[Path 2]</Path>
  ...
  <Conclusion>[Conclusion]</Conclusion>
</Parallel>

Key differences from previous version:
1. No attention masking - all paths see the same context
2. Paths are generated in parallel (same context for all)
3. No hierarchical parallel generation (no nested <Parallel> stages)
4. No special tokens - uses stop words instead
"""

import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import re


# Configuration
MODEL_PATH = "/storage/openpsi/models/zzy/Multiverse-20251030_154726"

# System prompt
SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    """Format content using ChatML template"""
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<||im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_start}"
    return formatted


def extract_outline_prefixes(goal_text: str) -> List[str]:
    """
    Extract outline prefixes from the goal text.
    
    The goal text contains numbered items in various formats:
    - Flat: "1:", "2:", "3:"
    - Hierarchical: "1.1:", "1.2:", "2.1.1:", etc.
    
    Args:
        goal_text: The generated goal text
    
    Returns:
        List of outline prefixes (e.g., ["1", "2", "3"] or ["1.1", "1.2", "2.1"])
    """
    # Pattern to match hierarchical numbered items like "1:", "1.1:", "2.1.1:", etc.
    pattern = r'^\s*([\d.]+)\s*.'
    
    # Find all matches across all lines
    matches = re.findall(pattern, goal_text, re.MULTILINE)
    
    # Clean up the prefixes (remove trailing dots if any)
    prefixes = [m.rstrip('.') for m in matches]
    print(f"[Debug] prefixes: {prefixes}")
    return prefixes


@sgl.function
def generate_normal_until_goal(s, problem: str, max_tokens: int = 30000):
    """Generate normally until </Goal> tag is encountered"""
    # Format initial prompt
    formatted_prompt = format_chat_template(SYSTEM_PROMPT, problem, "")
    s += formatted_prompt
    
    # Generate until </Goal>
    s += sgl.gen(
        "normal_text",
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</Goal>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


@sgl.function
def generate_path(s, problem: str, context: str, path_prefix: str, max_tokens: int = 30000):
    """Generate a single path given the context and prefix"""
    # Build the prompt with context
    assistant_content = context + f"\n<Path>\n{path_prefix}"
    formatted_prompt = format_chat_template(SYSTEM_PROMPT, problem, assistant_content)
    
    s += formatted_prompt
    
    # Generate path content until </Path>
    s += sgl.gen(
        "path_content",
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</Path>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


@sgl.function
def generate_conclusion(s, problem: str, context: str, max_tokens: int = 30000):
    """Generate conclusion given the context with all paths"""
    # Build the prompt with full context including all paths
    assistant_content = context + "\n<Conclusion>"
    formatted_prompt = format_chat_template(SYSTEM_PROMPT, problem, assistant_content)
    
    s += formatted_prompt
    
    # Generate conclusion until </Conclusion>
    s += sgl.gen(
        "conclusion_content",
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</Conclusion>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


@sgl.function
def continue_normal_generation(s, problem: str, accumulated_context: str, max_tokens: int = 30000):
    """Continue main generation after parallel stage"""
    formatted_prompt = format_chat_template(SYSTEM_PROMPT, problem, accumulated_context)
    
    s += formatted_prompt
    
    # Generate until next </Goal> or end
    s += sgl.gen(
        "normal_text",
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["</Goal>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


class MultiverseGeneratorNew:
    """
    Handles structured generation with parallel paths without attention masking.
    All paths see the same context and are generated independently.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, host: str = "127.0.0.1", port: int = 30008, use_existing_server: bool = False):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.host = host
        self.port = port
        self.use_existing_server = use_existing_server
        self.backend = None
    
    def process_parallel_stage(
        self,
        problem: str,
        current_context: str,
        max_path_tokens: int = 30000,
        max_conclusion_tokens: int = 30000
    ) -> Tuple[str, Dict]:
        """
        Process a parallel stage by generating paths and conclusion.
        
        Args:
            problem: The original problem
            current_context: The context up to and including </Goal>
            max_path_tokens: Max tokens per path
            max_conclusion_tokens: Max tokens for conclusion
        
        Returns:
            Tuple of (updated_context, stage_info)
        """
        print("="*70)
        print("Entering Parallel Stage Mode")
        print("="*70)
        
        # Step 1: Extract goal text from context
        goal_matches = re.findall(r'<Goal>(.*?)</Goal>', current_context, re.DOTALL)
        if not goal_matches:
            raise ValueError("No goal found in context. Expected </Goal> marker.")
        
        goal_text = goal_matches[-1].strip()
        print(f"\n[1] Goal extracted from context")
        print(f"Goal text: {goal_text}")
        
        # Step 2: Extract outline prefixes
        outline_prefixes = extract_outline_prefixes(goal_text)
        num_paths = len(outline_prefixes)
        print(f"Detected {num_paths} outline(s) in goal:")
        for i, prefix in enumerate(outline_prefixes):
            print(f"  Path {i + 1}: prefix = '{prefix}'")
        
        if num_paths == 0:
            print("WARNING: No outlines found in goal. Defaulting to 1 path with prefix '1'.")
            num_paths = 1
            outline_prefixes = ["1"]
        
        # Step 3: Generate paths independently
        paths = []
        shared_context = current_context  # Context before any path
        
        for path_idx in range(num_paths):
            print(f"\n[{path_idx + 2}] Generating Path {path_idx + 1} of {num_paths}...")
            
            path_prefix = outline_prefixes[path_idx]
            
            # Generate this path
            path_state = generate_path.run(
                problem=problem,
                context=shared_context,
                path_prefix=path_prefix,
                max_tokens=max_path_tokens
            )
            
            path_content = path_state["path_content"]
            
            paths.append({
                'index': path_idx + 1,
                'prefix': path_prefix,
                'content': path_content,
                'full_text': f"\n<Path>\n{path_prefix}{path_content}"
            })
            
            print(f"Path {path_idx + 1} generated")
            print(f"Path content: {path_content}")
        
        # Step 4: Concatenate all paths to context
        updated_context = current_context
        for path in paths:
            path_full = f"\n<Path>\n{path['prefix']}{path['content']}</Path>"
            updated_context += path_full
        
        # Step 5: Generate conclusion
        print(f"\n[{num_paths + 2}] Generating Conclusion...")
        
        conclusion_state = generate_conclusion.run(
            problem=problem,
            context=updated_context,
            max_tokens=max_conclusion_tokens
        )
        
        conclusion_content = conclusion_state["conclusion_content"]
        updated_context += f"\n<Conclusion>{conclusion_content}</Conclusion>"
        
        print(f"Conclusion generated")
        print(f"Conclusion content: {conclusion_content}")
        # Compile stage info
        stage_info = {
            'goal': goal_text,
            'num_outlines': num_paths,
            'outline_prefixes': outline_prefixes,
            'paths': paths,
            'conclusion': conclusion_content,
        }
        
        print("\n" + "="*70)
        print(f"Parallel Stage Complete! ({num_paths} paths generated)")
        print("="*70)
        
        return updated_context, stage_info
    
    def run_generation(
        self,
        problem: str,
        max_normal_tokens: int = 30000,
        max_path_tokens: int = 30000,
        max_conclusion_tokens: int = 30000,
        max_total_tokens: int = 30000,
        temperature: float = 0.7
    ) -> Dict:
        """
        Main function to generate with automatic parallel detection.
        
        Args:
            problem: The problem statement
            max_normal_tokens: Max tokens for normal generation
            max_path_tokens: Max tokens per path
            max_conclusion_tokens: Max tokens for conclusion
            max_total_tokens: Maximum total tokens
            temperature: Sampling temperature
        
        Returns:
            Dictionary with full generation result and stage information
        """
        print("="*70)
        print("Starting Generation with Auto Parallel Detection")
        print("="*70)
        
        # Set up backend
        if not self.use_existing_server:
            backend = RuntimeEndpoint(f"http://{self.host}:{self.port}")
            sgl.set_default_backend(backend)
            print(f"✅ Connected to SGLang server at {self.host}:{self.port}")
        else:
            backend = RuntimeEndpoint(f"http://{self.host}:{self.port}")
            sgl.set_default_backend(backend)
            print(f"🔗 Using existing server at {self.host}:{self.port}")
        
        print(f"\nOriginal problem: {problem}\n")
        
        # Track stages and context
        stages = []
        current_context = ""
        total_token_count = 0
        main_turn_count = 1
        
        while total_token_count < max_total_tokens:
            print(f"\n🔄 MAIN TURN {main_turn_count}: Starting generation...")
            
            # Generate until </Goal> or end
            if current_context == "":
                # First generation
                state = generate_normal_until_goal.run(
                    problem=problem,
                    max_tokens=max_normal_tokens
                )
            else:
                # Continue from previous context
                state = continue_normal_generation.run(
                    problem=problem,
                    accumulated_context=current_context,
                    max_tokens=max_normal_tokens
                )
            
            normal_text = state["normal_text"]
            
            if normal_text:
                current_context += normal_text
                token_count = len(self.tokenizer.encode(normal_text, add_special_tokens=False))
                total_token_count += token_count
                print(f"Generated (normal): {len(normal_text)} chars")
            
            # Check if </Goal> was encountered (stopped by </Goal>)
            if normal_text.strip().endswith("</Outline>"):
                # Add closing </Goal> tag if needed
                if not normal_text.strip().endswith("</Goal>"):
                    current_context += "</Goal>"
                    total_token_count += len(self.tokenizer.encode("</Goal>", add_special_tokens=False))
                
                print(f"\n{'='*70}")
                print(f"Detected </Goal> - entering Parallel Stage {len(stages) + 1}")
                print(f"{'='*70}")
                
                # Process parallel stage
                current_context, stage_info = self.process_parallel_stage(
                    problem=problem,
                    current_context=current_context,
                    max_path_tokens=max_path_tokens,
                    max_conclusion_tokens=max_conclusion_tokens
                )
                
                total_token_count = len(self.tokenizer.encode(current_context, add_special_tokens=False))
                stages.append(stage_info)
                
                print(f"\nReturning to normal generation after parallel stage...")
            else:
                # No </Goal> detected, check if we should stop
                if not normal_text or len(normal_text.strip()) == 0:
                    print("\nNo more content to generate. Stopping.")
                    break
                
                # Check if there's a goal tag that might continue
                # if "<Goal>" not in normal_text and "</Goal>" not in normal_text:
                #     print("\nNo parallel stage detected in this turn. Stopping.")
                #     break
            
            main_turn_count += 1
            if main_turn_count > 20:  # Safety limit
                print(f"⚠️ Reached maximum main turn limit (20)")
                break
        
        # Compile final result
        result = {
            'problem': problem,
            'full_text': current_context,
            'stages': stages,
            'num_stages': len(stages),
            'total_tokens': total_token_count,
        }
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total tokens (approx): {total_token_count}")
        print(f"Parallel stages encountered: {len(stages)}")
        
        return result
    
    def print_result(self, result: Dict):
        """Pretty print the generation result."""
        print("\n" + "="*70)
        print("GENERATION RESULT")
        print("="*70)
        print(f"\nProblem: {result['problem']}")
        print(f"\nTotal Tokens (approx): {result['total_tokens']}")
        print(f"\nParallel Stages: {result.get('num_stages', 0)}")
        
        # Print each stage if present
        if 'stages' in result:
            for idx, stage in enumerate(result['stages']):
                print(f"\n--- Stage {idx + 1} ---")
                print(f"Goal: {stage['goal']}")
                print(f"Paths ({len(stage['paths'])}):")
                for path in stage['paths']:
                    print(f"  Path {path['index']} (prefix: {path.get('prefix', 'N/A')}): {path['full_text']}")
                print(f"Conclusion: {stage['conclusion']}")
        
        print(f"\n--- Full Generated Reasoning ---")
        print(result['full_text'])
        print("\n" + "="*70)


# ============================================================================
# Usage Examples
# ============================================================================

def example_auto_parallel():
    """Example: Generate with automatic parallel detection."""
    generator = MultiverseGeneratorNew()
    
    problem = r"""A list of positive integers has the following properties:
$\bullet$ The sum of the items in the list is $30$.
$\bullet$ The unique mode of the list is $9$.
$\bullet$ The median of the list is a positive integer that does not appear in the list itself.
Find the sum of the squares of all the items in the list."""
    
    result = generator.run_generation(
        problem=problem,
        max_normal_tokens=10000,
        # max_goal_tokens=10000,
        max_path_tokens=10000,
        max_conclusion_tokens=10000,
        max_total_tokens=32768
    )
    
    generator.print_result(result)
    return result


if __name__ == "__main__":
    print("Multiverse Structured Generation with SGLang - New Version")
    print("="*70)
    print(f"Model Path: {MODEL_PATH}")
    print("="*70)
    
    # Run auto-detect parallel stages example
    print("\n\n### Auto Parallel Detection ###")
    try:
        result = example_auto_parallel()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\nNote: This now uses SGLang's Python API with s += gen() pattern!")