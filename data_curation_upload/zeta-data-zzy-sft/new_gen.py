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

import requests
import json
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer


# Configuration
SGLANG_SERVER_URL = "http://localhost:30005"
SGLANG_GENERATE_ENDPOINT = f"{SGLANG_SERVER_URL}/generate"


class MultiverseGeneratorNew:
    """
    Handles structured generation with parallel paths without attention masking.
    All paths see the same context and are generated independently.
    """
    
    def __init__(self, server_url: str = SGLANG_SERVER_URL, model_name: str = "/storage/openpsi/models/zzy/Multiverse-20251030_154726"):
        self.server_url = server_url
        self.generate_endpoint = f"{server_url}/generate"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def format_chat_template(self, problem: str, previous_reasoning: str = "") -> str:
        """
        Format the chat template for the model.
        
        THIS FUNCTION IS LEFT BLANK FOR YOU TO IMPLEMENT.
        
        Args:
            problem: The original problem statement
            previous_reasoning: Any previous reasoning context (empty string if first call)
        
        Returns:
            Formatted prompt string ready to be tokenized and sent to the model
        """
        # TODO: Implement this function
        # This should format the problem and previous reasoning into a prompt
        # that the model can understand and continue from
        
        return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{problem}\n<|im_end|>\n<|im_start|>assistant{previous_reasoning}"
    
    def _send_generation_request(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        stop_strings: Optional[List[str]] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        Send generation request to SGLang server.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            stop_strings: List of stop strings (words/phrases to stop at)
            temperature: Sampling temperature
        
        Returns:
            Response dictionary from server
        """
        # DEBUG: Print the input being sent
        print("\n" + "="*70)
        print("DEBUG: Sending to model for generation")
        print("="*70)
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Input length: {len(input_ids)} tokens")
        print(f"Max new tokens: {max_new_tokens}")
        if stop_strings:
            print(f"Stop strings: {stop_strings}")
        print(f"\nInput text:\n{input_text}")
        print("="*70 + "\n")
        
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "skip_special_tokens": False,
            }
        }
        
        # Add stop strings if specified
        if stop_strings:
            payload["sampling_params"]["stop"] = stop_strings
        
        response = requests.post(self.generate_endpoint, json=payload)
        result = response.json()
        
        # DEBUG: Print the generated output
        print("\n" + "-"*70)
        print("DEBUG: Model response")
        print("-"*70)
        print(f"Full response: {result}")
        
        # Extract generated text
        if 'text' in result:
            generated_text = result.get('text', '')
            print(f"Generated text: {generated_text}")
        else:
            print("WARNING: No 'text' field in response")
        
        print("-"*70 + "\n")
        
        return result
    
    def _extract_outline_prefixes(self, goal_text: str) -> List[str]:
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
        import re
        
        # Pattern to match hierarchical numbered items like "1:", "1.1:", "2.1.1:", etc.
        pattern = r'^\s*([\d.]+)\s*.'
        
        # Find all matches across all lines
        matches = re.findall(pattern, goal_text, re.MULTILINE)
        
        # Clean up the prefixes (remove trailing dots if any)
        prefixes = [m.rstrip('.') for m in matches]
        print(f"[Debug] prefixes: {prefixes}")
        return prefixes
    
    def generate_parallel_stage(
        self,
        input_ids: List[int],
        max_goal_tokens: int = 200,
        max_path_tokens: int = 150,
        max_conclusion_tokens: int = 100
    ) -> Tuple[List[int], Dict]:
        """
        Generate a complete parallel stage with goal, multiple paths, and conclusion.
        
        Key differences from previous version:
        - No attention masking
        - All paths are generated from the same context (context = prompt + goal)
        - Paths are generated independently and in parallel
        - No hierarchical/nested parallel stages
        
        Args:
            input_ids: Current sequence of token IDs (should end after detecting parallel trigger)
            max_goal_tokens: Maximum tokens for goal generation
            max_path_tokens: Maximum tokens for each path
            max_conclusion_tokens: Maximum tokens for conclusion
        
        Returns:
            Tuple of (updated input_ids, stage_info_dict)
        """
        print("="*70)
        print("Entering Parallel Stage Mode")
        print("="*70)
        
        # Step 1: Add <Parallel> and <Goal> markers
        parallel_start = "<Parallel>\n<Goal>\n"
        parallel_start_ids = self.tokenizer.encode(parallel_start, add_special_tokens=False)
        input_ids.extend(parallel_start_ids)
        
        print(f"\n[1] Generating Goal Section...")
        print(f"Current sequence length: {len(input_ids)}")
        
        # Step 2: Generate goal until </Goal>
        goal_result = self._send_generation_request(
            input_ids=input_ids,
            max_new_tokens=max_goal_tokens,
            stop_strings=["</Goal>"]
        )
        
        # Extract generated goal text
        goal_text = goal_result.get('text', '')
        goal_ids = self.tokenizer.encode(goal_text, add_special_tokens=False) if goal_text else []
        
        input_ids.extend(goal_ids)
        
        # Add closing </Goal> tag
        goal_close = "</Goal>"
        goal_close_ids = self.tokenizer.encode(goal_close, add_special_tokens=False)
        input_ids.extend(goal_close_ids)
        
        print(f"Goal generated: {goal_text}")
        print(f"Current sequence length: {len(input_ids)}")
        
        # Step 3: Extract outline prefixes to determine number of paths
        outline_prefixes = self._extract_outline_prefixes(goal_text)
        num_paths = len(outline_prefixes)
        print(f"Detected {num_paths} outline(s) in goal:")
        for i, prefix in enumerate(outline_prefixes):
            print(f"  Path {i + 1}: prefix = '{prefix}'")
        
        if num_paths == 0:
            print("WARNING: No outlines found in goal. Defaulting to 1 path with prefix '1'.")
            num_paths = 1
            outline_prefixes = ["1"]
        
        # Step 4: Save the context that all paths will use
        # This is the state BEFORE any path is generated
        shared_context_ids = input_ids.copy()
        
        # Step 5: Generate multiple paths independently from the same context
        paths = []
        
        for path_idx in range(num_paths):
            print(f"\n[{path_idx + 2}] Generating Path {path_idx + 1} of {num_paths}...")
            
            # Start fresh from the shared context (prompt + goal)
            # This ensures all paths see the same context
            path_input_ids = shared_context_ids.copy()
            
            # Prefill path start with the outline prefix
            path_prefix = f"\n<Path>\n{outline_prefixes[path_idx]}"
            path_prefix_ids = self.tokenizer.encode(path_prefix, add_special_tokens=False)
            path_input_ids.extend(path_prefix_ids)
            
            print(f"Generating from shared context")
            print(f"Prefilled: {path_prefix}")
            
            # Generate path content until </Path>
            path_result = self._send_generation_request(
                input_ids=path_input_ids,
                max_new_tokens=max_path_tokens,
                stop_strings=["</Path>"]
            )
            
            # Extract path text
            path_text = path_result.get('text', '')
            
            paths.append({
                'index': path_idx + 1,
                'prefix': outline_prefixes[path_idx],
                'content': path_text,
                'full_text': f"{path_prefix}{path_text}"
            })
            
            print(f"Path {path_idx + 1} generated: {path_prefix}{path_text}")
        
        # Step 6: Concatenate all paths to the shared context for conclusion
        # Now we build the full sequence with all paths
        for path in paths:
            path_full = f"\n<Path>\n{path['prefix']}{path['content']}</Path>"
            path_full_ids = self.tokenizer.encode(path_full, add_special_tokens=False)
            input_ids.extend(path_full_ids)
        
        print(f"Current sequence length after all paths: {len(input_ids)}")
        
        # Step 7: Generate conclusion (sees all paths)
        print(f"\n[{num_paths + 2}] Generating Conclusion...")
        
        # Prefill conclusion tag
        conclusion_prefix = "\n<Conclusion>"
        conclusion_prefix_ids = self.tokenizer.encode(conclusion_prefix, add_special_tokens=False)
        input_ids.extend(conclusion_prefix_ids)
        
        # Generate conclusion until </Conclusion>
        conclusion_result = self._send_generation_request(
            input_ids=input_ids,
            max_new_tokens=max_conclusion_tokens,
            stop_strings=["</Conclusion>"]
        )
        
        # Extract conclusion text
        conclusion_text = conclusion_result.get('text', '')
        conclusion_ids = self.tokenizer.encode(conclusion_text, add_special_tokens=False) if conclusion_text else []
        
        input_ids.extend(conclusion_ids)
        
        # Add closing tags
        close_tags = "</Conclusion>\n</Parallel>"
        close_tags_ids = self.tokenizer.encode(close_tags, add_special_tokens=False)
        input_ids.extend(close_tags_ids)
        
        print(f"Conclusion generated: {conclusion_text}")
        print(f"Final sequence length: {len(input_ids)}")
        
        # Compile stage info
        stage_info = {
            'goal': goal_text,
            'num_outlines': num_paths,
            'outline_prefixes': outline_prefixes,
            'paths': paths,
            'conclusion': conclusion_text,
        }
        
        print("\n" + "="*70)
        print(f"Parallel Stage Complete! ({num_paths} paths generated)")
        print("="*70)
        
        return input_ids, stage_info
    
    def generate_with_auto_parallel_detection(
        self,
        problem: str,
        max_normal_tokens: int = 30000,
        max_goal_tokens: int = 30000,
        max_path_tokens: int = 30000,
        max_conclusion_tokens: int = 30000,
        max_total_tokens: int = 30000,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate text normally, automatically detecting and handling parallel stages.
        
        When parallel trigger phrase is detected (e.g., "Let's think in parallel"):
        - Enter parallel mode
        - Generate goal (which contains outline items)
        - Count the number of outline items to determine number of paths
        - Generate all paths independently from the same context
        - Generate conclusion (which sees all paths)
        - After </Parallel>, continue normal generation
        - Repeat until max_total_tokens or natural end
        
        Args:
            problem: The problem statement
            max_normal_tokens: Max tokens to generate in normal mode before checking
            max_goal_tokens: Max tokens for goal in parallel stage
            max_path_tokens: Max tokens per path in parallel stage
            max_conclusion_tokens: Max tokens for conclusion in parallel stage
            max_total_tokens: Maximum total tokens in sequence
            temperature: Sampling temperature
        
        Returns:
            Dictionary with full generation result and stage information
        """
        print("="*70)
        print("Starting Generation with Auto Parallel Detection")
        print("="*70)
        
        # Format the initial prompt using the chat template function
        formatted_prompt = self.format_chat_template(problem, "\n")
        print(f"\nOriginal problem: {problem}")
        print(f"Formatted prompt: {formatted_prompt}\n")
        
        # Initialize with formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        stages = []
        generation_log = []
        
        print(f"Initial sequence length: {len(input_ids)}\n")
        
        # Parallel trigger phrases to detect
        parallel_triggers = [
            "Let's think in parallel.",
            "Let's think in parallel",
            "in parallel.",
            "in parallel"
        ]
        
        while len(input_ids) < max_total_tokens:
            # Generate normally
            print(f"\n[Normal Generation] Current position: {len(input_ids)} tokens")
            
            normal_result = self._send_generation_request(
                input_ids=input_ids,
                max_new_tokens=max_normal_tokens,
                temperature=temperature,
                stop_strings="<Parallel>"
            )
            
            # Extract normal text
            normal_text = normal_result.get('text', '')
            normal_ids = self.tokenizer.encode(normal_text, add_special_tokens=False) if normal_text else []
            
            if normal_text:
                input_ids.extend(normal_ids)
                generation_log.append({
                    'type': 'normal',
                    'text': normal_text,
                    'token_count': len(normal_ids)
                })
                print(f"Generated (normal): {normal_text}")
            
            # Check if any parallel trigger phrase appears at the end of generated text
            triggered = False
            for trigger in parallel_triggers:
                if normal_text.strip().endswith(trigger):
                    triggered = True
                    print(f"\n{'='*70}")
                    print(f"Detected parallel trigger: '{trigger}'")
                    print(f"Entering Parallel Stage {len(stages) + 1}")
                    print(f"{'='*70}")
                    break
            
            if triggered:
                # Process parallel stage
                input_ids, stage_info = self.generate_parallel_stage(
                    input_ids=input_ids,
                    max_goal_tokens=max_goal_tokens,
                    max_path_tokens=max_path_tokens,
                    max_conclusion_tokens=max_conclusion_tokens
                )
                
                stages.append(stage_info)
                generation_log.append({
                    'type': 'parallel_stage',
                    'stage_number': len(stages),
                    'info': stage_info
                })
                
                print(f"\nReturning to normal generation after parallel stage...")
            else:
                # No parallel trigger detected
                # Check if we should continue or stop
                if not normal_text or len(normal_text.strip()) == 0:
                    print("\nNo more content to generate. Stopping.")
                    break
        
        # Compile final result
        final_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        result = {
            'problem': problem,
            'full_text': final_text,
            'stages': stages,
            'num_stages': len(stages),
            'total_tokens': len(input_ids),
            'generation_log': generation_log
        }
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total tokens: {len(input_ids)}")
        print(f"Parallel stages encountered: {len(stages)}")
        
        return result
    
    def print_result(self, result: Dict):
        """Pretty print the generation result."""
        print("\n" + "="*70)
        print("GENERATION RESULT")
        print("="*70)
        print(f"\nProblem: {result['problem']}")
        print(f"\nTotal Tokens: {result['total_tokens']}")
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
        
        print(f"\n--- Full Text ---")
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
    
    result = generator.generate_with_auto_parallel_detection(
        problem=problem,
        max_normal_tokens=10000,
        max_goal_tokens=10000,
        max_path_tokens=10000,
        max_conclusion_tokens=10000,
        max_total_tokens=32768
    )
    
    generator.print_result(result)
    return result


if __name__ == "__main__":
    print("Multiverse Structured Generation with SGLang - New Version")
    print("="*70)
    print(f"Server URL: {SGLANG_SERVER_URL}")
    print("="*70)
    
    # Run auto-detect parallel stages example
    print("\n\n### Auto Parallel Detection ###")
    try:
        result = example_auto_parallel()
    except NotImplementedError as e:
        print(f"\nNote: {e}")
        print("Please implement the format_chat_template function before running.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\nNote: Make sure your SGLang server is running!")
    print("Remember to implement the format_chat_template function.")