
"""
Multiverse Structured Generation with SGLang
Handles parallel path generation with custom attention masking.

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
"""

import requests
import json
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import time
# Configuration
SGLANG_SERVER_URL = "http://localhost:30000"
SGLANG_GENERATE_ENDPOINT = f"{SGLANG_SERVER_URL}/generate"

# Special token IDs (replace these with actual token IDs from your tokenizer)
SPECIAL_TOKENS = {
    "<Parallel>": 151667,
    "</Parallel>": 151668,
    "<Goal>": 151669,
    "</Goal>": 151670,
    "<Outline>": 151671,
    "</Outline>": 151672,
    "<Path>": 151673,
    "</Path>": 151674,
    "<Conclusion>": 151675,
    "</Conclusion>": 151676,
}

# Reverse mapping for decoding
TOKEN_ID_TO_NAME = {v: k for k, v in SPECIAL_TOKENS.items()}


class MultiverseGenerator:
    """
    Handles structured generation with parallel paths and attention masking.
    """
    
    def __init__(self, server_url: str = SGLANG_SERVER_URL, model_name: str = "/storage/openpsi/models/zzy/Multiverse-20251030_154726"):
        self.server_url = server_url
        self.generate_endpoint = f"{server_url}/generate"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens to tokenizer if not already present
        self._setup_special_tokens()
    
    def _apply_chat_template(self, prompt: str) -> str:
        """
        Apply chat template to the prompt.
        
        Args:
            prompt: Raw prompt text
        
        Returns:
            Formatted prompt with chat template applied
        """
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted_prompt
    
    def _setup_special_tokens(self):
        """Add special tokens to the tokenizer."""
        special_tokens_list = list(SPECIAL_TOKENS.keys())
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_list
        })
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")
    
    def _send_generation_request(
        self,
        input_ids: List[int],
        attention_mask: Optional[List[int]] = None,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[int]] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        Send generation request to SGLang server with custom attention mask.
        """
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        
        # DEBUG: Print the input text being sent to the model
        print("\n" + "="*70)
        print("DEBUG: Sending to model for generation")
        print("="*70)
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Input length: {len(input_ids)} tokens")
        print(f"Max new tokens: {max_new_tokens}")
        if stop_tokens:
            stop_token_names = [TOKEN_ID_TO_NAME.get(tid, f"ID:{tid}") for tid in stop_tokens]
            print(f"Stop tokens: {stop_token_names}")
        print(f"\nInput text:\n{input_text}")
        print("="*70 + "\n")
        # time.sleep(5)
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "skip_special_tokens": False,  # Ensure special tokens are preserved
            }
        }
        
        # Add stop tokens if specified
        if stop_tokens:
            payload["sampling_params"]["stop_token_ids"] = stop_tokens
        
        response = requests.post(self.generate_endpoint, json=payload)
        result = response.json()
        
        # DEBUG: Print the generated output
        print("\n" + "-"*70)
        print("DEBUG: Model response")
        print("-"*70)
        print(f"Full response: {result}")
        
        # Try to get token IDs if available
        if 'output_ids' in result or 'token_ids' in result:
            generated_ids = result.get('output_ids', result.get('token_ids', []))
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            result['decoded_text_with_special_tokens'] = generated_text
            print(f"Generated token IDs: {generated_ids}")
            print(f"Generated text (with special tokens): {generated_text}")
        else:
            generated_text = result.get('text', '')
            print(f"Generated text (from server): {generated_text}")
            print("WARNING: Server returned text instead of token IDs. Special tokens may be lost!")
        
        print("-"*70 + "\n")
        
        return result
    
    def _create_masked_attention(
        self,
        base_mask: List[int],
        mask_ranges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Create attention mask by masking specific token ranges.
        
        Args:
            base_mask: Base attention mask (usually all 1s)
            mask_ranges: List of (start, end) tuples to mask
        
        Returns:
            Modified attention mask with masked positions set to 0
        """
        attention_mask = base_mask.copy()
        for start, end in mask_ranges:
            for i in range(start, end):
                if 0 <= i < len(attention_mask):
                    attention_mask[i] = 0
        return attention_mask
    
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
        # This matches lines that start with numbers (with dots) followed by a colon
        pattern = r'^\s*([\d.]+)\s*.'
        
        # Find all matches across all lines
        matches = re.findall(pattern, goal_text, re.MULTILINE)
        
        # Clean up the prefixes (remove trailing dots if any)
        prefixes = [m.rstrip('.') for m in matches]
        print (f"[Debug]prefixes: {prefixes}")
        return prefixes
    
    def _count_outlines_in_goal(self, goal_text: str) -> int:
        """
        Count the number of outlines in the goal text by parsing numbered items.
        
        The goal text contains numbered items in the format:
        1: Description
        2: Description
        or hierarchical:
        1.1: Description
        1.2: Description
        
        Args:
            goal_text: The generated goal text
        
        Returns:
            Number of numbered outline items found
        """
        prefixes = self._extract_outline_prefixes(goal_text)
        return len(prefixes)
    
    def generate_parallel_stage(
        self,
        input_ids: List[int],
        max_goal_tokens: int = 200,
        max_path_tokens: int = 150,
        max_conclusion_tokens: int = 100,
        parent_mask_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[List[int], Dict]:
        """
        Generate a complete parallel stage with goal, multiple paths, and conclusion.
        Assumes <Parallel> token has already been generated and is at the end of input_ids.
        
        The number of paths is determined by the number of <Outline> tags in the goal.
        Supports hierarchical parallel reasoning where paths can themselves contain parallel stages.
        
        Args:
            input_ids: Current sequence of token IDs (should end with <Parallel>)
            max_goal_tokens: Maximum tokens for goal generation
            max_path_tokens: Maximum tokens for each path
            max_conclusion_tokens: Maximum tokens for conclusion
            parent_mask_ranges: Ranges to mask from parent parallel stage (for nested reasoning)
        
        Returns:
            Tuple of (updated input_ids, stage_info_dict)
        """
        if parent_mask_ranges is None:
            parent_mask_ranges = []
        
        print("="*70)
        print("Entering Parallel Stage Mode")
        if parent_mask_ranges:
            print(f"Nested stage - inheriting {len(parent_mask_ranges)} mask ranges from parent")
        print("="*70)
        
        # Step 1: Add <Goal> token to start goal generation
        # input_ids.append(SPECIAL_TOKENS["<Goal>"])
        
        print(f"\n[1] Generating Goal Section...")
        print(f"Current sequence length: {len(input_ids)}")
        
        # Step 2: Generate goal until </Goal>
        goal_result = self._send_generation_request(
            input_ids=input_ids,
            max_new_tokens=max_goal_tokens,
            stop_tokens=[SPECIAL_TOKENS["</Goal>"]]
        )
        
        # Extract generated goal text and token IDs
        # Try to get token IDs first, fall back to text
        if 'text' in goal_result:
            # Server returned decoded text - encode it back
            goal_text = goal_result.get('text', '')
            goal_ids = self.tokenizer.encode(goal_text, add_special_tokens=False)
        elif 'output_ids' in goal_result:
            goal_ids = goal_result['output_ids']
            # Remove prefix tokens (the input_ids we sent)
            if len(goal_ids) > len(input_ids):
                goal_ids = goal_ids[len(input_ids):]
            goal_text = self.tokenizer.decode(goal_ids, skip_special_tokens=False)
        else:
            goal_ids = goal_result['token_ids']
            goal_text = self.tokenizer.decode(goal_ids, skip_special_tokens=False)
        
        
        input_ids.extend(goal_ids)
        input_ids.append(SPECIAL_TOKENS["</Goal>"])
        
        print(f"Goal generated: {goal_text}")
        print(f"Current sequence length: {len(input_ids)}")
        
        # Step 3: Extract outline prefixes to determine path numbering
        outline_prefixes = self._extract_outline_prefixes(goal_text)
        num_paths = len(outline_prefixes)
        print(f"Detected {num_paths} outline(s) in goal:")
        for i, prefix in enumerate(outline_prefixes):
            print(f"  Path {i + 1}: prefix = '{prefix}'")
        
        if num_paths == 0:
            print("WARNING: No outlines found in goal. Defaulting to 1 path with prefix '1'.")
            num_paths = 1
            outline_prefixes = ["1"]
        
        # Step 4: Generate multiple paths with masking
        paths = []
        path_ranges = []  # Track (start, end) positions of each path
        
        for path_idx in range(num_paths):
            print(f"\n[{path_idx + 2}] Generating Path {path_idx + 1} of {num_paths}...")
            
            # Record the starting position of this path
            path_start_pos = len(input_ids)
            
            # Prefill path start token with the outline prefix (not just an integer)
            path_prefix = f"\n<Path>\n{outline_prefixes[path_idx]}"
            path_prefix_ids = self.tokenizer.encode(path_prefix, add_special_tokens=False)
            input_ids.extend(path_prefix_ids)
            
            # Create attention mask that masks:
            # 1. All previous sibling paths in THIS stage
            # 2. All parent-level masked ranges (from parent parallel stages)
            base_attention_mask = [1] * len(input_ids)
            
            # Combine current stage's path masks with parent's mask ranges
            combined_mask_ranges = parent_mask_ranges + path_ranges
            attention_mask = self._create_masked_attention(base_attention_mask, combined_mask_ranges)
            
            masked_positions = [i for i, m in enumerate(attention_mask) if m == 0]
            print(f"Masked token positions: {len(masked_positions)} positions")
            print(f"  - Parent stage masks: {len(parent_mask_ranges)} ranges")
            print(f"  - Sibling path masks: {len(path_ranges)} ranges")
            print(f"Prefilled: {path_prefix}")
            
            # Generate path content until either </Path> or <Parallel>
            # This allows detecting nested parallel reasoning mid-path
            nested_stages = []
            path_content_parts = []  # Collect all parts of the path content
            
            while True:
                # Generate with both </Path> and <Parallel> as stop tokens
                path_result = self._send_generation_request(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_path_tokens,
                    stop_tokens=[SPECIAL_TOKENS["</Path>"], SPECIAL_TOKENS["<Parallel>"]]
                )
                
                # Extract generated text and token IDs
                if 'text' in path_result:
                    chunk_text = path_result.get('text', '')
                    chunk_ids = self.tokenizer.encode(chunk_text, add_special_tokens=False)
                elif 'output_ids' in path_result:
                    chunk_ids = path_result['output_ids']
                    if len(chunk_ids) > len(input_ids):
                        chunk_ids = chunk_ids[len(input_ids):]
                    chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=False)
                else:
                    chunk_ids = path_result['token_ids']
                    chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=False)
                
                # Add this chunk to the path
                input_ids.extend(chunk_ids)
                path_content_parts.append(chunk_text)
                
                # Check which stop condition was hit
                if chunk_text.strip().endswith("Let's think in parallel.") or chunk_text.strip().endswith("Let's think in parallel.\n") or chunk_text.strip().endswith("in parallel.") or chunk_text.strip().endswith("in parallel") or chunk_text.strip().endswith("in parallel "):
                    # Nested parallel reasoning detected
                    print(f"\n>>> Detected nested parallel reasoning in Path {path_idx + 1}!")
                    # time.sleep(5)
                    # Add <Parallel> token to start nested stage
                    input_ids.append(SPECIAL_TOKENS["<Parallel>"])
                    
                    # Generate nested parallel stage with current mask ranges
                    # This includes both parent masks and sibling path masks
                    nested_input_ids, nested_stage_info = self.generate_parallel_stage(
                        input_ids=input_ids,
                        max_goal_tokens=max_goal_tokens,
                        max_path_tokens=max_path_tokens,
                        max_conclusion_tokens=max_conclusion_tokens,
                        parent_mask_ranges=combined_mask_ranges
                    )
                    
                    # Update input_ids with the nested stage results
                    input_ids = nested_input_ids
                    nested_stages.append(nested_stage_info)
                    
                    # Update attention mask for continued generation after nested stage
                    base_attention_mask = [1] * len(input_ids)
                    attention_mask = self._create_masked_attention(base_attention_mask, combined_mask_ranges)
                    
                    print(f">>> Completed nested parallel stage in Path {path_idx + 1}")
                    print(f">>> Continuing to generate rest of Path {path_idx + 1}...")
                    
                    # Continue generating the rest of the path after the nested stage
                    continue
                else:
                    # Reached </Path> - path is complete
                    break
            
            # Combine all path content parts
            path_text = ''.join(path_content_parts)
            
            # Close the path
            input_ids.append(SPECIAL_TOKENS["</Path>"])
            
            # Record the range of this path (including prefix and content, excluding tags)
            path_end_pos = len(input_ids)
            path_ranges.append((path_start_pos, path_end_pos))
            
            paths.append({
                'index': path_idx + 1,
                'prefix': path_prefix,
                'content': path_text,
                'full_text': f"{path_prefix}{path_text}",
                'nested_stages': nested_stages
            })
            
            print(f"Path {path_idx + 1} generated: {path_prefix}{path_text}")
            if nested_stages:
                print(f"  (Contains {len(nested_stages)} nested parallel stage(s))")
            print(f"Current sequence length: {len(input_ids)}")
            # time.sleep(5)
        # Step 5: Generate conclusion (no masking needed)
        print(f"\n[{num_paths + 2}] Generating Conclusion...")
        # time.sleep(5)
        # Prefill conclusion tag
        conclusion_prefix = "\n<Conclusion>"
        conclusion_prefix_ids = self.tokenizer.encode(conclusion_prefix, add_special_tokens=False)
        input_ids.extend(conclusion_prefix_ids)
        
        # Generate conclusion until </Conclusion>
        conclusion_result = self._send_generation_request(
            input_ids=input_ids,
            max_new_tokens=max_conclusion_tokens,
            stop_tokens=[SPECIAL_TOKENS["</Conclusion>"]]
        )
        
        # Extract conclusion text and token IDs
        if 'text' in conclusion_result:
            conclusion_text = conclusion_result.get('text', '')
            conclusion_ids = self.tokenizer.encode(conclusion_text, add_special_tokens=False)
        elif 'output_ids' in conclusion_result:
            conclusion_ids = conclusion_result['output_ids']
            if len(conclusion_ids) > len(input_ids):
                conclusion_ids = conclusion_ids[len(input_ids):]
            conclusion_text = self.tokenizer.decode(conclusion_ids, skip_special_tokens=False)
        else:
            conclusion_ids = conclusion_result['token_ids']
            conclusion_text = self.tokenizer.decode(conclusion_ids, skip_special_tokens=False)
        
        
        input_ids.extend(conclusion_ids)
        input_ids.append(SPECIAL_TOKENS["</Conclusion>"])
        input_ids.extend(self.tokenizer.encode("\n", add_special_tokens=False))
        input_ids.append(SPECIAL_TOKENS["</Parallel>"])
        
        print(f"Conclusion generated: {conclusion_text}")
        print(f"Final sequence length: {len(input_ids)}")
        # time.sleep(5)
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
        if parent_mask_ranges:
            print("(Nested stage - returning to parent)")
        print("="*70)
        
        return input_ids, stage_info
    
    def generate_with_auto_parallel_detection(
        self,
        prompt: str,
        max_normal_tokens: int = 30000,
        max_goal_tokens: int = 30000,
        max_path_tokens: int = 30000,
        max_conclusion_tokens: int = 30000,
        max_total_tokens: int = 30000,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate text normally, automatically detecting and handling <Parallel> stages.
        
        When <Parallel> is generated:
        - Enter parallel mode
        - Generate goal (which contains <Outline> tags)
        - Count the number of <Outline> tags to determine number of paths
        - Generate that many paths (with masking)
        - Generate conclusion
        - After </Parallel>, continue normal generation
        - Repeat until max_total_tokens or natural end
        
        Args:
            prompt: Initial prompt text (will have chat template applied)
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
        
        # Apply chat template to the prompt
        formatted_prompt = self._apply_chat_template(prompt)
        print(f"\nOriginal prompt: {prompt}")
        print(f"Formatted prompt: {formatted_prompt}\n")
        
        # Initialize with formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        stages = []
        generation_log = []
        
        print(f"Initial sequence length: {len(input_ids)}\n")
        
        while len(input_ids) < max_total_tokens:
            # Generate normally until we hit <Parallel> or max tokens
            print(f"\n[Normal Generation] Current position: {len(input_ids)} tokens")
            
            normal_result = self._send_generation_request(
                input_ids=input_ids,
                max_new_tokens=max_normal_tokens,
                stop_tokens=[SPECIAL_TOKENS["<Parallel>"]],
                temperature=temperature
            )
            
            # Extract normal text and token IDs
            if 'text' in normal_result:
                normal_text = normal_result.get('text', '')
                normal_ids = self.tokenizer.encode(normal_text, add_special_tokens=False) if normal_text else []
            elif 'output_ids' in normal_result:
                normal_ids = normal_result['output_ids']
                if len(normal_ids) > len(input_ids):
                    normal_ids = normal_ids[len(input_ids):]
                normal_text = self.tokenizer.decode(normal_ids, skip_special_tokens=False)
            else:
                normal_ids = normal_result['token_ids']
                normal_text = self.tokenizer.decode(normal_ids, skip_special_tokens=False)
            
            
            if normal_text:
                input_ids.extend(normal_ids)
                generation_log.append({
                    'type': 'normal',
                    'text': normal_text,
                    'token_count': len(normal_ids)
                })
                print(f"Generated (normal): {normal_text}")
            
            # Check if we hit <Parallel> token
            # The stop_tokens parameter stops BEFORE generating the token, so we need to check
            # if the model would have generated it next
            
            #test_result = self._send_generation_request(
            #    input_ids=input_ids,
            #    max_new_tokens=1,
            #    temperature=0.0  # Greedy to check most likely next token
            #)
            
            # Check if next token would be <Parallel>
            #next_text = test_result.get('text', '')
            #next_ids = self.tokenizer.encode(next_text, add_special_tokens=False) if next_text else [] next_ids and next_ids[0] == SPECIAL_TOKENS["<Parallel>"]
            
            if  normal_text.strip().endswith("Let's think in parallel.") or normal_text.strip().endswith("in parallel") or normal_text.strip().endswith("in parallel "):
                # Add <Parallel> token
                print ("[DEBUG]: ENTERING PARALLEL STAGE")
                # input_ids.append(SPECIAL_TOKENS["<Parallel>"])
                print(f"\n{'='*70}")
                print(f"Detected <Parallel> token - Entering Parallel Stage {len(stages) + 1}")
                print(f"{'='*70}")
                input_ids.append (SPECIAL_TOKENS["<Parallel>"])
                # Process parallel stage (num_paths determined by outlines in goal)
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
                # No more <Parallel> tokens detected
                # Check if we should continue or stop
                if not normal_text or len(normal_text.strip()) == 0:
                    print("\nNo more content to generate. Stopping.")
                    break
        
        # Compile final result
        final_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        result = {
            'prompt': prompt,
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
        """Pretty print the generation result with support for nested stages."""
        print("\n" + "="*70)
        print("GENERATION RESULT")
        print("="*70)
        print(f"\nPrompt: {result['prompt']}")
        print(f"\nTotal Tokens: {result['total_tokens']}")
        print(f"\nParallel Stages: {result.get('num_stages', 0)}")
        
        # Print each stage if present
        if 'stages' in result:
            for idx, stage in enumerate(result['stages']):
                self._print_stage(stage, idx + 1, indent=0)
        
        print(f"\n--- Full Text ---")
        print(result['full_text'])
        print("\n" + "="*70)
    
    def _print_stage(self, stage: Dict, stage_num: int, indent: int = 0):
        """Recursively print a stage and its nested stages."""
        prefix = "  " * indent
        print(f"\n{prefix}--- Stage {stage_num} ---")
        print(f"{prefix}Goal: {stage['goal']}")
        print(f"{prefix}Paths ({len(stage['paths'])}):")
        for path in stage['paths']:
            print(f"{prefix}  Path {path['index']} (prefix: {path.get('prefix', 'N/A')}): {path['full_text']}")
            
            # Print nested stages if present
            if 'nested_stages' in path and path['nested_stages']:
                for nested_idx, nested_stage in enumerate(path['nested_stages']):
                    self._print_stage(nested_stage, f"{stage_num}.{nested_idx + 1}", indent + 2)
        
        print(f"{prefix}Conclusion: {stage['conclusion']}")


# ============================================================================
# Usage Examples
# ============================================================================

def example_auto_parallel():
    """Example: Generate with automatic parallel detection."""
    generator = MultiverseGenerator()
    
    prompt = r"""A list of positive integers has the following properties:
$\bullet$ The sum of the items in the list is $30$.
$\bullet$ The unique mode of the list is $9$.
$\bullet$ The median of the list is a positive integer that does not appear in the list itself.
Find the sum of the squares of all the items in the list."""
    
    result = generator.generate_with_auto_parallel_detection(
        prompt=prompt,
        max_normal_tokens=10000,
        max_goal_tokens=10000,
        max_path_tokens=10000,
        max_conclusion_tokens=10000,
        max_total_tokens=32768
    )
    
    generator.print_result(result)
    return result


if __name__ == "__main__":
    print("Multiverse Structured Generation with SGLang")
    print("="*70)
    print(f"Server URL: {SGLANG_SERVER_URL}")
    print("="*70)
    
    # Run auto-detect parallel stages example
    print("\n\n### Auto Parallel Detection ###")
    try:
        result = example_auto_parallel()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\nNote: Make sure your SGLang server is running and supports attention_mask parameter!")
    print("Update the SPECIAL_TOKENS dictionary with your actual token IDs.")

