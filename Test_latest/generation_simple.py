"""
Simple Multiverse Generator with SGLang
Just generates until stop tokens without any special parallel processing.

Compatible with new_batch_inference_new.py
"""

import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint
from typing import Dict
from transformers import AutoTokenizer


# System prompt
SYSTEM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""


def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    """Format content using ChatML template"""
    formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_start}"
    return formatted


@sgl.function
def generate_simple(s, problem: str, max_tokens: int = 30000, temperature: float = 0.7):
    """Generate normally until stop tokens"""
    # Format initial prompt
    formatted_prompt = format_chat_template(SYSTEM_PROMPT, problem, "")
    s += formatted_prompt
    
    # Generate until stop tokens
    s += sgl.gen(
        "output",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stop=["<|im_end|>", "<|endoftext|>", "</s>"]
    )


class MultiverseGeneratorNew:
    """
    Simple generator that just generates until stop tokens.
    Compatible with new_batch_inference_new.py interface.
    """
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 30000, use_existing_server: bool = False):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.host = host
        self.port = port
        self.use_existing_server = use_existing_server
        self.backend = None
    
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
        Main function to generate with simple stopping at end tokens.
        
        Args:
            problem: The problem statement
            max_normal_tokens: Max tokens for generation (used as max_tokens)
            max_path_tokens: Ignored (for compatibility)
            max_conclusion_tokens: Ignored (for compatibility)
            max_total_tokens: Maximum total tokens (used as max_tokens)
            temperature: Sampling temperature
        
        Returns:
            Dictionary with full generation result
        """
        print("="*70)
        print("Starting Simple Generation")
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
        
        print(f"\nProblem: {problem}\n")
        
        # Use max_total_tokens as the limit, fallback to max_normal_tokens
        max_tokens = min(max_total_tokens, max_normal_tokens) if max_total_tokens else max_normal_tokens
        
        # Generate
        state = generate_simple.run(
            problem=problem,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        output = state["output"]
        
        # Count tokens
        total_tokens = len(self.tokenizer.encode(output, add_special_tokens=False))
        
        # Compile result (compatible with new_batch_inference_new.py)
        result = {
            'problem': problem,
            'full_text': output,
            'stages': [],  # No stages in simple generation
            'num_stages': 0,
            'total_tokens': total_tokens,
        }
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total tokens (approx): {total_tokens}")
        print(f"Output length: {len(output)} characters")
        
        return result
    
    def print_result(self, result: Dict):
        """Pretty print the generation result."""
        print("\n" + "="*70)
        print("GENERATION RESULT")
        print("="*70)
        print(f"\nProblem: {result['problem']}")
        print(f"\nTotal Tokens (approx): {result['total_tokens']}")
        print(f"\n--- Full Generated Text ---")
        print(result['full_text'])
        print("\n" + "="*70)


# ============================================================================
# Usage Example
# ============================================================================

def example_simple_generation():
    """Example: Simple generation."""
    generator = MultiverseGeneratorNew(
        model_path="/storage/openpsi/models/zzy/Multiverse-20251030_154726",
        host="localhost",
        port=30001,
        use_existing_server=False
    )
    
    problem = r"""A list of positive integers has the following properties:
$\bullet$ The sum of the items in the list is $30$.
$\bullet$ The unique mode of the list is $9$.
$\bullet$ The median of the list is a positive integer that does not appear in the list itself.
Find the sum of the squares of all the items in the list."""
    
    result = generator.run_generation(
        problem=problem,
        max_normal_tokens=10000,
        max_total_tokens=32768,
        temperature=0.6
    )
    
    generator.print_result(result)
    return result


if __name__ == "__main__":
    print("Simple Multiverse Generator with SGLang")
    print("="*70)
    
    try:
        result = example_simple_generation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
