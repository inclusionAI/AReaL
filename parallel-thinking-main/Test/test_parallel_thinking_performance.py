import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from problems import problems

def find_latest_checkpoint(base_dir):
    """Find the latest checkpoint in the model directory"""
    checkpoints = []
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if item.startswith("checkpoint-"):
                match = re.search(r'checkpoint-(\d+)', item)
                if match:
                    checkpoints.append((int(match.group(1)), item))
    
    if checkpoints:
        latest_checkpoint_num, latest_checkpoint = max(checkpoints)
        return os.path.join(base_dir, latest_checkpoint)
    return None

def check_parallel_thinking_tokens(text):
    """Check if text contains parallel thinking tokens"""
    tokens = [
        "<reasoning_process>",
        "</reasoning_process>",  # Added closing tag
        "<parallel_processing>", 
        "<launch_threads>",
        "<thread id='0'>",
        "<think: type = ''>"
    ]
    
    found_tokens = {}
    for token in tokens:
        found_tokens[token] = token in text
    
    return found_tokens

def calculate_additional_metrics(text):
    """Calculate additional metrics for parallel thinking analysis"""
    metrics = {
        'number_of_threads': 0,
        'max_width': 0,
        'token_count': 0,
        'word_count': 0
    }
    
    # 1. Number of threads: max number of <thread id='n'>, where n is number, the number of threads is n+1
    thread_pattern = r"<thread id='(\d+)'>"
    thread_matches = re.findall(thread_pattern, text)
    if thread_matches:
        max_thread_id = max(int(match) for match in thread_matches)
        metrics['number_of_threads'] = max_thread_id + 1
    
    # 2. Max width: maximum number of <task> tokens between each pair of <launch_threads> </launch_threads>
    launch_pattern = r"<launch_threads>(.*?)</launch_threads>"
    launch_blocks = re.findall(launch_pattern, text, re.DOTALL)
    
    max_tasks = 0
    for block in launch_blocks:
        task_count = block.count("<task>")
        max_tasks = max(max_tasks, task_count)
    metrics['max_width'] = max_tasks
    
    # 3. Token count: number of tokens (using simple whitespace split)
    # This is an approximation - for exact token count you'd need the tokenizer
    tokens = text.split()
    metrics['token_count'] = len(tokens)
    
    # 4. Word count: number of words, regarding each special token as one word
    special_tokens = [
        "<reasoning_process>", "</reasoning_process>", "<parallel_processing>", 
        "<launch_threads>", "</launch_threads>", "<task>", "</task>",
        "<thread", "</thread>", "<think:", "</think>"
    ]
    
    # Count regular words
    word_count = len(text.split())
    
    # Adjust for special tokens (they might be counted as multiple words in split())
    for token_pattern in ["<[^>]+>"]:  # Match any tag
        special_token_matches = re.findall(token_pattern, text)
        # Each special token should count as 1 word
        for match in special_token_matches:
            # Subtract the extra words this token might have been counted as
            token_words = len(match.split()) if len(match.split()) > 1 else 1
            word_count = word_count - token_words + 1
    
    metrics['word_count'] = word_count
    
    return metrics

def test_model_performance(model, tokenizer, problems, max_problems=None):
    """Test model performance on the problem set"""
    special_token = "<reasoning_process>\n<think: type = ''>\n"  
    results = []
    total_problems = len(problems) if max_problems is None else min(max_problems, len(problems))
    
    print(f"Testing model on {total_problems} problems...")
    print("=" * 80)
    
    for i, problem in enumerate(problems[:total_problems]):
        print(f"\nProblem {i+1}/{total_problems}:")
        print("-" * 40)
        
        try:
            # Prepare input
            problem_with_special_token = problem + special_token
            inputs = tokenizer(problem_with_special_token, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_response = special_token + full_response
            response = full_response[len(problem):].strip()
            
            # Check for parallel thinking tokens
            token_results = check_parallel_thinking_tokens(response)
            found_tokens = [token for token, found in token_results.items() if found]
            has_parallel_thinking = len(found_tokens) > 0
            
            # Calculate additional metrics
            additional_metrics = calculate_additional_metrics(response)
            
            # Store result
            result = {
                'problem_id': i + 1,
                'problem': problem[:100] + "..." if len(problem) > 100 else problem,
                'response_length': len(response),
                'has_parallel_thinking': has_parallel_thinking,
                'token_results': token_results,  # Individual token results
                'found_tokens': found_tokens,
                'full_response': response,
                'metrics': additional_metrics  # New metrics
            }
            results.append(result)
            
            # Print summary for this problem
            print(f"Response length: {len(response)} characters")
            print(f"Parallel thinking tokens found: {found_tokens}")
            print(f"Has parallel thinking: {'✓' if has_parallel_thinking else '✗'}")
            
            # Show individual token results
            for token, found in token_results.items():
                status = "✓" if found else "✗"
                print(f"  {token}: {status}")
            
            # Show additional metrics
            print(f"Additional metrics:")
            print(f"  Number of threads: {additional_metrics['number_of_threads']}")
            print(f"  Max width (tasks): {additional_metrics['max_width']}")
            print(f"  Token count: {additional_metrics['token_count']}")
            print(f"  Word count: {additional_metrics['word_count']}")
            
            # Show first part of response
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"Response preview: {preview}")
            
        except Exception as e:
            print(f"Error processing problem {i+1}: {e}")
            result = {
                'problem_id': i + 1,
                'problem': problem[:100] + "..." if len(problem) > 100 else problem,
                'error': str(e),
                'has_parallel_thinking': False,
                'token_results': {token: False for token in [
                    "<reasoning_process>", "</reasoning_process>", "<parallel_processing>", 
                    "<launch_threads>", "<thread id='0'>", "<think: type = ''>"
                ]},
                'found_tokens': [],
                'metrics': {'number_of_threads': 0, 'max_width': 0, 'token_count': 0, 'word_count': 0}
            }
            results.append(result)
    
    return results

def print_summary_stats(results):
    """Print summary statistics"""
    total_problems = len(results)
    successful_problems = [r for r in results if 'error' not in r]
    problems_with_parallel_thinking = [r for r in successful_problems if r['has_parallel_thinking']]
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total problems tested: {total_problems}")
    print(f"Successfully processed: {len(successful_problems)}")
    print(f"Problems with parallel thinking tokens: {len(problems_with_parallel_thinking)}")
    print(f"Overall parallel thinking success rate: {len(problems_with_parallel_thinking)/len(successful_problems)*100:.1f}%")
    
    # Individual token analysis
    token_names = [
        "<reasoning_process>",
        "</reasoning_process>",  # Added closing tag
        "<parallel_processing>", 
        "<launch_threads>",
        "<thread id='0'>",
        "<think: type = ''>"
    ]
    
    print(f"\nINDIVIDUAL TOKEN STATISTICS:")
    print("-" * 50)
    
    for token in token_names:
        count = sum(1 for result in successful_problems 
                   if result['token_results'].get(token, False))
        ratio = count / len(successful_problems) * 100
        print(f"{token}")
        print(f"  Found in: {count}/{len(successful_problems)} problems ({ratio:.1f}%)")
        print()
    
    # Additional metrics analysis
    print("ADDITIONAL METRICS ANALYSIS:")
    print("-" * 50)
    
    # Collect metrics from all successful problems
    thread_counts = [r['metrics']['number_of_threads'] for r in successful_problems]
    max_widths = [r['metrics']['max_width'] for r in successful_problems]
    token_counts = [r['metrics']['token_count'] for r in successful_problems]
    word_counts = [r['metrics']['word_count'] for r in successful_problems]
    
    # Problems with threads (non-zero thread count)
    problems_with_threads = [r for r in successful_problems if r['metrics']['number_of_threads'] > 0]
    problems_with_tasks = [r for r in successful_problems if r['metrics']['max_width'] > 0]
    
    print(f"Thread Analysis:")
    print(f"  Problems with threads: {len(problems_with_threads)}/{len(successful_problems)} ({len(problems_with_threads)/len(successful_problems)*100:.1f}%)")
    if thread_counts:
        print(f"  Average threads per response: {sum(thread_counts)/len(thread_counts):.1f}")
        print(f"  Max threads in any response: {max(thread_counts)}")
        print(f"  Thread count distribution: {dict(zip(*zip(*[(tc, thread_counts.count(tc)) for tc in set(thread_counts)])))}") 
    
    print(f"\nTask Width Analysis:")
    print(f"  Problems with task blocks: {len(problems_with_tasks)}/{len(successful_problems)} ({len(problems_with_tasks)/len(successful_problems)*100:.1f}%)")
    if max_widths:
        print(f"  Average max width per response: {sum(max_widths)/len(max_widths):.1f}")
        print(f"  Highest max width: {max(max_widths)}")
        print(f"  Width distribution: {dict(zip(*zip(*[(mw, max_widths.count(mw)) for mw in set(max_widths)])))}") 
    
    print(f"\nLength Analysis:")
    if token_counts:
        print(f"  Average token count: {sum(token_counts)/len(token_counts):.1f}")
        print(f"  Average word count: {sum(word_counts)/len(word_counts):.1f}")
        print(f"  Max token count: {max(token_counts)}")
        print(f"  Max word count: {max(word_counts)}")
    
    # Additional analysis for reasoning process tags
    print("\nREASONING PROCESS TAG ANALYSIS:")
    print("-" * 40)
    open_tag_count = sum(1 for result in successful_problems 
                        if result['token_results'].get("<reasoning_process>", False))
    close_tag_count = sum(1 for result in successful_problems 
                         if result['token_results'].get("</reasoning_process>", False))
    both_tags_count = sum(1 for result in successful_problems 
                         if result['token_results'].get("<reasoning_process>", False) and 
                            result['token_results'].get("</reasoning_process>", False))
    
    print(f"Problems with opening <reasoning_process>: {open_tag_count} ({open_tag_count/len(successful_problems)*100:.1f}%)")
    print(f"Problems with closing </reasoning_process>: {close_tag_count} ({close_tag_count/len(successful_problems)*100:.1f}%)")
    print(f"Problems with both tags (complete reasoning blocks): {both_tags_count} ({both_tags_count/len(successful_problems)*100:.1f}%)")
    
    # Token co-occurrence analysis
    print("\nTOKEN CO-OCCURRENCE ANALYSIS:")
    print("-" * 40)
    
    # Count problems with multiple tokens
    multiple_tokens_count = {}
    for num_tokens in range(1, len(token_names) + 1):
        count = sum(1 for result in successful_problems 
                   if sum(result['token_results'].values()) == num_tokens)
        if count > 0:
            multiple_tokens_count[num_tokens] = count
    
    for num_tokens, count in multiple_tokens_count.items():
        ratio = count / len(successful_problems) * 100
        print(f"Problems with exactly {num_tokens} token(s): {count} ({ratio:.1f}%)")
    
    # Most common token combinations
    print(f"\nMOST COMMON TOKEN COMBINATIONS:")
    print("-" * 40)
    
    combination_counts = {}
    for result in problems_with_parallel_thinking:
        combo = tuple(sorted(result['found_tokens']))
        combination_counts[combo] = combination_counts.get(combo, 0) + 1
    
    # Sort by frequency
    sorted_combos = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)
    
    for combo, count in sorted_combos[:5]:  # Show top 5 combinations
        ratio = count / len(successful_problems) * 100
        combo_str = ", ".join(combo)
        print(f"[{combo_str}]: {count} times ({ratio:.1f}%)")
    
    # Show examples of successful parallel thinking
    print(f"\nEXAMPLES OF PROBLEMS WITH PARALLEL THINKING:")
    print("-" * 50)
    for i, result in enumerate(problems_with_parallel_thinking[:3]):
        print(f"\nExample {i+1} (Problem {result['problem_id']}):")
        print(f"Problem: {result['problem']}")
        print(f"Found tokens: {result['found_tokens']}")
        print(f"Metrics: Threads={result['metrics']['number_of_threads']}, "
              f"Width={result['metrics']['max_width']}, "
              f"Tokens={result['metrics']['token_count']}, "
              f"Words={result['metrics']['word_count']}")
        
        # Show which specific tokens were found
        print("Token breakdown:")
        for token, found in result['token_results'].items():
            status = "✓" if found else "✗"
            print(f"  {token}: {status}")
        
        print(f"Response preview: {result['full_response']}...")
        print("-" * 30)

def main():
    # Find model checkpoint
    base_dir = "/nvme0n1/zzy_model/fresh_mixed_741556/deepseek-parallel-thinking/"
    model_path = find_latest_checkpoint(base_dir)
    
    if not model_path:
        print("No checkpoints found!")
        return
    
    print(f"Using model checkpoint: {model_path}")
    
    try:
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        print("Model loaded successfully!")
        
        # Test on problems (you can limit the number for faster testing)
        # Use max_problems=5 for quick testing, None for all problems
        results = test_model_performance(model, tokenizer, problems, max_problems=100)
        
        # Print summary statistics
        print_summary_stats(results)
        
        # Save detailed results to file
        import json
        with open('parallel_thinking_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to 'parallel_thinking_test_results.json'")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()