"""
Batch Inference Script for Multiverse Generation
Processes problems from a JSONL file, runs inference, and evaluates answers.

Usage:
    python run_batch_inference.py --input data.jsonl --output-dir results --port 30000
"""

import json
import argparse
import re
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import traceback

from inference_heir_new import MultiverseGenerator


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from \boxed{} notation.
    
    Args:
        text: Generated text containing \boxed{answer}
    
    Returns:
        Extracted answer string or None if not found
    """
    # Pattern to match \boxed{...} with proper brace matching
    # This handles nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed answer found (usually the final answer)
        return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    - Remove whitespace
    - Convert to lowercase
    - Remove common LaTeX formatting
    
    Args:
        answer: Raw answer string
    
    Returns:
        Normalized answer string
    """
    if answer is None:
        return ""
    
    # Remove whitespace
    normalized = answer.strip()
    
    # Remove LaTeX formatting that doesn't affect the value
    normalized = normalized.replace("\\text{", "").replace("}", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\:", "")
    normalized = normalized.replace("\\;", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace(" ", "")
    
    # Remove dollar signs
    normalized = normalized.replace("$", "")
    
    return normalized.lower()


def extract_integer(answer: str) -> Optional[int]:
    """
    Extract integer from answer string.
    
    Args:
        answer: Answer string that should contain an integer
    
    Returns:
        Integer value or None if extraction fails
    """
    if not answer:
        return None
    
    # Try to extract integer from string
    # Handle negative numbers and remove non-digit characters
    match = re.search(r'-?\d+', answer)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    
    return None


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth.
    For integer answers, extracts and compares the numeric values.
    
    Args:
        predicted: Predicted answer (from \boxed{})
        ground_truth: Ground truth answer from dataset
    
    Returns:
        True if answers match, False otherwise
    """
    # First try exact match after normalization
    pred_norm = normalize_answer(predicted) if predicted else ""
    gt_norm = normalize_answer(ground_truth) if ground_truth else ""
    
    if pred_norm == gt_norm:
        return True
    
    # Try integer comparison
    pred_int = extract_integer(predicted) if predicted else None
    gt_int = extract_integer(ground_truth) if ground_truth else None
    
    if pred_int is not None and gt_int is not None:
        return pred_int == gt_int
    
    return False


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load problems from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of problem dictionaries
    """
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                problem = json.loads(line)
                problem['line_number'] = line_num
                problems.append(problem)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    
    return problems


def save_answer_to_file(output_dir: str, index: int, full_response: str):
    """
    Save the full response to a text file.
    
    Args:
        output_dir: Directory to save the answer
        index: Problem index
        full_response: The full generated response
    """
    answer_file = os.path.join(output_dir, f"answer{index}.txt")
    with open(answer_file, 'w', encoding='utf-8') as f:
        f.write(full_response)


def save_grading_report(output_dir: str, results: List[Dict], stats: Dict):
    """
    Save a grading report summarizing all results.
    
    Args:
        output_dir: Directory to save the report
        results: List of result dictionaries
        stats: Statistics dictionary
    """
    report_file = os.path.join(output_dir, "grading_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("GRADING REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total problems: {stats['total']}\n")
        f.write(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)\n")
        f.write(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)\n")
        f.write(f"No answer extracted: {stats['no_answer_extracted']} "
                f"({stats['no_answer_extracted']/stats['total']*100:.1f}%)\n")
        f.write(f"Errors: {stats['errors']} ({stats['errors']/stats['total']*100:.1f}%)\n")
        
        if stats['correct'] + stats['incorrect'] > 0:
            accuracy = stats['correct'] / (stats['correct'] + stats['incorrect']) * 100
            f.write(f"\nAccuracy (excluding errors): {accuracy:.1f}%\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for result in results:
            idx = result['index']
            f.write(f"Problem {idx}:\n")
            f.write(f"  Line Number: {result.get('line_number', 'N/A')}\n")
            f.write(f"  Problem: {result['problem'][:100]}{'...' if len(result['problem']) > 100 else ''}\n")
            f.write(f"  Ground Truth: {result['ground_truth']}\n")
            f.write(f"  Predicted Answer: {result.get('predicted_answer', 'N/A')}\n")
            f.write(f"  Status: {result['status']}\n")
            
            if result['status'] == 'success':
                status_icon = "✓" if result['is_correct'] else "✗"
                f.write(f"  Correctness: {status_icon} {'CORRECT' if result['is_correct'] else 'INCORRECT'}\n")
                f.write(f"  Total Tokens: {result['generation_info']['total_tokens']}\n")
                f.write(f"  Parallel Stages: {result['generation_info']['num_stages']}\n")
            elif result['status'] == 'error':
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            elif result['status'] == 'no_answer_extracted':
                f.write(f"  Warning: Could not extract answer from \\boxed{{}}\n")
                f.write(f"  Total Tokens: {result['generation_info']['total_tokens']}\n")
            
            f.write(f"  Answer File: answer{idx}.txt\n")
            f.write(f"  Timestamp: {result['timestamp']}\n")
            f.write("\n" + "-"*80 + "\n\n")
        
        # Footer
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def save_metadata(output_dir: str, results: List[Dict]):
    """
    Save metadata as JSON for programmatic access.
    
    Args:
        output_dir: Directory to save metadata
        results: List of result dictionaries
    """
    metadata_file = os.path.join(output_dir, "metadata.json")
    
    metadata = {
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def run_inference_on_problem(
    generator: MultiverseGenerator,
    problem: str,
    max_normal_tokens: int = 10000,
    max_goal_tokens: int = 10000,
    max_path_tokens: int = 10000,
    max_conclusion_tokens: int = 10000,
    max_total_tokens: int = 32768,
    temperature: float = 0.6
) -> Dict:
    """
    Run inference on a single problem.
    
    Args:
        generator: MultiverseGenerator instance
        problem: Problem text
        max_*_tokens: Token limits for generation
        temperature: Sampling temperature
    
    Returns:
        Generation result dictionary
    """
    result = generator.generate_with_auto_parallel_detection(
        prompt=problem,
        max_normal_tokens=max_normal_tokens,
        max_goal_tokens=max_goal_tokens,
        max_path_tokens=max_path_tokens,
        max_conclusion_tokens=max_conclusion_tokens,
        max_total_tokens=max_total_tokens,
        temperature=temperature
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run batch inference on problems from JSONL file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input JSONL file with problems and answers'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Base directory for output (timestamp will be appended)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=30001,
        help='SGLang server port (default: 30000)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/storage/openpsi/models/zzy/Multiverse-20251030_154726',
        help='Path to model'
    )
    parser.add_argument(
        '--max-normal-tokens',
        type=int,
        default=10000,
        help='Max tokens for normal generation (default: 10000)'
    )
    parser.add_argument(
        '--max-goal-tokens',
        type=int,
        default=10000,
        help='Max tokens for goal generation (default: 10000)'
    )
    parser.add_argument(
        '--max-path-tokens',
        type=int,
        default=10000,
        help='Max tokens for path generation (default: 10000)'
    )
    parser.add_argument(
        '--max-conclusion-tokens',
        type=int,
        default=10000,
        help='Max tokens for conclusion generation (default: 10000)'
    )
    parser.add_argument(
        '--max-total-tokens',
        type=int,
        default=32768,
        help='Max total tokens (default: 10000)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start index (0-based) for processing problems (default: 0)'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index (exclusive) for processing problems (default: None, process all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    server_url = f"http://localhost:{args.port}"
    print("="*70)
    print("Batch Inference for Multiverse Generation")
    print("="*70)
    print(f"Input file: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Server URL: {server_url}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: normal={args.max_normal_tokens}, goal={args.max_goal_tokens}, "
          f"path={args.max_path_tokens}, conclusion={args.max_conclusion_tokens}, "
          f"total={args.max_total_tokens}")
    print("="*70)
    
    # Load problems
    print(f"\nLoading problems from {args.input}...")
    problems = load_jsonl(args.input)
    print(f"Loaded {len(problems)} problems")
    
    # Apply start/end index filtering
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(problems)
    problems = problems[start_idx:end_idx]
    print(f"Processing problems {start_idx} to {end_idx-1} ({len(problems)} problems)")
    
    # Initialize generator
    print(f"\nInitializing generator...")
    generator = MultiverseGenerator(server_url=server_url, model_name=args.model)
    
    # Statistics
    stats = {
        'total': len(problems),
        'correct': 0,
        'incorrect': 0,
        'no_answer_extracted': 0,
        'errors': 0
    }
    
    # Store all results for final report
    all_results = []
    
    # Process each problem
    print(f"\nStarting inference...")
    print("="*70)
    
    for idx, problem_data in enumerate(problems):
        actual_idx = start_idx + idx
        problem = problem_data.get('problem', '')
        ground_truth = problem_data.get('answer', '')
        
        print(f"\n[{actual_idx + 1}/{start_idx + len(problems)}] Processing problem...")
        print(f"Problem: {problem[:100]}..." if len(problem) > 100 else f"Problem: {problem}")
        print(f"Ground truth: {ground_truth}")
        
        result_entry = {
            'index': actual_idx,
            'line_number': problem_data.get('line_number'),
            'problem': problem,
            'ground_truth': ground_truth,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Run inference
            generation_result = run_inference_on_problem(
                generator=generator,
                problem=problem,
                max_normal_tokens=args.max_normal_tokens,
                max_goal_tokens=args.max_goal_tokens,
                max_path_tokens=args.max_path_tokens,
                max_conclusion_tokens=args.max_conclusion_tokens,
                max_total_tokens=args.max_total_tokens,
                temperature=args.temperature
            )
            
            # Extract answer
            full_text = generation_result['full_text']
            predicted_answer = extract_boxed_answer(full_text)
            
            # Compare answers
            if predicted_answer is not None:
                is_correct = compare_answers(predicted_answer, ground_truth)
                
                result_entry.update({
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'full_response': full_text,
                    'generation_info': {
                        'total_tokens': generation_result['total_tokens'],
                        'num_stages': generation_result.get('num_stages', 0),
                        'stages': generation_result.get('stages', [])
                    },
                    'status': 'success'
                })
                
                if is_correct:
                    stats['correct'] += 1
                    print(f"✓ CORRECT - Predicted: {predicted_answer}")
                else:
                    stats['incorrect'] += 1
                    print(f"✗ INCORRECT - Predicted: {predicted_answer}, Expected: {ground_truth}")
            else:
                stats['no_answer_extracted'] += 1
                result_entry.update({
                    'predicted_answer': None,
                    'is_correct': False,
                    'full_response': full_text,
                    'generation_info': {
                        'total_tokens': generation_result['total_tokens'],
                        'num_stages': generation_result.get('num_stages', 0),
                        'stages': generation_result.get('stages', [])
                    },
                    'status': 'no_answer_extracted',
                    'error': 'Could not extract answer from \\boxed{}'
                })
                print(f"⚠ NO ANSWER EXTRACTED")
            
            # Save the full response to file
            save_answer_to_file(output_dir, actual_idx, full_text)
        
        except Exception as e:
            stats['errors'] += 1
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            result_entry.update({
                'predicted_answer': None,
                'is_correct': False,
                'full_response': None,
                'status': 'error',
                'error': error_msg,
                'error_trace': error_trace
            })
            
            print(f"✗ ERROR: {error_msg}")
            print(f"Traceback:\n{error_trace}")
            
            # Save error information to file
            error_text = f"ERROR: {error_msg}\n\nTraceback:\n{error_trace}"
            save_answer_to_file(output_dir, actual_idx, error_text)
        
        # Add to results list
        all_results.append(result_entry)
        
        # Print progress
        print(f"\nProgress: {idx + 1}/{len(problems)} problems processed")
        print(f"Stats: ✓ {stats['correct']} | ✗ {stats['incorrect']} | "
              f"⚠ {stats['no_answer_extracted']} | ✗✗ {stats['errors']}")
        print("-"*70)
    
    # Save grading report and metadata
    print("\nSaving grading report and metadata...")
    save_grading_report(output_dir, all_results, stats)
    save_metadata(output_dir, all_results)
    
    # Final statistics
    print("\n" + "="*70)
    print("BATCH INFERENCE COMPLETE")
    print("="*70)
    print(f"Total problems: {stats['total']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)")
    print(f"No answer extracted: {stats['no_answer_extracted']} "
          f"({stats['no_answer_extracted']/stats['total']*100:.1f}%)")
    print(f"Errors: {stats['errors']} ({stats['errors']/stats['total']*100:.1f}%)")
    
    if stats['correct'] + stats['incorrect'] > 0:
        accuracy = stats['correct'] / (stats['correct'] + stats['incorrect']) * 100
        print(f"\nAccuracy (excluding errors): {accuracy:.1f}%")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Answer files: answer0.txt, answer1.txt, ...")
    print(f"  - Grading report: grading_report.txt")
    print(f"  - Metadata: metadata.json")
    print("="*70)


if __name__ == "__main__":
    main()
