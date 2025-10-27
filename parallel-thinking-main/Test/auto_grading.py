#!/usr/bin/env python3
# filepath: /home/zhangzy/parallel-thinking/evaluation/AReaL-main/evaluation/grade_independent_results.py

import argparse
import json
import os
from typing import Dict, Any, List
from datetime import datetime
import sys
sys.path.append("/storage/openpsi/users/zzy/parallel_thinking/evaluation")

from parser import extract_answer, strip_string, STRIP_EXCEPTIONS
from grader import math_equal

def extract_answer_from_response(content: str, data_name: str = "math") -> str:
    """
    Extract the final answer from model response using parser.py functions.
    This replaces the custom extraction in independent_sglang.py
    """
    try:
        # Use the robust extract_answer function from parser.py
        extracted = extract_answer(content, data_name, use_last_number=True)
        return extracted if extracted else None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def compare_answers(model_answer: str, real_answer: str, data_name: str = "math") -> bool:
    """
    Compare model answer with real answer using grader.py functions.
    This replaces the custom comparison in independent_sglang.py
    """
    if model_answer is None:
        return False
    
    try:
        # Use the robust math_equal function from grader.py
        return math_equal(model_answer, real_answer, timeout=True)
    except Exception as e:
        print(f"Error comparing answers: {e}")
        return False

def auto_grading(base_directory: str, real_answers: List[str], data_name: str = "math") -> Dict[str, Any]:
    """
    Automatically grade the model responses by comparing with correct answers.
    Uses functions from parser.py and grader.py for robust evaluation.
    
    Args:
        base_directory: Directory containing problem subdirectories
        real_answers: List of correct answers
        data_name: Dataset name for proper parsing (default: "math")
    
    Returns:
        Dictionary containing grading results and statistics
    """
    print(f"\n🔍 Starting auto-grading for directory: {base_directory}")
    print(f"📊 Dataset: {data_name}")
    
    # Initialize grading statistics
    total_problems = 0
    correct_answers = 0
    failed_extractions = 0
    grading_results = []
    
    # Grade each problem
    for i in range(len(real_answers)):
        problem_dir = os.path.join(base_directory, f"problem_{i:02d}")
        main_answer_file = os.path.join(problem_dir, "main_answer.txt")
        
        if not os.path.exists(main_answer_file):
            print(f"Problem {i} file not found: {main_answer_file}")
            continue
        
        total_problems += 1
        
        try:
            # Read the main answer file
            with open(main_answer_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract answer from the model response using parser.py
            model_answer = extract_answer_from_response(content, data_name)
            real_answer = real_answers[i]
            
            # Process the real answer using parser.py functions
            if data_name not in STRIP_EXCEPTIONS:
                processed_real_answer = strip_string(real_answer, skip_unit=data_name == "carp_en")
            else:
                processed_real_answer = real_answer.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
            
            # Compare answers using grader.py
            is_correct = compare_answers(model_answer, processed_real_answer, data_name)
            
            if model_answer is None:
                failed_extractions += 1
                result = "EXTRACTION_FAILED"
            elif is_correct:
                correct_answers += 1
                result = "CORRECT"
            else:
                result = "INCORRECT"
            
            grading_results.append({
                'problem': i,
                'model_answer': model_answer,
                'real_answer': processed_real_answer,
                'original_real_answer': real_answer,
                'result': result
            })
            
            print(f"Problem {i:2d}: Model='{model_answer}', Real='{processed_real_answer}', Result={result}")
            
        except Exception as e:
            print(f"Error grading problem {i}: {str(e)}")
            failed_extractions += 1
            grading_results.append({
                'problem': i,
                'model_answer': None,
                'real_answer': real_answers[i],
                'original_real_answer': real_answers[i],
                'result': "ERROR"
            })
    
    # Generate grading report
    accuracy = (correct_answers / total_problems * 100) if total_problems > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"GRADING SUMMARY")
    print(f"=" * 60)
    print(f"Dataset: {data_name}")
    print(f"Total Problems: {total_problems}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Incorrect Answers: {total_problems - correct_answers - failed_extractions}")
    print(f"Failed Extractions: {failed_extractions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"=" * 60)
    
    # Save detailed grading report
    grading_file = os.path.join(base_directory, "grading_report.txt")
    with open(grading_file, 'w', encoding='utf-8') as f:
        f.write(f"Automatic Grading Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {base_directory}\n")
        f.write(f"Dataset: {data_name}\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"Total Problems: {total_problems}\n")
        f.write(f"Correct Answers: {correct_answers}\n")
        f.write(f"Incorrect Answers: {total_problems - correct_answers - failed_extractions}\n")
        f.write(f"Failed Extractions: {failed_extractions}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        
        f.write(f"DETAILED RESULTS:\n")
        f.write("=" * 80 + "\n")
        
        for result in grading_results:
            f.write(f"Problem {result['problem']:2d}: ")
            f.write(f"Model='{result['model_answer']}', ")
            f.write(f"Real='{result['real_answer']}', ")
            f.write(f"Result={result['result']}\n")
    
    # Save detailed grading report as JSON
    json_grading_file = os.path.join(base_directory, "grading_report.json")
    grading_summary = {
        'total': total_problems,
        'correct': correct_answers,
        'incorrect': total_problems - correct_answers - failed_extractions,
        'failed_extractions': failed_extractions,
        'accuracy': accuracy,
        'dataset': data_name,
        'timestamp': datetime.now().isoformat(),
        'results': grading_results
    }
    
    with open(json_grading_file, 'w', encoding='utf-8') as f:
        json.dump(grading_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed grading report saved to: {grading_file}")
    print(f"JSON grading report saved to: {json_grading_file}")
    
    return grading_summary

def grade_single_problem(problem_dir: str, real_answer: str, data_name: str = "math") -> Dict[str, Any]:
    """
    Grade a single problem directory.
    
    Args:
        problem_dir: Path to the problem directory
        real_answer: The correct answer
        data_name: Dataset name for proper parsing
    
    Returns:
        Dictionary containing grading result for this problem
    """
    main_answer_file = os.path.join(problem_dir, "main_answer.txt")
    
    if not os.path.exists(main_answer_file):
        return {
            'problem_dir': problem_dir,
            'model_answer': None,
            'real_answer': real_answer,
            'result': "FILE_NOT_FOUND",
            'error': f"File not found: {main_answer_file}"
        }
    
    try:
        # Read the main answer file
        with open(main_answer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract answer from the model response
        model_answer = extract_answer_from_response(content, data_name)
        
        # Process the real answer
        if data_name not in STRIP_EXCEPTIONS:
            processed_real_answer = strip_string(real_answer, skip_unit=data_name == "carp_en")
        else:
            processed_real_answer = real_answer.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
        
        # Compare answers
        is_correct = compare_answers(model_answer, processed_real_answer, data_name)
        
        if model_answer is None:
            result = "EXTRACTION_FAILED"
        elif is_correct:
            result = "CORRECT"
        else:
            result = "INCORRECT"
        
        return {
            'problem_dir': problem_dir,
            'model_answer': model_answer,
            'real_answer': processed_real_answer,
            'original_real_answer': real_answer,
            'result': result
        }
        
    except Exception as e:
        return {
            'problem_dir': problem_dir,
            'model_answer': None,
            'real_answer': real_answer,
            'original_real_answer': real_answer,
            'result': "ERROR",
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Grade model outputs from independent text files")
    parser.add_argument("base_directory", help="Base directory containing problem subdirectories")
    parser.add_argument("--answers", nargs='+', required=True, help="List of correct answers")
    parser.add_argument("--answers-file", help="JSON file containing list of correct answers")
    parser.add_argument("--data-name", default="math", 
                       help="Dataset name for proper parsing (default: math)")
    parser.add_argument("--single-problem", help="Grade only a specific problem directory")
    parser.add_argument("--single-answer", help="Correct answer for single problem")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed results for each problem")
    
    args = parser.parse_args()
    
    # Handle answers input
    if args.answers_file:
        try:
            with open(args.answers_file, 'r', encoding='utf-8') as f:
                real_answers = json.load(f)
        except Exception as e:
            print(f"Error loading answers file: {e}")
            return
    else:
        real_answers = args.answers
    
    # Single problem mode
    if args.single_problem:
        if not args.single_answer:
            print("Error: --single-answer is required when using --single-problem")
            return
        
        result = grade_single_problem(args.single_problem, args.single_answer, args.data_name)
        
        print(f"Single Problem Grading Result:")
        print(f"Problem Dir: {result['problem_dir']}")
        print(f"Model Answer: {result['model_answer']}")
        print(f"Real Answer: {result['real_answer']}")
        print(f"Result: {result['result']}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        return
    
    # Grade all problems
    if not os.path.exists(args.base_directory):
        print(f"Error: Directory '{args.base_directory}' not found.")
        return
    
    grading_results = auto_grading(args.base_directory, real_answers, args.data_name)
    
    # Print detailed results if verbose
    if args.verbose:
        print("\nDetailed Results:")
        for result in grading_results['results']:
            status = "✓" if result['result'] == 'CORRECT' else "✗"
            print(f"Problem {result['problem']:2d}: {status}")
            print(f"  Model Answer: {result['model_answer']}")
            print(f"  Real Answer: {result['real_answer']}")
            print(f"  Result: {result['result']}")
            if result['result'] in ['ERROR'] and 'error' in result:
                print(f"  Error: {result['error']}")
            print()

if __name__ == "__main__":
    main()

# === New grading function for final_answer.txt and JSONL ===
def grade_with_jsonl(base_dir, jsonl_filename):
    """
    Grade problems in base_dir using final_answer.txt and a JSONL file.
    Each line in the JSONL corresponds to problem_00, problem_01, ...
    """
    import json
    import os
    jsonl_path = os.path.join(base_dir, jsonl_filename)
    report_path = os.path.join(base_dir, "grading_report.txt")

    # Read all answers from jsonl
    answers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            answers.append(obj.get("answer"))

    results = []
    for idx in range(30):
        prob_dir = os.path.join(base_dir, f"problem_{idx:02d}")
        final_ans_path = os.path.join(prob_dir, "final_answer.txt")
        if not os.path.isdir(prob_dir) or not os.path.isfile(final_ans_path):
            continue  # skip missing problems
        with open(final_ans_path, "r", encoding="utf-8") as f:
            true_ans = f.read().strip()
        # Get predicted answer
        if idx >= len(answers):
            pred_ans = None
        else:
            pred_ans = str(answers[idx]).strip()
        # Compare answers numerically
        try:
            correct = float(true_ans) == float(pred_ans)
        except Exception:
            correct = true_ans == pred_ans
        results.append({
            "problem": f"problem_{idx:02d}",
            "true_answer": true_ans,
            "predicted_answer": pred_ans,
            "correct": correct
        })

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['problem']}: true={r['true_answer']} pred={r['predicted_answer']} correct={r['correct']}\n")
        total = len(results)
        correct_count = sum(r['correct'] for r in results)
        f.write(f"\nTotal: {correct_count}/{total} correct\n")

# CLI entry point for the new grading function
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1].endswith('/') and sys.argv[2].endswith('.jsonl'):
        grade_with_jsonl(sys.argv[1], sys.argv[2])