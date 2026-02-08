"""
Aggregate Results from Multiple Test Runs

This script aggregates results from multiple test runs within a Test_{timestamp} directory.
Each run creates a subdirectory with a timestamp, containing grading_report.txt.

The script generates:
1. A summary report listing accuracy for each trial
2. Average accuracy across all trials
3. Detailed statistics (min, max, std deviation)
4. Per-problem accuracy and theoretical variance

Usage:
    python aggregate_results.py --test-dir Test_20260113_120000
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from datetime import datetime
import math
import sys

# Add AReaL to path to import math_parser
areal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AReaL')
if areal_path not in sys.path:
    sys.path.insert(0, areal_path)

try:
    from areal.reward.math_parser import math_equal, extract_answer
except ImportError:
    print("Warning: Could not import math_parser from AReaL. Make sure AReaL directory exists.")
    # Provide a fallback simple comparison
    def math_equal(pred, ref, include_percentage=True, is_close=True, timeout=False):
        """Fallback comparison if math_parser is not available."""
        return str(pred).strip() == str(ref).strip()
    
    def extract_answer(text, data_name="math", use_last_number=True):
        """Fallback answer extraction."""
        return text.strip()


def parse_grading_report(report_path: str) -> Dict:
    """
    Parse a grading_report.txt file and recalculate statistics based on math_equal comparison.
    
    Args:
        report_path: Path to grading_report.txt
    
    Returns:
        Dictionary with recalculated statistics (total, correct, incorrect, accuracy, etc.)
    """
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse individual problem results using math_equal
    problem_results = parse_problem_results(report_path)
    
    if not problem_results:
        return None
    
    # Recalculate statistics based on actual comparisons
    stats = {}
    stats['total'] = len(problem_results)
    stats['correct'] = sum(1 for is_correct in problem_results.values() if is_correct)
    stats['incorrect'] = stats['total'] - stats['correct']
    
    # Calculate percentages
    if stats['total'] > 0:
        stats['correct_pct'] = (stats['correct'] / stats['total']) * 100
        stats['incorrect_pct'] = (stats['incorrect'] / stats['total']) * 100
        stats['accuracy'] = stats['correct_pct']
    else:
        stats['correct_pct'] = 0.0
        stats['incorrect_pct'] = 0.0
        stats['accuracy'] = 0.0
    
    # For compatibility, set these to 0 (we're not tracking them separately anymore)
    stats['no_answer'] = 0
    stats['no_answer_pct'] = 0.0
    stats['errors'] = 0
    stats['errors_pct'] = 0.0
    
    return stats


def parse_problem_results(report_path: str) -> Dict[str, bool]:
    """
    Parse individual problem results from grading_report.txt.
    Uses math_equal to compare Ground Truth and Predicted Answer directly.
    
    Args:
        report_path: Path to grading_report.txt
    
    Returns:
        Dictionary mapping problem_id to correctness (True/False)
        Note: "No answer extracted" cases are treated as False (incorrect)
    """
    if not os.path.exists(report_path):
        return {}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    problem_results = {}
    
    # Split into problem sections
    problem_sections = re.split(r'\n-{80,}\n', content)
    
    for section in problem_sections:
        # Extract problem number
        problem_match = re.search(r'^Problem\s+(\d+):', section, re.MULTILINE)
        if problem_match:
            problem_id = problem_match.group(1)
            
            # Extract Ground Truth
            ground_truth_match = re.search(r'Ground Truth:\s*(.+?)$', section, re.MULTILINE)
            # Extract Predicted Answer
            predicted_match = re.search(r'Predicted Answer:\s*(.+?)$', section, re.MULTILINE)
            
            if ground_truth_match and predicted_match:
                ground_truth = ground_truth_match.group(1).strip()
                predicted_answer = predicted_match.group(1).strip()
                
                # Handle "None" case (no answer extracted)
                if predicted_answer.lower() in ['none', '']:
                    problem_results[problem_id] = False
                else:
                    # Use math_equal to compare
                    try:
                        is_correct = math_equal(predicted_answer, ground_truth, 
                                              include_percentage=True, 
                                              is_close=True, 
                                              timeout=False)
                        problem_results[problem_id] = is_correct
                    except Exception as e:
                        # If comparison fails, treat as incorrect
                        print(f"Warning: Error comparing problem {problem_id}: {e}")
                        problem_results[problem_id] = False
            else:
                # If we can't extract both values, treat as incorrect
                problem_results[problem_id] = False
    
    return problem_results


def find_all_runs(test_dir: str) -> List[Tuple[str, str]]:
    """
    Find all run subdirectories within the test directory.
    
    Args:
        test_dir: Path to Test_{timestamp} directory
    
    Returns:
        List of tuples (run_name, grading_report_path)
    """
    runs = []
    test_path = Path(test_dir)
    
    if not test_path.exists():
        print(f"Error: Test directory {test_dir} does not exist")
        return runs
    
    # Find all subdirectories with grading_report.txt
    for subdir in sorted(test_path.iterdir()):
        if subdir.is_dir():
            grading_report = subdir / "grading_report.txt"
            if grading_report.exists():
                runs.append((subdir.name, str(grading_report)))
    
    return runs


def calculate_per_problem_accuracy(all_problem_results: List[Dict[str, bool]]) -> Dict[str, float]:
    """
    Calculate accuracy for each problem across all runs.
    
    Args:
        all_problem_results: List of dictionaries mapping problem_id to correctness
    
    Returns:
        Dictionary mapping problem_id to accuracy (0.0 to 1.0)
    """
    if not all_problem_results:
        return {}
    
    # Get all unique problem IDs
    all_problem_ids = set()
    for results in all_problem_results:
        all_problem_ids.update(results.keys())
    
    # Calculate accuracy for each problem
    problem_accuracy = {}
    for problem_id in sorted(all_problem_ids):
        correct_count = 0
        total_count = 0
        
        for results in all_problem_results:
            if problem_id in results:
                total_count += 1
                if results[problem_id]:
                    correct_count += 1
        
        if total_count > 0:
            problem_accuracy[problem_id] = correct_count / total_count
        else:
            problem_accuracy[problem_id] = 0.0
    
    return problem_accuracy


def calculate_theoretical_variance(problem_accuracy: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate theoretical variance and standard deviation based on per-problem accuracy.
    
    For each problem with accuracy P, the variance is: P(1-P)^2 + (1-P)P^2 = P(1-P)
    Total variance is the sum of individual variances (independent problems).
    
    Args:
        problem_accuracy: Dictionary mapping problem_id to accuracy (0.0 to 1.0)
    
    Returns:
        Tuple of (total_variance, standard_deviation)
    """
    total_variance = 0.0
    
    for problem_id, p in problem_accuracy.items():
        # Variance for this problem: P(1-P)^2 + (1-P)P^2 = P(1-P)[1-P + P] = P(1-P)
        # This simplifies to P(1-P), which is the variance of a Bernoulli random variable
        variance = p * (1 - p)
        total_variance += variance
    
    std_dev = math.sqrt(total_variance)
    
    return total_variance, std_dev


def calculate_accuracy_per_group(problem_accuracy: Dict[str, float], group_size: int = 30) -> Dict[str, Dict]:
    """
    Calculate accuracy for groups of problems.
    
    Args:
        problem_accuracy: Dictionary mapping problem_id to accuracy (0.0 to 1.0)
        group_size: Number of problems per group (default: 30)
    
    Returns:
        Dictionary mapping group label to statistics
    """
    if not problem_accuracy:
        return {}
    
    # Sort problem IDs numerically
    sorted_problem_ids = sorted(problem_accuracy.keys(), key=lambda x: int(x))
    
    group_stats = {}
    
    # Group problems
    for i in range(0, len(sorted_problem_ids), group_size):
        group_problems = sorted_problem_ids[i:i + group_size]
        start_idx = int(group_problems[0])
        end_idx = int(group_problems[-1])
        
        group_label = f"{start_idx}-{end_idx}"
        
        # Calculate average accuracy for this group
        accuracies = [problem_accuracy[pid] for pid in group_problems]
        avg_accuracy = statistics.mean(accuracies) if accuracies else 0.0
        
        # Calculate variance and std dev for this group
        group_variance = sum(p * (1 - p) for p in accuracies)
        group_std = math.sqrt(group_variance)
        
        group_stats[group_label] = {
            'start': start_idx,
            'end': end_idx,
            'count': len(group_problems),
            'avg_accuracy': avg_accuracy,
            'variance': group_variance,
            'std_dev': group_std,
            'problem_ids': group_problems
        }
    
    return group_stats


def generate_summary_report(test_dir: str, runs_stats: List[Dict], 
                           problem_accuracy: Dict[str, float] = None,
                           theoretical_variance: float = None,
                           theoretical_std: float = None,
                           output_file: str = "summary_report.txt"):
    """
    Generate a summary report aggregating all runs.
    
    Args:
        test_dir: Path to Test_{timestamp} directory
        runs_stats: List of statistics dictionaries from each run
        problem_accuracy: Dictionary mapping problem_id to accuracy
        theoretical_variance: Theoretical variance based on per-problem accuracy
        theoretical_std: Theoretical standard deviation
        output_file: Name of output file
    """
    output_path = os.path.join(test_dir, output_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("SUMMARY REPORT - MULTIPLE RUNS AGGREGATION\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Directory: {test_dir}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runs: {len(runs_stats)}\n\n")
        
        f.write("="*80 + "\n\n")
        
        # Individual run results
        f.write("INDIVIDUAL RUN RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        accuracies = []
        correct_counts = []
        for idx, stats in enumerate(runs_stats, 1):
            f.write(f"Run {idx}: {stats['run_name']}\n")
            f.write(f"  Total Problems: {stats.get('total', 'N/A')}\n")
            f.write(f"  Correct: {stats.get('correct', 'N/A')} ({stats.get('correct_pct', 'N/A')}%)\n")
            f.write(f"  Incorrect: {stats.get('incorrect', 'N/A')} ({stats.get('incorrect_pct', 'N/A')}%)\n")
            f.write(f"  No Answer: {stats.get('no_answer', 'N/A')} ({stats.get('no_answer_pct', 'N/A')}%)\n")
            f.write(f"  Errors: {stats.get('errors', 'N/A')} ({stats.get('errors_pct', 'N/A')}%)\n")
            
            # Recalculate accuracy using total_problems as denominator
            # Treat "no answer extracted" as incorrect (already counted in total)
            if 'correct' in stats and 'total' in stats and stats['total'] > 0:
                # Accuracy = correct / total (where total includes all problems)
                accuracy = (stats['correct'] / stats['total']) * 100
                f.write(f"  Accuracy: {accuracy:.2f}%\n")
                accuracies.append(accuracy)
                correct_counts.append(stats['correct'])
            else:
                f.write(f"  Accuracy: N/A\n")
            
            f.write("\n")
        
        f.write("-"*80 + "\n\n")
        
        # Aggregate statistics
        f.write("AGGREGATE STATISTICS\n")
        f.write("-"*80 + "\n\n")
        f.write("Note: 'No answer extracted' cases are treated as incorrect.\n\n")
        
        if accuracies:
            avg_accuracy = statistics.mean(accuracies)
            f.write(f"Average Accuracy: {avg_accuracy:.2f}%\n")
            
            if len(accuracies) > 1:
                # Empirical standard deviation
                std_accuracy = statistics.stdev(accuracies)
                min_accuracy = min(accuracies)
                max_accuracy = max(accuracies)
                median_accuracy = statistics.median(accuracies)
                
                f.write(f"Empirical Standard Deviation (Accuracy): {std_accuracy:.2f}%\n")
                f.write(f"Minimum Accuracy: {min_accuracy:.2f}%\n")
                f.write(f"Maximum Accuracy: {max_accuracy:.2f}%\n")
                f.write(f"Median Accuracy: {median_accuracy:.2f}%\n")
                
                # Empirical standard deviation of correct counts
                std_correct = statistics.stdev(correct_counts)
                f.write(f"\nEmpirical Standard Deviation (Correct Count): {std_correct:.4f}\n")
            
            f.write("\n")
            f.write(f"Accuracy Range: [{min(accuracies):.2f}% - {max(accuracies):.2f}%]\n")
        else:
            f.write("No valid accuracy data found.\n")
        
        # Accuracy per group of 30 problems
        if problem_accuracy:
            f.write("\n")
            f.write("ACCURACY PER GROUP (30 PROBLEMS EACH)\n")
            f.write("-"*80 + "\n\n")
            f.write("Note: 'No answer extracted' cases are treated as incorrect.\n\n")
            
            group_stats = calculate_accuracy_per_group(problem_accuracy, group_size=30)
            
            f.write(f"{'Group':<15} {'Count':<8} {'Avg Accuracy':<15} {'Percentage':<12} {'Variance':<12} {'Std Dev':<12}\n")
            f.write("-"*80 + "\n")
            
            for group_label in sorted(group_stats.keys(), key=lambda x: group_stats[x]['start']):
                stats = group_stats[group_label]
                f.write(f"{group_label:<15} {stats['count']:<8} {stats['avg_accuracy']:<15.4f} {stats['avg_accuracy']*100:<12.2f} {stats['variance']:<12.6f} {stats['std_dev']:<12.6f}\n")
            
            f.write("\n")
        
        # Per-problem accuracy (moved before theoretical statistics for better organization)
        if problem_accuracy:
            f.write("\n")
            f.write("PER-PROBLEM ACCURACY\n")
            f.write("-"*80 + "\n\n")
            f.write("Note: 'No answer extracted' cases are treated as incorrect.\n\n")
            f.write(f"{'Problem ID':<15} {'Accuracy':<12} {'Percentage':<12} {'Variance':<12} {'Std Dev':<12}\n")
            f.write("-"*80 + "\n")
            
            total_variance_check = 0.0
            for problem_id in sorted(problem_accuracy.keys(), key=lambda x: int(x)):
                p = problem_accuracy[problem_id]
                variance = p * (1 - p)
                std_dev = math.sqrt(variance)
                total_variance_check += variance
                
                f.write(f"{problem_id:<15} {p:<12.4f} {p*100:<12.2f} {variance:<12.6f} {std_dev:<12.6f}\n")
            
            f.write("-"*80 + "\n")
            f.write(f"Total Variance (sum): {total_variance_check:.6f}\n")
            f.write(f"Total Std Dev (sqrt of sum): {math.sqrt(total_variance_check):.6f}\n")
            f.write("\n")
        
        # Theoretical variance and standard deviation
        if problem_accuracy is not None and theoretical_variance is not None:
            f.write("\n")
            f.write("THEORETICAL STATISTICS (Based on Per-Problem Accuracy)\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Number of Problems: {len(problem_accuracy)}\n")
            f.write(f"Theoretical Variance (Correct Count): {theoretical_variance:.4f}\n")
            f.write(f"Theoretical Standard Deviation (Correct Count): {theoretical_std:.4f}\n")
            
            # Calculate expected correct count
            expected_correct = sum(problem_accuracy.values())
            f.write(f"Expected Correct Count: {expected_correct:.4f}\n")
            
            # Compare with empirical
            if correct_counts:
                empirical_mean = statistics.mean(correct_counts)
                f.write(f"\nComparison:\n")
                f.write(f"  Empirical Mean (Correct Count): {empirical_mean:.4f}\n")
                f.write(f"  Expected Mean (Correct Count): {expected_correct:.4f}\n")
                if len(correct_counts) > 1:
                    empirical_std = statistics.stdev(correct_counts)
                    f.write(f"  Empirical Std Dev (Correct Count): {empirical_std:.4f}\n")
                    f.write(f"  Theoretical Std Dev (Correct Count): {theoretical_std:.4f}\n")
                    f.write(f"\n  Difference (Empirical - Theoretical): {empirical_std - theoretical_std:.4f}\n")
                    if theoretical_std > 0:
                        ratio = empirical_std / theoretical_std
                        f.write(f"  Ratio (Empirical / Theoretical): {ratio:.4f}\n")
        
        # Aggregate totals
        f.write("\n")
        f.write("TOTAL ACROSS ALL RUNS\n")
        f.write("-"*80 + "\n\n")
        
        total_problems = sum(s.get('total', 0) for s in runs_stats)
        total_correct = sum(s.get('correct', 0) for s in runs_stats)
        total_incorrect = sum(s.get('incorrect', 0) for s in runs_stats)
        total_no_answer = sum(s.get('no_answer', 0) for s in runs_stats)
        total_errors = sum(s.get('errors', 0) for s in runs_stats)
        
        f.write(f"Total Problems Processed: {total_problems}\n")
        f.write(f"Total Correct: {total_correct}\n")
        f.write(f"Total Incorrect: {total_incorrect}\n")
        f.write(f"Total No Answer: {total_no_answer}\n")
        f.write(f"Total Errors: {total_errors}\n")
        
        if total_problems > 0:
            overall_accuracy = (total_correct / total_problems) * 100
            f.write(f"\nOverall Accuracy: {overall_accuracy:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF SUMMARY REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {output_path}")


def generate_csv_report(test_dir: str, runs_stats: List[Dict], output_file: str = "summary_report.csv"):
    """
    Generate a CSV report for easy analysis in spreadsheet software.
    
    Args:
        test_dir: Path to Test_{timestamp} directory
        runs_stats: List of statistics dictionaries from each run
        output_file: Name of output CSV file
    """
    output_path = os.path.join(test_dir, output_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("Run,Total,Correct,Incorrect,No_Answer,Errors,Accuracy\n")
        
        # Data rows
        for stats in runs_stats:
            run_name = stats.get('run_name', 'Unknown')
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            incorrect = stats.get('incorrect', 0)
            no_answer = stats.get('no_answer', 0)
            errors = stats.get('errors', 0)
            accuracy = stats.get('accuracy', 0.0)
            
            f.write(f"{run_name},{total},{correct},{incorrect},{no_answer},{errors},{accuracy:.2f}\n")
    
    print(f"CSV report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from multiple test runs'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Path to Test_{timestamp} directory containing run subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='summary_report.txt',
        help='Output filename for summary report (default: summary_report.txt)'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Also generate CSV report'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Aggregate Results - Multiple Test Runs")
    print("="*70)
    print(f"Test Directory: {args.test_dir}")
    print("="*70)
    
    # Find all runs
    print("\nSearching for run subdirectories...")
    runs = find_all_runs(args.test_dir)
    
    if not runs:
        print(f"Error: No run subdirectories with grading_report.txt found in {args.test_dir}")
        return
    
    print(f"Found {len(runs)} run(s):")
    for run_name, report_path in runs:
        print(f"  - {run_name}")
    
    # Parse each grading report
    print("\nParsing grading reports...")
    runs_stats = []
    all_problem_results = []
    
    for run_name, report_path in runs:
        print(f"  Processing {run_name}...")
        stats = parse_grading_report(report_path)
        problem_results = parse_problem_results(report_path)
        
        if stats:
            stats['run_name'] = run_name
            stats['report_path'] = report_path
            runs_stats.append(stats)
            all_problem_results.append(problem_results)
            print(f"    ✓ Accuracy: {stats.get('accuracy', 'N/A'):.2f}%")
            if problem_results:
                print(f"    ✓ Parsed {len(problem_results)} individual problem results")
        else:
            print(f"    ✗ Failed to parse report")
    
    if not runs_stats:
        print("\nError: No valid grading reports could be parsed")
        return
    
    # Calculate per-problem accuracy
    print("\nCalculating per-problem accuracy...")
    problem_accuracy = calculate_per_problem_accuracy(all_problem_results)
    
    theoretical_variance = None
    theoretical_std = None
    
    if problem_accuracy:
        print(f"  ✓ Calculated accuracy for {len(problem_accuracy)} problems")
        
        # Calculate accuracy per group
        print("\nCalculating accuracy per group (30 problems each)...")
        group_stats = calculate_accuracy_per_group(problem_accuracy, group_size=30)
        print(f"  ✓ Created {len(group_stats)} group(s)")
        for group_label in sorted(group_stats.keys(), key=lambda x: group_stats[x]['start']):
            stats = group_stats[group_label]
            print(f"    Group {group_label}: {stats['avg_accuracy']*100:.2f}% (n={stats['count']})")
        
        # Calculate theoretical variance
        print("\nCalculating theoretical variance...")
        theoretical_variance, theoretical_std = calculate_theoretical_variance(problem_accuracy)
        print(f"  ✓ Theoretical variance: {theoretical_variance:.4f}")
        print(f"  ✓ Theoretical std dev: {theoretical_std:.4f}")
    else:
        print("  ⚠ Could not calculate per-problem accuracy (may need to adjust parsing)")
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(
        args.test_dir, 
        runs_stats, 
        problem_accuracy,
        theoretical_variance,
        theoretical_std,
        args.output
    )
    
    # Generate CSV if requested
    if args.csv:
        print("Generating CSV report...")
        generate_csv_report(args.test_dir, runs_stats)
    
    # Print summary to console
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    accuracies = [s['accuracy'] for s in runs_stats if 'accuracy' in s]
    correct_counts = [s['correct'] for s in runs_stats if 'correct' in s]
    
    if accuracies:
        avg_accuracy = statistics.mean(accuracies)
        print(f"Total Runs: {len(runs_stats)}")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        
        if len(accuracies) > 1:
            std_accuracy = statistics.stdev(accuracies)
            print(f"Empirical Std Dev (Accuracy): {std_accuracy:.2f}%")
            print(f"Range: [{min(accuracies):.2f}% - {max(accuracies):.2f}%]")
            
        if len(correct_counts) > 1:
            std_correct = statistics.stdev(correct_counts)
            mean_correct = statistics.mean(correct_counts)
            print(f"\nEmpirical Mean (Correct Count): {mean_correct:.4f}")
            print(f"Empirical Std Dev (Correct Count): {std_correct:.4f}")
    
    # Display theoretical statistics
    if theoretical_variance is not None and theoretical_std is not None:
        print(f"\nTheoretical Variance (Correct Count): {theoretical_variance:.4f}")
        print(f"Theoretical Std Dev (Correct Count): {theoretical_std:.4f}")
        
        if problem_accuracy:
            expected_correct = sum(problem_accuracy.values())
            print(f"Expected Correct Count: {expected_correct:.4f}")
    
    # Print aggregate totals
    total_problems = sum(s.get('total', 0) for s in runs_stats)
    total_correct = sum(s.get('correct', 0) for s in runs_stats)
    total_incorrect = sum(s.get('incorrect', 0) for s in runs_stats)
    total_no_answer = sum(s.get('no_answer', 0) for s in runs_stats)
    
    print(f"\nAggregate Totals:")
    print(f"  Total Problems: {total_problems}")
    print(f"  Correct: {total_correct}")
    print(f"  Incorrect: {total_incorrect}")
    print(f"  Failed Extraction: {total_no_answer}")
    
    print("="*70)
    print(f"\nReports saved to {args.test_dir}/")
    print("  - " + args.output)
    if args.csv:
        print("  - summary_report.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()