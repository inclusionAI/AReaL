#!/usr/bin/env python3
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


def parse_grading_report(report_path: str) -> Dict:
    """
    Parse a grading_report.txt file to extract statistics.
    
    Args:
        report_path: Path to grading_report.txt
    
    Returns:
        Dictionary with statistics (total, correct, incorrect, accuracy, etc.)
    """
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract statistics using regex
    stats = {}
    
    # Total problems
    match = re.search(r'Total problems:\s*(\d+)', content)
    if match:
        stats['total'] = int(match.group(1))
    
    # Correct
    match = re.search(r'Correct:\s*(\d+)\s*\(([\d.]+)%\)', content)
    if match:
        stats['correct'] = int(match.group(1))
        stats['correct_pct'] = float(match.group(2))
    
    # Incorrect
    match = re.search(r'Incorrect:\s*(\d+)\s*\(([\d.]+)%\)', content)
    if match:
        stats['incorrect'] = int(match.group(1))
        stats['incorrect_pct'] = float(match.group(2))
    
    # No answer extracted
    match = re.search(r'No answer extracted:\s*(\d+)\s*\(([\d.]+)%\)', content)
    if match:
        stats['no_answer'] = int(match.group(1))
        stats['no_answer_pct'] = float(match.group(2))
    
    # Errors
    match = re.search(r'Errors:\s*(\d+)\s*\(([\d.]+)%\)', content)
    if match:
        stats['errors'] = int(match.group(1))
        stats['errors_pct'] = float(match.group(2))
    
    # Accuracy (excluding errors)
    match = re.search(r'Accuracy \(excluding errors\):\s*([\d.]+)%', content)
    if match:
        stats['accuracy'] = float(match.group(1))
    else:
        # Calculate if not found
        if 'correct' in stats and 'total' in stats:
            if stats['total'] > 0:
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100
    
    return stats if stats else None


def parse_problem_results(report_path: str) -> Dict[str, bool]:
    """
    Parse individual problem results from grading_report.txt.
    
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
    
    # Look for patterns like "Correctness: ✓ CORRECT" or "Correctness: ✗ INCORRECT"
    # Match Problem ID followed by Correctness status
    # Pattern: Problem X: ... Correctness: ✓ CORRECT or ✗ INCORRECT
    
    # Split into problem sections
    problem_sections = re.split(r'\n-{80,}\n', content)
    
    for section in problem_sections:
        # Extract problem number
        problem_match = re.search(r'^Problem\s+(\d+):', section, re.MULTILINE)
        if problem_match:
            problem_id = problem_match.group(1)
            
            # Check correctness
            if re.search(r'Correctness:\s*✓\s*CORRECT', section):
                problem_results[problem_id] = True
            elif re.search(r'Correctness:\s*✗\s*INCORRECT', section):
                problem_results[problem_id] = False
            # Treat "No answer extracted" as incorrect
            elif re.search(r'Status:\s*no_answer_extracted', section):
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
        
        # Per-problem accuracy (moved before theoretical statistics for better organization)
        if problem_accuracy:
            f.write("\n")
            f.write("PER-PROBLEM ACCURACY\n")
            f.write("-"*80 + "\n\n")
            f.write("Note: 'No answer extracted' cases are treated as incorrect.\n\n")
            f.write(f"{'Problem ID':<15} {'Accuracy':<12} {'Percentage':<12} {'Variance':<12} {'Std Dev':<12}\n")
            f.write("-"*80 + "\n")
            
            total_variance_check = 0.0
            for problem_id in sorted(problem_accuracy.keys()):
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
