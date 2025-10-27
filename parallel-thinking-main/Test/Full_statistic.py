import json
import os
import re
from datetime import datetime

def find_timestamped_directories(base_path):
    """Find all directories with timestamp format YYYY-MM-DD_HH-MM-SS"""
    directories = []
    timestamp_pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and re.match(timestamp_pattern, item):
            directories.append(item_path)
    
    return sorted(directories)

def get_length_statistics(directory):
    """Get length statistics from a specific directory"""
    sum_main_len = 0
    sum_thread_len = 0
    sum_total_len = 0
    total_threads = 0
    total_launches = 0
    count = 0
    
    for i in range(500):
        if i < 10:
            problem_dir = os.path.join(directory, f"problem_0{i}")
        else:
            problem_dir = os.path.join(directory, f"problem_{i}")
        
        try:
            with open(os.path.join(problem_dir, "statistics.json"), "r") as f:
                statistics = json.load(f)
        except:
            continue
        
        main_len = statistics["main_answer_length"]
        thread_len = statistics["thread_answers_total_length"]
        number_thread = statistics["total_threads"]
        number_launches = len(statistics["launch_widths"])
        
        if main_len != 1:
            total_len = main_len + thread_len
            count += 1
            sum_main_len += main_len
            sum_thread_len += thread_len
            sum_total_len += total_len
            total_threads += number_thread
            total_launches += number_launches
    
    return {
        'count': count,
        'sum_main_len': sum_main_len,
        'sum_thread_len': sum_thread_len,
        'sum_total_len': sum_total_len,
        'total_threads': total_threads,
        'total_launches': total_launches
    }

def get_grading_statistics(directory):
    """Get grading statistics from grading_report.json"""
    grading_file = os.path.join(directory, "grading_report.json")
    
    try:
        with open(grading_file, "r") as f:
            grading_data = json.load(f)
        return {
            'total': grading_data.get('total', 0),
            'correct': grading_data.get('correct', 0),
            'accuracy': grading_data.get('accuracy', 0.0)
        }
    except:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}

def aggregate_all_statistics(base_directory, output_file=None):
    """Aggregate statistics from all timestamped directories"""
    directories = find_timestamped_directories(base_directory)
    
    # Create output file name if not provided
    if output_file is None:
        output_file = os.path.join(base_directory, "REPORT.txt")
    
    # Open file for writing
    with open(output_file, 'w') as f:
        def write_output(text):
            """Helper function to write to both console and file"""
            print(text)
            f.write(text + '\n')
        
        if not directories:
            write_output(f"No timestamped directories found in {base_directory}")
            return
        
        write_output(f"STATISTICS REPORT - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_output(f"Base directory: {base_directory}")
        write_output("="*80)
        write_output(f"\nFound {len(directories)} directories:")
        for dir_path in directories:
            write_output(f"  - {os.path.basename(dir_path)}")
        write_output("")
        
        # Aggregate length statistics
        total_length_stats = {
            'count': 0,
            'sum_main_len': 0,
            'sum_thread_len': 0,
            'sum_total_len': 0,
            'total_threads': 0,
            'total_launches': 0
        }
        
        # Aggregate grading statistics
        total_grading_stats = {
            'total': 0,
            'correct': 0
        }
        
        valid_dirs = 0
        
        for directory in directories:
            write_output(f"Processing {os.path.basename(directory)}...")
            
            # Get length statistics
            length_stats = get_length_statistics(directory)
            if length_stats['count'] > 0:
                for key in total_length_stats:
                    total_length_stats[key] += length_stats[key]
            
            # Get grading statistics
            grading_stats = get_grading_statistics(directory)
            if grading_stats['total'] > 0:
                total_grading_stats['total'] += grading_stats['total']
                total_grading_stats['correct'] += grading_stats['correct']
                valid_dirs += 1
                write_output(f"  - Problems: {grading_stats['correct']}/{grading_stats['total']} correct ({grading_stats['accuracy']:.1f}%)")
            else:
                write_output(f"  - No grading report found")
        
        write_output("\n" + "="*80)
        write_output("AGGREGATED RESULTS")
        write_output("="*80)
        
        # Print length statistics
        if total_length_stats['count'] > 0:
            write_output("\nLENGTH STATISTICS:")
            write_output(f"Total problems processed: {total_length_stats['count']}")
            write_output(f"Average main length: {total_length_stats['sum_main_len'] / total_length_stats['count']:.2f}")
            write_output(f"Average thread length: {total_length_stats['sum_thread_len'] / total_length_stats['count']:.2f}")
            write_output(f"Average total length: {total_length_stats['sum_total_len'] / total_length_stats['count']:.2f}")
            write_output(f"Average threads per problem: {total_length_stats['total_threads'] / total_length_stats['count']:.2f}")
            write_output(f"Average launches per problem: {total_length_stats['total_launches'] / total_length_stats['count']:.2f}")
            if total_length_stats['total_launches'] > 0:
                write_output(f"Average threads per launch: {total_length_stats['total_threads'] / total_length_stats['total_launches']:.2f}")
        
        # Print grading statistics
        if total_grading_stats['total'] > 0:
            overall_accuracy = (total_grading_stats['correct'] / total_grading_stats['total']) * 100
            write_output(f"\nGRADING STATISTICS:")
            write_output(f"Total problems graded: {total_grading_stats['total']}")
            write_output(f"Total correct answers: {total_grading_stats['correct']}")
            write_output(f"Overall accuracy: {overall_accuracy:.2f}%")
            write_output(f"Directories with grading data: {valid_dirs}")
        
        write_output(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    # Get the parent directory of the current script
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for timestamped directories in the current directory
    aggregate_all_statistics("/storage/openpsi/experiments/logs/admin/zzy_test/result/new_format_parallel_ratio/")