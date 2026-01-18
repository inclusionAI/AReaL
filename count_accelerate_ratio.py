import os
import re
from pathlib import Path
from typing import Tuple, List
import argparse

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

def calculate_latency(text: str) -> int:
    """
    Calculate the maximum latency for parallel reasoning patterns.
    
    For sequential content, latency = word count
    For parallel content in <Parallel>...</Parallel>, latency = max(path
latencies)
    Handles hierarchical parallel stages recursively.
    """
    # Remove content outside <Think> tags for cleaner processing
    think_match = re.search(r'<Think>(.*?)</Think>', text, re.DOTALL)
    if think_match:
        content = think_match.group(1)
    else:
        content = text
    
    return _calculate_section_latency(content)

def _calculate_section_latency(content: str) -> int:
    """
    Calculate latency for a section of content.
    Only processes the outermost <Parallel>...</Parallel> groups.
    Nested <Parallel> tags are treated as normal words.
    """
    total_latency = 0
    pos = 0
    
    while pos < len(content):
        # Find next parallel block (only outermost)
        parallel_start = content.find('<Parallel>', pos)
        
        if parallel_start == -1:
            # No more parallel blocks, count remaining sequential content
            remaining = content[pos:]
            total_latency += count_words(remaining)
            break
        
        # Count sequential content before parallel block
        sequential_before = content[pos:parallel_start]
        total_latency += count_words(sequential_before)
        
        # Find matching closing tag (only for outermost level)
        parallel_end = _find_outermost_closing_tag(content, parallel_start)
        
        if parallel_end == -1:
            # Unclosed parallel block - treat rest of content as if it has
            # </Parallel> at the end
            parallel_content = content[parallel_start + len('<Parallel>'):]
            
            # Calculate latency for this parallel block
            parallel_latency =_calculate_parallel_block_latency(parallel_content)
            total_latency += parallel_latency
            
            # Done processing
            break
        
        # Extract parallel block content (excluding the tags themselves)
        parallel_content = content[parallel_start +
len('<Parallel>'):parallel_end]
        
        # Calculate latency for this parallel block
        parallel_latency = _calculate_parallel_block_latency(parallel_content)
        total_latency += parallel_latency
        
        # Move position past this parallel block
        pos = parallel_end + len('</Parallel>')
    
    return total_latency

def _calculate_parallel_block_latency(parallel_content: str) -> int:
    """
    Calculate latency for a parallel block.
    Latency = max(path latencies) + content outside paths
    """
    # Extract all path blocks and their positions
    path_info = []
    path_pos = 0
    
    while True:
        path_start = parallel_content.find('<Path>', path_pos)
        if path_start == -1:
            break
        
        path_end = parallel_content.find('</Path>', path_start)
        if path_end == -1:
            break
        
        # Store the path start, end, and content
        path_content = parallel_content[path_start + len('<Path>'):path_end]
        path_info.append({
            'start': path_start,
            'end': path_end + len('</Path>'),
            'latency': count_words(path_content)
        })
        
        path_pos = path_end + len('</Path>')
    
    # Calculate content outside all paths
    content_outside_paths = parallel_content
    for path in reversed(path_info):  # Remove from end to start to preserve positions
        content_outside_paths = content_outside_paths[:path['start']] + content_outside_paths[path['end']:]
    
    outside_latency = count_words(content_outside_paths)
    
    # Get max path latency
    max_path_latency = max([p['latency'] for p in path_info]) if path_info else 0
    
    # Total latency is max path latency + content outside paths
    return max_path_latency + outside_latency

def _find_outermost_closing_tag(content: str, start_pos: int) -> int:
    """
    Find the position of the closing tag for the outermost <Parallel> block.
    Only considers the first-level nesting.
    """
    depth = 1
    pos = start_pos + len('<Parallel>')
    
    while pos < len(content) and depth > 0:
        next_open = content.find('<Parallel>', pos)
        next_close = content.find('</Parallel>', pos)
        
        if next_close == -1:
            # No closing tag found
            return -1
        
        if next_open != -1 and next_open < next_close:
            # Found another opening tag before the closing tag
            depth += 1
            pos = next_open + len('<Parallel>')
        else:
            # Found a closing tag
            depth -= 1
            if depth == 0:
                return next_close
            pos = next_close + len('</Parallel>')
    
    return -1

def process_file(file_path: Path, require_boxed: bool = False) -> Tuple[int, int, bool]:
    """
    Process a single file and return (total_words, latency, is_valid).
    
    Args:
        file_path: Path to the file to process
        require_boxed: If True, skip files that don't contain \\boxed{
    
    Returns:
        Tuple of (total_words, latency, is_valid)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if boxed answer is required and present
        if require_boxed and r'\boxed{' not in content:
            return 0, 0, False
        
        total_words = count_words(content)
        latency = calculate_latency(content)
        
        return total_words, latency, True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, False

def process_directory(directory: str, require_boxed: bool = False) -> None:
    """
    Process all answer files in subdirectories and calculate statistics.
    
    Args:
        directory: Directory containing timestamp subdirectories with answer files
        require_boxed: If True, skip files that don't contain \\boxed{
    """
    base_path = Path(directory)
    
    if not base_path.exists():
        print(f"Directory {directory} does not exist!")
        return
    
    # Find all timestamp subdirectories
    timestamp_dirs = [d for d in base_path.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    
    if not timestamp_dirs:
        print(f"No timestamp directories found in {directory}")
        return
    
    if require_boxed:
        print(f"Found {len(timestamp_dirs)} timestamp directories (filtering for \\boxed{{}} answers)\n")
    else:
        print(f"Found {len(timestamp_dirs)} timestamp directories\n")
    
    grand_total_words = 0
    grand_total_latency = 0
    problem_results = []
    skipped_count = 0
    
    for timestamp_dir in sorted(timestamp_dirs):
        print(f"Processing directory: {timestamp_dir.name}")
        
        # Find all answer*.txt files
        answer_files = sorted(timestamp_dir.glob('answer*.txt'))
        
        for answer_file in answer_files:
            total_words, latency, is_valid = process_file(answer_file, require_boxed)
            
            if not is_valid:
                skipped_count += 1
                if require_boxed:
                    print(f"  {answer_file.name}: SKIPPED (no \\boxed{{}} found)")
                continue
            
            if total_words > 0 and latency > 0:
                speedup = total_words / latency if latency > 0 else 0
                problem_results.append({
                    'file': answer_file.name,
                    'dir': timestamp_dir.name,
                    'words': total_words,
                    'latency': latency,
                    'speedup': speedup
                })
                
                grand_total_words += total_words
                grand_total_latency += latency
                
                print(f"  {answer_file.name}: {total_words} words, {latency} latency, speedup: {speedup:.2f}x")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total problems processed: {len(problem_results)}")
    if skipped_count > 0:
        print(f"Problems skipped: {skipped_count}")
    print(f"Total words: {grand_total_words}")
    print(f"Total latency: {grand_total_latency}")
    
    if len(problem_results) > 0:
        average_latency = grand_total_latency / len(problem_results)
        print(f"Average latency per problem: {average_latency:.2f}")
    
    if grand_total_latency > 0:
        overall_speedup = grand_total_words / grand_total_latency
        print(f"Overall speedup (total words / total latency): {overall_speedup:.2f}x")
    else:
        print("No valid data to calculate speedup")
    
    print("\n" + "="*80)
    print("INDIVIDUAL RESULTS")
    print("="*80)
    for result in problem_results:
        print(f"{result['dir']}/{result['file']}: {result['words']} words, "
              f"{result['latency']} latency, {result['speedup']:.2f}x speedup")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count parallel reasoning pattern acceleration')
    parser.add_argument('directory', help='Directory containing timestamp subdirectories with answer files')
    parser.add_argument('--require-boxed', action='store_true', 
                        help='Skip answers that do not contain \\boxed{}')
    
    args = parser.parse_args()
    process_directory(args.directory, args.require_boxed)