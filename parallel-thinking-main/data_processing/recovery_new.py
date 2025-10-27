import json
import os
import re
from pathlib import Path

def fix_xml_content_for_jsonl(xml_content):
    """
    Fix XML content for JSONL output by:
    1. Removing "The model " from objectives that start with it
    2. Removing empty launch_threads blocks
    3. Detecting parallel_processing blocks without launch_threads (for filtering)
    
    Args:
        xml_content (str): The original XML content
        
    Returns:
        tuple: (fixed_xml_content, should_exclude, statistics)
    """
    stats = {
        'objectives_fixed': 0,
        'empty_launch_threads_removed': 0,
        'parallel_processing_without_launch_threads': 0
    }
    
    should_exclude = False
    
    # Check for parallel_processing blocks without launch_threads
    parallel_blocks = re.findall(r'<parallel_processing>(.*?)</parallel_processing>', xml_content, re.DOTALL)
    for block in parallel_blocks:
        if '<launch_threads>' not in block:
            stats['parallel_processing_without_launch_threads'] += 1
            should_exclude = True
    
    # If we should exclude this entry, return early
    if should_exclude:
        return xml_content, True, stats
    
    # Fix 1: Remove "The model " from objectives that start with it
    def fix_objective(match):
        objective_content = match.group(1)
        if objective_content.startswith("The model "):
            stats['objectives_fixed'] += 1
            new_content = objective_content[len("The model "):]  # Remove "The model " (with space)
            return f"<objective>\n{new_content}\n</objective>"
        return match.group(0)
    
    # Apply objective fixes
    objective_pattern = r'<objective>\s*(.*?)\s*</objective>'
    fixed_content = re.sub(objective_pattern, fix_objective, xml_content, flags=re.DOTALL)
    
    # Fix 2: Remove empty launch_threads blocks
    lines = fixed_content.split('\n')
    filtered_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line contains <launch_threads>
        if '<launch_threads>' in line:
            # Look for empty launch_threads pattern
            start_idx = i
            found_end = False
            content_between = ""
            
            # Look ahead to find </launch_threads>
            j = i
            while j < len(lines):
                current_line = lines[j].strip()
                
                if '</launch_threads>' in current_line:
                    found_end = True
                    # Check if there's only whitespace between tags
                    if not content_between.strip():
                        stats['empty_launch_threads_removed'] += 1
                        i = j + 1  # Skip all these lines
                        break
                    else:
                        # There's content, keep the lines
                        break
                elif j > i:  # Don't include the opening tag line in content check
                    content_between += current_line
                
                j += 1
            
            if not found_end or content_between.strip():
                # Either didn't find closing tag or there's content - keep the line
                filtered_lines.append(lines[i])
                i += 1
        else:
            filtered_lines.append(lines[i])
            i += 1
    
    final_content = '\n'.join(filtered_lines)
    
    # Clean up any extra newlines
    final_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', final_content)
    
    return final_content, False, stats

def create_combined_jsonl(xml_results_file, original_data_file, output_file="combined_data.jsonl"):
    """
    Create a JSONL file combining XML results with original problem data.
    Excludes entries with parallel_processing blocks that have no launch_threads.
    Fixes XML content for remaining entries.
    
    Args:
        xml_results_file (str): Path to the XML generation results JSONL file
        original_data_file (str): Path to the original data JSONL file with Problem and CoT
        output_file (str): Path for the output JSONL file
    """
    
    # Statistics for XML fixes and filtering
    total_stats = {
        'files_processed': 0,
        'entries_excluded': 0,
        'objectives_fixed': 0,
        'empty_launch_threads_removed': 0,
        'parallel_processing_without_launch_threads': 0
    }
    
    # First, load all original data into memory for faster lookup
    print("Loading original data...")
    original_data = {}
    try:
        with open(original_data_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    data = json.loads(line.strip())
                    # Store by index (line_idx + 1 = problem_index)
                    original_data[line_idx + 1] = {
                        'Problem': data.get('Problem', ''),
                        'CoT': data.get('CoT', '')
                    }
        print(f"Loaded {len(original_data)} original data entries")
    except FileNotFoundError:
        print(f"Error: Original data file {original_data_file} not found")
        return
    except Exception as e:
        print(f"Error loading original data: {e}")
        return
    
    # Process XML results and create combined data
    print("Processing XML results...")
    successful_count = 0
    failed_count = 0
    excluded_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        try:
            with open(xml_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    try:
                        xml_data = json.loads(line.strip())
                        
                        # Skip failed entries
                        if xml_data.get('status') != 'success':
                            failed_count += 1
                            continue
                        
                        problem_index = xml_data.get('problem_index')
                        main_xml_path = xml_data.get('main_xml_path')
                        thread_xml_path = xml_data.get('thread_xml_path')
                        
                        if not all([problem_index, main_xml_path, thread_xml_path]):
                            print(f"Warning: Missing data for problem {problem_index}")
                            failed_count += 1
                            continue
                        
                        # Get original problem data
                        original = original_data.get(problem_index, {})
                        original_problem = original.get('Problem', f'[Problem {problem_index} not found]')
                        original_cot = original.get('CoT', f'[CoT {problem_index} not found]')
                        
                        # Load and check XML files for exclusion criteria
                        main_thread_content = ""
                        thread_1_content = ""
                        should_exclude_entry = False
                        
                        try:
                            total_stats['files_processed'] += 1
                            
                            if os.path.exists(main_xml_path):
                                with open(main_xml_path, 'r', encoding='utf-8') as xml_f:
                                    original_main_content = xml_f.read()
                                
                                # Check and fix main XML content
                                main_thread_content, should_exclude_main, main_stats = fix_xml_content_for_jsonl(original_main_content)
                                
                                if should_exclude_main:
                                    should_exclude_entry = True
                                    print(f"Excluding problem {problem_index}: main XML has parallel_processing without launch_threads")
                                
                                # Update statistics
                                for key in ['objectives_fixed', 'empty_launch_threads_removed', 'parallel_processing_without_launch_threads']:
                                    total_stats[key] += main_stats[key]
                                
                            else:
                                main_thread_content = f"[File not found: {main_xml_path}]"
                                
                            if os.path.exists(thread_xml_path):
                                with open(thread_xml_path, 'r', encoding='utf-8') as xml_f:
                                    original_thread_content = xml_f.read()
                                
                                # Check and fix thread XML content
                                thread_1_content, should_exclude_thread, thread_stats = fix_xml_content_for_jsonl(original_thread_content)
                                
                                if should_exclude_thread:
                                    should_exclude_entry = True
                                    print(f"Excluding problem {problem_index}: thread XML has parallel_processing without launch_threads")
                                
                                # Update statistics
                                for key in ['objectives_fixed', 'empty_launch_threads_removed', 'parallel_processing_without_launch_threads']:
                                    total_stats[key] += thread_stats[key]
                                        
                            else:
                                thread_1_content = f"[File not found: {thread_xml_path}]"
                            
                            # If we should exclude this entry, skip it
                            if should_exclude_entry:
                                excluded_count += 1
                                total_stats['entries_excluded'] += 1
                                continue
                                
                        except Exception as e:
                            print(f"Error reading XML files for problem {problem_index}: {e}")
                            main_thread_content = f"[Error reading file: {e}]"
                            thread_1_content = f"[Error reading file: {e}]"
                        
                        # Create combined entry (only if not excluded)
                        combined_entry = {
                            "problem_index": problem_index,
                            "original_problem": original_problem,
                            "original_CoT": original_cot,
                            "main_thread": main_thread_content,
                            "thread_1": thread_1_content
                        }
                        
                        # Write to output file
                        json.dump(combined_entry, out_f, ensure_ascii=False)
                        out_f.write('\n')
                        
                        successful_count += 1
                        
                        if successful_count % 100 == 0:
                            print(f"Processed {successful_count} entries...")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        failed_count += 1
                        continue
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                        failed_count += 1
                        continue
                        
        except FileNotFoundError:
            print(f"Error: XML results file {xml_results_file} not found")
            return
        except Exception as e:
            print(f"Error processing XML results: {e}")
            return
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed entries: {failed_count}")
    print(f"Excluded entries (parallel_processing without launch_threads): {excluded_count}")
    print(f"Output file: {output_file}")
    print(f"Total output entries: {successful_count}")
    
    print(f"\n=== XML FIXES SUMMARY ===")
    print(f"Total XML files processed: {total_stats['files_processed']}")
    print(f"Entries excluded: {total_stats['entries_excluded']}")
    print(f"Objectives fixed (removed 'The model '): {total_stats['objectives_fixed']}")
    print(f"Empty launch_threads blocks removed: {total_stats['empty_launch_threads_removed']}")
    print(f"Parallel_processing blocks without launch_threads found: {total_stats['parallel_processing_without_launch_threads']}")

def main():
    # File paths (adjust these to match your actual file locations)
    xml_results_file = "xml_generation_batched_size41_delay1min.jsonl"
    original_data_file = "out_open_math_reasoning.jsonl"  # Adjust this path
    output_file = "combined_parallel_thinking_data.jsonl"
    
    # Check if files exist
    if not os.path.exists(xml_results_file):
        print(f"Error: XML results file '{xml_results_file}' not found")
        return
    
    if not os.path.exists(original_data_file):
        print(f"Error: Original data file '{original_data_file}' not found")
        print("Please provide the correct path to your original JSONL file with Problem and CoT fields")
        return
    
    print(f"Creating combined JSONL file with XML filtering and fixes...")
    print(f"XML results: {xml_results_file}")
    print(f"Original data: {original_data_file}")
    print(f"Output: {output_file}")
    print("-" * 50)
    
    create_combined_jsonl(xml_results_file, original_data_file, output_file)

if __name__ == "__main__":
    main()