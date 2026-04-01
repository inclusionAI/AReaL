import json
import re
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, '/Users/zzy/Downloads/datasync/')
from api_call import call_model_claude

# Thread-safe print lock
print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)


def summarize_path(path_content, path_number, retries=8):
    """Call LLM to summarize the objective of a path."""
    prompt = (
        "Below is a reasoning path from a mathematical/scientific problem-solving trace. "
        "Summarize its objective in ONE concise sentence (20-40 words). "
        "Start directly with an action verb (e.g. 'Compute', 'Verify', 'Derive', 'Explore', 'Apply', etc.). "
        "Do NOT include any preamble, explanation, or extra text — output only the one-sentence summary.\n\n"
        f"<Path>\n{path_content.strip()}\n</Path>"
    )
    for attempt in range(retries):
        try:
            result = call_model_claude(prompt, max_tokens=2048)
            if result:
                summary = result['choices'][0]['message']['content'].strip()
                # Remove any accidental leading numbering the model might add
                summary = re.sub(r'^\d+\.\s*', '', summary)
                return f"{path_number}. {summary}"
        except Exception as e:
            safe_print(f"  [Attempt {attempt+1}] Error calling LLM for path {path_number}: {e}")
            time.sleep(2 ** attempt)
    return None


def fill_parallel_block(parallel_content, line_num, p_idx):
    """
    Given the inner content of a <Parallel> block, find empty <Trial> tags,
    look up the corresponding <Path> content, call the LLM, and return the
    updated parallel content.
    """
    goal_match = re.search(r'(<Goal>)(.*?)(</Goal>)', parallel_content, re.DOTALL)
    if not goal_match:
        return parallel_content

    goal_open, goal_content, goal_close = goal_match.group(1), goal_match.group(2), goal_match.group(3)

    # Split trials preserving delimiters
    trial_parts = re.split(r'(<Trial>.*?</Trial>)', goal_content, flags=re.DOTALL)
    # Collect all paths
    paths = re.findall(r'<Path>(.*?)</Path>', parallel_content, re.DOTALL)

    # Identify which trial indices are empty
    trial_tags = re.findall(r'<Trial>(.*?)</Trial>', goal_content, re.DOTALL)
    empty_indices = [i for i, t in enumerate(trial_tags) if not t.strip()]

    if not empty_indices:
        return parallel_content

    safe_print(f"  Line {line_num}, Parallel {p_idx}: filling {len(empty_indices)} empty trial(s): "
               f"{[i+1 for i in empty_indices]}")

    # Fetch summaries in parallel for this block
    summaries = {}
    tasks = {}
    with ThreadPoolExecutor(max_workers=min(len(empty_indices), 5)) as executor:
        for trial_idx in empty_indices:
            path_number = trial_idx + 1  # 1-indexed
            if path_number <= len(paths):
                future = executor.submit(summarize_path, paths[trial_idx], path_number)
                tasks[future] = trial_idx
            else:
                safe_print(f"  WARNING: No path #{path_number} found (only {len(paths)} paths). Skipping.")

        for future in as_completed(tasks):
            trial_idx = tasks[future]
            summary = future.result()
            if summary:
                summaries[trial_idx] = summary
            else:
                safe_print(f"  WARNING: Failed to get summary for trial #{trial_idx+1}. Leaving empty.")

    # Rebuild goal content: replace empty <Trial></Trial> with filled content
    trial_counter = [0]  # mutable counter for the split-based rebuild

    def replace_empty_trial(m):
        inner = m.group(1)
        current_idx = trial_counter[0]
        trial_counter[0] += 1
        if not inner.strip() and current_idx in summaries:
            return f"<Trial>\n{summaries[current_idx]}\n</Trial>"
        return m.group(0)

    new_goal_content = re.sub(r'<Trial>(.*?)</Trial>', replace_empty_trial, goal_content, flags=re.DOTALL)
    new_goal = goal_open + new_goal_content + goal_close
    new_parallel = parallel_content[:goal_match.start()] + new_goal + parallel_content[goal_match.end():]
    return new_parallel


def process_output(output, line_num):
    """Process the output field of one JSONL entry, filling all empty Trials."""
    # Find all <Parallel> blocks and replace them
    parallel_idx = [0]

    def replace_parallel(m):
        p_idx = parallel_idx[0] + 1
        parallel_idx[0] += 1
        inner = m.group(1)
        # Check if this parallel has any empty trials at all
        goal = re.search(r'<Goal>(.*?)</Goal>', inner, re.DOTALL)
        if not goal:
            return m.group(0)
        trials = re.findall(r'<Trial>(.*?)</Trial>', goal.group(1), re.DOTALL)
        if not any(not t.strip() for t in trials):
            return m.group(0)
        new_inner = fill_parallel_block(inner, line_num, p_idx)
        return f"<Parallel>{new_inner}</Parallel>"

    new_output = re.sub(r'<Parallel>(.*?)</Parallel>', replace_parallel, output, flags=re.DOTALL)
    return new_output


def main():
    input_path = '/Users/zzy/Downloads/datasync/trial-fixed.jsonl'
    output_path = '/Users/zzy/Downloads/datasync/trial-filled.jsonl'

    # Load all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = [l for l in f if l.strip()]

    safe_print(f"Loaded {len(raw_lines)} lines. Starting to fill empty <Trial> tags...\n")

    results = []
    total_filled = 0

    for line_num, raw_line in enumerate(raw_lines, start=1):
        data = json.loads(raw_line)
        output = data['output']

        # Quick check: does this line have any empty trials?
        parallels = re.findall(r'<Parallel>(.*?)</Parallel>', output, re.DOTALL)
        has_empty = False
        for p in parallels:
            goal = re.search(r'<Goal>(.*?)</Goal>', p, re.DOTALL)
            if goal:
                trials = re.findall(r'<Trial>(.*?)</Trial>', goal.group(1), re.DOTALL)
                if any(not t.strip() for t in trials):
                    has_empty = True
                    break

        if has_empty:
            safe_print(f"Processing line {line_num}...")
            new_output = process_output(output, line_num)
            # Count how many got filled
            old_empty = len(re.findall(r'<Trial>\s*</Trial>', output))
            new_empty = len(re.findall(r'<Trial>\s*</Trial>', new_output))
            filled = old_empty - new_empty
            total_filled += filled
            safe_print(f"  Line {line_num}: filled {filled} trial(s), {new_empty} still empty.\n")
            data['output'] = new_output
        
        results.append(data)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in results:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    safe_print(f"\nDone! Total trials filled: {total_filled}")
    safe_print(f"Output written to: {output_path}")


if __name__ == '__main__':
    main()
