#!/usr/bin/env python3
"""
Analyze test logs to compare response lengths between base and trained models.
"""

import re
import sys
from pathlib import Path


def parse_log_file(log_path):
    """Extract statistics from a test log file."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract model name
    model_match = re.search(r'Testing model: (.+)', content)
    model_name = model_match.group(1) if model_match else "Unknown"
    
    # Extract all questions and their stats
    questions = []
    
    # Pattern to match question blocks
    question_pattern = r'--- Question (\d+) ---.*?Stop reason: (.+?); new_tokens: (\d+)/(\d+)'
    
    matches = re.finditer(question_pattern, content, re.DOTALL)
    for match in matches:
        q_num = int(match.group(1))
        stop_reason = match.group(2).strip()
        new_tokens = int(match.group(3))
        max_tokens = int(match.group(4))
        
        # Extract generated text
        gen_match = re.search(r'Generated \(full, from full_output\):\n(.*?)(?=\nCorrect Answer|\n--- Question|$)', content[match.end():], re.DOTALL)
        generated_text = gen_match.group(1).strip() if gen_match else ""
        
        # Extract result
        result_match = re.search(r'Result: (.+)', content[match.end():])
        is_correct = "✓ CORRECT" in (result_match.group(1) if result_match else "")
        
        questions.append({
            'number': q_num,
            'stop_reason': stop_reason,
            'new_tokens': new_tokens,
            'max_tokens': max_tokens,
            'generated_length': len(generated_text),
            'generated_words': len(generated_text.split()),
            'is_correct': is_correct,
        })
    
    # Extract final accuracy
    acc_match = re.search(r'ACCURACY: ([\d.]+)% \((\d+)/(\d+)\)', content)
    accuracy = float(acc_match.group(1)) if acc_match else 0.0
    correct_count = int(acc_match.group(2)) if acc_match else 0
    total_count = int(acc_match.group(3)) if acc_match else len(questions)
    
    return {
        'model': model_name,
        'questions': questions,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
    }


def print_statistics(stats):
    """Print statistics in a readable format."""
    model = stats['model']
    questions = stats['questions']
    
    if not questions:
        print(f"\nNo questions found in log for {model}")
        return
    
    # Calculate averages
    avg_tokens = sum(q['new_tokens'] for q in questions) / len(questions)
    avg_chars = sum(q['generated_length'] for q in questions) / len(questions)
    avg_words = sum(q['generated_words'] for q in questions) / len(questions)
    
    # Count stop reasons
    stop_reasons = {}
    max_token_hits = 0
    eos_stops = 0
    hash_stops = 0
    
    for q in questions:
        reason = q['stop_reason']
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        
        if q['new_tokens'] >= q['max_tokens']:
            max_token_hits += 1
        if 'eos' in reason.lower():
            eos_stops += 1
        if 'hash' in reason.lower():
            hash_stops += 1
    
    # Find min/max
    min_tokens = min(q['new_tokens'] for q in questions)
    max_tokens = max(q['new_tokens'] for q in questions)
    
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    print(f"Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"\nResponse Length Statistics:")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Average characters: {avg_chars:.1f}")
    print(f"  Average words: {avg_words:.1f}")
    print(f"  Min tokens: {min_tokens}")
    print(f"  Max tokens: {max_tokens}")
    print(f"\nStop Reasons:")
    for reason, count in sorted(stop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nSummary:")
    print(f"  Hit max tokens: {max_token_hits}/{len(questions)} ({max_token_hits/len(questions)*100:.1f}%)")
    print(f"  Stopped at EOS: {eos_stops}/{len(questions)} ({eos_stops/len(questions)*100:.1f}%)")
    print(f"  Stopped at ####: {hash_stops}/{len(questions)} ({hash_stops/len(questions)*100:.1f}%)")


def compare_logs(log1_path, log2_path):
    """Compare two log files."""
    stats1 = parse_log_file(log1_path)
    stats2 = parse_log_file(log2_path)
    
    print("\n" + "="*60)
    print("LOG COMPARISON")
    print("="*60)
    
    print_statistics(stats1)
    print_statistics(stats2)
    
    # Direct comparison
    if stats1['questions'] and stats2['questions']:
        avg_tokens1 = sum(q['new_tokens'] for q in stats1['questions']) / len(stats1['questions'])
        avg_tokens2 = sum(q['new_tokens'] for q in stats2['questions']) / len(stats2['questions'])
        
        avg_chars1 = sum(q['generated_length'] for q in stats1['questions']) / len(stats1['questions'])
        avg_chars2 = sum(q['generated_length'] for q in stats2['questions']) / len(stats2['questions'])
        
        max_hits1 = sum(1 for q in stats1['questions'] if q['new_tokens'] >= q['max_tokens'])
        max_hits2 = sum(1 for q in stats2['questions'] if q['new_tokens'] >= q['max_tokens'])
        
        print(f"\n{'='*60}")
        print("DIRECT COMPARISON")
        print(f"{'='*60}")
        print(f"Average tokens per response:")
        print(f"  {stats1['model']}: {avg_tokens1:.1f}")
        print(f"  {stats2['model']}: {avg_tokens2:.1f}")
        print(f"  Difference: {avg_tokens2 - avg_tokens1:.1f} tokens ({((avg_tokens2/avg_tokens1 - 1)*100):.1f}%)")
        print(f"\nAverage characters per response:")
        print(f"  {stats1['model']}: {avg_chars1:.1f}")
        print(f"  {stats2['model']}: {avg_chars2:.1f}")
        print(f"  Difference: {avg_chars2 - avg_chars1:.1f} chars ({((avg_chars2/avg_chars1 - 1)*100):.1f}%)")
        print(f"\nQuestions hitting max tokens:")
        print(f"  {stats1['model']}: {max_hits1}/{len(stats1['questions'])} ({max_hits1/len(stats1['questions'])*100:.1f}%)")
        print(f"  {stats2['model']}: {max_hits2}/{len(stats2['questions'])} ({max_hits2/len(stats2['questions'])*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print("ASSESSMENT")
        print(f"{'='*60}")
        # Calculate reduction: (base - trained) / base * 100
        # Assuming stats1 is trained, stats2 is base (when called as compare_logs(trained, base))
        reduction = ((avg_tokens2 - avg_tokens1) / avg_tokens2) * 100 if avg_tokens2 > 0 else 0
        if reduction > 50:
            print(f"✓ Training significantly shortened responses by ~{reduction:.1f}%")
            print(f"  - Base model averaged {avg_tokens2:.0f} tokens per response")
            print(f"  - Trained model averaged {avg_tokens1:.0f} tokens per response")
            print(f"  - This is a {reduction:.1f}% reduction in response length")
        elif reduction > 20:
            print(f"✓ Training moderately shortened responses by ~{reduction:.1f}%")
        else:
            print(f"Training had minimal impact on response length ({reduction:.1f}% reduction)")
        
        if max_hits2 > max_hits1:
            print(f"\n✓ Training reduced the number of responses hitting max tokens:")
            print(f"  - Base: {max_hits2}/{len(stats2['questions'])} questions hit max tokens")
            print(f"  - Trained: {max_hits1}/{len(stats1['questions'])} questions hit max tokens")
        
        print(f"\nThis suggests the trained model learned to:")
        print(f"  - Generate more concise, direct answers")
        print(f"  - Stop generation appropriately (via #### or natural EOS)")
        print(f"  - Follow the GSM8K format more closely")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare_logs(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        stats = parse_log_file(sys.argv[1])
        print_statistics(stats)
    else:
        # Default: compare the two most recent logs
        log_dir = Path("examples/local_gsm8k/logs")
        logs = sorted(log_dir.glob("test_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if len(logs) >= 2:
            print(f"Comparing two most recent logs:")
            print(f"  1. {logs[0].name} (most recent)")
            print(f"  2. {logs[1].name} (second most recent)")
            compare_logs(str(logs[1]), str(logs[0]))  # Second most recent vs most recent
        else:
            print("Need at least 2 log files to compare")

