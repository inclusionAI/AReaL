#!/usr/bin/env python3
"""
Run batch inference with automatic server launch.
This script launches the SGLang server and runs inference all in one command.

Usage:
    python run_inference.py -m <model-path> -i <input-file> [-o <output-dir>] [--tp-size 8] [--port 30000]

Example:
    python run_inference.py -m /path/to/model -i S1-parallel/AIME2425.jsonl -o results
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run batch inference with automatic SGLang server launch'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: <model>/Test_<timestamp>)'
    )
    parser.add_argument(
        '--tp-size',
        type=int,
        default=8,
        help='Tensor parallelism size (default: 8)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=30000,
        help='Port for SGLang server (default: 30000)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start index for processing problems (default: 0)'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index for processing problems (default: None, process all)'
    )
    parser.add_argument(
        '--max-normal-tokens',
        type=int,
        default=10000,
        help='Max tokens for normal generation (default: 10000)'
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
        help='Max total tokens (default: 32768)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature (default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.model}/Test_{timestamp}"
    else:
        output_dir = args.output_dir
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Batch Inference with Server Launch")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"TP Size: {args.tp_size}")
    print(f"Port: {args.port}")
    print("="*70)
    
    # Build command
    cmd = [
        'python3', 'new_batch_inference_new.py',
        '--input', args.input,
        '--output-dir', output_dir,
        '--model', args.model,
        '--tp-size', str(args.tp_size),
        '--port', str(args.port),
        '--start-idx', str(args.start_idx),
        '--max-normal-tokens', str(args.max_normal_tokens),
        '--max-path-tokens', str(args.max_path_tokens),
        '--max-conclusion-tokens', str(args.max_conclusion_tokens),
        '--max-total-tokens', str(args.max_total_tokens),
        '--temperature', str(args.temperature)
    ]
    
    if args.end_idx is not None:
        cmd.extend(['--end-idx', str(args.end_idx)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run inference
    try:
        result = subprocess.run(cmd, check=True)
        
        # Run aggregation if successful
        print("\n" + "="*70)
        print("Running aggregation...")
        print("="*70)
        
        agg_cmd = ['python3', 'aggregate_accuracy.py', '--test-dir', output_dir]
        subprocess.run(agg_cmd, check=True)
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        print(f"Check {output_dir}/summary_report.txt for results")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
