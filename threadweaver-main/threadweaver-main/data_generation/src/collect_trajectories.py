# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script references code from Multiverse's data processing script https://github.com/Multiverse4FM/Multiverse/blob/main/data/src/data/collect_1k.py

The original script as well as the part from the original script used in this script are under Apache License 2.0 https://github.com/Multiverse4FM/Multiverse/blob/main/LICENSE
"""

from datasets import load_dataset
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Collect samples from a dataset and save to files.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory or JSONL file.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to collect (default: all).')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing existing output directories/files.')
    args = parser.parse_args()

    collect_path = os.path.join(args.output_path, 'collected.jsonl')
    output_dir_exists = os.path.exists(args.output_path)
    if output_dir_exists and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output_path}. Use --overwrite to replace it.")
    
    # Determine if we're loading from JSONL or dataset based on file extension
    if args.dataset_path.endswith('.jsonl'):
        # Assert that dataset_path is a file
        assert os.path.isfile(args.dataset_path), f"dataset_path must be a file when using JSONL: {args.dataset_path}"
        
        # Load data from JSONL file
        collect_data = []
        with open(args.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                collect_data.append(data)
                
                if args.max_samples is not None and i + 1 >= args.max_samples:
                    break
        
        print(f"Loaded {len(collect_data)} samples from JSONL file {args.dataset_path}")
    else:
        # Assert that dataset_path is a directory
        assert os.path.isdir(args.dataset_path), f"dataset_path must be a directory when not using JSONL: {args.dataset_path}"
        
        # Load data from dataset
        dataset = load_dataset(args.dataset_path, split='train')

        collect_data = []
        for i in range(len(dataset)):
            data = dataset[i]
            d = dict()
            d['uuid'] = os.path.basename(args.dataset_path) + '-' + str(i)
            d['problem'] = data['question']
            d['solution'] = None
            # The dataset is filtered already
            d['correctness'] = True
            d['thinking'] = data['response_reasoning']
            d['output'] = data['response_content']
            collect_data.append(d)

            if args.max_samples is not None and i + 1 >= args.max_samples:
                break

        print(f"Collected {len(collect_data)} samples from dataset {args.dataset_path}")

    # Common logic for both cases: save collected.jsonl and individual text files
    os.makedirs(args.output_path, exist_ok=True)

    with open(collect_path, 'w') as f:
        for d in collect_data:
            f.write(json.dumps(d) + '\n')

    for i, d in enumerate(collect_data):
        with open(os.path.join(args.output_path, f'{i}.txt'), 'w') as f:
            f.write(d['thinking'] + '\n')

    print(f"Saved {len(collect_data)} samples to {collect_path} and individual text files in {args.output_path}")

if __name__ == "__main__":
    main()
