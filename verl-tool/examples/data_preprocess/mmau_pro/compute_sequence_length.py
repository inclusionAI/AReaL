"""
Analyze input sequence length statistics for train and test datasets.
Uses Qwen2_5OmniProcessor to process multimodal inputs and calculate sequence length statistics.
"""

import re
import numpy as np
import pandas as pd
from transformers import Qwen2_5OmniProcessor
import fire
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _build_messages(sample):
    """
    Build messages from sample following the same logic as rl_dataset.py _build_messages.
    Converts content strings with <audio> tokens to structured format.
    
    Args:
        sample: Sample dict with 'prompt' and 'audios' fields
        
    Returns:
        list: Processed messages with structured content
    """
    # Extract messages from prompt (make a copy to avoid modifying original)
    messages = [msg.copy() for msg in sample['prompt']]
    
    # If there are audios, process messages to convert <audio> tokens to structured format
    if 'audios' in sample and sample.get('audios') is not None:
        for message in messages:
            content = message["content"]
            # Only process string content (not already structured)
            if isinstance(content, str):
                content_list = []
                # Split content by special tokens (<image>, <video>, <audio>)
                segments = re.split("(<image>|<video>|<audio>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    elif segment == "<audio>":
                        content_list.append({"type": "audio"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                
                message["content"] = content_list
    
    return messages


def process_sample(processor, sample, audio_start_special_token_id, audio_end_special_token_id):
    """
    Process a single sample with Qwen2_5OmniProcessor and return sequence length.
    Follows the pattern from rl_dataset.py doc2len function.
    
    Args:
        processor: Qwen2_5OmniProcessor instance
        sample: Sample dict with 'prompt' and 'audios' fields
        
    Returns:
        int: Sequence length (input_ids length), or None if processing fails
    """
    try:
        # Build messages using the same logic as rl_dataset._build_messages
        # This converts <audio> tokens in content strings to structured format
        messages = _build_messages(sample)
        
        # Apply chat template to get raw prompt text
        raw_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if isinstance(raw_prompt, list):
            raw_prompt = raw_prompt[0]
        
        # Process audio files using process_audio from audio_utils
        # This converts audio paths to numpy arrays that the processor expects
        from verl_tool.utils.dataset.audio_utils import process_audio
        
        audios = None
        if 'audios' in sample and sample.get('audios') is not None:
            audios = [process_audio(audio) for audio in sample['audios']]
        
        # Process with Qwen2_5OmniProcessor
        # Pass text as a list (processor expects list format)
        inputs = processor(text=[raw_prompt], audio=audios)
        
        # Extract sequence length from input_ids
        input_ids = inputs["input_ids"][0]
        total_length = len(input_ids)
        # keep audio sequence length calculation for future analysis
        # audio_start_pos = np.where(input_ids == audio_start_special_token_id)[0]
        # audio_end_pos = np.where(input_ids == audio_end_special_token_id)[0]
        # audio_length = audio_end_pos - audio_start_pos + 1 

        return total_length
        
    except Exception as e:
        print(f"Warning: Failed to process sample: {e}")
        return None

def calculate_statistics(sequence_lengths):
    """
    Calculate statistics for sequence lengths.
    
    Args:
        sequence_lengths: List of sequence lengths
        
    Returns:
        dict: Statistics including min, max, q1, q2 (median), q3, and mean
    """
    if not sequence_lengths:
        return None
    
    arr = np.array(sequence_lengths)
    stats = {
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'q1': int(np.percentile(arr, 25)),
        'q2': int(np.percentile(arr, 50)),  # median
        'q3': int(np.percentile(arr, 75)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'count': len(sequence_lengths)
    }
    return stats


def process_samples_parallel(
    samples, processor, model_name, audio_start_special_token_id, 
    audio_end_special_token_id, max_workers=16, desc="Processing"
):
    """
    Process samples in parallel using ThreadPoolExecutor.
    
    Args:
        samples: List of samples to process
        processor: Qwen2_5OmniProcessor instance (shared across threads - thread-safe for read ops)
        model_name: Model name (kept for compatibility, not used)
        audio_start_special_token_id: Audio start token ID
        audio_end_special_token_id: Audio end token ID
        max_workers: Maximum number of worker threads
        desc: Description for progress bar
        
    Returns:
        list: List of sequence lengths in the same order as input samples (None for failed samples)
    """
    def process_single_sample(sample):
        """Process a single sample using shared processor."""
        return process_sample(
            processor, sample, 
            audio_start_special_token_id, audio_end_special_token_id
        )
    
    # Process samples in parallel
    # Note: transformers processors are thread-safe for read operations (tokenization, processing)
    sequence_lengths = [None] * len(samples)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their indices to maintain order
        future_to_index = {
            executor.submit(process_single_sample, sample): idx 
            for idx, sample in enumerate(samples)
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_index), total=len(samples), desc=desc):
            idx = future_to_index[future]
            seq_len = future.result()
            sequence_lengths[idx] = seq_len
    
    return sequence_lengths


def main(
    train_parquet_path='/data/zhenxiong/data/mmau_pro/train.parquet',
    test_parquet_path='/data/zhenxiong/data/mmau_pro/test.parquet',
    model_name='qwen/qwen2.5-omni-3b',
    max_samples=None,  # Limit number of samples for faster analysis (None = all)
    max_workers=32,  # Maximum number of worker threads
):
    """
    Analyze sequence lengths for train and test datasets.
    
    Args:
        train_parquet_path: Path to train.parquet file
        test_parquet_path: Path to test.parquet file
        model_name: HuggingFace model name for Qwen2_5OmniProcessor
        max_samples: Maximum number of samples to process (None for all)
        max_workers: Maximum number of worker threads for parallel processing (default: 16)
    """
    print(f"Loading Qwen2_5OmniProcessor from {model_name}...", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    audio_start_special_token_id = processor.tokenizer.encode("<|audio_bos|>")[0]
    audio_special_token_id = processor.tokenizer.encode("<|AUDIO|>")[0]
    audio_end_special_token_id = processor.tokenizer.encode("<|audio_eos|>")[0]
    print(audio_start_special_token_id, audio_special_token_id, audio_end_special_token_id)
    
    # Process train dataset
    print(f"\nLoading train dataset from {train_parquet_path}...", flush=True)
    train_df = pd.read_parquet(train_parquet_path)
    train_samples = train_df.to_dict('records')
    
    if max_samples is not None:
        train_samples = train_samples[:max_samples]
    
    print(f"Processing {len(train_samples)} train samples with {max_workers} workers...", flush=True)
    train_sequence_lengths = process_samples_parallel(
        train_samples, processor, model_name, 
        audio_start_special_token_id, audio_end_special_token_id,
        max_workers=max_workers, desc="Train"
    )
    
    # Add seq_length column to train dataframe
    train_df_with_seq = train_df.copy()
    # Initialize seq_length column with None if it doesn't exist
    if 'seq_length' not in train_df_with_seq.columns:
        train_df_with_seq['seq_length'] = None
    
    if max_samples is not None:
        # Only update the processed samples (use iloc for position-based indexing)
        train_df_with_seq.iloc[:max_samples, train_df_with_seq.columns.get_loc('seq_length')] = train_sequence_lengths
    else:
        # Update all samples
        train_df_with_seq['seq_length'] = train_sequence_lengths
    # Filter out None values for statistics
    train_sequence_lengths_filtered = [x for x in train_sequence_lengths if x is not None]
    
    # Process test dataset
    print(f"\nLoading test dataset from {test_parquet_path}...", flush=True)
    test_df = pd.read_parquet(test_parquet_path)
    test_samples = test_df.to_dict('records')
    
    if max_samples is not None:
        test_samples = test_samples[:max_samples]
    
    print(f"Processing {len(test_samples)} test samples with {max_workers} workers...", flush=True)
    test_sequence_lengths = process_samples_parallel(
        test_samples, processor, model_name,
        audio_start_special_token_id, audio_end_special_token_id,
        max_workers=max_workers, desc="Test"
    )
    
    # Add seq_length column to test dataframe
    test_df_with_seq = test_df.copy()
    # Initialize seq_length column with None if it doesn't exist
    if 'seq_length' not in test_df_with_seq.columns:
        test_df_with_seq['seq_length'] = None
    
    if max_samples is not None:
        # Only update the processed samples (use iloc for position-based indexing)
        test_df_with_seq.iloc[:max_samples, test_df_with_seq.columns.get_loc('seq_length')] = test_sequence_lengths
    else:
        # Update all samples
        test_df_with_seq['seq_length'] = test_sequence_lengths
    # Filter out None values for statistics
    test_sequence_lengths_filtered = [x for x in test_sequence_lengths if x is not None]
    
    # Write updated dataframes back to parquet files
    train_df_with_seq.to_parquet("/data/zhenxiong/data/mmau_pro/train_w_length.parquet", index=False)
    processed_count = len([x for x in train_sequence_lengths if x is not None])
    print(f"Train dataset updated with seq_length column. Total samples: {len(train_df_with_seq)}, "
          f"Processed: {processed_count}")
    
    test_df_with_seq.to_parquet("/data/zhenxiong/data/mmau_pro/test_w_length.parquet", index=False)
    processed_count = len([x for x in test_sequence_lengths if x is not None])
    print(f"Test dataset updated with seq_length column. Total samples: {len(test_df_with_seq)}, "
          f"Processed: {processed_count}")
    
    # Calculate and print statistics
    print("\n" + "="*60)
    print("SEQUENCE LENGTH STATISTICS")
    print("="*60)
    
    train_stats = calculate_statistics(train_sequence_lengths_filtered)
    if train_stats:
        print("\nTrain Dataset:")
        print(f"  Count:    {train_stats['count']}")
        print(f"  Min:      {train_stats['min']}")
        print(f"  Q1:       {train_stats['q1']}")
        print(f"  Q2 (Med): {train_stats['q2']}")
        print(f"  Q3:       {train_stats['q3']}")
        print(f"  Max:      {train_stats['max']}")
        print(f"  Mean:     {train_stats['mean']:.2f}")
        print(f"  Std:      {train_stats['std']:.2f}")
        failed_count = len(train_sequence_lengths) - train_stats['count']
        if failed_count > 0:
            print(f"  Failed:   {failed_count} samples could not be processed")
    
    test_stats = calculate_statistics(test_sequence_lengths_filtered)
    if test_stats:
        print("\nTest Dataset:")
        print(f"  Count:    {test_stats['count']}")
        print(f"  Min:      {test_stats['min']}")
        print(f"  Q1:       {test_stats['q1']}")
        print(f"  Q2 (Med): {test_stats['q2']}")
        print(f"  Q3:       {test_stats['q3']}")
        print(f"  Max:      {test_stats['max']}")
        print(f"  Mean:     {test_stats['mean']:.2f}")
        print(f"  Std:      {test_stats['std']:.2f}")
        failed_count = len(test_sequence_lengths) - test_stats['count']
        if failed_count > 0:
            print(f"  Failed:   {failed_count} samples could not be processed")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    fire.Fire(main)
