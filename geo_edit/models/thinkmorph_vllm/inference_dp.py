# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph multi-process Data Parallel inference

import os
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThinkMorphDP:
    """
    Multi-process Data Parallel inference for ThinkMorph.

    Distributes samples across multiple GPUs, with each GPU running
    an independent model instance. Each GPU processes samples sequentially
    to support full interleaved text-image generation (visual thinking).

    Args:
        model_path: Path to ThinkMorph model
        num_gpus: Number of GPUs to use
        max_mem_per_gpu: Maximum memory per GPU
    """

    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_mem_per_gpu: str = "140GiB",
    ):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.max_mem_per_gpu = max_mem_per_gpu

        logger.info(f"ThinkMorphDP initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  GPUs: {self.num_gpus}")
        logger.info(f"  Parallelism: {self.num_gpus} samples concurrently")

    def infer_dataset(
        self,
        dataset,
        image_field: str = "image",
        text_field: str = "text",
        id_field: str = "id",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a HuggingFace Dataset.

        Args:
            dataset: HuggingFace Dataset object
            image_field: Field name for images
            text_field: Field name for text
            id_field: Field name for sample IDs
            **kwargs: Additional inference parameters

        Returns:
            List of result dicts with 'original_idx', 'id', and 'output' keys
        """
        # Convert dataset to samples list
        samples = []
        for idx, item in enumerate(dataset):
            samples.append({
                'image': item.get(image_field),
                'text': item.get(text_field),
                'id': item.get(id_field, idx),
            })

        return self.infer_samples(samples, **kwargs)

    def infer_samples(
        self,
        samples: List[Dict[str, Any]],
        think: bool = True,
        understanding_output: bool = True,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a list of samples.

        Args:
            samples: List of dicts with 'image', 'text', and optional 'id' keys
            think: Enable thinking mode
            understanding_output: Text-only output
            show_progress: Show progress bar

        Returns:
            List of result dicts sorted by original index
        """
        if not samples:
            return []

        logger.info(f"Starting DP inference on {len(samples)} samples with {self.num_gpus} GPUs")

        # Distribute samples to GPUs (round-robin)
        chunks = [[] for _ in range(self.num_gpus)]
        for i, sample in enumerate(samples):
            chunks[i % self.num_gpus].append((i, sample))

        # Prepare kwargs for workers
        worker_kwargs = {
            'think': think,
            'understanding_output': understanding_output,
            'show_progress': show_progress and True,  # Only rank 0 shows progress
        }

        # Start multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        ctx = mp.get_context('spawn')
        output_queue = ctx.Queue()

        # Launch worker processes
        processes = []
        for rank in range(self.num_gpus):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    rank,
                    self.model_path,
                    self.max_mem_per_gpu,
                    chunks[rank],
                    output_queue,
                    worker_kwargs,
                )
            )
            p.start()
            processes.append(p)

        # Collect results from queue - each sample updates progress immediately
        # Workers send each result individually, then None when done
        from tqdm import tqdm

        all_results = []
        num_workers_with_samples = sum(1 for c in chunks if c)
        workers_done = 0
        total_samples = len(samples)

        # Show single progress bar for total samples
        pbar = tqdm(total=total_samples, desc="DP Inference", disable=not show_progress)
        while workers_done < num_workers_with_samples:
            result = output_queue.get()  # Blocks until data available
            if result is None:
                # Worker finished
                workers_done += 1
            else:
                # Got a single sample result
                all_results.append(result)
                pbar.update(1)
        pbar.close()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Sort by original index
        all_results.sort(key=lambda x: x['original_idx'])

        logger.info(f"DP inference complete: {len(all_results)} results")
        return all_results


def _worker_fn(
    rank: int,
    model_path: str,
    max_mem_per_gpu: str,
    indexed_samples: List,
    output_queue: mp.Queue,
    kwargs: Dict[str, Any],
):
    """
    Worker function for each GPU process.

    Each worker processes samples sequentially using ThinkMorphInference,
    which supports full interleaved text-image generation (visual thinking).

    Args:
        rank: GPU rank (0 to num_gpus-1)
        model_path: Path to model
        max_mem_per_gpu: Memory limit per GPU
        indexed_samples: List of (original_idx, sample) tuples
        output_queue: Queue to send results back
        kwargs: Inference parameters
    """
    import torch
    from tqdm import tqdm

    # Set device for this process
    torch.cuda.set_device(rank)

    # Import ThinkMorphInference for full interleaved generation support
    from .inference import ThinkMorphInference

    logger.info(f"[GPU {rank}] Loading model...")

    # Initialize inferencer with specific device
    inferencer = ThinkMorphInference(
        model_path=model_path,
        max_mem_per_gpu=max_mem_per_gpu,
        device=rank,  # Load model only on this GPU
    )

    logger.info(f"[GPU {rank}] Processing {len(indexed_samples)} samples...")

    # Process samples sequentially, send each result immediately for real-time progress
    for original_idx, sample in indexed_samples:
        try:
            # Run single-sample inference with full interleaved generation
            output = inferencer.infer_single(
                image=sample.get('image'),
                text=sample.get('text'),
                think=kwargs.get('think', True),
                understanding_output=kwargs.get('understanding_output', False),
            )

            # Send result immediately (for real-time progress bar)
            output_queue.put({
                'original_idx': original_idx,
                'id': sample.get('id', original_idx),
                'output': output,
            })

        except Exception as e:
            logger.error(f"[GPU {rank}] Error processing sample {original_idx}: {e}")
            output_queue.put({
                'original_idx': original_idx,
                'id': sample.get('id', original_idx),
                'output': [f"Error: {str(e)}"],
            })

    # Signal that this worker is done
    output_queue.put(None)
    logger.info(f"[GPU {rank}] Done!")
