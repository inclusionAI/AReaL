# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph multi-process Data Parallel inference

import os
import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThinkMorphDP:
    """
    Multi-process Data Parallel inference for ThinkMorph.

    Distributes samples across multiple GPUs, with each GPU running
    multiple model instances. Each model processes samples sequentially
    to support full interleaved text-image generation (visual thinking).

    Args:
        model_path: Path to ThinkMorph model
        num_gpus: Number of GPUs to use
        max_mem_per_gpu: Maximum memory per GPU
        models_per_gpu: Number of model instances per GPU (default 1)
            With 140GB GPU and ~30GB per model, can fit up to 4 models
        inference_config: Optional dict of inference parameters to override defaults
            Keys: max_think_tokens, text_temperature, cfg_text_scale, cfg_img_scale,
                  cfg_interval, timestep_shift, num_timesteps, cfg_renorm_min,
                  cfg_renorm_type, image_shapes, max_rounds
    """

    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_mem_per_gpu: str = "140GiB",
        models_per_gpu: int = 1,
        inference_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.max_mem_per_gpu = max_mem_per_gpu
        self.models_per_gpu = models_per_gpu
        self.inference_config = inference_config or {}

        total_parallelism = self.num_gpus * self.models_per_gpu
        logger.info(f"ThinkMorphDP initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  GPUs: {self.num_gpus}")
        logger.info(f"  Models per GPU: {self.models_per_gpu}")
        logger.info(f"  Total parallelism: {total_parallelism} samples concurrently")
        if self.inference_config:
            logger.info(f"  Config overrides: {list(self.inference_config.keys())}")

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
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a list of samples.

        Args:
            samples: List of dicts with 'image', 'text', and optional 'id' keys
            think: Enable thinking mode
            understanding_output: Text-only output
            show_progress: Show progress bar
            on_result: Callback function called immediately when each result is ready.
                       Signature: on_result(result_dict) -> None
                       Use this to save results incrementally (e.g., save images immediately).

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
            'models_per_gpu': self.models_per_gpu,
            'inference_config': self.inference_config,
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
                # Got a single sample result - call callback immediately if provided
                if on_result is not None:
                    on_result(result)
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

    Each worker loads multiple model instances and uses threading to process
    samples concurrently. Each model supports full interleaved text-image
    generation (visual thinking).

    Args:
        rank: GPU rank (0 to num_gpus-1)
        model_path: Path to model
        max_mem_per_gpu: Memory limit per GPU
        indexed_samples: List of (original_idx, sample) tuples
        output_queue: Queue to send results back
        kwargs: Inference parameters (including 'models_per_gpu')
    """
    import torch
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Set device for this process
    torch.cuda.set_device(rank)

    # Import ThinkMorphInference for full interleaved generation support
    from .inference import ThinkMorphInference

    models_per_gpu = kwargs.get('models_per_gpu', 1)

    if not indexed_samples:
        output_queue.put(None)
        return

    # Load multiple model instances
    inference_config = kwargs.get('inference_config', {})
    logger.info(f"[GPU {rank}] Loading {models_per_gpu} model instance(s)...")
    inferencers = []
    for i in range(models_per_gpu):
        logger.info(f"[GPU {rank}] Loading model instance {i+1}/{models_per_gpu}...")
        inf = ThinkMorphInference(
            model_path=model_path,
            max_mem_per_gpu=max_mem_per_gpu,
            device=rank,  # Load model only on this GPU
            **inference_config,  # Pass inference config overrides
        )
        inferencers.append(inf)

    logger.info(f"[GPU {rank}] Processing {len(indexed_samples)} samples with {models_per_gpu} model(s)...")

    # Create a lock for each inferencer to ensure thread-safe access
    inferencer_locks = [threading.Lock() for _ in range(models_per_gpu)]

    def process_sample(inferencer_idx: int, original_idx: int, sample: dict):
        """Process a single sample with a specific inferencer."""
        inferencer = inferencers[inferencer_idx]
        lock = inferencer_locks[inferencer_idx]

        with lock:
            try:
                output = inferencer.infer_single(
                    image=sample.get('image'),
                    text=sample.get('text'),
                    think=kwargs.get('think', True),
                    understanding_output=kwargs.get('understanding_output', False),
                )
                return {
                    'original_idx': original_idx,
                    'id': sample.get('id', original_idx),
                    'output': output,
                }
            except Exception as e:
                logger.error(f"[GPU {rank}] Error processing sample {original_idx}: {e}")
                return {
                    'original_idx': original_idx,
                    'id': sample.get('id', original_idx),
                    'output': [f"Error: {str(e)}"],
                }

    if models_per_gpu == 1:
        # Single model: process sequentially (no threading overhead)
        for original_idx, sample in indexed_samples:
            result = process_sample(0, original_idx, sample)
            output_queue.put(result)
    else:
        # Multiple models: use ThreadPoolExecutor for concurrent inference
        # Distribute samples to inferencers in round-robin fashion
        with ThreadPoolExecutor(max_workers=models_per_gpu) as executor:
            futures = []
            for i, (original_idx, sample) in enumerate(indexed_samples):
                inferencer_idx = i % models_per_gpu
                future = executor.submit(process_sample, inferencer_idx, original_idx, sample)
                futures.append(future)

            # Send results as they complete
            for future in as_completed(futures):
                result = future.result()
                output_queue.put(result)

    # Signal that this worker is done
    output_queue.put(None)
    logger.info(f"[GPU {rank}] Done!")
