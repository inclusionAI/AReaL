# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph batch inference with Data Parallel (DP) support

import os
import json
import logging
from copy import deepcopy
from typing import List, Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from local thinkmorph_src
from .thinkmorph_src.transforms import ImageTransform
from .thinkmorph_src.data_utils import pil_img2rgb, add_special_tokens
from .thinkmorph_src.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel, NaiveCache
)
from .thinkmorph_src.qwen2 import Qwen2Tokenizer
from .thinkmorph_src.autoencoder import load_ae


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""
    batch_size: int = 4
    max_think_tokens: int = 4096
    do_sample: bool = True
    text_temperature: float = 0.3
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 2.0
    cfg_interval: Tuple[float, float] = (0.0, 1.0)
    timestep_shift: float = 3.0
    num_timesteps: int = 50
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "text_channel"
    num_workers: int = 4


class ThinkMorphBatchInference:
    """
    ThinkMorph batch inference with Data Parallel support.

    Supports:
    - Batch text generation for multiple samples
    - Multi-GPU inference with DataParallel
    - Efficient memory usage
    """

    def __init__(
        self,
        model_path: str,
        max_mem_per_gpu: str = "40GiB",
        use_dp: bool = True,  # Enable DataParallel
        config: BatchInferenceConfig = None,
    ):
        """
        Initialize batch inference.

        Args:
            model_path: Path to ThinkMorph model
            max_mem_per_gpu: Maximum GPU memory per device
            use_dp: Whether to use DataParallel for multi-GPU
            config: Batch inference configuration
        """
        logger.info(f"Initializing ThinkMorphBatchInference with model: {model_path}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")

        self.model_path = model_path
        self.use_dp = use_dp and torch.cuda.device_count() > 1
        self.config = config or BatchInferenceConfig()

        # Load model components
        self._load_model_components(model_path, max_mem_per_gpu)

        logger.info("Initialization complete!")

    def _load_model_components(self, model_path: str, max_mem_per_gpu: str):
        """Load model components with optional DP wrapping."""

        # 1. Load configs
        logger.info("Loading configs...")
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # 2. Load VAE
        logger.info("Loading VAE...")
        self.vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
        self.vae_model = self.vae_model.cuda().to(torch.bfloat16).eval()

        # 3. Create Bagel config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        # 4. Initialize model
        logger.info("Initializing model structure...")
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # 5. Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # 6. Image transforms
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

        # 7. Load weights with device map
        logger.info("Loading model weights...")
        num_gpus = torch.cuda.device_count()

        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(num_gpus)},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        # Ensure modules on same device
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device

        self.model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "model.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        self.model = self.model.eval()

        logger.info(f"Model loaded across {num_gpus} GPU(s)")

    def _init_gen_context(self):
        """Initialize generation context."""
        return {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    @torch.no_grad()
    def _update_context_text(self, text, gen_context):
        """Update context with text."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def _update_context_image(self, image, gen_context, vae=True, vit=True):
        """Update context with image."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def _gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        """Generate text."""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def infer_single(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        think: bool = True,
        understanding_output: bool = True,
    ) -> List[Union[str, Image.Image]]:
        """Single sample inference."""
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided")

        output_list = []
        gen_context = self._init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Add system prompt if thinking
            if think:
                gen_context = self._update_context_text(VLM_THINK_SYSTEM_PROMPT, gen_context)

            # Process inputs
            if image is not None:
                image_input = self.vae_transform.resize_transform(pil_img2rgb(image))
                gen_context = self._update_context_image(image_input, gen_context, vae=not understanding_output)

            if text is not None:
                gen_context = self._update_context_text(text, gen_context)

            # Generate text
            gen_text = self._gen_text(
                gen_context,
                do_sample=self.config.do_sample,
                temperature=self.config.text_temperature,
                max_length=self.config.max_think_tokens
            )
            output_list.append(gen_text)

        return output_list

    @torch.no_grad()
    def infer_batch(
        self,
        samples: List[Dict[str, Any]],
        think: bool = True,
        understanding_output: bool = True,
        show_progress: bool = True,
    ) -> List[List[Union[str, Image.Image]]]:
        """
        Batch inference on multiple samples.

        Args:
            samples: List of dicts with 'image' and/or 'text' keys
            think: Enable thinking mode
            understanding_output: Text-only output
            show_progress: Show progress bar

        Returns:
            List of outputs for each sample
        """
        results = []

        iterator = tqdm(samples, desc="Batch inference") if show_progress else samples

        for sample in iterator:
            image = sample.get('image')
            text = sample.get('text')

            try:
                outputs = self.infer_single(
                    image=image,
                    text=text,
                    think=think,
                    understanding_output=understanding_output
                )
                results.append(outputs)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                results.append([f"Error: {str(e)}"])

        return results

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """Callable interface for single sample."""
        return self.infer_single(image=image, text=text, **kwargs)

    def _init_batch_gen_context(self, batch_size: int):
        """Initialize generation context for batch processing."""
        return {
            'kv_lens': [0] * batch_size,
            'ropes': [0] * batch_size,
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

    @torch.no_grad()
    def _update_batch_context_text(self, texts: List[str], gen_context):
        """Update context with batch of texts."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=texts,
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def _update_batch_context_images(self, images: List, gen_context, vae: bool = True, vit: bool = True):
        """
        Update context with batch of images.

        All images are processed together in a single batch call.
        Images can have different sizes - they will be packed together.
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)

        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=images,
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def _gen_batch_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        """Generate text for batch."""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )

        # Decode each sample in batch
        batch_size = len(kv_lens)
        outputs = []
        for i in range(batch_size):
            output = self.tokenizer.decode(unpacked_latent[:, i])
            # Parse output
            if '<|im_end|>' in output:
                output = output.split('<|im_end|>')[0]
            if '<|im_start|>' in output:
                output = output.split('<|im_start|>')[-1]
            outputs.append(output)

        return outputs

    @torch.no_grad()
    def infer_batch_parallel(
        self,
        samples: List[Dict[str, Any]],
        think: bool = True,
        understanding_output: bool = True,
        show_progress: bool = True,
    ) -> List[List[Union[str, Image.Image]]]:
        """
        True batch inference: process multiple samples simultaneously.

        This method processes multiple samples in a single forward pass,
        providing better GPU utilization than sequential processing.

        Note: For batches with images, ALL samples must have images.
        Mixed batches (some with images, some without) fall back to sequential.

        Args:
            samples: List of dicts with 'image' and/or 'text' keys
            think: Enable thinking mode
            understanding_output: Text-only output
            show_progress: Show progress bar

        Returns:
            List of outputs for each sample
        """
        if not samples:
            return []

        batch_size = len(samples)

        # Check image configuration
        has_images = [sample.get('image') is not None for sample in samples]
        all_have_images = all(has_images)
        none_have_images = not any(has_images)

        # Mixed batch (some with images, some without) - fall back to sequential
        if not all_have_images and not none_have_images:
            logger.info("Mixed batch (some with images, some without), using sequential processing")
            return self.infer_batch(
                samples,
                think=think,
                understanding_output=understanding_output,
                show_progress=show_progress,
            )

        results = [[] for _ in range(batch_size)]

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Initialize batch context
            gen_context = self._init_batch_gen_context(batch_size)

            # Add system prompt if thinking
            if think:
                system_prompts = [VLM_THINK_SYSTEM_PROMPT] * batch_size
                gen_context = self._update_batch_context_text(system_prompts, gen_context)

            # Batch process images if ALL samples have images
            if all_have_images:
                images = []
                for sample in samples:
                    image = sample.get('image')
                    image_input = self.vae_transform.resize_transform(pil_img2rgb(image))
                    images.append(image_input)

                gen_context = self._update_batch_context_images(
                    images, gen_context, vae=not understanding_output
                )

            # Batch process text prompts
            texts = [sample.get('text', '') for sample in samples]
            # Filter empty texts
            non_empty_texts = [t if t else ' ' for t in texts]  # Use space for empty
            gen_context = self._update_batch_context_text(non_empty_texts, gen_context)

            # Generate text in batch
            gen_texts = self._gen_batch_text(
                gen_context,
                do_sample=self.config.do_sample,
                temperature=self.config.text_temperature,
                max_length=self.config.max_think_tokens
            )

            # Collect results
            for i, text in enumerate(gen_texts):
                results[i].append(text)

        return results


def run_batch_evaluation(
    model_path: str,
    dataset,
    output_dir: str,
    batch_size: int = 4,
    max_mem_per_gpu: str = "40GiB",
    image_field: str = "image",
    text_field: str = "text",
    max_samples: Optional[int] = None,
):
    """
    Run batch evaluation on a dataset.

    Args:
        model_path: Path to ThinkMorph model
        dataset: HuggingFace dataset
        output_dir: Output directory
        batch_size: Batch size for inference
        max_mem_per_gpu: GPU memory limit
        image_field: Image field name in dataset
        text_field: Text field name in dataset
        max_samples: Maximum samples to process
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize inferencer
    config = BatchInferenceConfig(batch_size=batch_size)
    inferencer = ThinkMorphBatchInference(
        model_path=model_path,
        max_mem_per_gpu=max_mem_per_gpu,
        config=config,
    )

    # Prepare samples
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples = []
    for item in dataset:
        samples.append({
            'image': item.get(image_field),
            'text': item.get(text_field),
        })

    # Run batch inference
    logger.info(f"Running batch inference on {len(samples)} samples...")
    results = inferencer.infer_batch(samples, understanding_output=True)

    # Save results
    output_results = []
    for i, (sample, outputs) in enumerate(zip(samples, results)):
        output_results.append({
            'id': i,
            'text': sample.get('text', ''),
            'response': outputs[0] if outputs else '',
        })

    results_path = os.path.join(output_dir, "batch_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {results_path}")
    return output_results
