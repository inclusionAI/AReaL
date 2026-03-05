# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Simplified inference wrapper for ThinkMorph using vLLM

import os
import json
import logging
from typing import List, Union, Dict, Any, Optional
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Classes and Functions (extracted from ThinkMorph)
# ============================================================================

class MaxLongEdgeMinShortEdgeResize(nn.Module):
    """Resize image to fit within size constraints while maintaining aspect ratio."""

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    """Image transformation pipeline for ThinkMorph."""

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


def pil_img2rgb(image):
    """Convert PIL image to RGB format."""
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    return image


# System prompts for thinking mode
VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


# ============================================================================
# Main Inference Class
# ============================================================================

class VLLMInterleavedInference:
    """
    Simplified ThinkMorph inference using vLLM.

    This class provides a clean API for interleaved image-text generation
    using ThinkMorph model with vLLM backend for efficient inference.
    """

    def __init__(
        self,
        model_path: str = "ThinkMorph/ThinkMorph-7B",
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        # Inference configuration
        max_think_tokens: int = 4096,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        image_shapes: tuple = (1024, 1024),
        max_rounds: int = 3,
        **vllm_kwargs
    ):
        """
        Initialize the VLLMInterleavedInference.

        Args:
            model_path: Path or HF model ID for ThinkMorph model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for model weights
            max_model_len: Maximum context length
            max_think_tokens: Maximum tokens for thinking/reasoning
            text_temperature: Sampling temperature for text generation
            cfg_text_scale: Classifier-free guidance scale for text conditioning
            cfg_img_scale: Classifier-free guidance scale for image conditioning
            cfg_interval: Interval for applying CFG [start, end]
            timestep_shift: Shift for diffusion timestep schedule
            num_timesteps: Number of denoising steps for image generation
            cfg_renorm_min: Minimum value for CFG renormalization
            cfg_renorm_type: Type of CFG renormalization (global/channel/text_channel)
            image_shapes: Default image generation size (H, W)
            max_rounds: Maximum rounds of interleaved generation
            **vllm_kwargs: Additional arguments for vLLM
        """
        logger.info(f"Initializing VLLMInterleavedInference with model: {model_path}")

        # Load model with vLLM
        logger.info("Loading model with vLLM...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,  # Required for Bagel custom code
            **vllm_kwargs
        )

        # Get the underlying Bagel model for image generation
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Initialize image transforms
        self.vae_transform = ImageTransform(1024, 512, 16)  # For generation
        self.vit_transform = ImageTransform(980, 224, 14)   # For understanding

        # Store inference configuration
        if cfg_interval is None:
            cfg_interval = [0.0, 1.0]

        self.config = {
            "max_think_tokens": max_think_tokens,
            "text_temperature": text_temperature,
            "cfg_text_scale": cfg_text_scale,
            "cfg_img_scale": cfg_img_scale,
            "cfg_interval": cfg_interval,
            "timestep_shift": timestep_shift,
            "num_timesteps": num_timesteps,
            "cfg_renorm_min": cfg_renorm_min,
            "cfg_renorm_type": cfg_renorm_type,
            "image_shapes": image_shapes,
            "max_rounds": max_rounds,
        }

        logger.info("Initialization complete!")

    def infer_single(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        think: bool = True,
        understanding_output: bool = False,
    ) -> List[Union[str, Image.Image]]:
        """
        Run inference on a single image-text pair.

        Args:
            image: Input PIL Image (optional)
            text: Input text prompt (optional)
            think: Whether to enable thinking mode with step-by-step reasoning
            understanding_output: If True, only output text; if False, generate images

        Returns:
            List of alternating text and images
        """
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided")

        # Build input list
        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        # Run interleaved inference
        return self._interleave_inference(
            input_list,
            think=think,
            understanding_output=understanding_output
        )

    def infer_dataset(
        self,
        dataset,
        output_dir: str,
        image_field: str = "image",
        text_field: str = "text",
        id_field: str = "id",
        think: bool = True,
        understanding_output: bool = False,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a HuggingFace dataset.

        Args:
            dataset: HuggingFace Dataset object
            output_dir: Directory to save results
            image_field: Field name for images in dataset
            text_field: Field name for text in dataset
            id_field: Field name for sample IDs
            think: Whether to enable thinking mode
            understanding_output: Whether to only output text
            resume: Whether to resume from checkpoint

        Returns:
            Dictionary with statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        # Checkpoint management
        checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        processed_ids = self._load_checkpoint(checkpoint_file) if resume else set()

        logger.info(f"Processing dataset with {len(dataset)} samples...")
        logger.info(f"Output directory: {output_dir}")

        # Process each sample
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            sample_id = sample.get(id_field, str(idx))

            # Skip if already processed
            if sample_id in processed_ids:
                logger.debug(f"Skipping already processed sample: {sample_id}")
                continue

            try:
                # Parse input
                image_data = sample.get(image_field)
                image = self._parse_image(image_data) if image_data is not None else None
                text = sample.get(text_field, "")

                # Run inference
                outputs = self.infer_single(image, text, think=think, understanding_output=understanding_output)

                # Save result
                result = self._save_result(sample_id, outputs, output_dir)
                results.append(result)

                # Update checkpoint
                processed_ids.add(sample_id)
                self._save_checkpoint(checkpoint_file, processed_ids)

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
                continue

        logger.info(f"Processing complete! Processed {len(results)} samples.")
        return {"processed": len(results), "results": results}

    def _interleave_inference(
        self,
        input_list: List[Union[str, Image.Image]],
        think: bool = True,
        understanding_output: bool = False,
    ) -> List[Union[str, Image.Image]]:
        """
        Core interleaved inference logic.

        This method handles the alternating text-image generation process:
        1. Build multimodal prompt with images and text
        2. Generate text response (using vLLM)
        3. If text contains <image_start>, generate image (using Bagel's VAE)
        4. Add generated image to context and continue
        5. Repeat until max_rounds or no more image generation
        """
        output_list = []

        # Build initial prompt
        prompt = self._build_prompt(input_list, think=think)

        # Track image shapes for generation
        image_shapes = self.config["image_shapes"]
        if len(input_list) > 0 and isinstance(input_list[0], Image.Image):
            # Use input image size as reference
            image_shapes = input_list[0].size[::-1]  # (H, W)

        # Generation loop
        max_rounds = self.config["max_rounds"]
        rounds = 0

        with torch.no_grad():
            while rounds < max_rounds:
                # Generate text using vLLM
                sampling_params = SamplingParams(
                    temperature=self.config["text_temperature"],
                    max_tokens=self.config["max_think_tokens"],
                    stop=["<|im_end|>"],
                )

                vllm_outputs = self.llm.generate([prompt], sampling_params)
                generated_text = vllm_outputs[0].outputs[0].text

                # Clean up text
                generated_text = generated_text.split('<|im_end|>')[0]
                output_list.append(generated_text)

                # Check if we need to generate image
                if not understanding_output and "<image_start>" in generated_text:
                    logger.info(f"Generating image for round {rounds + 1}")

                    # TODO: Implement image generation using Bagel's generate_image method
                    # This requires accessing the model's VAE and implementing the diffusion process
                    # For now, we'll add a placeholder
                    logger.warning("Image generation not yet implemented - placeholder returned")

                    # Placeholder: create a blank image
                    placeholder_image = Image.new('RGB', image_shapes[::-1], color=(128, 128, 128))
                    output_list.append(placeholder_image)

                    # Update prompt with generated image
                    # prompt = self._update_prompt_with_image(prompt, placeholder_image)
                    rounds += 1
                else:
                    # No more image generation needed
                    break

        return output_list

    def _build_prompt(
        self,
        input_list: List[Union[str, Image.Image]],
        think: bool = True
    ) -> str:
        """
        Build a text prompt from input list.

        For now, this is a simplified version that only handles text.
        Full implementation would need to handle image tokens properly.
        """
        prompt_parts = []

        # Add system prompt if thinking is enabled
        if think:
            prompt_parts.append(VLM_THINK_SYSTEM_PROMPT)

        # Process input list
        for item in input_list:
            if isinstance(item, str):
                prompt_parts.append(item)
            elif isinstance(item, Image.Image):
                # TODO: Properly encode image as tokens
                prompt_parts.append("<image>")

        return "\n".join(prompt_parts)

    def _parse_image(self, image_data: Union[str, Image.Image, Any]) -> Optional[Image.Image]:
        """Parse image from various formats."""
        if image_data is None:
            return None

        if isinstance(image_data, Image.Image):
            return pil_img2rgb(image_data)

        if isinstance(image_data, str):
            # Assume it's a file path
            return pil_img2rgb(Image.open(image_data))

        # Try to convert from bytes or other formats
        try:
            from io import BytesIO
            if isinstance(image_data, bytes):
                return pil_img2rgb(Image.open(BytesIO(image_data)))
        except:
            pass

        logger.warning(f"Could not parse image data of type {type(image_data)}")
        return None

    def _save_result(
        self,
        sample_id: str,
        outputs: List[Union[str, Image.Image]],
        output_dir: str
    ) -> Dict[str, Any]:
        """Save inference results to disk."""
        result = {
            "sample_id": sample_id,
            "outputs": []
        }

        for idx, output in enumerate(outputs):
            if isinstance(output, str):
                # Save text
                result["outputs"].append({
                    "type": "text",
                    "content": output
                })
            elif isinstance(output, Image.Image):
                # Save image
                image_filename = f"{sample_id}_image_{idx}.png"
                image_path = os.path.join(output_dir, image_filename)
                output.save(image_path)
                result["outputs"].append({
                    "type": "image",
                    "path": image_filename
                })

        # Save result metadata as JSON
        result_filename = f"{sample_id}_result.json"
        result_path = os.path.join(output_dir, result_filename)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _load_checkpoint(self, checkpoint_file: str) -> set:
        """Load processed sample IDs from checkpoint."""
        if not os.path.exists(checkpoint_file):
            return set()

        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, checkpoint_file: str, processed_ids: set):
        """Save processed sample IDs to checkpoint."""
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")
