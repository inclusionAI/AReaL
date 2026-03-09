# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# ThinkMorph inference wrapper using manual model loading (like original ThinkMorph)

import os
import sys
import json
import logging
from copy import deepcopy
from typing import List, Union, Dict, Any, Optional

from PIL import Image
import torch
from tqdm import tqdm
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ThinkMorph Path Setup
# ============================================================================

def _find_thinkmorph_path():
    """Find ThinkMorph source path from environment or common locations."""
    # Priority 1: Environment variable
    if "THINKMORPH_PATH" in os.environ:
        path = os.environ["THINKMORPH_PATH"]
        if os.path.exists(path):
            return path

    # Priority 2: Common locations
    common_paths = [
        "/storage/openpsi/users/lichangye.lcy/antoinegg1/ThinkMorph",
        "/ThinkMorph",
        os.path.expanduser("~/ThinkMorph"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ThinkMorph"),
    ]

    for path in common_paths:
        path = os.path.abspath(path)
        if os.path.exists(path) and os.path.exists(os.path.join(path, "modeling", "bagel")):
            return path

    return None

THINKMORPH_PATH = _find_thinkmorph_path()
if THINKMORPH_PATH is None:
    raise ImportError(
        "Could not find ThinkMorph source directory. Please set THINKMORPH_PATH environment variable:\n"
        "  export THINKMORPH_PATH=/path/to/ThinkMorph\n"
        "The directory should contain 'modeling/bagel' and 'data' subdirectories."
    )

logger.info(f"Using ThinkMorph from: {THINKMORPH_PATH}")
if THINKMORPH_PATH not in sys.path:
    sys.path.insert(0, THINKMORPH_PATH)

# Import from ThinkMorph
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae


# ============================================================================
# System Prompts
# ============================================================================

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

GEN_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


# ============================================================================
# Main Inference Class (Manual Loading)
# ============================================================================

class ThinkMorphInference:
    """
    ThinkMorph inference using manual model loading.

    This class loads the model components separately (llm_config, vit_config, vae)
    like the original ThinkMorph implementation, avoiding the config.json parsing issue.
    """

    def __init__(
        self,
        model_path: str = "ThinkMorph/ThinkMorph-7B",
        max_mem_per_gpu: str = "40GiB",
        # Inference configuration
        max_think_tokens: int = 4096,
        do_sample: bool = True,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: Optional[List[float]] = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        image_shapes: tuple = (1024, 1024),
        max_rounds: int = 3,
    ):
        """
        Initialize ThinkMorphInference with manual component loading.

        Args:
            model_path: Path to ThinkMorph model directory
            max_mem_per_gpu: Maximum memory per GPU (e.g., "40GiB", "80GiB")
            max_think_tokens: Maximum tokens for thinking/reasoning
            do_sample: Whether to use sampling for text generation
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
        """
        logger.info(f"[{self._timestamp()}] INFO inference.py: Initializing ThinkMorphInference with model: {model_path}")

        self.model_path = model_path

        # Store inference configuration
        if cfg_interval is None:
            cfg_interval = [0.0, 1.0]

        self.inference_config = {
            "max_think_token_n": max_think_tokens,
            "do_sample": do_sample,
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

        # Load model components manually
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading model components manually...")
        self._load_model_components(model_path, max_mem_per_gpu)

        logger.info(f"[{self._timestamp()}] INFO inference.py: Initialization complete!")

    def _timestamp(self):
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _load_model_components(self, model_path: str, max_mem_per_gpu: str):
        """
        Load model components separately, like original ThinkMorph.

        This avoids the config.json parsing issue by loading:
        - llm_config.json
        - vit_config.json
        - ae.safetensors (VAE)
        - model.safetensors (weights)
        """
        # 1. Load LLM config
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading LLM config from llm_config.json...")
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # 2. Load ViT config
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading ViT config from vit_config.json...")
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # 3. Load VAE
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading VAE from ae.safetensors...")
        self.vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # 4. Create Bagel config
        logger.info(f"[{self._timestamp()}] INFO inference.py: Creating BagelConfig...")
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

        # 5. Initialize model with empty weights
        logger.info(f"[{self._timestamp()}] INFO inference.py: Initializing model structure...")
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # 6. Load tokenizer
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading tokenizer...")
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # 7. Setup image transforms
        logger.info(f"[{self._timestamp()}] INFO inference.py: Setting up image transforms...")
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

        # 8. Setup device map and load weights
        logger.info(f"[{self._timestamp()}] INFO inference.py: Setting up device map (max_mem_per_gpu={max_mem_per_gpu})...")
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        logger.info(f"[{self._timestamp()}] INFO inference.py: Device map: {device_map}")

        # Ensure certain modules are on same device
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # 9. Load checkpoint
        logger.info(f"[{self._timestamp()}] INFO inference.py: Loading model weights from model.safetensors...")
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
        logger.info(f"[{self._timestamp()}] INFO inference.py: Model loaded successfully!")

    def _init_gen_context(self):
        """Initialize generation context for interleaved inference."""
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def _update_context_text(self, text, gen_context):
        """Update generation context with text."""
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
        """Update generation context with image."""
        assert vae or vit
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
    def _gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        """Generate image using diffusion process."""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        # Text CFG
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        # Image CFG
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self._decode_image(unpacked_latent[0], image_shape)
        return image

    def _decode_image(self, latent, image_shape):
        """Decode latent to image using VAE."""
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def _gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        """Generate text using the language model."""
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
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think: bool = False,
        understanding_output: bool = False,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Optional[List[float]] = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        image_shapes: tuple = (1024, 1024),
        enable_taylorseer: bool = False,
        max_rounds: int = 3,
    ) -> List[Union[str, Image.Image]]:
        """
        Run interleaved text-image inference.

        Args:
            input_lists: List of input items (strings or PIL Images)
            think: Whether to enable thinking mode
            understanding_output: If True, only output text
            max_think_token_n: Maximum tokens for generation
            do_sample: Whether to use sampling
            text_temperature: Temperature for text generation
            cfg_text_scale: CFG scale for text
            cfg_img_scale: CFG scale for images
            cfg_interval: CFG interval [start, end]
            timestep_shift: Timestep shift for diffusion
            num_timesteps: Number of denoising steps
            cfg_renorm_min: CFG renorm minimum
            cfg_renorm_type: CFG renorm type
            image_shapes: Default image shape (H, W)
            enable_taylorseer: Enable TaylorSeer optimization
            max_rounds: Maximum generation rounds

        Returns:
            List of generated outputs (text strings and/or PIL Images)
        """
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        output_list = []
        gen_context = self._init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self._update_context_text(system_prompt, gen_context)
                cfg_img_context = self._update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self._update_context_text(input_term, gen_context)
                    cfg_img_context = self._update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self._update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self._gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                rounds = 0
                while rounds < max_rounds:
                    gen_text = self._gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    output_list.append(gen_text)
                    gen_context = self._update_context_text(gen_text, gen_context)

                    if "<image_start>" in gen_text:
                        img = self._gen_image(
                            image_shapes,
                            gen_context,
                            cfg_text_precontext=cfg_text_context,
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            timestep_shift=timestep_shift,
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                        )
                        output_list.append(img)

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        gen_context = self._update_context_image(img_input, gen_context, vae=not understanding_output)
                        rounds += 1
                    else:
                        break

        return output_list

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
            think: Whether to enable thinking mode
            understanding_output: If True, only output text

        Returns:
            List of generated outputs
        """
        if image is None and text is None:
            raise ValueError("At least one of image or text must be provided")

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        return self.interleave_inference(
            input_list,
            think=think,
            understanding_output=understanding_output,
            **self.inference_config
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

        checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        processed_ids = self._load_checkpoint(checkpoint_file) if resume else set()

        logger.info(f"Processing dataset with {len(dataset)} samples...")
        logger.info(f"Output directory: {output_dir}")

        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            sample_id = sample.get(id_field, str(idx))

            if sample_id in processed_ids:
                continue

            try:
                image_data = sample.get(image_field)
                image = pil_img2rgb(image_data) if image_data is not None else None
                text = sample.get(text_field, "")

                outputs = self.infer_single(image, text, think=think, understanding_output=understanding_output)

                result = self._save_result(sample_id, outputs, output_dir)
                results.append(result)

                processed_ids.add(sample_id)
                self._save_checkpoint(checkpoint_file, processed_ids)

            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
                continue

        logger.info(f"Processing complete! Processed {len(results)} samples.")
        return {"processed": len(results), "results": results}

    def _save_result(self, sample_id: str, outputs: List, output_dir: str) -> Dict[str, Any]:
        """Save inference results to disk."""
        result = {"sample_id": sample_id, "outputs": []}

        for idx, output in enumerate(outputs):
            if isinstance(output, str):
                result["outputs"].append({"type": "text", "content": output})
            elif isinstance(output, Image.Image):
                image_filename = f"{sample_id}_image_{idx}.png"
                image_path = os.path.join(output_dir, image_filename)
                output.save(image_path)
                result["outputs"].append({"type": "image", "path": image_filename})

        result_path = os.path.join(output_dir, f"{sample_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _load_checkpoint(self, checkpoint_file: str) -> set:
        """Load processed sample IDs from checkpoint."""
        if not os.path.exists(checkpoint_file):
            return set()
        try:
            with open(checkpoint_file, 'r') as f:
                return set(json.load(f).get("processed_ids", []))
        except:
            return set()

    def _save_checkpoint(self, checkpoint_file: str, processed_ids: set):
        """Save processed sample IDs to checkpoint."""
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        input_list: Optional[List] = None,
        **kwargs
    ) -> List[Union[str, Image.Image]]:
        """
        Callable interface for inference.

        Args:
            image: Input PIL Image
            text: Input text prompt
            input_list: List of inputs (overrides image/text)
            **kwargs: Additional inference parameters

        Returns:
            List of generated outputs
        """
        if input_list is None:
            if image is None and text is None:
                logger.warning('Please provide at least one input: either an image or text.')
                return []

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)

        # Merge with default config
        config = {**self.inference_config, **kwargs}
        return self.interleave_inference(input_list, **config)


# Alias for backward compatibility
VLLMInterleavedInference = ThinkMorphInference
