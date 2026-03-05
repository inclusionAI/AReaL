# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Configuration presets for ThinkMorph inference

"""
Inference configuration presets for ThinkMorph.

These configurations control various aspects of the interleaved generation:
- Text generation (temperature, max tokens)
- Image generation (diffusion steps, CFG scales)
- CFG renormalization strategies
"""

# Default balanced configuration
DEFAULT_CONFIG = {
    "max_think_tokens": 4096,
    "text_temperature": 0.3,
    "cfg_text_scale": 4.0,      # Text conditioning strength (typical: 4.0-8.0)
    "cfg_img_scale": 2.0,        # Image conditioning strength (typical: 1.0-2.0)
    "cfg_interval": [0.0, 1.0],  # Apply CFG throughout entire generation
    "timestep_shift": 3.0,       # Shift for diffusion timestep schedule
    "num_timesteps": 50,         # Number of denoising steps
    "cfg_renorm_min": 0.0,       # Minimum value for CFG renormalization
    "cfg_renorm_type": "text_channel",  # Renormalization strategy
    "image_shapes": (1024, 1024),  # Default image size (H, W)
    "max_rounds": 3,             # Maximum rounds of interleaved generation
}

# Fast inference configuration - fewer denoising steps
FAST_CONFIG = {
    **DEFAULT_CONFIG,
    "num_timesteps": 25,         # Fewer steps = faster
    "cfg_text_scale": 3.0,       # Slightly lower CFG for speed
}

# High quality configuration - more denoising steps
HIGH_QUALITY_CONFIG = {
    **DEFAULT_CONFIG,
    "num_timesteps": 100,        # More steps = better quality
    "cfg_text_scale": 6.0,       # Higher CFG for stronger conditioning
    "cfg_img_scale": 2.5,
}

# Reasoning-heavy configuration - more tokens for thinking
REASONING_CONFIG = {
    **DEFAULT_CONFIG,
    "max_think_tokens": 8192,    # More tokens for complex reasoning
    "text_temperature": 0.2,     # Lower temperature for more focused reasoning
}

# Editing configuration - optimized for image editing tasks
EDITING_CONFIG = {
    **DEFAULT_CONFIG,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_renorm_type": "global",  # Global renorm works better for editing
    "num_timesteps": 50,
}

"""
Configuration Parameter Guide:

1. max_think_tokens (int):
   - Maximum tokens for thinking/reasoning process
   - Typical: 4096, Range: 1024-8192
   - Higher values allow more detailed reasoning but slower

2. text_temperature (float):
   - Sampling temperature for text generation
   - Typical: 0.3, Range: 0.0-1.0
   - Lower = more deterministic, Higher = more creative

3. cfg_text_scale (float):
   - Classifier-Free Guidance scale for text conditioning
   - Typical: 4.0-8.0, Range: 1.0-15.0
   - 1.0 = no guidance, Higher = stronger text adherence
   - Too high can cause artifacts

4. cfg_img_scale (float):
   - Classifier-Free Guidance scale for image conditioning
   - Typical: 1.0-2.0, Range: 1.0-5.0
   - Controls how much generated images preserve input image details

5. cfg_interval (list):
   - [start, end] fractions of generation where CFG is applied
   - Typical: [0.0, 1.0] (apply throughout)
   - [0.4, 1.0] applies CFG only in later steps (saves computation)

6. timestep_shift (float):
   - Shifts the distribution of denoising timesteps
   - Typical: 3.0, Range: 1.0-5.0
   - Higher = more steps at start (affects layout)
   - Lower = more steps at end (improves details)

7. num_timesteps (int):
   - Number of denoising steps for image generation
   - Typical: 50, Range: 20-100
   - More steps = better quality but slower

8. cfg_renorm_min (float):
   - Minimum value for CFG renormalization
   - Typical: 0.0, Range: 0.0-1.0
   - 1.0 disables renormalization
   - Lower values allow stronger renormalization

9. cfg_renorm_type (str):
   - Method for CFG renormalization
   - Options: "global", "channel", "text_channel"
   - "global": Normalize over all tokens and channels (default for T2I)
   - "channel": Normalize per channel for each token
   - "text_channel": Like channel, but only for text condition (good for editing, may blur)
   - If images are blurry, try "global" or decrease cfg_renorm_min

10. image_shapes (tuple):
    - Default generation size (H, W)
    - Typical: (1024, 1024)
    - Must be divisible by 16 (VAE downsample factor)

11. max_rounds (int):
    - Maximum rounds of text-image interleaved generation
    - Typical: 3, Range: 1-10
    - Each round can generate one image
"""
