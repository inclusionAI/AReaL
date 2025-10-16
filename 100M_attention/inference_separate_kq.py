"""Inference utilities with separate positional encoding for keys and queries.

This script demonstrates how to run text generation with Hugging Face
transformers while providing different RoPE encodings for keys and queries.
This is achieved by creating a custom rotary embedding module that returns
separate cos/sin values for keys and queries, and patching the attention
mechanism to use them appropriately.

Example usage (requires internet access for pretrained weights):

	python inference_separate_kq.py --prompt "Once upon a time" --max-new-tokens 64

To run an offline self-test:

	python inference_separate_kq.py --self-test
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import (
	AutoConfig,
	AutoModelForCausalLM,
	AutoTokenizer,
	GPT2Config,
	GPT2LMHeadModel,
	Qwen2Config,
	Qwen2ForCausalLM,
	PreTrainedModel,
	PreTrainedTokenizerBase,
	set_seed,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
	"""Container holding generation outputs."""

	prompt: str
	generated_text: str
	full_text: str


class CustomRotaryPositionalEncodingSeparateKQ(nn.Module):
	"""Custom rotary positional embedding with separate encodings for keys and queries.
	
	This module returns four tensors instead of two:
	- cos_k, sin_k: for keys
	- cos_q, sin_q: for queries
	"""

	def __init__(
		self,
		inv_freq: torch.Tensor,
		max_length: int,
		attention_scaling: float = 1.0,
		dropout: float = 0.0,
		learned_scaling: bool = True,
	) -> None:
		super().__init__()
		if inv_freq.ndim != 1:
			raise ValueError("`inv_freq` is expected to be a 1-D tensor.")

		head_dim = inv_freq.numel() * 2
		if head_dim % 2 != 0:
			raise ValueError("Rotary head dimension must be even.")

		self.head_dim = head_dim
		self.max_seq_len_cached = max_length
		self.original_max_seq_len = max_length
		self.attention_scaling = attention_scaling
		self.dropout = nn.Dropout(dropout)
		self.learned_scaling = learned_scaling

		inv_freq = inv_freq.detach().to(dtype=torch.float32)
		self.register_buffer("inv_freq", inv_freq, persistent=False)

		if learned_scaling:
			self.alpha = nn.Parameter(torch.ones_like(inv_freq))
			self.beta = nn.Parameter(torch.zeros_like(inv_freq))
		else:
			self.register_buffer("alpha", torch.ones_like(inv_freq), persistent=False)
			self.register_buffer("beta", torch.zeros_like(inv_freq), persistent=False)

	def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Forward pass returning separate cos/sin for keys and queries.
		
		Returns:
			Tuple of (cos_k, sin_k, cos_q, sin_q)
		"""
		if position_ids.dtype != torch.long:
			position_ids = position_ids.long()

		batch, seq_len = position_ids.shape
		inv_freq = self.inv_freq.to(device=x.device)
		position = position_ids.to(device=x.device, dtype=torch.float32)

		freqs = torch.einsum("bs,d->bsd", position, inv_freq)
		alpha = self.alpha.to(device=x.device)
		beta = self.beta.to(device=x.device)
		freqs = freqs * alpha + beta

		emb = torch.cat((freqs, freqs), dim=-1)
		emb = self.dropout(emb)

		# Apply modified RoPE: 
		# - First m dimensions: original RoPE values
		# - Last (n-m) dimensions: special cosine/sine patterns
		# n is the embedding dimension (self.head_dim), ensure both n and m are even
		n = self.head_dim
		# Ensure n is even (it should be by construction)
		if n % 2 != 0:
			n = n - 1
		
		# Calculate m as approximately 0.8 * n, ensuring it's even
		m = int(0.8 * n)
		if m % 2 != 0:
			m = m - 1
		
		# Create cos and sin from the original embeddings
		cos_original = emb.cos()
		sin_original = emb.sin()
		
		# Initialize separate tensors for keys and queries
		cos_k = cos_original.clone()
		sin_k = sin_original.clone()
		cos_q = cos_original.clone()
		sin_q = sin_original.clone()
		
		# For the remaining (n-m) dimensions, create the special pattern
		if n > m:
			n_minus_m = n - m
			# Create dimension indices: 0, 1, 2, ..., (n-m-1)
			dim_indices = torch.arange(n_minus_m, device=x.device, dtype=torch.float32)
			
			# position_ids has shape [batch, seq_len], we need to index into our pattern
			# For each position t and dimension i, we compute angles based on position modulo (n-m)
			# This creates a cyclic pattern
			
			# Get position modulo (n-m) for indexing into the pattern
			# Shape: [batch, seq_len]
			pos_mod = position_ids.float() % n_minus_m
			
			# For each dimension i (0 to n-m-1), compute angle = 2π * pos * i / (n-m)
			# pos_mod: [batch, seq_len] -> [batch, seq_len, 1]
			# dim_indices: [n-m] -> [1, 1, n-m]
			# Result: [batch, seq_len, n-m]
			angles = 2 * math.pi * pos_mod.unsqueeze(-1) * dim_indices.unsqueeze(0).unsqueeze(0) / n_minus_m
			
			# For keys: use cos and sin directly
			cos_pattern_k_expanded = torch.cos(angles).to(dtype=cos_k.dtype)
			sin_pattern_k_expanded = torch.sin(angles).to(dtype=sin_k.dtype)
			
			# For queries: swap cos and sin
			cos_pattern_q_expanded = torch.sin(angles).to(dtype=cos_q.dtype)
			sin_pattern_q_expanded = torch.cos(angles).to(dtype=sin_q.dtype)
			
			# Assign to the last (n-m) dimensions
			cos_k[:, :, m:n] = cos_pattern_k_expanded
			sin_k[:, :, m:n] = sin_pattern_k_expanded
			cos_q[:, :, m:n] = cos_pattern_q_expanded
			sin_q[:, :, m:n] = sin_pattern_q_expanded
		
		# Apply attention scaling
		cos_k = cos_k * self.attention_scaling
		sin_k = sin_k * self.attention_scaling
		cos_q = cos_q * self.attention_scaling
		sin_q = sin_q * self.attention_scaling

		return cos_k.to(dtype=x.dtype), sin_k.to(dtype=x.dtype), cos_q.to(dtype=x.dtype), sin_q.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
	"""Rotates half the hidden dims of the input."""
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_k: torch.Tensor, sin_k: torch.Tensor, 
						  cos_q: torch.Tensor, sin_q: torch.Tensor, 
						  unsqueeze_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Apply rotary position embeddings with separate encodings for keys and queries.
	
	Args:
		q: Query tensor
		k: Key tensor
		cos_k, sin_k: Cosine and sine for keys
		cos_q, sin_q: Cosine and sine for queries
		unsqueeze_dim: Dimension to unsqueeze cos/sin
	
	Returns:
		Tuple of (rotated_q, rotated_k)
	"""
	cos_k = cos_k.unsqueeze(unsqueeze_dim)
	sin_k = sin_k.unsqueeze(unsqueeze_dim)
	cos_q = cos_q.unsqueeze(unsqueeze_dim)
	sin_q = sin_q.unsqueeze(unsqueeze_dim)
	
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	
	return q_embed, k_embed


def _retrieve_module_and_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
	parent = root
	parts = name.split(".")
	for part in parts[:-1]:
		parent = getattr(parent, part)
	return parent, parts[-1]


def patch_qwen2_attention(model: PreTrainedModel) -> None:
	"""Patch Qwen2 attention layers to use separate key/query positional encodings."""
	
	for name, module in model.named_modules():
		if module.__class__.__name__ == "Qwen2Attention":
			# Store original forward method
			original_forward = module.forward
			
			def create_patched_forward(attn_module, orig_forward):
				def patched_forward(
					hidden_states: torch.Tensor,
					attention_mask: Optional[torch.Tensor] = None,
					position_ids: Optional[torch.LongTensor] = None,
					past_key_value = None,
					output_attentions: bool = False,
					use_cache: bool = False,
					cache_position: Optional[torch.LongTensor] = None,
					position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
					**kwargs,
				) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
					bsz, q_len, _ = hidden_states.size()

					query_states = attn_module.q_proj(hidden_states)
					key_states = attn_module.k_proj(hidden_states)
					value_states = attn_module.v_proj(hidden_states)

					query_states = query_states.view(bsz, q_len, 12, attn_module.head_dim).transpose(1, 2)
					key_states = key_states.view(bsz, q_len, 2, attn_module.head_dim).transpose(1, 2)
					value_states = value_states.view(bsz, q_len, 2, attn_module.head_dim).transpose(1, 2)

					# Check if we have separate k/q encodings (4 tensors) or standard (2 tensors)
					if position_embeddings is not None:
						if len(position_embeddings) == 4:
							# Separate encodings for keys and queries
							cos_k, sin_k, cos_q, sin_q = position_embeddings
							query_states, key_states = apply_rotary_pos_emb(
								query_states, key_states, cos_k, sin_k, cos_q, sin_q, unsqueeze_dim=1
							)
						else:
							# Standard RoPE with same encoding for both
							cos, sin = position_embeddings
							cos = cos.unsqueeze(1)
							sin = sin.unsqueeze(1)
							query_states = (query_states * cos) + (rotate_half(query_states) * sin)
							key_states = (key_states * cos) + (rotate_half(key_states) * sin)

					if past_key_value is not None:
						cache_kwargs = {"sin": sin_k if len(position_embeddings) == 4 else sin, 
										"cos": cos_k if len(position_embeddings) == 4 else cos,
										"cache_position": cache_position}
						key_states, value_states = past_key_value.update(key_states, value_states, attn_module.layer_idx, cache_kwargs)

					# Repeat k/v heads if necessary
					key_states = attn_module.repeat_kv(key_states, attn_module.num_key_value_groups)
					value_states = attn_module.repeat_kv(value_states, attn_module.num_key_value_groups)

					attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)

					if attention_mask is not None:
						causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
						attn_weights = attn_weights + causal_mask

					attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
					attn_weights = nn.functional.dropout(attn_weights, p=attn_module.attention_dropout, training=attn_module.training)
					attn_output = torch.matmul(attn_weights, value_states)

					attn_output = attn_output.transpose(1, 2).contiguous()
					attn_output = attn_output.reshape(bsz, q_len, -1)
					attn_output = attn_module.o_proj(attn_output)

					if not output_attentions:
						attn_weights = None

					return (attn_output, attn_weights, past_key_value)
				
				return patched_forward
			
			# Apply the patch
			module.forward = create_patched_forward(module, original_forward)
			module.repeat_kv = staticmethod(lambda hidden_states, n_rep: hidden_states.repeat_interleave(n_rep, dim=1) if n_rep > 1 else hidden_states)
			LOGGER.info(f"Patched attention layer: {name}")


def inject_custom_positional_encoding(
	model: PreTrainedModel,
	max_length: Optional[int] = None,
	dropout: float = 0.0,
	learned_scaling: bool = True,
) -> Tuple[nn.Module, str]:
	"""Replace the model's positional embedding with the custom module supporting separate k/q encodings."""

	config = model.config
	max_length = (
		max_length
		or getattr(config, "max_position_embeddings", None)
		or getattr(config, "n_positions", None)
		or getattr(config, "n_ctx", None)
	)
	if max_length is None:
		raise ValueError("Unable to infer `max_length` for positional encoding.")

	primary_device = next(model.parameters()).device
	primary_dtype = next(model.parameters()).dtype

	for name, module in model.named_modules():
		cls_name = module.__class__.__name__.lower()
		if "rotary" in cls_name and hasattr(module, "inv_freq"):
			inv_freq = module.inv_freq.detach().to(device=primary_device, dtype=torch.float32)
			rope_max = max_length or getattr(module, "max_seq_len_cached", None)
			if rope_max is None:
				rope_max = max_length
			if rope_max is None:
				rope_max = max_length or config.max_position_embeddings

			attention_scaling = getattr(module, "attention_scaling", 1.0)
			custom_rotary = CustomRotaryPositionalEncodingSeparateKQ(
				inv_freq=inv_freq,
				max_length=rope_max,
				attention_scaling=attention_scaling,
				dropout=dropout,
				learned_scaling=learned_scaling,
			).to(device=primary_device, dtype=primary_dtype)

			parent_module, attribute_name = _retrieve_module_and_parent(model, name)
			setattr(parent_module, attribute_name, custom_rotary)
			LOGGER.info("Replaced rotary positional embedding module at %s", name)
			
			# Patch attention layers to handle 4-tensor output
			patch_qwen2_attention(model)
			
			return custom_rotary, name

	raise RuntimeError("Failed to locate a positional embedding to replace.")


def load_model_and_tokenizer(
	model_name_or_path: str,
	device: torch.device,
	use_pretrained: bool = True,
	local_files_only: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
	"""Load a causal language model and tokenizer.
	
	Args:
		model_name_or_path: Path to local directory or HuggingFace model ID
		device: Device to load the model on
		use_pretrained: Whether to load pretrained weights or random init
		local_files_only: If True, only load from local directory without internet access
	"""

	if use_pretrained:
		LOGGER.info("Loading pretrained model '%s'%s", 
					model_name_or_path,
					" (local files only)" if local_files_only else "")
		model = AutoModelForCausalLM.from_pretrained(
			model_name_or_path,
			local_files_only=local_files_only,
			trust_remote_code=True,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_name_or_path,
			use_fast=True,
			local_files_only=local_files_only,
			trust_remote_code=True,
		)
	else:
		LOGGER.info("Instantiating a randomly initialised model from config '%s'", model_name_or_path)
		config = AutoConfig.from_pretrained(
			model_name_or_path,
			local_files_only=local_files_only,
			trust_remote_code=True,
		)
		model = AutoModelForCausalLM.from_config(config)
		tokenizer = AutoTokenizer.from_pretrained(
			model_name_or_path,
			use_fast=True,
			local_files_only=local_files_only,
			trust_remote_code=True,
		)

	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	tokenizer.padding_side = "left"
	model.to(device)
	model.eval()
	return model, tokenizer


def generate_text(
	model: PreTrainedModel,
	tokenizer: PreTrainedTokenizerBase,
	prompt: str,
	max_new_tokens: int = 64,
	temperature: float = 0.8,
	top_p: float = 0.95,
	top_k: int = 0,
	do_sample: bool = True,
	use_chat_template: bool = False,
) -> GenerationResult:
	"""Generate text from a prompt, optionally using chat template for instruct models.
	
	Args:
		model: The language model to use for generation
		tokenizer: The tokenizer for the model
		prompt: The user's prompt text
		max_new_tokens: Maximum number of new tokens to generate
		temperature: Sampling temperature
		top_p: Nucleus sampling parameter
		top_k: Top-k sampling parameter
		do_sample: Whether to use sampling or greedy decoding
		use_chat_template: If True, format prompt as a chat message for instruct models
		
	Returns:
		GenerationResult containing the prompt, generated text, and full output
	"""
	if use_chat_template:
		# Format the prompt as a chat message for instruct models
		if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
			messages = [{"role": "user", "content": prompt}]
			formatted_prompt = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True
			)
			LOGGER.info("Using chat template. Formatted prompt: %s", formatted_prompt[:200])
		else:
			LOGGER.warning(
				"Chat template requested but tokenizer does not support it. "
				"Falling back to raw prompt."
			)
			formatted_prompt = prompt
	else:
		formatted_prompt = prompt
	
	inputs = tokenizer(formatted_prompt, return_tensors="pt")
	input_ids = inputs["input_ids"].to(model.device)
	attention_mask = inputs["attention_mask"].to(model.device)

	with torch.no_grad():
		generated_ids = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=max_new_tokens,
			do_sample=do_sample,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k if top_k > 0 else None,
			pad_token_id=tokenizer.pad_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
	# Extract only the generated portion after the formatted prompt
	generated_text = full_text[len(formatted_prompt):] if not use_chat_template else full_text.split(formatted_prompt)[-1]
	return GenerationResult(prompt=prompt, generated_text=generated_text, full_text=full_text)


def _self_test_qwen2(device: torch.device, max_length: int) -> None:
	config = Qwen2Config(
		vocab_size=256,
		hidden_size=128,
		intermediate_size=512,
		num_hidden_layers=2,
		num_attention_heads=4,
		num_key_value_heads=4,
		rms_norm_eps=1e-5,
		max_position_embeddings=max_length,
	)

	model = Qwen2ForCausalLM(config).to(device)
	if model.config.pad_token_id is None:
		model.config.pad_token_id = 0
	if model.config.eos_token_id is None:
		model.config.eos_token_id = 0

	inject_custom_positional_encoding(model, max_length=max_length)

	input_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)
	attention_mask = torch.ones_like(input_ids)

	with torch.no_grad():
		outputs = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=6,
			do_sample=False,
			pad_token_id=model.config.pad_token_id,
			eos_token_id=model.config.eos_token_id,
		)

	expected_length = input_ids.size(1) + 6
	assert outputs.size(1) == expected_length, "Qwen self-test generation length mismatch."
	LOGGER.info("Qwen self-test passed: generated shape %s", tuple(outputs.shape))


def run_self_test(device: torch.device, max_length: int = 64) -> None:
	"""Quick integration test covering Qwen-style rotary models with separate k/q encodings."""
	_self_test_qwen2(device, max_length)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt to feed the model.")
	parser.add_argument(
		"--model-name-or-path",
		type=str,
		default="Qwen/Qwen2-0.5B",
		help="Hugging Face model identifier or local path (must be a Qwen2 model).",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=64,
		dest="max_new_tokens",
		help="Number of tokens to sample beyond the prompt.",
	)
	parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
	parser.add_argument("--top-p", type=float, default=0.95, dest="top_p", help="Nucleus sampling top-p.")
	parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling filter. 0 disables it.")
	parser.add_argument(
		"--dropout", type=float, default=0.0, help="Dropout applied inside the custom positional encoding."
	)
	parser.add_argument(
		"--no-learned-scaling",
		action="store_true",
		help="Disable the learned affine transformation on positional encodings.",
	)
	parser.add_argument(
		"--max-position-embeddings",
		type=int,
		default=None,
		help="Override the maximum positional length handled by the custom encoding.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device to run generation on.",
	)
	parser.add_argument(
		"--no-pretrained",
		action="store_true",
		help="Instantiate the model from config instead of loading pretrained weights.",
	)
	parser.add_argument(
		"--greedy",
		action="store_true",
		help="Disable sampling for deterministic generation.",
	)
	parser.add_argument(
		"--self-test",
		action="store_true",
		help="Run an offline integration test instead of full inference.",
	)
	parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="Load model from local directory only, without internet access.",
	)
	parser.add_argument(
		"--use-chat-template",
		action="store_true",
		help="Format the prompt using chat template for instruct models.",
	)
	return parser.parse_args()


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
	args = parse_args()

	device = torch.device(args.device)
	set_seed(args.seed)

	if args.self_test:
		LOGGER.info("Running self-test on device %s", device)
		run_self_test(device)
		return

	model, tokenizer = load_model_and_tokenizer(
		model_name_or_path=args.model_name_or_path,
		device=device,
		use_pretrained=not args.no_pretrained,
		local_files_only=args.local_files_only,
	)

	max_length = args.max_position_embeddings
	custom_module, replaced_name = inject_custom_positional_encoding(
		model,
		max_length=max_length,
		dropout=args.dropout,
		learned_scaling=not args.no_learned_scaling,
	)

	LOGGER.info(
		"Custom rotary encoding with separate K/Q injected at %s (head_dim=%s, max_seq_len=%s)",
		replaced_name,
		custom_module.head_dim,
		getattr(custom_module, "max_seq_len_cached", None),
	)

	result = generate_text(
		model=model,
		tokenizer=tokenizer,
		prompt=args.prompt,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		top_k=args.top_k,
		do_sample=not args.greedy,
		use_chat_template=args.use_chat_template,
	)

	LOGGER.info("Prompt: %s", result.prompt)
	LOGGER.info("Generated continuation: %s", result.generated_text)
	print(result.full_text)


if __name__ == "__main__":
	main()
