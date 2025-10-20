"""Inference utilities with custom positional encoding.

This script demonstrates how to run text generation with Hugging Face
transformers while swapping the model's positional embeddings for a custom
PyTorch module. By default it targets GPT-style causal language models, but the
replacement logic is generic enough to support any architecture that exposes a
learned positional `nn.Embedding` compatible with the provided hidden size.

Example usage (requires internet access for pretrained weights):

	python Inference.py --prompt "Once upon a time" --max-new-tokens 64

To run an offline self-test that instantiates a tiny randomly initialised GPT2
model and validates the positional-encoding swap:

	python Inference.py --self-test
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


class CustomPositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding with learned per-dimension scaling.

	The module produces deterministic sinusoidal embeddings and then applies a
	lightweight, trainable affine transformation (per-dimension scale and
	offset). This keeps the positional information explicit while granting the
	model flexibility to adapt the encoding distribution during fine-tuning.
	"""

	def __init__(
		self,
		hidden_size: int,
		max_length: int,
		dropout: float = 0.0,
		learned_scaling: bool = True,
	) -> None:
		super().__init__()
		if hidden_size % 2 != 0:
			raise ValueError(
				"CustomPositionalEncoding expects an even hidden size; received"
				f" {hidden_size}."
			)

		self.hidden_size = hidden_size
		self.embedding_dim = hidden_size
		self.num_embeddings = max_length
		self.dropout = nn.Dropout(dropout)
		self.learned_scaling = learned_scaling

		pe = self._build_sinusoidal_table(max_length, hidden_size)
		self.register_buffer("pe", pe, persistent=False)

		if learned_scaling:
			self.alpha = nn.Parameter(torch.ones(hidden_size))
			self.beta = nn.Parameter(torch.zeros(hidden_size))
		else:
			self.register_buffer("alpha", torch.ones(hidden_size), persistent=False)
			self.register_buffer("beta", torch.zeros(hidden_size), persistent=False)

	@staticmethod
	def _build_sinusoidal_table(max_len: int, hidden_size: int) -> torch.Tensor:
		position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, hidden_size, 2, dtype=torch.float32)
			* (-math.log(10000.0) / hidden_size)
		)
		pe = torch.zeros(max_len, hidden_size, dtype=torch.float32)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe

	@property
	def weight(self) -> torch.Tensor:
		"""Expose the internal table to mimic `nn.Embedding`'s API."""

		return self.pe

	def forward(self, position_ids: torch.LongTensor) -> torch.Tensor:
		if position_ids.dtype != torch.long:
			position_ids = position_ids.long()

		max_pos = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
		if max_pos > self.num_embeddings:
			raise ValueError(
				f"Position ids require length {max_pos}, but buffer is built"
				f" for {self.num_embeddings}. Increase `max_length`."
			)

		device = position_ids.device
		pe = self.pe.to(device)
		embeddings = pe[position_ids]

		alpha = self.alpha.to(device)
		beta = self.beta.to(device)
		embeddings = embeddings * alpha + beta
		embeddings = self.dropout(embeddings)
		return embeddings


class CustomRotaryPositionalEncoding(nn.Module):
	"""Custom rotary positional embedding compatible with Qwen-style models."""

	def __init__(
		self,
		inv_freq: torch.Tensor,
		max_length: int,
		attention_scaling: float = 1.0,
		dropout: float = 0.0,
		learned_scaling: bool = True,
		period: int = 128000,
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
		self.period = period  # Periodicity for positional encoding

		inv_freq = inv_freq.detach().to(dtype=torch.float32)
		self.register_buffer("inv_freq", inv_freq, persistent=False)

		if learned_scaling:
			self.alpha = nn.Parameter(torch.ones_like(inv_freq))
			self.beta = nn.Parameter(torch.zeros_like(inv_freq))
		else:
			self.register_buffer("alpha", torch.ones_like(inv_freq), persistent=False)
			self.register_buffer("beta", torch.zeros_like(inv_freq), persistent=False)

	def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if position_ids.dtype != torch.long:
			position_ids = position_ids.long()

		batch, seq_len = position_ids.shape
		inv_freq = self.inv_freq.to(device=x.device)
		
		# Apply modulo operation to create periodic positional encoding
		# For any index i and i+period, they will have the same encoding
		position = position_ids.to(device=x.device, dtype=torch.float32)
		position = position % self.period  # Apply periodicity

		freqs = torch.einsum("bs,d->bsd", position, inv_freq)
		alpha = self.alpha.to(device=x.device)
		beta = self.beta.to(device=x.device)
		freqs = freqs * alpha + beta

		emb = torch.cat((freqs, freqs), dim=-1)
		emb = self.dropout(emb)

		# Apply masking: zero out dimensions beyond m, where m ≈ 0.8 * n
		# n is the embedding dimension (self.head_dim), ensure both n and m are even
		n = self.head_dim
		# Ensure n is even (it should be by construction)
		if n % 2 != 0:
			n = n - 1
		
		# Calculate m as approximately 0.8 * n, ensuring it's even
		m = int(0.8 * n)
		if m % 2 != 0:
			m = m - 1
		
		# Create a mask that zeros out dimensions beyond m
		# emb shape: [batch, seq_len, head_dim]
		mask = torch.zeros(n, device=emb.device, dtype=emb.dtype)
		mask[:m] = 1.0
		
		# Apply the mask to zero out the higher dimensions
		emb = emb * mask.unsqueeze(0).unsqueeze(0)  # broadcast to [1, 1, head_dim]

		cos = emb.cos() * self.attention_scaling
		sin =  emb.sin()* self.attention_scaling

		return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _retrieve_module_and_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
	parent = root
	parts = name.split(".")
	for part in parts[:-1]:
		parent = getattr(parent, part)
	return parent, parts[-1]


def inject_custom_positional_encoding(
	model: PreTrainedModel,
	max_length: Optional[int] = None,
	dropout: float = 0.0,
	learned_scaling: bool = True,
) -> Tuple[CustomPositionalEncoding, str]:
	"""Replace the model's positional embedding with the custom module."""

	config = model.config
	hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
	if hidden_size is None:
		raise ValueError("Unable to determine the model's hidden size for PE replacement.")

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

	custom_module = CustomPositionalEncoding(
		hidden_size=hidden_size,
		max_length=max_length,
		dropout=dropout,
		learned_scaling=learned_scaling,
	).to(device=primary_device, dtype=primary_dtype)

	target_name = None
	for name, module in model.named_modules():
		if isinstance(module, nn.Embedding):
			num_embeddings = getattr(module, "num_embeddings", None)
			embedding_dim = getattr(module, "embedding_dim", None)
			if num_embeddings == max_length and embedding_dim == hidden_size:
				parent_module, attribute_name = _retrieve_module_and_parent(model, name)
				setattr(parent_module, attribute_name, custom_module)
				LOGGER.info("Replaced positional embedding module at %s", name)
				return custom_module, name

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
			custom_rotary = CustomRotaryPositionalEncoding(
				inv_freq=inv_freq,
				max_length=rope_max,
				attention_scaling=attention_scaling,
				dropout=dropout,
				learned_scaling=learned_scaling,
			).to(device=primary_device, dtype=primary_dtype)

			parent_module, attribute_name = _retrieve_module_and_parent(model, name)
			setattr(parent_module, attribute_name, custom_rotary)
			LOGGER.info("Replaced rotary positional embedding module at %s", name)
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


def _self_test_gpt2(device: torch.device, max_length: int) -> None:
	config = GPT2Config(
		n_positions=max_length,
		n_ctx=max_length,
		n_layer=2,
		n_head=2,
		n_embd=64,
		vocab_size=256,
	)
	model = GPT2LMHeadModel(config).to(device)
	if model.config.pad_token_id is None:
		model.config.pad_token_id = 0
	if model.config.eos_token_id is None:
		model.config.eos_token_id = 0

	inject_custom_positional_encoding(model, max_length=max_length)

	input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
	attention_mask = torch.ones_like(input_ids)

	with torch.no_grad():
		outputs = model.generate(
			input_ids=input_ids,
			attention_mask=attention_mask,
			max_new_tokens=8,
			do_sample=False,
			pad_token_id=model.config.pad_token_id,
			eos_token_id=model.config.eos_token_id,
		)

	expected_length = input_ids.size(1) + 8
	assert outputs.size(1) == expected_length, "GPT-2 self-test generation length mismatch."
	LOGGER.info("GPT-2 self-test passed: generated shape %s", tuple(outputs.shape))


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
	"""Quick integration test covering GPT-2 and Qwen-style rotary models."""

	_self_test_gpt2(device, max_length)
	_self_test_qwen2(device, max_length)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt to feed the model.")
	parser.add_argument(
		"--prompt-file",
		type=str,
		default=None,
		help="Path to a text file containing the prompt. If provided, overrides --prompt.",
	)
	parser.add_argument(
		"--model-name-or-path",
		type=str,
		default="gpt2",
		help="Hugging Face model identifier or local path.",
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

	# Load prompt from file if specified
	if args.prompt_file:
		LOGGER.info("Loading prompt from file: %s", args.prompt_file)
		try:
			with open(args.prompt_file, 'r', encoding='utf-8') as f:
				prompt = f.read().strip()
			LOGGER.info("Loaded prompt from file (length: %d characters)", len(prompt))
		except FileNotFoundError:
			LOGGER.error("Prompt file not found: %s", args.prompt_file)
			return
		except Exception as e:
			LOGGER.error("Error reading prompt file: %s", e)
			return
	else:
		prompt = args.prompt

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

	if hasattr(custom_module, "num_embeddings"):
		LOGGER.info(
			"Custom positional encoding injected at %s (num_embeddings=%s, hidden_size=%s)",
			replaced_name,
			custom_module.num_embeddings,
			custom_module.hidden_size,
		)
	elif hasattr(custom_module, "head_dim"):
		LOGGER.info(
			"Custom rotary encoding injected at %s (head_dim=%s, max_seq_len=%s)",
			replaced_name,
			custom_module.head_dim,
			getattr(custom_module, "max_seq_len_cached", None),
		)
	else:
		LOGGER.info("Custom positional encoding injected at %s", replaced_name)

	result = generate_text(
		model=model,
		tokenizer=tokenizer,
		prompt=prompt,
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