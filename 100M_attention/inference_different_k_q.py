import torch
import torch.nn as nn
import math
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv

# --- Configuration ---

# ‼️ MUST-CHANGE: Update this to your local model path
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen2.5-1.5B" 

# Set the sentence length 'n' for chunking
# Example: 64 tokens per sentence
N_SENTENCE_LENGTH = 128000

# --- Custom Encoding Functions ---

def _custom_encode_query(q_tensor, position_ids, n, original_rope):
    """
    Applies the custom positional encoding to the Query tensor.
    - Top 80%: Standard RoPE with intra-sentence position (i % n).
    - Bottom 20%: Additive 'sin' encoding with sentence index (i // n).
    """
    # q_tensor shape: (batch_size, num_heads, seq_len, head_dim)
    # position_ids shape: (batch_size, seq_len)
    
    head_dim = q_tensor.shape[-1]
    
    # Calculate split point, ensuring dimensions are even for RoPE
    s_80 = int(head_dim * 0.95)
    s_80 = (s_80 // 2) * 2  # Make even
    p = head_dim - s_80
    
    if p == 0: # Handle cases where 80% is the whole dim
        s_80 = head_dim
        
    q_80 = q_tensor[..., :s_80]
    
    # 1. Encode top 80% with standard RoPE (using pos % n)
    intra_pos = position_ids % n
    
    # Get the pre-computed inverse frequencies from the original RoPE module
    inv_freq_80 = original_rope.inv_freq[:s_80//2].to(q_tensor.device).to(q_tensor.dtype)
    
    # Calculate RoPE embeddings
    t = intra_pos.to(q_tensor.dtype).unsqueeze(1).unsqueeze(-1) # (batch, 1, seq_len, 1)
    freqs_80 = t @ inv_freq_80.unsqueeze(0)          # (batch, 1, seq_len, s_80/2)
    emb_80 = torch.cat((freqs_80, freqs_80), dim=-1) # (batch, 1, seq_len, s_80)
    
    # Manual RoPE application (same as apply_rotary_pos_emb does internally)
    # emb_80 already has shape (batch, 1, seq_len, s_80), which will broadcast correctly
    # with q_80 shape (batch, num_heads, seq_len, s_80)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = emb_80.cos()  # (batch, 1, seq_len, s_80)
    sin = emb_80.sin()  # (batch, 1, seq_len, s_80)
    q_80_encoded = (q_80 * cos) + (rotate_half(q_80) * sin)
    
    if p == 0:
        return q_80_encoded
        
    # ---
    
    # 2. Encode bottom 20% with custom 'sin' encoding
    q_20 = q_tensor[..., s_80:]
    sentence_indices = (position_ids // n).to(q_tensor.dtype).unsqueeze(1).unsqueeze(-1) # (batch, 1, seq_len, 1)
    
    # Create dimension indices: [0, 1, ..., p-1]
    dim_indices_p = torch.arange(0, p, device=q_tensor.device, dtype=q_tensor.dtype).reshape(1, 1, 1, p)
    
    # Calculate argument: 2 * pi * dim * sent_idx / p
    arg_p = (2 * math.pi / p) * dim_indices_p * sentence_indices
    
    # Per prompt: "substitute ... to sin" for Query
    # We use factor 1.0 for the linear combination (additive)
    q_20_encoding = arg_p.sin()
    q_20_encoded = q_20 + q_20_encoding
    
    return torch.cat((q_80_encoded, q_20_encoded), dim=-1)


def _custom_encode_key(k_tensor, position_ids, n, original_rope):
    """
    Applies the custom positional encoding to the Key tensor.
    - Top 80%: Standard RoPE with intra-sentence position (i % n).
    - Bottom 20%: Additive 'cos' encoding with sentence index (i // n).
    """
    # k_tensor shape: (batch_size, num_kv_heads, seq_len, head_dim)
    
    head_dim = k_tensor.shape[-1]
    s_80 = int(head_dim * 0.95)
    s_80 = (s_80 // 2) * 2 # Make even
    p = head_dim - s_80

    if p == 0:
        s_80 = head_dim
        
    k_80 = k_tensor[..., :s_80]

    # 1. Encode top 80% (same as query)
    intra_pos = position_ids % n
    inv_freq_80 = original_rope.inv_freq[:s_80//2].to(k_tensor.device).to(k_tensor.dtype)
    t = intra_pos.to(k_tensor.dtype).unsqueeze(1).unsqueeze(-1)
    freqs_80 = t @ inv_freq_80.unsqueeze(0)
    emb_80 = torch.cat((freqs_80, freqs_80), dim=-1)
    
    # Manual RoPE application (same as apply_rotary_pos_emb does internally)
    # emb_80 already has shape (batch, 1, seq_len, s_80), which will broadcast correctly
    # with k_80 shape (batch, num_kv_heads, seq_len, s_80)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = emb_80.cos()  # (batch, 1, seq_len, s_80)
    sin = emb_80.sin()  # (batch, 1, seq_len, s_80)
    k_80_encoded = (k_80 * cos) + (rotate_half(k_80) * sin)

    if p == 0:
        return k_80_encoded
        
    # ---

    # 2. Encode bottom 20% with custom 'cos' encoding
    k_20 = k_tensor[..., s_80:]
    sentence_indices = (position_ids // n).to(k_tensor.dtype).unsqueeze(1).unsqueeze(-1)
    dim_indices_p = torch.arange(0, p, device=k_tensor.device, dtype=k_tensor.dtype).reshape(1, 1, 1, p)
    
    # Calculate argument: 2 * pi * dim * sent_idx / p
    arg_p = (2 * math.pi / p) * dim_indices_p * sentence_indices
    
    # Per prompt: use 'cos' for Key
    k_20_encoding = arg_p.cos()
    k_20_encoded = k_20 + k_20_encoding
    
    return torch.cat((k_80_encoded, k_20_encoded), dim=-1)


# --- Monkey-Patching Factory ---

def create_custom_forward(n_sentence_length, original_rope_object):
    """
    This factory returns a new 'forward' function that will
    replace the original Qwen2Attention.forward method.
    """
    
    # This is the new function that will be bound to each attention module
    def custom_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None, # Qwen2-specific kwarg
        **kwargs,
    ):
        """
        This is a re-implementation of Qwen2Attention.forward,
        with the RoPE application swapped for our custom functions.
        """
        
        bsz, q_len, _ = hidden_states.size()

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        
        # --- ‼️ CUSTOM ENCODING REPLACES RoPE ‼️ ---
        
        # Original code:
        # cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # New code:
        query_states = _custom_encode_query(query_states, position_ids, n_sentence_length, original_rope_object)
        key_states = _custom_encode_key(key_states, position_ids, n_sentence_length, original_rope_object)
        
        # --- End of Custom Encoding ---

        
        # KV Cache handling (updated for modern transformers Cache API)
        if past_key_value is not None:
            # For Cache objects, we need dummy cos/sin for the cache update
            # Since we already applied custom encoding, these won't be used for RoPE
            dummy_cos = torch.ones_like(key_states[..., :1])
            dummy_sin = torch.zeros_like(key_states[..., :1])
            cache_kwargs = {"sin": dummy_sin, "cos": dummy_cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: Repeat K/V heads (use imported repeat_kv function, not class method)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Standard Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.config.num_attention_heads, q_len, key_states.shape[2]):
            raise ValueError(f"Attention weights shape mismatch: {attn_weights.size()}")

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Output projection
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    # Return the new function, ready to be bound
    return custom_attention_forward


# --- Main Execution ---

def main():
    print(f"Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {MODEL_PATH}")
    # We load with 'auto' device_map and bf16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # Force "eager" attention implementation
        attn_implementation="eager" 
    )
    model.eval()

    # 1. Get the original RoPE module (for its `inv_freq` buffer)
    # This path is standard for Qwen2 models
    try:
        original_rope = model.model.rotary_emb
        print(f"Successfully located original RoPE module at `model.model.rotary_emb`")
    except AttributeError:
        print("Error: Could not find `model.model.rotary_emb`. Aborting.")
        return

    # 2. Create the new 'forward' function
    new_forward = create_custom_forward(
        n_sentence_length=N_SENTENCE_LENGTH,
        original_rope_object=original_rope
    )

    # 3. Monkey-patch all attention layers
    patch_count = 0
    for layer in model.model.layers:
        # Bind the new 'forward' function as a method to the layer's attention module
        layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
        patch_count += 1
    
    print(f"Successfully patched {patch_count} attention layers with custom encoding.")

    # 4. Run generation
    print("\n--- Starting Generation with Custom Encoding ---")
    
    # A prompt that is longer than N_SENTENCE_LENGTH to test both encodings
    prompt = (
        "This is the first sentence, it has several tokens. " # Part of sentence 0
        "This is still the first sentence, let's make it go "
        "past our n=64 limit. Now we are finally in the " # This part will be i // n = 1
        "second sentence, and the encoding should change." # Part of sentence 1
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id
        )
    
    print("\n--- Generation Complete ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()