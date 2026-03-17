"""HybridEngine side: load BailingMoeV2.5 model, run forward pass, extract per-layer outputs.

Run on a pod with antllm + Megatron-LM + HybridEngine installed.

Usage:
    PYTHONPATH=/path/to/Asystem-HybridEngine:/path/to/antllm:/path/to/Megatron-LM:$PYTHONPATH \
    USE_MAX_V2=1 HYBRID_IGNORE_LOAD_CHECK=1 \
    torchrun --nproc_per_node=8 compare_hybrid_engine.py \
        --model_path /path/to/mini-v25-hgf \
        --dcp_path /path/to/mini-v25-dcp \
        --input /tmp/comparison/test_input.pt \
        --output /tmp/comparison/hybrid_outputs.pt

Environment:
    - antllm (provides bailing_moe_model_provider, process_bailing_args)
    - Megatron-LM (internal fork from code.alipay.com/Arc/Megatron-LM)
    - Asystem-HybridEngine (provides MegatronBackend)
    - flash-linear-attention (fla, for Lightning Attention kernel)
"""

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist

# Ensure USE_MAX_V2 is set for BailingMoe models
os.environ.setdefault("USE_MAX_V2", "1")
os.environ.setdefault("HYBRID_IGNORE_LOAD_CHECK", "1")


# ============================================================================
# Config for mini v2.5 model
# Based on: Asystem-HybridEngine/tests/data/configs/mini_v2.5.yaml
# ============================================================================
MINI_V25_CONFIG = {
    "num_layers": 20,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_query_groups": 16,
    "ffn_hidden_size": 5120,
    "num_experts": 256,
    "moe_ffn_hidden_size": 512,
    "moe_shared_expert_intermediate_size": 512,
    "first_k_dense_replace": 1,
    "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # v2.5 specific
    "layer_group_size": 5,
    "linear_attn_num_query_groups": 16,
    "linear_attn_norm_group_size": 4,
    "multi_latent_attention": True,
    "group_query_attention": False,
    "kv_lora_rank": 512,
    "qk_head_dim": 128,
    "qk_pos_emb_head_dim": 64,
    "v_head_dim": 128,
    # Parallelism (overridden by CLI args)
    "expert_model_parallel_size": 8,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
    # RoPE
    "rotary_base": 10000,
    "rotary_percent": 0.5,
    "position_embedding_type": "rope",
    "apply_rope_fusion": False,
    "use_rotary_position_embeddings": True,
    # Vocabulary
    "vocab_size": 157184,
    "make_vocab_size_divisible_by": 128,
    # MoE settings
    "moe_router_topk": 8,
    "moe_router_score_function": "sigmoid",
    "moe_router_num_groups": 8,
    "moe_router_group_topk": 4,
    "moe_router_topk_scaling_factor": 2.5,
    "moe_router_enable_expert_bias": True,
    "moe_grouped_gemm": True,
    "moe_token_dispatcher_type": "alltoall",
    "moe_shared_expert_overlap": True,
    "moe_router_dtype": "fp32",
    # Training / precision
    "bf16": True,
    "normalization": "RMSNorm",
    "norm_epsilon": 1e-6,
    "qk_layernorm": True,
    "swiglu": True,
    "add_bias_linear": False,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "attention_softmax_in_fp32": True,
    "init_method_std": 0.006,
    "untie_embeddings_and_output_weights": True,
    "use_flash_attn": True,
    "use_mcore_models": True,
    "sequence_parallel": True,
    # Optimizer (needed by Megatron init but not used for eval)
    "lr": 1e-5,
    "lr_decay_style": "constant",
    "weight_decay": 0.0,
    "clip_grad": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "micro_batch_size": 1,
    "global_batch_size": 8,
    "seq_length": 4096,
    "max_position_embeddings": 4096,
    "use_distributed_optimizer": True,
    "overlap_grad_reduce": True,
    "overlap_p2p_comm": True,
    # Recompute
    "recompute_granularity": None,  # Disable for eval to get all layer outputs
    # Checkpoint
    "auto_detect_ckpt_format": True,
    "no_load_optim": True,
    "no_load_rng": True,
    "no_save_optim": True,
    # Logging
    "seed": 42,
    "tensorboard_log_interval": 1,
    "enable_one_logger": False,
    "use_random_logits": False,
    "tokenizer_type": "HuggingFaceTokenizer",
}


def register_hooks(model):
    """Register forward hooks on all decoder layers."""
    layer_outputs = {}
    attn_outputs = {}
    hooks = []

    # Access layers through the model hierarchy
    # For Megatron GPTModel: model.decoder.layers
    decoder = None
    m = model
    while hasattr(m, 'module'):
        m = m.module
    if hasattr(m, 'decoder'):
        decoder = m.decoder
    elif hasattr(m, 'language_model') and hasattr(m.language_model, 'decoder'):
        decoder = m.language_model.decoder

    if decoder is None:
        print("[WARN] Could not find decoder layers for hooks")
        return layer_outputs, attn_outputs, hooks

    for i, layer in enumerate(decoder.layers):
        def make_layer_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                layer_outputs[idx] = hs.detach().cpu()
            return hook_fn

        h = layer.register_forward_hook(make_layer_hook(i))
        hooks.append(h)

        if hasattr(layer, 'self_attention'):
            def make_attn_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    attn_outputs[idx] = out.detach().cpu()
                return hook_fn

            h2 = layer.self_attention.register_forward_hook(make_attn_hook(i))
            hooks.append(h2)

    print(f"Registered hooks on {len(decoder.layers)} layers")
    return layer_outputs, attn_outputs, hooks


def forward_step_func_eval(data_iterator, model):
    """Forward step function that returns logits (no loss computation)."""
    def loss_func(output_tensor):
        return torch.tensor(1.0, device=torch.cuda.current_device()), {'logits': output_tensor}

    data = next(data_iterator)
    tokens = data['tokens'].to(torch.cuda.current_device())
    attention_mask = data['attention_mask'].to(torch.cuda.current_device())
    position_ids = data['position_ids'].to(torch.cuda.current_device())
    output_tensor = model(tokens, position_ids, attention_mask)
    return output_tensor, loss_func


def forward_step_func_loss(data_iterator, model):
    """Forward step function that computes loss."""
    import megatron.core.parallel_state as mpu

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor['loss'].float() if isinstance(output_tensor, dict) else output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        if mpu.get_context_parallel_world_size() > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])},
        )

    from functools import partial
    data = next(data_iterator)
    tokens = data['tokens'].to(torch.cuda.current_device())
    attention_mask = data['attention_mask'].to(torch.cuda.current_device())
    position_ids = data['position_ids'].to(torch.cuda.current_device())
    labels = data['labels'].to(torch.cuda.current_device())
    loss_mask = data['loss_mask'].to(torch.cuda.current_device())

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
    return output_tensor, partial(loss_func, loss_mask)


def main():
    parser = argparse.ArgumentParser(description="HybridEngine comparison forward pass")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model path (tokenizer_model for Megatron)")
    parser.add_argument("--dcp_path", type=str, default=None,
                        help="DCP checkpoint path (Megatron distributed checkpoint)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input data file from generate_test_data.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file to save results")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=8, help="Expert parallel size")
    args = parser.parse_args()

    # Build config
    config = dict(MINI_V25_CONFIG)
    config["tokenizer_model"] = args.model_path
    if args.dcp_path:
        config["load"] = args.dcp_path
    config["tensor_model_parallel_size"] = args.tp
    config["pipeline_model_parallel_size"] = args.pp
    config["expert_model_parallel_size"] = args.ep
    config["seq_length"] = 4096  # Will be overridden by actual input len
    config["tensor_inspect_interval"] = None

    # Load test data
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    seq_len = data["seq_len"]
    config["seq_length"] = seq_len
    config["max_position_embeddings"] = max(seq_len, 4096)

    # Format data for HybridEngine (same format as get_mocked_data)
    test_data = [{
        "tokens": data["input_ids"],
        "attention_mask": data["attention_mask"],
        "position_ids": data["position_ids"],
        "labels": data["labels"],
        "loss_mask": data["loss_mask"],
    }]

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"Config: TP={args.tp}, PP={args.pp}, EP={args.ep}")
        print(f"Input: seq_len={seq_len}, batch_size={data['batch_size']}")
        print(f"Model path: {args.model_path}")
        print(f"DCP path: {args.dcp_path}")

    # Initialize MegatronBackend
    from asystem_runtime.backend.megatron_backend import MegatronBackend
    from asystem_runtime.utils.megatron_model_provider import get_model_provider

    # Import antllm for BailingMoe support
    from antllm.arguments.moe_args import process_bailing_args
    config['extra_args_provider'] = process_bailing_args

    backend = MegatronBackend(config)
    backend.initialize()

    model_arch = backend.hf_config.architectures[0]
    use_max_v2 = os.environ.get("USE_MAX_V2", "1") == "1"
    backend.setup_model(model_provide_func=get_model_provider(use_max_v2, model_arch))

    if rank == 0:
        print(f"Model architecture: {model_arch}")
        print(f"Model loaded successfully")

    # Get the underlying model for hooks
    model = backend.model[0] if hasattr(backend, 'model') else backend.models[0]

    # Register hooks
    layer_outputs, attn_outputs, hooks = register_hooks(model)

    # Run forward pass to get logits
    if rank == 0:
        print("Running forward pass (logits)...")

    outputs = backend.forward(test_data, seq_len, forward_step_func_eval)

    # Also run with loss
    if rank == 0:
        print("Running forward pass (loss)...")

    backend.set_global_step(1)
    loss_outputs = backend.train(test_data, forward_step_func_loss)

    # Extract results
    import megatron.core.parallel_state as mpu
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_last = mpu.get_pipeline_model_parallel_world_size() - 1
    dp_rank = mpu.get_data_parallel_rank()

    logits_tensor = None
    loss_value = None

    if pp_rank == pp_last and outputs:
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict) and 'logits' in outputs[0]:
                logits_tensor = outputs[0]['logits'].detach().cpu()

    if loss_outputs is not None:
        # Extract loss from training output
        if isinstance(loss_outputs, dict) and 'lm loss' in loss_outputs:
            sum_loss, total_tokens = loss_outputs['lm loss']
            loss_value = (sum_loss / total_tokens).item()
        elif isinstance(loss_outputs, list) and len(loss_outputs) > 0:
            # Try first element
            lo = loss_outputs[0]
            if isinstance(lo, dict) and 'lm loss' in lo:
                sum_loss, total_tokens = lo['lm loss']
                loss_value = (sum_loss / total_tokens).item()

    results = {
        "layer_outputs": {k: v.float() for k, v in layer_outputs.items()},
        "attn_outputs": {k: v.float() for k, v in attn_outputs.items()},
        "logits": logits_tensor.float() if logits_tensor is not None else None,
        "loss": loss_value,
        "ppl": math.exp(loss_value) if loss_value is not None and loss_value < 100 else None,
        "engine": "hybrid_engine",
        "model_path": args.model_path,
        "parallel": {"tp": args.tp, "pp": args.pp, "ep": args.ep},
    }

    if dp_rank == 0:
        save_path = args.output
        if args.pp > 1:
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_pp{pp_rank}{ext}"

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(results, save_path)
        print(f"[Rank {rank}, PP {pp_rank}] Saved results to {save_path}")
        print(f"  Layer outputs: {list(layer_outputs.keys())}")
        print(f"  Attn outputs: {list(attn_outputs.keys())}")
        if loss_value is not None:
            print(f"  Loss: {loss_value:.6f}")
            print(f"  PPL: {results['ppl']:.4f}")

    # Cleanup
    for h in hooks:
        h.remove()

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
