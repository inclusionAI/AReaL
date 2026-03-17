"""AReaL side: load BailingMoeV2.5 model, run forward pass, extract per-layer outputs.

Usage:
    PYTHONPATH=/storage/openpsi/codes/chucai.dzq/gh/AReaL:$PYTHONPATH \
    AREAL_SPMD_MODE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    torchrun --nproc_per_node=8 examples/single-controller/compare_areal.py \
        --model_path /storage/openpsi/models/moe-mini-v25-e256-ep8tp1pp1-fp8-structure-fitted-adamw-new-3T \
        --input /tmp/comparison/test_input.pt \
        --output /tmp/comparison/areal_outputs.pt
"""

import argparse
import os

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize torch distributed using torchrun env vars."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def init_megatron_parallel(world_size: int, tp: int, pp: int, ep: int, seed: int = 42):
    """Initialize megatron-core parallel state and RNG tracker."""
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )
    # Initialize RNG tracker — required by mcore layers that use tensor_parallel rng fork
    model_parallel_cuda_manual_seed(seed)
    return mpu


def load_model(model_path: str, dtype: torch.dtype):
    """Load BailingMoeV2.5 model using mbridge AutoBridge."""
    import mbridge
    # Import our bailing_moe_bridge to register bailing_moe_linear/bailing_hybrid types
    import areal.models.mcore.bailing_moe_bridge  # noqa: F401

    bridge = mbridge.AutoBridge.from_pretrained(model_path, trust_remote_code=True)
    bridge.dtype = dtype

    models = bridge.get_model(
        weight_path=model_path,
        wrap_with_ddp=False,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
    )
    models = list(models)
    return models, bridge


def get_gpt_model(models):
    """Unwrap to get the raw GPTModel."""
    from megatron.core.models.gpt.gpt_model import GPTModel
    model = models[0]
    # Unwrap DDP/Float16Module if present
    while hasattr(model, 'module'):
        model = model.module
    assert isinstance(model, GPTModel), f"Expected GPTModel, got {type(model)}"
    return model


def register_hooks(gpt_model):
    """Register forward hooks to capture per-layer hidden states."""
    layer_outputs = {}
    attn_outputs = {}
    hooks = []

    # Hook on each decoder layer
    for i, layer in enumerate(gpt_model.decoder.layers):
        def make_layer_hook(layer_idx):
            def hook_fn(module, input, output):
                # TransformerLayer output: (hidden_states, context)
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                # Move to CPU and detach to save memory
                layer_outputs[layer_idx] = hs.detach().cpu()
            return hook_fn

        h = layer.register_forward_hook(make_layer_hook(i))
        hooks.append(h)

        # Hook on self_attention to get attention output (before MLP/MoE)
        def make_attn_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                attn_outputs[layer_idx] = attn_out.detach().cpu()
            return hook_fn

        h2 = layer.self_attention.register_forward_hook(make_attn_hook(i))
        hooks.append(h2)

    return layer_outputs, attn_outputs, hooks


def forward_pass(gpt_model, data, device):
    """Run model forward pass and compute loss."""
    input_ids = data["input_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)

    gpt_model.eval()
    with torch.no_grad():
        # GPTModel forward: (input_ids, position_ids, attention_mask, labels=None)
        # When labels is provided, returns loss tensor
        # When labels is None, returns logits
        output = gpt_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Also get logits (forward without labels)
        logits = gpt_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    return output, logits


def compute_loss_from_logits(logits, labels, loss_mask):
    """Compute cross-entropy loss from logits (same as HybridEngine)."""
    # logits: [S, B, vocab] or [B, S, vocab]
    # Ensure [B, S, vocab]
    if logits.dim() == 3 and logits.shape[0] != labels.shape[0]:
        logits = logits.transpose(0, 1)
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    # Cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.view(shift_labels.shape)
    masked_loss = (losses * shift_mask).sum() / shift_mask.sum()
    return masked_loss.item()


def main():
    parser = argparse.ArgumentParser(description="AReaL comparison forward pass")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument("--input", type=str, required=True, help="Input data file from generate_test_data.py")
    parser.add_argument("--output", type=str, required=True, help="Output file to save results")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=8, help="Expert parallel size")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}, TP={args.tp}, PP={args.pp}, EP={args.ep}")

    mpu = init_megatron_parallel(world_size, args.tp, args.pp, args.ep)

    if rank == 0:
        print(f"Loading model from {args.model_path}...")

    models, bridge = load_model(args.model_path, dtype)
    gpt_model = get_gpt_model(models)

    if rank == 0:
        print(f"Model loaded. Decoder layers: {len(gpt_model.decoder.layers)}")
        print(f"Pre_process: {gpt_model.pre_process}, Post_process: {gpt_model.post_process}")

    # Load test data
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    if rank == 0:
        print(f"Loaded test data: input_ids shape={data['input_ids'].shape}")

    # Register hooks
    layer_outputs, attn_outputs, hooks = register_hooks(gpt_model)

    # Run forward pass
    if rank == 0:
        print("Running forward pass...")

    output, logits = forward_pass(gpt_model, data, device)

    # Compute loss
    if gpt_model.post_process:
        if isinstance(logits, torch.Tensor):
            logits_cpu = logits.detach().cpu()
            loss = compute_loss_from_logits(
                logits_cpu,
                data["labels"],
                data["loss_mask"],
            )
        else:
            logits_cpu = None
            loss = None
    else:
        logits_cpu = None
        loss = None

    # Get embedding output
    embedding = None
    if gpt_model.pre_process and hasattr(gpt_model, 'embedding'):
        # Try to get embedding output via hook
        pass  # Will get from layer_outputs[0] input

    # Collect results (only on the last PP rank for logits/loss)
    results = {
        "layer_outputs": {k: v.float() for k, v in layer_outputs.items()},
        "attn_outputs": {k: v.float() for k, v in attn_outputs.items()},
        "logits": logits_cpu.float() if logits_cpu is not None else None,
        "loss": loss,
        "ppl": None,
        "engine": "areal",
        "model_path": args.model_path,
        "parallel": {"tp": args.tp, "pp": args.pp, "ep": args.ep},
        "dtype": args.dtype,
    }

    if loss is not None:
        import math
        results["ppl"] = math.exp(loss) if loss < 100 else float('inf')
        if rank == 0:
            print(f"Loss: {loss:.6f}, PPL: {results['ppl']:.4f}")

    # Save results (only rank 0 or last PP rank)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_last = mpu.get_pipeline_model_parallel_world_size() - 1
    dp_rank = mpu.get_data_parallel_rank()

    if dp_rank == 0:
        save_path = args.output
        if args.pp > 1:
            # Save per-PP rank (each has different layers)
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_pp{pp_rank}{ext}"

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(results, save_path)
        print(f"[Rank {rank}, PP {pp_rank}] Saved results to {save_path}")
        print(f"  Layer outputs: {list(layer_outputs.keys())}")
        print(f"  Attn outputs: {list(attn_outputs.keys())}")
        if loss is not None:
            print(f"  Loss: {loss:.6f}")

    # Clean up hooks
    for h in hooks:
        h.remove()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
