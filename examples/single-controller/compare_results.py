"""Compare AReaL vs HybridEngine outputs offline.

Usage:
    python compare_results.py \
        --areal /tmp/comparison/areal_outputs.pt \
        --hybrid /tmp/comparison/hybrid_outputs.pt
"""

import argparse
import math

import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor):
    """Compare two tensors and print metrics."""
    if a is None or b is None:
        print(f"  {name}: SKIPPED (one side is None)")
        return

    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={a.shape} b={b.shape}")
        return

    diff = (a.float() - b.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    l2_norm = diff.norm().item()
    rel_l2 = l2_norm / (a.float().norm().item() + 1e-12)
    cos = cosine_sim(a, b)

    status = "OK" if max_abs < 1e-2 else ("WARN" if max_abs < 1e-1 else "FAIL")

    print(f"  {name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, "
          f"rel_L2={rel_l2:.6e}, cosine={cos:.8f} [{status}]")


def main():
    parser = argparse.ArgumentParser(description="Compare AReaL vs HybridEngine outputs")
    parser.add_argument("--areal", type=str, required=True, help="AReaL outputs file")
    parser.add_argument("--hybrid", type=str, required=True, help="HybridEngine outputs file")
    args = parser.parse_args()

    print("Loading outputs...")
    areal = torch.load(args.areal, map_location="cpu", weights_only=False)
    hybrid = torch.load(args.hybrid, map_location="cpu", weights_only=False)

    print("\n" + "=" * 70)
    print("AReaL vs HybridEngine Comparison Results")
    print("=" * 70)

    # 1. Loss and PPL comparison
    print("\n--- End-to-End Metrics ---")
    a_loss = areal.get("loss")
    h_loss = hybrid.get("loss")
    if a_loss is not None and h_loss is not None:
        loss_diff = abs(a_loss - h_loss)
        loss_rel = loss_diff / (abs(h_loss) + 1e-12)
        a_ppl = math.exp(a_loss) if a_loss < 100 else float('inf')
        h_ppl = math.exp(h_loss) if h_loss < 100 else float('inf')
        ppl_diff = abs(a_ppl - h_ppl)
        ppl_rel = ppl_diff / (h_ppl + 1e-12)
        print(f"  AReaL  loss={a_loss:.6f}  ppl={a_ppl:.4f}")
        print(f"  Hybrid loss={h_loss:.6f}  ppl={h_ppl:.4f}")
        print(f"  Loss diff: abs={loss_diff:.6e}, rel={loss_rel:.6e}")
        print(f"  PPL  diff: abs={ppl_diff:.6e}, rel={ppl_rel:.6e}")
        status = "PASS" if loss_rel < 0.01 else "FAIL"
        print(f"  Status: {status} (threshold: <1% relative difference)")

    # 2. Logits comparison
    print("\n--- Logits ---")
    a_logits = areal.get("logits")
    h_logits = hybrid.get("logits")
    if a_logits is not None and h_logits is not None:
        compare_tensors("logits", a_logits, h_logits)
    else:
        print("  Logits not available on one or both sides")

    # 3. Per-layer hidden states comparison
    a_layers = areal.get("layer_outputs", {})
    h_layers = hybrid.get("layer_outputs", {})

    if a_layers and h_layers:
        all_layer_ids = sorted(set(list(a_layers.keys()) + list(h_layers.keys())))
        print(f"\n--- Per-Layer Hidden States ({len(all_layer_ids)} layers) ---")
        for layer_id in all_layer_ids:
            a_hs = a_layers.get(layer_id)
            h_hs = h_layers.get(layer_id)
            compare_tensors(f"layer_{layer_id}", a_hs, h_hs)
    else:
        print("\n--- Per-Layer Hidden States ---")
        print("  Layer outputs not available on one or both sides")

    # 4. Attention-only outputs (before MLP/MoE)
    a_attn = areal.get("attn_outputs", {})
    h_attn = hybrid.get("attn_outputs", {})

    if a_attn and h_attn:
        all_layer_ids = sorted(set(list(a_attn.keys()) + list(h_attn.keys())))
        print(f"\n--- Per-Layer Attention Outputs ({len(all_layer_ids)} layers) ---")
        for layer_id in all_layer_ids:
            a_out = a_attn.get(layer_id)
            h_out = h_attn.get(layer_id)
            compare_tensors(f"attn_{layer_id}", a_out, h_out)

    # 5. Embedding comparison
    print("\n--- Embedding ---")
    compare_tensors("embedding", areal.get("embedding"), hybrid.get("embedding"))

    # 6. Final layernorm comparison
    print("\n--- Final LayerNorm ---")
    compare_tensors("final_layernorm", areal.get("final_layernorm"), hybrid.get("final_layernorm"))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  AReaL output keys: {list(areal.keys())}")
    print(f"  Hybrid output keys: {list(hybrid.keys())}")
    if a_loss is not None and h_loss is not None:
        print(f"  Loss match: {'YES' if abs(a_loss - h_loss) / (abs(h_loss) + 1e-12) < 0.01 else 'NO'}")


if __name__ == "__main__":
    main()
