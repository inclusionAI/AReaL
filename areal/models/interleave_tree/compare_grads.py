import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-grad", type=str, required=True,
                        help="Path to baseline gradient file")
    parser.add_argument("--exp-grad", type=str, required=True,
                        help="Path to experimental gradient file")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file to save the comparison table")
    args = parser.parse_args()

    # -------- load gradients --------
    base_grads = torch.load(args.baseline_grad, map_location="cpu")
    exp_grads = torch.load(args.exp_grad, map_location="cpu")

    base_keys = set(base_grads.keys())
    exp_keys = set(exp_grads.keys())

    common_keys = sorted(base_keys & exp_keys)
    missing_in_exp = base_keys - exp_keys
    missing_in_base = exp_keys - base_keys

    lines = []

    if missing_in_exp:
        lines.append(f"[Warning] Missing in exp-grad ({len(missing_in_exp)}):")
        for k in sorted(missing_in_exp):
            lines.append(f"  {k}")

    if missing_in_base:
        lines.append(f"[Warning] Missing in baseline-grad ({len(missing_in_base)}):")
        for k in sorted(missing_in_base):
            lines.append(f"  {k}")

    lines.append(f"\nComparing {len(common_keys)} common parameters\n")

    # -------- per-parameter comparison --------
    results = []
    eps = 1e-6
    for name in common_keys:
        g0 = base_grads[name]
        g1 = exp_grads[name]

        if g0 is None or g1 is None:
            continue

        g0 = g0.float()
        g1 = g1.float()

        base_norm = torch.norm(g0)
        diff_norm = torch.norm(g1 - g0)

        ratio = diff_norm / (base_norm + eps)

        results.append((name, ratio.item(), base_norm.item()))

    # -------- sort by ratio (descending) --------
    results.sort(key=lambda x: x[1], reverse=True)

    # -------- format output --------
    header = f"{'Parameter':60s} {'|Δg|/|g|':>12s} {'|g_baseline|':>12s}"
    sep = "-" * 104
    lines.append(header)
    lines.append(sep)

    for name, ratio, base_norm in results:
        lines.append(
            f"{name:60s} "
            f"{ratio:12.4e} "
            f"{base_norm:12.4e}"
        )

    output = "\n".join(lines)

    # -------- write to file if needed --------
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\n[Saved] comparison table written to {args.out}")


if __name__ == "__main__":
    main()
