"""Compare evaluation results between trajectory and direct inference.

Uses trajectory results as primary (only compares samples that exist in trajectory).

Usage:
    python -m geo_edit.evaluation.compare_eval_results \
        --traj_eval /path/to/trajectory_eval \
        --direct_eval /path/to/direct_eval
"""

import argparse
import json
import os


def load_eval_results(path: str) -> dict:
    """Load eval_result.jsonl from path or directory."""
    if os.path.isdir(path):
        path = os.path.join(path, "eval_result.jsonl")
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                results[str(record["id"])] = record
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare trajectory vs direct eval results.")
    parser.add_argument("--traj_eval", type=str, required=True, help="Path to trajectory eval results.")
    parser.add_argument("--direct_eval", type=str, required=True, help="Path to direct eval results.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    args = parser.parse_args()

    # Load results
    print(f"Loading trajectory: {args.traj_eval}")
    traj = load_eval_results(args.traj_eval)
    print(f"  {len(traj)} samples")

    print(f"Loading direct: {args.direct_eval}")
    direct = load_eval_results(args.direct_eval)
    print(f"  {len(direct)} samples")

    # Compare using trajectory IDs as primary
    both_correct = both_wrong = traj_only = direct_only = 0
    traj_correct = direct_correct = 0
    missing_in_direct = 0

    for sid, t in traj.items():
        if sid not in direct:
            missing_in_direct += 1
            continue

        t_ok = t.get("result") == 1.0
        d_ok = direct[sid].get("result") == 1.0

        traj_correct += t_ok
        direct_correct += d_ok

        if t_ok and d_ok:
            both_correct += 1
        elif not t_ok and not d_ok:
            both_wrong += 1
        elif t_ok:
            traj_only += 1
        else:
            direct_only += 1

    total = both_correct + both_wrong + traj_only + direct_only
    if total == 0:
        print("\nNo common samples!")
        return

    t_acc = traj_correct / total
    d_acc = direct_correct / total
    diff = t_acc - d_acc

    # Print results
    print("\n" + "=" * 50)
    print("COMPARISON (trajectory as primary)")
    print("=" * 50)
    print(f"Common samples: {total}")
    if missing_in_direct:
        print(f"  (Missing in direct: {missing_in_direct})")
    print(f"trajectory: {traj_correct}/{total} ({t_acc:.4f})")
    print(f"direct: {direct_correct}/{total} ({d_acc:.4f})")
    print("-" * 50)
    print(f"Diff (traj - direct): {diff:+.4f}")
    print(f"Both correct: {both_correct}")
    print(f"Both wrong: {both_wrong}")
    print(f"Trajectory only: {traj_only}")
    print(f"Direct only: {direct_only}")
    print("=" * 50)

    # Save if requested
    if args.output:
        report = {
            "total": total,
            "trajectory": {"correct": traj_correct, "accuracy": t_acc},
            "direct": {"correct": direct_correct, "accuracy": d_acc},
            "diff": diff,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "traj_only": traj_only,
            "direct_only": direct_only,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
