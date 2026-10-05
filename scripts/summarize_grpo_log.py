"""Table of the per-step GRPO metrics in a training log.

tqdm writes the progress bar with carriage returns on the same lines as the
metric dicts, so the dicts are pulled out by pattern rather than line by line.

    python scripts/summarize_grpo_log.py logs/grpo_v4b_test.log [--cols kl spatial ...]
"""

import argparse
import ast
import re

COLS = {
    "kl": "kl",
    "lr": "learning_rate",
    "spatial": "rewards/thk_spatial_reward",
    "tpoint": "rewards/thk_temporal_point_reward",
    "self": "rewards/motion_self_consistency_reward",
    "traj": "rewards/motion_trajectory_reward_v4",
    "acc": "rewards/ans_acc_reward",
    "fmt": "rewards/format_reward_v4",
    "copied": "grounding/copied_box_rate",
    "len": "completion_length",
    "rstd": "reward_std",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--cols", nargs="+", default=["kl", "lr", "spatial", "self", "traj", "copied", "fmt", "len", "rstd"])
    args = ap.parse_args()
    text = open(args.log, errors="replace").read()
    rows = []
    for m in re.finditer(r"\{'loss':.*?\}", text):
        try:
            rows.append(ast.literal_eval(m.group(0)))
        except (ValueError, SyntaxError):
            continue
    if not rows:
        print("no metric lines found")
        return
    print("epoch  " + " ".join(f"{c:>9s}" for c in args.cols))
    for d in rows:
        vals = []
        for c in args.cols:
            v = d.get(COLS.get(c, c))
            try:
                v = float(v)
                vals.append(f"{v:9.2e}" if c in ("kl", "lr") else f"{v:9.3f}" if abs(v) < 100 else f"{v:9.0f}")
            except (TypeError, ValueError):
                vals.append(f"{'-':>9s}")
        print(f"{float(d.get('epoch', 'nan')):.3f}  " + " ".join(vals))
    k = max(1, len(rows) // 3)
    print("\nmean of first vs last third:")
    for c in args.cols:
        key = COLS.get(c, c)
        a = [float(d[key]) for d in rows[:k] if key in d]
        b = [float(d[key]) for d in rows[-k:] if key in d]
        if a and b:
            print(f"  {c:8s} {sum(a) / len(a):10.4g} -> {sum(b) / len(b):10.4g}")


if __name__ == "__main__":
    main()
