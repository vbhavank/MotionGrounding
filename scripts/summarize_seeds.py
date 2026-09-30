#!/usr/bin/env python3
"""
Mean +- std over seeds, and Welch's t-test.

1) Aggregate numeric leaves of several result jsons (one per seed):
     python scripts/summarize_seeds.py agg seed42.json seed43.json seed44.json [--keys overall_accuracy ...]
2) Compare two runs' per-seed scores with Welch's t-test:
     python scripts/summarize_seeds.py welch --a 35.1 35.6 34.9 --b 34.2 34.8 34.5
3) Welch's t-test from summary statistics (e.g. Fig. S6 token counts):
     python scripts/summarize_seeds.py welch-stats 123 90 100 138 90 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import mcot_metrics as mm  # noqa: E402


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agg")
    a.add_argument("files", nargs="+")
    a.add_argument("--keys", nargs="*", default=None, help="substring filters on flattened keys")
    w = sub.add_parser("welch")
    w.add_argument("--a", nargs="+", type=float, required=True)
    w.add_argument("--b", nargs="+", type=float, required=True)
    ws = sub.add_parser("welch-stats")
    ws.add_argument("m1", type=float); ws.add_argument("s1", type=float); ws.add_argument("n1", type=int)
    ws.add_argument("m2", type=float); ws.add_argument("s2", type=float); ws.add_argument("n2", type=int)
    args = ap.parse_args()

    if args.cmd == "agg":
        per_key = {}
        for f in args.files:
            for k, v in leaves(json.load(open(f))):
                if "per_sample" in k or "predictions" in k:
                    continue
                if args.keys and not any(s in k for s in args.keys):
                    continue
                per_key.setdefault(k, []).append(v)
        for k, vs in sorted(per_key.items()):
            if len(vs) == len(args.files):
                m, s = mm.mean_std(vs)
                print(f"{k:70s} {m:9.4f} +- {s:.4f}  (n={len(vs)})")
    elif args.cmd == "welch":
        m1, s1 = mm.mean_std(args.a)
        m2, s2 = mm.mean_std(args.b)
        print(json.dumps({"a": [m1, s1, len(args.a)], "b": [m2, s2, len(args.b)],
                          **mm.welch_t(m1, s1, len(args.a), m2, s2, len(args.b))}, default=float))
    else:
        print(json.dumps(mm.welch_t(args.m1, args.s1, args.n1, args.m2, args.s2, args.n2), default=float))


if __name__ == "__main__":
    main()
