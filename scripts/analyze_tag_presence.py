#!/usr/bin/env python3
"""
Tag presence under benchmark prompts (offline, no GPU).

    rho_tag = P(output contains >= 1 well-formed <motion/> tag)
    P(correct | tag)  vs  P(correct | no tag)   (+ two-proportion z-test)

Reads the result json written by scripts/eval_{mvbench,motionbench,tvbench}.py
(they now store the full generated text under "predictions"), or any json list
of {"pred_text", "correct"}. If rho_tag ~ 0 under the letter-only prompts of
Appendix F.4, benchmark gains cannot come from MCoT at inference time.

    python scripts/analyze_tag_presence.py results/mvbench_*.json [--by subset]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import mcot_metrics as mm  # noqa: E402


def load_preds(path):
    obj = json.load(open(path))
    if isinstance(obj, list):
        return obj
    for key in ("predictions", "per_sample", "predictions_sample"):
        if key in obj:
            if key == "predictions_sample":
                print(f"WARNING {path}: only 'predictions_sample' found (first 100, text truncated); "
                      "re-run the eval script to get full predictions.")
            return obj[key]
    raise ValueError(f"no predictions in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--text_key", default="pred_text")
    ap.add_argument("--correct_key", default="correct")
    ap.add_argument("--by", default=None, help="group key, e.g. subset or question_type")
    args = ap.parse_args()

    for f in args.files:
        preds = load_preds(f)
        tk = args.text_key if any(args.text_key in p for p in preds) else "pred_raw"
        print(f"\n== {f}")
        print(json.dumps(mm.tag_presence(preds, tk, args.correct_key), indent=2))
        if args.by:
            groups = defaultdict(list)
            for p in preds:
                groups[p.get(args.by)].append(p)
            for g, ps in sorted(groups.items(), key=lambda x: str(x[0])):
                r = mm.tag_presence(ps, tk, args.correct_key)
                print(f"  {g}: rho_tag={r['rho_tag']:.3f} acc|tag={r['acc_given_tag']} "
                      f"acc|no_tag={r['acc_given_no_tag']} n={r['n']}")


if __name__ == "__main__":
    main()
