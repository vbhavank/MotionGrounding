#!/usr/bin/env python3
"""
Paired accuracy on reversal-contrast pairs.

    PA = 1/n sum 1[f(V_j) = A_j  and  f(V_j^rev) = A_j^rev]

A model that answers from static appearance gives the same answer to both
clips, so its PA is exactly 0; uniform guessing on binary pairs gives 25%.
Also reports the tag-level reversal check d_hat(V_rev) = d_hat(V) + 180 deg,
c_hat(V_rev) = flip(c_hat(V)) for the queried object (rho_reverse).

    python scripts/eval_reversal_pairs.py --model_path M --pairs_json pairs.json \
        --prompt mcot --output_file pa.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
import motion_core as mc  # noqa: E402
import mcot_metrics as mm  # noqa: E402


def question(p, t0, t1):
    q = p["template"].format(t0=f"{t0:.1f}", t1=f"{t1:.1f}", obj=p["obj"])
    return q + "\nOptions:\n" + "\n".join(p["options"]) + \
        "\nOutput only the option letter inside <answer> </answer>."


def obj_tag(text, obj):
    key = mc.normalize_obj_name(obj)
    tags = [t for t in mc.parse_tags(text) if t["well_formed"] and (t["obj"] == key or key in t["obj"])]
    return tags[-1] if tags else None


def main():
    ap = argparse.ArgumentParser()
    from mcot_eval_common import add_model_args
    add_model_args(ap)
    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--prompt", choices=["mcot", "letter"], default="mcot")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    from mcot_eval_common import (LETTER_SYSTEM_PROMPT, MOTION_SYSTEM_PROMPT, build_prompt, extract_letter,
                                  generate, load_video, make_llm)
    system = MOTION_SYSTEM_PROMPT if args.prompt == "mcot" else LETTER_SYSTEM_PROMPT
    pairs = json.load(open(args.pairs_json))
    pairs = pairs[: args.max_pairs] if args.max_pairs else pairs
    llm, sp, processor = make_llm(args)

    jobs, kept = [], []
    for p in pairs:
        try:
            v, fps = load_video(p["video_path_full"], args.video_max_pixels, args.video_max_frames)
        except Exception as e:
            print(f"skip {p['id']}: {e}")
            continue
        duration = len(v) / fps
        v_rev = v[::-1].copy()
        jobs.append((build_prompt(processor, system, question(p, p["t0"], p["t1"])), v))
        jobs.append((build_prompt(processor, system, question(p, duration - p["t1"], duration - p["t0"])), v_rev))
        kept.append(p)
    texts = generate(llm, sp, jobs, args.batch_size)

    rows, tag_checks = [], []
    for i, p in enumerate(kept):
        fwd, rev = texts[2 * i], texts[2 * i + 1]
        n_opt = len(p["options"])
        row = {"id": p["id"], "kind": p["kind"], "gt": p["answer"], "gt_rev": p["answer_rev"],
               "pred": extract_letter(fwd, n_opt), "pred_rev": extract_letter(rev, n_opt),
               "text": fwd, "text_rev": rev}
        tf, tr = obj_tag(fwd, p["obj"]), obj_tag(rev, p["obj"])
        if tf and tr:
            row["tag_reversal_consistency"] = mc.rho_consistency("reverse", tf, tr)
            tag_checks.append(row["tag_reversal_consistency"])
        rows.append(row)

    summary = {"all": mm.paired_accuracy([{"pred": r["pred"], "gt": r["gt"], "pred_rev": r["pred_rev"],
                                           "gt_rev": r["gt_rev"]} for r in rows])}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        summary[kind] = mm.paired_accuracy([{"pred": r["pred"], "gt": r["gt"], "pred_rev": r["pred_rev"],
                                             "gt_rev": r["gt_rev"]} for r in sub])
    summary["tag_reversal_consistency"] = sum(tag_checks) / len(tag_checks) if tag_checks else None
    summary["n_tag_pairs"] = len(tag_checks)
    summary["chance_PA_binary"] = 0.25
    json.dump({"summary": summary, "args": vars(args), "rows": rows}, open(args.output_file, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
