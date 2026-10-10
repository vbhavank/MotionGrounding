#!/usr/bin/env python3
"""Can the model turn boxes into a motion tag? (box -> tag probe)

For every GT object with >= 2 keyframes, the assistant turn is pre-filled with
that object's GT grounded claims (pixel boxes of the frames the model sees, as
in the SFT targets) followed by the opening of its tag:

    <think><obj>man</obj><box>[..]</box>at<t>2.0</t>s <obj>man</obj><box>[..]</box>at<t>4.0</t>s<motion obj="man" dir="

and the model writes the rest of the tag (greedy, a few tokens). The label is
M(those boxes), i.e. the GT motion. Accuracy here is the box -> tag rule alone,
with perception taken out; compare with the majority baseline and between
models. ``--shuffle_boxes`` pairs each object with another object's boxes:
if accuracy does not drop, the tag ignores the boxes.

    python scripts/probe_box_to_tag.py --backend hf --model_path M \\
        --dataset_json .../eval_heldout_clean.json --max_samples 200 \\
        --output_file results/probe_<name>.json
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
import motion_core as mc  # noqa: E402
import mcot_metrics as mm  # noqa: E402

ATTR_RE = {a: re.compile(rf'{a}="([^"]*)"') for a in ("dir", "speed", "scale")}


def claims_text(name, track, size):
    parts = []
    for t, b in track:
        px = mc.to_pixels(b, size)
        parts.append(f"<obj>{name}</obj><box>[{','.join(str(int(round(v))) for v in px)}]</box>at<t>{t:.1f}</t>s")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    from mcot_eval_common import add_model_args
    add_model_args(ap)
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--shuffle_boxes", action="store_true", help="control: boxes from another object")
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()
    args.max_tokens = 24

    from mcot_eval_common import MOTION_SYSTEM_PROMPT, build_prompt, clip_size, generate, load_clip, make_llm
    data = [s for s in json.load(open(args.dataset_json)) if os.path.isfile(s.get("video_path_full", ""))]
    items = []
    for s in data:
        tracks = {n: tr for n, tr in mc.tracks_from_key_items(s.get("key_items") or {},
                                                              s.get("key_frames") or []).items() if len(tr) >= 2}
        if tracks:
            items.append((s, tracks))
    items = items[: args.max_samples] if args.max_samples else items
    pool = [tr for _, tracks in items for tr in tracks.values()]
    rng = random.Random(args.seed)

    llm, sp, processor = make_llm(args)
    jobs, meta = [], []
    for s, tracks in items:
        try:
            clip = load_clip(args, s["video_path_full"], s)
        except Exception as e:
            print(f"skip {s.get('id')}: {e}")
            continue
        size = clip_size(clip)
        for name, tr in tracks.items():
            shown = rng.choice(pool) if args.shuffle_boxes else tr
            prefix = "<think>" + claims_text(name, shown, size) + f'<motion obj="{name}" dir="'
            jobs.append((build_prompt(processor, MOTION_SYSTEM_PROMPT, s["question"], prefix, clip=clip), clip))
            target = mc.motion_descriptor([(t, mc.to_pixels(b, size)) for t, b in shown])
            meta.append({"id": s["id"], "obj": name, "target": target})
    texts = generate(llm, sp, jobs, args.batch_size)

    rows = []
    for m, cont in zip(meta, texts):
        tag = 'dir="' + (cont or "")
        pred = {a: (ATTR_RE[a].search(tag).group(1) if ATTR_RE[a].search(tag) else None) for a in ATTR_RE}
        rows.append({**m, "pred": pred, "continuation": cont})
    metrics = {"n": len(rows), "shuffle_boxes": args.shuffle_boxes}
    for a in ("dir", "speed", "scale"):
        metrics[a] = mm.classification_report([(r["pred"][a], r["target"][a]) for r in rows], a)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    json.dump({"args": vars(args), "metrics": metrics, "rows": rows}, open(args.output_file, "w"), indent=2, default=str)
    for a in ("dir", "speed", "scale"):
        r = metrics[a]
        print(f"{a:5s} n={r.get('n')} exact {r.get('exact', float('nan')):.3f} | BAcc {r.get('balanced_acc', float('nan')):.3f}"
              f" | majority {r.get('majority_baseline_acc', float('nan')):.3f} ({r.get('majority_class')})")


if __name__ == "__main__":
    main()
