#!/usr/bin/env python3
"""
Build reversal-contrast pairs (V, q, A) / (V_rev, q_rev, A_rev) with A_rev != A.

Questions are generated from dense GT tracks (M(T_gt)) so the answer under
time reversal is known exactly (rho_reverse): left<->right, up<->down,
closer<->farther, entering<->leaving. Objects that are STAT / stable on the
queried axis are skipped because their answer does not flip. Option order is
shuffled per pair so a letter prior cannot score.

    python scripts/build_reversal_pairs.py --dataset_json EVAL.json --output pairs.json \
        [--exclude_json TRAIN.json] [--max_per_video 2]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402

H_DIR = {"E": "right", "NE": "right", "SE": "right", "W": "left", "NW": "left", "SW": "left"}
V_DIR = {"N": "up", "NE": "up", "NW": "up", "S": "down", "SE": "down", "SW": "down"}
FLIP = {"left": "right", "right": "left", "up": "down", "down": "up", "closer": "farther",
        "farther": "closer", "entering": "leaving", "leaving": "entering"}
TEMPLATES = {
    "horizontal": ("Between {t0}s and {t1}s, does the {obj} move toward the left or the right side of the frame?",
                   ["left", "right"]),
    "vertical": ("Between {t0}s and {t1}s, does the {obj} move up or down in the frame?", ["up", "down"]),
    "depth": ("Between {t0}s and {t1}s, does the {obj} get closer to the camera or farther from it?",
              ["closer", "farther"]),
    "enter_leave": ("Between {t0}s and {t1}s, is the {obj} entering or leaving the frame?",
                    ["entering", "leaving"]),
}


def at_edge(b, eps=0.02):
    return b[0] <= eps or b[1] <= eps or b[2] >= 1 - eps or b[3] >= 1 - eps


def inside(b, eps=0.05):
    return b[0] > eps and b[1] > eps and b[2] < 1 - eps and b[3] < 1 - eps


def candidates(track):
    d = mc.motion_descriptor(track)
    out = []
    if d["dir"] in H_DIR:
        out.append(("horizontal", H_DIR[d["dir"]]))
    if d["dir"] in V_DIR:
        out.append(("vertical", V_DIR[d["dir"]]))
    if d["scale"] == "approaching":
        out.append(("depth", "closer"))
    elif d["scale"] == "receding":
        out.append(("depth", "farther"))
    first, last = track[0][1], track[-1][1]
    if mc.is_normalized(first) and mc.is_normalized(last):
        if at_edge(first) and inside(last):
            out.append(("enter_leave", "entering"))
        elif inside(first) and at_edge(last):
            out.append(("enter_leave", "leaving"))
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude_json", default=None)
    ap.add_argument("--max_per_video", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    exclude = {s.get("id") for s in json.load(open(args.exclude_json))} if args.exclude_json else set()
    pairs = []
    for s in json.load(open(args.dataset_json)):
        if s.get("id") in exclude or not s.get("key_items"):
            continue
        tracks = mc.tracks_from_key_items(s["key_items"], s.get("key_frames") or [])
        made = 0
        for obj, tr in tracks.items():
            if len(tr) < 2 or made >= args.max_per_video:
                continue
            desc, cands = candidates(tr)
            for kind, ans in cands:
                if made >= args.max_per_video:
                    break
                template, opts = TEMPLATES[kind]
                opts = opts[:]
                rng.shuffle(opts)
                letters = [chr(ord("A") + i) for i in range(len(opts))]
                pairs.append({
                    "id": f"{s.get('id')}::{obj}::{kind}",
                    "video_path_full": s.get("video_path_full"),
                    "obj": obj, "kind": kind, "template": template,
                    "t0": tr[0][0], "t1": tr[-1][0],
                    "options": [f"({L}) {o}" for L, o in zip(letters, opts)],
                    "answer": letters[opts.index(ans)],
                    "answer_rev": letters[opts.index(FLIP[ans])],
                    "gt_motion": desc,
                })
                made += 1
    json.dump(pairs, open(args.output, "w"), indent=2)
    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"{len(pairs)} pairs -> {args.output}  {kinds}")


if __name__ == "__main__":
    main()
