"""Find GT tracks that merge two objects under one name, and drop them from eval.

STGR annotates boxes per keyframe by object *name*. When two instances share a
name ("dog", "person"), ``tracks_from_key_items`` joins them into one track that
jumps between them, and M(T_gt) reports fast motion for objects that do not
move (e.g. plm_009912: a seated dog labeled SW/fast/receding because the "dog"
boxes alternate between the dog on the bed and a second dog).

Flags per track (normalized boxes; >= 3 observations for the alternation test):
  alternating : box i and box i+2 overlap (IoU >= 0.3) while box i+1 overlaps
                neither (IoU < 0.05): A-B-A, two instances.
  teleport    : consecutive boxes do not overlap (IoU < 0.05) and the centre moves
                more than --jump of the frame within --jump_dt seconds. Can also
                be real fast motion; reported separately.

    python scripts/audit_gt_tracks.py --inputs $J/splits_v4/eval_heldout.json \
        $J/splits_v4/sft_v4.json $J/splits_v4/rl_v4_sub800_kf.json \
        --clean $J/splits_v4/eval_heldout.json --drop alternating
    -> eval_heldout_clean.json (--suffix): flagged objects removed from key_items (their GT
       label and oracle boxes disappear; other objects in the sample stay)
"""

import argparse
import collections
import copy
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def track_flags(track, jump=0.35, jump_dt=2.0):
    boxes = [b for _, b in track]
    times = [t for t, _ in track]
    flags = set()
    for i in range(len(boxes) - 2):
        if iou(boxes[i], boxes[i + 2]) >= 0.3 and iou(boxes[i], boxes[i + 1]) < 0.05 \
                and iou(boxes[i + 1], boxes[i + 2]) < 0.05:
            flags.add("alternating")
    for i in range(len(boxes) - 1):
        (x0, y0), (x1, y1) = centre(boxes[i]), centre(boxes[i + 1])
        if iou(boxes[i], boxes[i + 1]) < 0.05 and math.hypot(x1 - x0, y1 - y0) > jump \
                and times[i + 1] - times[i] <= jump_dt:
            flags.add("teleport")
    return flags


def audit_rows(rows, args):
    per_src = collections.defaultdict(collections.Counter)
    flagged = {}
    for r in rows:
        items, frames = r.get("key_items") or {}, r.get("key_frames") or []
        if not items or not frames:
            continue
        tracks = mc.tracks_from_key_items(items, frames)
        labels = mc.gt_motion_from_key_items(items, frames)
        src = r.get("source")
        for name, tr in tracks.items():
            if len(tr) < 2:
                continue
            boxes_ok = all(mc.is_normalized(b) for _, b in tr)
            fl = track_flags(tr, args.jump, args.jump_dt) if boxes_ok else set()
            c = per_src[src]
            c["tracks"] += 1
            c["moving label"] += labels.get(name, {}).get("dir", "STAT") != "STAT"
            for f in fl:
                c[f] += 1
            if fl:
                c["any flag"] += 1
                c["flagged & moving label"] += labels.get(name, {}).get("dir", "STAT") != "STAT"
                flagged.setdefault(str(r.get("id")), {})[name] = sorted(fl)
    return per_src, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--clean", nargs="*", default=[], help="files to write *_clean.json for")
    ap.add_argument("--drop", choices=["alternating", "any"], default="alternating")
    ap.add_argument("--suffix", default="_clean", help="output name: <input><suffix>.json")
    ap.add_argument("--jump", type=float, default=0.35)
    ap.add_argument("--jump_dt", type=float, default=2.0)
    ap.add_argument("--show", type=int, default=3, help="print this many flagged examples per file")
    args = ap.parse_args()

    for path in args.inputs:
        rows = json.load(open(path))
        per_src, flagged = audit_rows(rows, args)
        print(f"\n{path}")
        tot = collections.Counter()
        for src, c in sorted(per_src.items()):
            tot.update(c)
            print(f"  {src:18s} tracks {c['tracks']:5d} | alternating {c['alternating']:4d} | "
                  f"teleport {c['teleport']:4d} | any {c['any flag']:4d} | "
                  f"moving labels {c['moving label']:4d}, of which flagged {c['flagged & moving label']:4d}")
        if tot["tracks"]:
            print(f"  TOTAL tracks {tot['tracks']}: alternating {tot['alternating']} "
                  f"({100 * tot['alternating'] / tot['tracks']:.1f}%), any flag {tot['any flag']} "
                  f"({100 * tot['any flag'] / tot['tracks']:.1f}%); "
                  f"{tot['flagged & moving label']} of {tot['moving label']} moving labels are flagged")
        for rid, objs in list(flagged.items())[:args.show]:
            print(f"    e.g. id {rid}: {objs}")

        if path in args.clean:
            out, dropped = [], 0
            for r in rows:
                objs = flagged.get(str(r.get("id")), {})
                kill = {n for n, fl in objs.items() if args.drop == "any" or "alternating" in fl}
                if kill:
                    r = copy.deepcopy(r)
                    r["key_items"] = {k: {n: b for n, b in (v or {}).items() if n not in kill}
                                      for k, v in r["key_items"].items()}
                    dropped += len(kill)
                out.append(r)
            dst = re.sub(r"\.json$", f"{args.suffix}.json", path)
            json.dump(out, open(dst, "w"), indent=1)
            print(f"  -> {dst}: removed {dropped} flagged objects ({args.drop})")


if __name__ == "__main__":
    main()
