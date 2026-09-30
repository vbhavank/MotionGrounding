#!/usr/bin/env python3
"""
Relabel MCoT data with v4 rules.

* Tags only objects with >= 2 annotated timestamps (the paper's rule). The v2
  augmenter defaults to ``tag_single_frame=True``, which writes STAT /
  stationary / stable for every single-frame object; this inflates the STAT
  class in Fig. S5 and teaches the model to emit tags without motion evidence
  (the seashell case in Fig. S8).
* ``--piecewise``: split tracks at heading reversals (motion_core.segment_track)
  and emit one tag per segment with ``from``/``to``.
* Recomputes ``gt_motion`` from key_items with the same >= 2 rule.
* ``--report`` prints the class distribution before/after, which is what the
  majority-class baseline should be computed from.

    python scripts/relabel_motion_v4.py --input STGR-SFT-motion-mixed.json \
        --output STGR-SFT-motion-v4.json [--piecewise] --report
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402

TAG_V2 = re.compile(r"\s*<motion\s+[^<>]*?/>")
ELIGIBLE = {"temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"}


def fmt_tag(obj, d, seg=None):
    s = f'<motion obj="{obj}" dir="{d["dir"]}" speed="{d["speed"]}" scale="{d["scale"]}"'
    if seg is not None:
        s += f' from="{seg[0]:g}" to="{seg[1]:g}"'
    return s + "/>"


def insertion_point(text, obj, t_max=None):
    """End of the last ``<obj>obj</obj>...at<t>t</t>s`` with t <= t_max."""
    best = None
    for m in re.finditer(rf"<obj>{re.escape(obj)}</obj>(?:<box>\[.*?\]</box>)+at<t>([\d.]+)</t>s", text):
        if t_max is None or float(m.group(1)) <= t_max + 1e-6:
            best = m.end()
    return best


def relabel(sample, piecewise=False, min_obs=2):
    out = dict(sample)
    if sample.get("task") not in ELIGIBLE:
        return out, []
    tracks = mc.tracks_from_key_items(sample.get("key_items") or {}, sample.get("key_frames") or [])
    labels = {}
    for name, tr in tracks.items():
        if len(tr) >= min_obs:
            labels[name] = mc.motion_descriptor(tr)
    out["gt_motion"] = labels or None
    emitted = []
    r = sample.get("reasoning_process")
    if r:
        r = TAG_V2.sub("", r)
        for name, tr in tracks.items():
            if name not in labels:
                continue
            segs = mc.segment_track(tr) if piecewise else [(tr[0][0], tr[-1][0])]
            # insert from the last segment backwards so offsets stay valid
            for seg in reversed(segs):
                d = mc.motion_descriptor(mc.restrict_track(tr, *seg)) if len(segs) > 1 else labels[name]
                pos = insertion_point(r, name, seg[1] if len(segs) > 1 else None)
                if pos is None:
                    continue
                r = r[:pos] + fmt_tag(name, d, seg if len(segs) > 1 else None) + r[pos:]
                emitted.append(d)
        out["reasoning_process"] = r
    return out, emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--min-observations", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.input))
    before = Counter()
    after = Counter()
    new = []
    for s in data:
        for m in re.finditer(r'<motion\s+[^<>]*?dir="([^"]+)"\s+speed="([^"]+)"\s+scale="([^"]+)"', s.get("reasoning_process") or ""):
            before.update([f"dir={m.group(1)}", f"speed={m.group(2)}", f"scale={m.group(3)}"])
        o, emitted = relabel(s, args.piecewise, args.min_observations)
        for d in emitted:
            after.update([f"dir={d['dir']}", f"speed={d['speed']}", f"scale={d['scale']}"])
        new.append(o)
    json.dump(new, open(args.output, "w"), indent=2)
    print(f"wrote {len(new)} samples -> {args.output}")
    if args.report:
        for attr in ("dir", "speed", "scale"):
            for name, c in (("before", before), ("after", after)):
                row = {k.split("=")[1]: v for k, v in c.items() if k.startswith(attr + "=")}
                tot = sum(row.values()) or 1
                maj = max(row.items(), key=lambda x: x[1]) if row else ("-", 0)
                print(f"{attr:6s} {name:6s} n={tot:6d} majority={maj[0]} ({100 * maj[1] / tot:.1f}%) {dict(sorted(row.items()))}")


if __name__ == "__main__":
    main()
