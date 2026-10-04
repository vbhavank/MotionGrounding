"""Restore the keyframe timestamps that STGR-RL rows leave out.

An STGR-RL row lists a single entry in ``key_frames`` (e.g. idx 2 at 3.5 s)
while ``key_items`` keeps the boxes of the earlier keyframes ("0", "1", "2").
Without their timestamps every GT track has one observation, so M(T_gt) is
undefined and r_traj / the equivariance targets are 0 for every rollout.

Keyframe images are named ``<sample>_<idx>_time_<seconds>.jpg``. Missing
``(sample, idx)`` timestamps are recovered from (a) the ``key_frames`` of every
row in the given json files and (b) the image files in the keyframe folders.
Only entries whose image exists are added (the trainer inserts them into the
frame sequence). ``gt_motion`` is recomputed from the repaired tracks.

    python scripts/repair_rl_keyframes.py \
        --kf_roots $OO3/videos/stgr/plm/kfs $OO3/videos/stgr/temporal_grounding/kfs \
        --index_json $J/splits_v4/sft_v4.json $J/splits_v4/rl_v4.json \
        --inputs $J/splits_v4/rl_v4_sub800.json $J/splits_v4/rl_v4_notag_sub800.json
    -> *_kf.json next to each input
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402

KF_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)_time_(?P<time>\d+(?:\.\d+)?)\.(?:jpg|jpeg|png)$", re.I)


def parse_name(path):
    m = KF_RE.match(os.path.basename(path or ""))
    return (m.group("prefix"), int(m.group("idx")), float(m.group("time"))) if m else None


def scan_roots(roots):
    index, exists = {}, set()
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = parse_name(f)
                if p:
                    rel = os.path.relpath(os.path.join(dirpath, f), root)
                    index.setdefault((p[0], p[1]), (p[2], rel))
                    exists.add(rel)
    return index, exists


def max_obs(row):
    tr = mc.tracks_from_key_items(row.get("key_items") or {}, row.get("key_frames") or [])
    return max((len(v) for v in tr.values()), default=0)


def repair(row, index, exists):
    kfs = list(row.get("key_frames") or [])
    if not kfs or not row.get("key_items"):
        return row, 0
    prefixes = {p[0] for p in (parse_name(k.get("path")) for k in kfs) if p}
    if len(prefixes) != 1:
        return row, 0
    prefix = prefixes.pop()
    have = {str(k["idx"]) for k in kfs}
    added = 0
    for key in row["key_items"]:
        if key in have or not str(key).isdigit():
            continue
        hit = index.get((prefix, int(key)))
        if hit is None or (exists and hit[1] not in exists):
            continue
        kfs.append({"idx": int(key), "time": hit[0], "path": hit[1]})
        added += 1
    if not added:
        return row, 0
    out = dict(row)
    out["key_frames"] = sorted(kfs, key=lambda k: float(k["time"]))
    tracks = mc.tracks_from_key_items(out["key_items"], out["key_frames"])
    if "gt_motion" in row:  # absent in *_notag files
        out["gt_motion"] = {n: mc.motion_descriptor(t) for n, t in tracks.items() if len(t) >= 2} or None
    return out, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf_roots", nargs="+", required=True)
    ap.add_argument("--index_json", nargs="*", default=[], help="extra rows whose key_frames give timestamps")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--suffix", default="_kf")
    args = ap.parse_args()

    index, exists = scan_roots(args.kf_roots)
    print(f"keyframe images indexed: {len(exists)}")
    for path in args.index_json + args.inputs:
        for r in json.load(open(path)):
            for k in r.get("key_frames") or []:
                p = parse_name(k.get("path"))
                if p:
                    index.setdefault((p[0], int(k["idx"])), (float(k["time"]), k["path"]))
    print(f"(sample, idx) timestamps known: {len(index)}")

    for path in args.inputs:
        rows = json.load(open(path))
        before, after, n_added = collections.Counter(), collections.Counter(), 0
        out = []
        for r in rows:
            r2, a = repair(r, index, exists)
            n_added += a
            if r.get("task") == "temporal-spatial free-form QA":
                src = r.get("source")
                before[src] += max_obs(r) >= 2
                after[src] += max_obs(r2) >= 2
            out.append(r2)
        dst = re.sub(r"\.json$", f"{args.suffix}.json", path)
        json.dump(out, open(dst, "w"), indent=1)
        print(f"\n{path}\n  keyframes added: {n_added} -> {dst}")
        print("  rows with a GT object at >= 2 keyframes (before -> after):")
        for s in sorted(set(before) | set(after)):
            print(f"    {s}: {before[s]} -> {after[s]}")


if __name__ == "__main__":
    main()
