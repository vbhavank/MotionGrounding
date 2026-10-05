"""Which STGR rows can be trained on, and which media is missing.

For each (source, task) of STGR-SFT.json / STGR-RL.json: number of rows, rows
whose video/image exists (same path rules as grpo_trainer_v4._prepare_sample),
rows whose keyframes also exist (grounded task), and the folder the missing
media is expected in, with the number of distinct missing files.

    python scripts/data_coverage.py --json $J/STGR-SFT.json $J/STGR-RL.json
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def roots(data_root):
    v = os.path.join(data_root, "videos")
    return {
        "gqa": os.path.join(v, "gqa"),
        "timerft": os.path.join(v, "timerft"),
        "tvg": os.path.join(v, "tvg_r1"),
        "vespresso": v,
        "vespresso_kf": os.path.join(v, "videoespresso/kfs"),
        "str": os.path.join(v, "stgr/temporal_grounding/videos"),
        "str_kf": os.path.join(v, "stgr/temporal_grounding/kfs"),
        "plm": os.path.join(v, "stgr/plm/videos"),
        "plm_kf": os.path.join(v, "stgr/plm/kfs"),
        "videor1": os.path.join(v, "videor1"),
    }


def media_path(r, R):
    """(media path, keyframe root or None), mirroring grpo_trainer_v4._prepare_sample."""
    src = r.get("source") or ""
    if src == "videoespresso_train_video":
        return os.path.join(R["vespresso"], r.get("video_path", "")), R["vespresso_kf"]
    if src == "timerft":
        return os.path.join(R["timerft"], r.get("video_path", "")), None
    if src == "gqa":
        return os.path.join(R["gqa"], r.get("image_path", "")), None
    if "STR" in src:
        plm = "STR_plm" in src
        return os.path.join(R["plm" if plm else "str"], r.get("video_path", "")), R["plm_kf" if plm else "str_kf"]
    if "TVG" in src:
        return os.path.join(R["tvg"], r.get("video_path", "")), None
    if "videor1" in src:
        return os.path.join(R["videor1"], r.get("video_path", "")), None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--data_root", default=None, help="default: configs/data_root.py DATA_ROOT")
    args = ap.parse_args()
    if args.data_root is None:
        from configs.data_root import DATA_ROOT
        args.data_root = DATA_ROOT
    R = roots(args.data_root)
    print(f"DATA_ROOT = {args.data_root}")

    exists = {}

    def ok(p):
        if p not in exists:
            exists[p] = os.path.isfile(p)
        return exists[p]

    for path in args.json:
        rows = json.load(open(path))
        stat = collections.defaultdict(collections.Counter)
        missing_dirs = collections.defaultdict(set)
        for r in rows:
            key = (r.get("source"), r.get("task"))
            c = stat[key]
            c["rows"] += 1
            p, kf_root = media_path(r, R)
            if p is None:
                c["unknown source"] += 1
                continue
            if not ok(p):
                missing_dirs[os.path.dirname(os.path.dirname(p)) if "/" in (r.get("video_path") or r.get("image_path") or "")
                             else os.path.dirname(p)].add(p)
                continue
            c["media ok"] += 1
            if r.get("task") == "temporal-spatial free-form QA" and kf_root:
                kfs = r.get("key_frames") or []
                if kfs and all(ok(os.path.join(kf_root, k.get("path", ""))) for k in kfs):
                    c["media+kf ok"] += 1
        tot = collections.Counter()
        print(f"\n{path}: {len(rows)} rows")
        print(f"  {'source':28s} {'task':32s} {'rows':>6s} {'media ok':>9s} {'+kf ok':>7s}")
        for (src, task), c in sorted(stat.items(), key=lambda kv: -kv[1]["rows"]):
            tot.update(c)
            kf = c["media+kf ok"] if task == "temporal-spatial free-form QA" else ""
            print(f"  {str(src)[:28]:28s} {str(task)[:32]:32s} {c['rows']:6d} {c['media ok']:9d} {kf!s:>7s}")
        print(f"  {'TOTAL':61s} {tot['rows']:6d} {tot['media ok']:9d} {tot['media+kf ok']:7d}")
        if missing_dirs:
            print("  missing media, by folder (distinct files):")
            for d, files in sorted(missing_dirs.items(), key=lambda kv: -len(kv[1])):
                print(f"    {len(files):7d}  {d}{'' if os.path.isdir(d) else '   [folder does not exist]'}")


if __name__ == "__main__":
    main()
