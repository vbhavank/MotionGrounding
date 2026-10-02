#!/usr/bin/env python3
"""
Build leakage-free SFT / RL / held-out files for MCoT v4 from what is on disk.

* PLM rows in the SFT file are replaced by their dense versions (same ids).
* A held-out set of VIDEOS (not sample ids) is drawn from the grounded
  temporal-spatial rows, proportionally per source; every SFT and RL row whose
  video is held out is removed, so no held-out video is ever trained on.
* Held-out rows keep only samples with at least one object observed at >= 2
  timestamps (a motion target exists); the dense version is used when present.
* Writes the no-tag control copies alongside.

    python scripts/build_v4_splits.py --sft STGR-SFT-v4.json --plm_dense PLM-dense-v4.json \
        --rl STGR-RL-v4.json --out_dir splits_v4 --heldout_videos 400
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_core as mc  # noqa: E402
from strip_motion_tags import strip_tags  # noqa: E402

GROUNDED = "temporal-spatial free-form QA"


def has_motion_target(r):
    return bool(mc.gt_motion_from_key_items(r.get("key_items") or {}, r.get("key_frames") or []))


def notag(rows):
    out = []
    for r in rows:
        r = dict(r)
        if r.get("reasoning_process"):
            r["reasoning_process"] = strip_tags(r["reasoning_process"])
        r.pop("gt_motion", None)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--plm_dense", default=None)
    ap.add_argument("--rl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--heldout_videos", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft = json.load(open(args.sft))
    rl = json.load(open(args.rl))
    dense = {r["id"]: r for r in json.load(open(args.plm_dense))} if args.plm_dense else {}
    sft = [dense.get(r.get("id"), r) for r in sft]
    print(f"SFT rows {len(sft)} ({sum(r.get('id') in dense for r in sft)} replaced by dense PLM), RL rows {len(rl)}")

    # candidate held-out videos: grounded rows with a motion target, per source
    by_source = defaultdict(set)
    for r in sft + rl:
        if r.get("task") == GROUNDED and r.get("video_path") and has_motion_target(r):
            by_source[r["source"]].add(r["video_path"])
    total = sum(len(v) for v in by_source.values())
    held = set()
    for s, vids in sorted(by_source.items()):
        k = max(1, round(args.heldout_videos * len(vids) / total)) if total else 0
        held |= set(rng.sample(sorted(vids), min(k, len(vids))))

    def split(rows):
        keep = [r for r in rows if r.get("video_path") not in held]
        gone = [r for r in rows if r.get("video_path") in held]
        return keep, gone

    sft_keep, sft_gone = split(sft)
    rl_keep, rl_gone = split(rl)
    eval_rows, seen = [], set()
    for r in sft_gone + rl_gone:  # SFT first so dense PLM versions win
        key = (r.get("video_path"), r.get("question"))
        if r.get("task") == GROUNDED and key not in seen and has_motion_target(r):
            seen.add(key)
            eval_rows.append(r)

    files = {
        "sft_v4.json": sft_keep, "sft_v4_notag.json": notag(sft_keep),
        "rl_v4.json": rl_keep, "rl_v4_notag.json": notag(rl_keep),
        "eval_heldout.json": eval_rows,
    }
    for name, rows in files.items():
        json.dump(rows, open(out / name, "w"), indent=1)
    json.dump(sorted(held), open(out / "heldout_videos.json", "w"), indent=1)

    leak = {r.get("video_path") for r in sft_keep + rl_keep} & held
    print(f"held-out videos {len(held)}; leakage check: {len(leak)} held-out videos in training files")
    for name, rows in files.items():
        if name.endswith("notag.json"):
            continue
        print(f"{name:20s} {len(rows):6d}  {dict(Counter(r.get('source') for r in rows).most_common())}")
    sv = {(r.get("video_path"), r.get("question")) for r in sft_keep}
    rv = {(r.get("video_path"), r.get("question")) for r in rl_keep}
    print(f"SFT/RL rows sharing (video, question): {len(sv & rv)}")


if __name__ == "__main__":
    main()
