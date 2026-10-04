"""Check r_traj (motion_trajectory_reward_v4) plumbing on RL rows.

Rows go through the same in-memory Arrow round trip as training (union of keys,
None fill) and the trainer's None clean-up. For each row it reports the GT
objects with >= 2 keyframes and, when the row carries a reference chain
(``reasoning_process``), scores that chain as if it were a rollout:
name-only matches vs the matches with the IoU fallback, and the resulting r_traj.

    python scripts/debug_traj_reward.py --json $J/splits_v4/rl_v4_sub800.json --n 200
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402
import motion_reward_v4 as mr  # noqa: E402
from data_loader_v4 import load_json_dataset  # noqa: E402


def clean(row):
    items = {k: v for k, v in (row.get("key_items") or {}).items() if v is not None}
    items = {k: {o: b for o, b in v.items() if b is not None} for k, v in items.items() if isinstance(v, dict)}
    return items, row.get("key_frames") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--show", type=int, default=3, help="print details for this many rows")
    args = ap.parse_args()

    ds = load_json_dataset(args.json)
    stats = collections.Counter()
    scores, shown = [], 0
    for row in ds.select(range(min(args.n, len(ds)))):
        if row.get("task") not in mr.MOTION_TASKS:
            continue
        stats["motion rows"] += 1
        items, frames = clean(row)
        tracks = mc.tracks_from_key_items(items, frames)
        multi = {k: v for k, v in tracks.items() if len(v) >= 2}
        stats["rows with a >=2-keyframe GT object"] += bool(multi)
        chain = row.get("reasoning_process") or ""
        if not chain:
            stats["rows without reasoning_process"] += 1
            continue
        claims = mc.parse_claims(chain)
        gt_times = sorted({float(f["time"]) for f in frames})
        names = {mc.normalize_obj_name(n) for n in tracks}
        by_name = {c["obj"] for c in claims} & names
        matched, aliases = mr._match_claims_to_gt(claims, tracks, gt_times, row.get("image_size"))
        stats["chains whose claim names equal a GT name"] += bool(by_name)
        stats["chains matched to a GT object (name or IoU)"] += bool(matched)
        rows = mr.trajectory_details(f"{chain}", items, frames, row.get("image_size"))
        stats["chains with a scored object"] += bool(rows)
        if rows:
            scores.append(np.mean([r["score"] for r in rows]))
        if shown < args.show:
            shown += 1
            print(f"--- {row.get('source')} | {row.get('video_path')}")
            print(f"  GT objects (n keyframes): { {k: len(v) for k, v in tracks.items()} }")
            print(f"  claim names: {sorted({c['obj'] for c in claims})[:8]}")
            print(f"  tags: {[t['raw'] for t in mc.parse_tags(chain)][:4]}")
            print(f"  matched GT -> rollout names: { {k: sorted(v) for k, v in aliases.items()} }")
            print(f"  r_traj rows: {[(r['obj'], r['score']) for r in rows]}")
    print("\n".join(f"{v:6d}  {k}" for k, v in stats.items()))
    if scores:
        print(f"r_traj of the reference chains: mean {np.mean(scores):.3f} over {len(scores)} rows")


if __name__ == "__main__":
    main()
