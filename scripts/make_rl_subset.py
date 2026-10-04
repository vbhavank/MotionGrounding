"""Draw the same RL subset from rl_v4.json and rl_v4_notag.json (single-GPU budgets).

The two files are row-aligned (build_v4_splits.py writes notag(rl_keep)), so the
tagged and no-tag RL runs see identical samples. PLM rows (dense boxes, most
motion) get ``--plm_frac`` of the budget; the rest is split over the other
sources in proportion to their size, so temporal QA keeps its natural share.

    python scripts/make_rl_subset.py --splits $J/splits_v4 --n 800
    -> rl_v4_sub800.json, rl_v4_notag_sub800.json
"""

import argparse
import collections
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="directory with rl_v4.json and rl_v4_notag.json")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--plm_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.splits)
    rows = json.load(open(d / "rl_v4.json"))
    notag = json.load(open(d / "rl_v4_notag.json"))
    assert len(rows) == len(notag), "rl_v4 and rl_v4_notag are not row-aligned"

    rng = random.Random(args.seed)
    by_src = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_src[r.get("source")].append(i)
    plm = [s for s in by_src if "plm" in (s or "").lower()]
    other = [s for s in by_src if s not in plm]

    quota = {}
    n_plm = min(round(args.n * args.plm_frac), sum(len(by_src[s]) for s in plm))
    for group, budget in ((plm, n_plm), (other, args.n - n_plm)):
        total = sum(len(by_src[s]) for s in group)
        for s in group:
            quota[s] = min(len(by_src[s]), round(budget * len(by_src[s]) / total)) if total else 0

    picked = []
    for s, q in quota.items():
        picked += rng.sample(by_src[s], q)
    rng.shuffle(picked)

    tag = f"sub{args.n}"
    json.dump([rows[i] for i in picked], open(d / f"rl_v4_{tag}.json", "w"), indent=1)
    json.dump([notag[i] for i in picked], open(d / f"rl_v4_notag_{tag}.json", "w"), indent=1)
    print(f"picked {len(picked)} rows -> rl_v4_{tag}.json, rl_v4_notag_{tag}.json")
    for s, q in sorted(quota.items(), key=lambda kv: -kv[1]):
        print(f"  {q:5d} / {len(by_src[s]):5d}  {s}")


if __name__ == "__main__":
    main()
