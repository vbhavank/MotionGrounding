#!/usr/bin/env python3
"""
No-tag control data: remove every <motion .../> tag (v2 self-closing and legacy
<motion>...</motion>) from ``reasoning_process`` and drop ``gt_motion``.

Everything else (samples, boxes, timestamps, dense keyframes) is unchanged, so
a model trained on the output with r_motion = 0 differs from Motion-o only in
MCoT. Run on both the SFT and RL json files.

    python scripts/strip_motion_tags.py --input STGR-SFT-motion-mixed.json \
        --output STGR-SFT-dense-notag.json
"""

import argparse
import json
import re

TAG_V2 = re.compile(r"\s*<motion\s+[^<>]*?/>")
TAG_V1 = re.compile(r"\s*<motion>[^<]*</motion>")


def strip_tags(text: str) -> str:
    return TAG_V1.sub("", TAG_V2.sub("", text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    n_tags = 0
    for s in data:
        r = s.get("reasoning_process")
        if r:
            n_tags += len(TAG_V2.findall(r)) + len(TAG_V1.findall(r))
            s["reasoning_process"] = strip_tags(r)
        s.pop("gt_motion", None)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"{len(data)} samples, removed {n_tags} motion tags -> {args.output}")


if __name__ == "__main__":
    main()
