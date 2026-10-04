#!/usr/bin/env python3
"""
<motion/> tag evaluation v2 (replaces the Table S4 protocol).

Differences from scripts/eval_motion_tags.py
  * ``--exclude_json``: drop samples whose id is in the training json. The v1
    script evaluates on the SFT json itself.
  * ``--gt_source key_items`` (default): GT = M(T_gt) recomputed from dense
    key_items for objects with >= 2 observations. ``tags`` reproduces v1 (GT
    parsed from reasoning_process, including single-frame STAT tags).
  * Reports exact, adjacent (+-45 deg / +-1 rank), balanced accuracy, macro-F1
    and the majority-class baseline on the same labels.
  * Self-consistency SC and schema-violation rate from the model's own boxes.
  * Localization error Err(M(T_hat), M(T_gt)).
  * ``--oracle_boxes``: pre-fill <think> with the GT boxes of every tracked
    object and let the model emit the tags -> motion-reasoning error given
    perfect localization.
  * Saves raw generations; ``--from_predictions`` recomputes metrics offline.

    python scripts/eval_motion_tags_v2.py --model_path M --dataset_json EVAL.json \
        --exclude_json STGR-SFT-motion-mixed.json --output_file out.json [--oracle_boxes]

  * Input format follows training (timestamped frames, training frame size);
    ``--frame_format video --video_max_pixels 2097152`` reproduces v1, and
    ``--insert_keyframes`` adds the annotated keyframes as in training.
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
import motion_core as mc  # noqa: E402
import mcot_metrics as mm  # noqa: E402


def name_sim(a, b):
    a, b = mc.normalize_obj_name(a), mc.normalize_obj_name(b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def gt_labels(sample, source):
    if source == "tags":
        return {t["obj"]: t for t in mc.parse_tags(sample.get("reasoning_process", "")) if t["well_formed"]}
    return mc.gt_motion_from_key_items(sample.get("key_items") or {}, sample.get("key_frames") or [],
                                       min_observations=2)


def oracle_prefix(sample, image_size=None):
    """<think> pre-filled with GT grounded claims for every tracked object.

    With ``image_size`` (W, H) the boxes are written in pixels of the frames the
    model sees, as in the SFT targets; otherwise normalized boxes stay as is."""
    tracks = mc.tracks_from_key_items(sample.get("key_items") or {}, sample.get("key_frames") or [])
    parts = []
    for name, tr in tracks.items():
        if len(tr) < 2:
            continue
        for t, b in tr:
            b = mc.to_pixels(b, image_size)
            box = ",".join(f"{v:.3f}" if mc.is_normalized(b) else f"{int(round(v))}" for v in b)
            parts.append(f"<obj>{name}</obj><box>[{box}]</box>at<t>{t:.1f}</t>s")
    return "<think>" + " ".join(parts) + " "


def score_sample(sample, text, gt, oracle_text=None):
    think = mc.extract_think(text) or text
    tags = {t["obj"]: t for t in mc.parse_tags(think) if t["well_formed"]}
    claims = mc.parse_claims(think)
    oracle_tags = {}
    if oracle_text is not None:
        oracle_tags = {t["obj"]: t for t in mc.parse_tags(oracle_text) if t["well_formed"]}
    records = []
    for obj, g in gt.items():
        best, sim = None, 0.5
        for p in tags:
            s = name_sim(obj, p)
            if s > sim:
                best, sim = p, s
        own = [c for c in claims if best is not None and c["obj"] == best]
        implied = mc.motion_descriptor(mc.claims_to_track(own)) if len({c["t"] for c in own}) >= 2 else None
        o = None
        for p in oracle_tags:
            if name_sim(obj, p) > 0.5:
                o = oracle_tags[p]
        records.append({"obj": obj, "gt": {a: g[a] for a in mm.ATTRS},
                        "pred": {a: tags[best][a] for a in mm.ATTRS} if best else None,
                        "implied": implied,
                        "oracle_pred": {a: o[a] for a in mm.ATTRS} if o else None})
    return records


def compute_metrics(per_sample):
    records = [r for s in per_sample for r in s["records"]]
    out = {"n_samples": len(per_sample), "n_gt_objects": len(records),
           "tag_match_rate": sum(r["pred"] is not None for r in records) / len(records) if records else None}
    for a in mm.ATTRS:
        out[a] = {
            "all_gt (unmatched=wrong)": mm.classification_report(
                [(r["pred"][a] if r["pred"] else None, r["gt"][a]) for r in records], a),
            "matched_only": mm.classification_report(
                [(r["pred"][a], r["gt"][a]) for r in records if r["pred"]], a),
        }
        if any(r["oracle_pred"] for r in records):
            out[a]["oracle_boxes"] = mm.classification_report(
                [(r["oracle_pred"][a], r["gt"][a]) for r in records if r["oracle_pred"]], a)
    out["error_decomposition"] = mm.error_decomposition(records)
    out["self_consistency"] = mm.self_consistency([s["text"] for s in per_sample])
    return out


def main():
    ap = argparse.ArgumentParser()
    from mcot_eval_common import add_model_args
    add_model_args(ap)
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--exclude_json", default=None, help="training json; its ids are excluded")
    ap.add_argument("--gt_source", choices=["key_items", "tags"], default="key_items")
    ap.add_argument("--oracle_boxes", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--from_predictions", default=None, help="recompute metrics from a saved output_file")
    args = ap.parse_args()

    if args.from_predictions:
        per_sample = json.load(open(args.from_predictions))["per_sample"]
        data = {s["id"]: s for s in json.load(open(args.dataset_json))}
        for s in per_sample:
            s["records"] = score_sample(data[s["id"]], s["text"], gt_labels(data[s["id"]], args.gt_source),
                                        s.get("oracle_text"))
        metrics = compute_metrics(per_sample)
        json.dump({"metrics": metrics, "per_sample": per_sample}, open(args.output_file, "w"), indent=2)
        print(json.dumps(metrics, indent=2, default=str)[:4000])
        return

    from mcot_eval_common import MOTION_SYSTEM_PROMPT, build_prompt, clip_size, generate, load_clip, make_llm
    data = json.load(open(args.dataset_json))
    exclude = set()
    if args.exclude_json:
        exclude = {s.get("id") for s in json.load(open(args.exclude_json))}
    samples = []
    for s in data:
        if s.get("id") in exclude or not os.path.isfile(s.get("video_path_full", "")):
            continue
        if gt_labels(s, args.gt_source):
            samples.append(s)
    samples = samples[: args.max_samples] if args.max_samples else samples
    print(f"{len(samples)} eval samples ({len(exclude)} training ids excluded)")

    llm, sp, processor = make_llm(args)
    jobs, oracle_jobs, kept = [], [], []
    for s in samples:
        try:
            clip = load_clip(args, s["video_path_full"], s)
        except Exception as e:
            print(f"skip {s.get('id')}: {e}")
            continue
        kept.append(s)
        jobs.append((build_prompt(processor, MOTION_SYSTEM_PROMPT, s["question"], clip=clip), clip))
        if args.oracle_boxes:
            size = clip_size(clip) if clip["format"] == "images" else None
            oracle_jobs.append((build_prompt(processor, MOTION_SYSTEM_PROMPT, s["question"],
                                             oracle_prefix(s, size), clip=clip), clip))
    texts = generate(llm, sp, jobs, args.batch_size)
    oracle_texts = generate(llm, sp, oracle_jobs, args.batch_size) if oracle_jobs else [None] * len(kept)

    per_sample = []
    for s, t, o in zip(kept, texts, oracle_texts):
        per_sample.append({"id": s["id"], "question": s["question"], "text": t, "oracle_text": o,
                           "records": score_sample(s, t, gt_labels(s, args.gt_source), o)})
    metrics = compute_metrics(per_sample)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    json.dump({"model_path": args.model_path, "dataset_json": args.dataset_json, "args": vars(args),
               "metrics": metrics, "per_sample": per_sample}, open(args.output_file, "w"), indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
