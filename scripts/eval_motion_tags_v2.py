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


MATCH_IOU = 0.3   # name-free fallback, as in r_traj v4
MATCH_DT = 2.0    # seconds between a claim and the GT keyframe it is compared with


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _track_iou(claims, track, frame_size):
    """Mean IoU of claimed boxes with the GT box at the nearest keyframe (<= MATCH_DT)."""
    vals = []
    for c in claims:
        near = [(abs(t - c["t"]), b) for t, b in track if abs(t - c["t"]) <= MATCH_DT]
        if near:
            b = min(near, key=lambda x: x[0])[1]
            vals.append(_iou(mc.to_pixels(c["box"], frame_size), mc.to_pixels(b, frame_size)))
    return sum(vals) / len(vals) if vals else 0.0


def score_sample(sample, text, gt, oracle_text=None, frame_size=None):
    """One record per GT object. The model's tag for it is found by object name
    (similarity > 0.5); otherwise, with ``frame_size`` (W, H of the frames the
    model saw), by box overlap: the tagged model object whose boxes overlap the GT
    track most (mean IoU >= MATCH_IOU). Name-only matching misses synonyms
    ("man" vs "person in blue")."""
    think = mc.extract_think(text) or text
    tags = {t["obj"]: t for t in mc.parse_tags(think) if t["well_formed"]}
    claims = mc.parse_claims(think)
    oracle_tags = {}
    if oracle_text is not None:
        oracle_tags = {t["obj"]: t for t in mc.parse_tags(oracle_text) if t["well_formed"]}
    tracks = mc.tracks_from_key_items(sample.get("key_items") or {}, sample.get("key_frames") or []) \
        if frame_size else {}
    best_by_obj, used = {}, set()
    for obj in gt:
        best, sim = None, 0.5
        for p in tags:
            s = name_sim(obj, p)
            if s > sim:
                best, sim = p, s
        if best is not None:
            best_by_obj[obj] = (best, "name")
            used.add(best)
    for obj in gt:
        if obj in best_by_obj or obj not in tracks:
            continue
        scored = {p: _track_iou([c for c in claims if c["obj"] == p], tracks[obj], frame_size)
                  for p in tags if p not in used}
        if scored:
            p = max(scored, key=scored.get)
            if scored[p] >= MATCH_IOU:
                best_by_obj[obj] = (p, "iou")
                used.add(p)
    records = []
    for obj, g in gt.items():
        best, how = best_by_obj.get(obj, (None, None))
        own = [c for c in claims if best is not None and c["obj"] == best]
        implied = mc.motion_descriptor(mc.claims_to_track(own)) if len({c["t"] for c in own}) >= 2 else None
        o = None
        for p in oracle_tags:
            if name_sim(obj, p) > 0.5:
                o = oracle_tags[p]
        records.append({"obj": obj, "gt": {a: g[a] for a in mm.ATTRS},
                        "pred": {a: tags[best][a] for a in mm.ATTRS} if best else None,
                        "match": how, "implied": implied,
                        "oracle_pred": {a: o[a] for a in mm.ATTRS} if o else None})
    return records


def frame_size_for(sample, saved_args):
    """(W, H) of the frames the model saw, re-derived from the video with the
    settings of the saved run (files written before frame_size was stored).
    Results written before --frame_format existed used the v1 video input."""
    import types
    from mcot_eval_common import clip_size, load_clip
    fmt = saved_args.get("frame_format", "video")
    a = types.SimpleNamespace(frame_format=fmt, video_max_frames=saved_args.get("video_max_frames", 16),
                              video_max_pixels=saved_args.get("video_max_pixels") or
                              (128 * 28 * 28 if fmt == "images" else 2097152),
                              insert_keyframes=False, kf_roots=None)
    try:
        return clip_size(load_clip(a, sample["video_path_full"]))
    except Exception as e:
        print(f"frame size unavailable for {sample.get('id')}: {e}")
        return None


def compute_metrics(per_sample):
    records = [r for s in per_sample for r in s["records"]]
    out = {"n_samples": len(per_sample), "n_gt_objects": len(records),
           "tag_match_rate": sum(r["pred"] is not None for r in records) / len(records) if records else None,
           "matched_by": {k: sum(r.get("match") == k for r in records) for k in ("name", "iou")}}
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
    ap.add_argument("--no_iou_match", action="store_true",
                    help="match GT objects to tags by name only (v2 before this change)")
    args = ap.parse_args()

    if args.from_predictions:
        per_sample = json.load(open(args.from_predictions))["per_sample"]
        data = {s["id"]: s for s in json.load(open(args.dataset_json))}
        saved_args = json.load(open(args.from_predictions)).get("args", {})
        for s in per_sample:
            if s["id"] not in data:
                s["records"] = []
                continue
            size = None if args.no_iou_match else s.get("frame_size") or frame_size_for(data[s["id"]], saved_args)
            s["frame_size"] = size
            s["records"] = score_sample(data[s["id"]], s["text"], gt_labels(data[s["id"]], args.gt_source),
                                        s.get("oracle_text"), size)
        metrics = compute_metrics(per_sample)
        json.dump({"metrics": metrics, "per_sample": per_sample, "args": saved_args},
                  open(args.output_file, "w"), indent=2, default=str)
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
    jobs, oracle_jobs, kept, sizes = [], [], [], []
    for s in samples:
        try:
            clip = load_clip(args, s["video_path_full"], s)
        except Exception as e:
            print(f"skip {s.get('id')}: {e}")
            continue
        kept.append(s)
        sizes.append(clip_size(clip))
        jobs.append((build_prompt(processor, MOTION_SYSTEM_PROMPT, s["question"], clip=clip), clip))
        if args.oracle_boxes:
            size = clip_size(clip) if clip["format"] == "images" else None
            oracle_jobs.append((build_prompt(processor, MOTION_SYSTEM_PROMPT, s["question"],
                                             oracle_prefix(s, size), clip=clip), clip))
    texts = generate(llm, sp, jobs, args.batch_size)
    oracle_texts = generate(llm, sp, oracle_jobs, args.batch_size) if oracle_jobs else [None] * len(kept)

    per_sample = []
    for s, t, o, size in zip(kept, texts, oracle_texts, sizes):
        size = None if args.no_iou_match else size
        per_sample.append({"id": s["id"], "question": s["question"], "text": t, "oracle_text": o,
                           "frame_size": size,
                           "records": score_sample(s, t, gt_labels(s, args.gt_source), o, size)})
    metrics = compute_metrics(per_sample)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    json.dump({"model_path": args.model_path, "dataset_json": args.dataset_json, "args": vars(args),
               "metrics": metrics, "per_sample": per_sample}, open(args.output_file, "w"), indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
