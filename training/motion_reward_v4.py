"""
Motion rewards v4.

Changes relative to motion_reward_v3 (see revision/README.md for the rationale):

1. ``motion_self_consistency_reward``  r_self: scores each <motion/> tag against
   M(T_hat_o), the motion implied by the model's *own* grounded boxes, with the
   same adjacency-aware bin matching as r_traj.
2. ``format_reward_v4``: r_fmt - beta * 1[a tag is emitted for an object with
   fewer than two grounded timestamps in its evidence window].
3. ``motion_trajectory_reward_v4``: r_traj with GT recomputed from key_items for
   objects with >= 2 observations (single-frame objects are no longer "STAT"
   targets), a missing tag for a grounded multi-timestamp object scores 0, and
   piecewise tags (``from``/``to``) are scored duration-weighted per segment.
   Because GT is recomputed from ``key_items``, the same function scores
   rollouts on a transformed video gV once the annotations are transformed with
   ``motion_core.transform_key_annotations`` -- that is r_ground'.
4. ``motion_equivariance_consistency_reward``: the label-free variant
   1[M_hat(gV) = rho_g(M_hat(V))], averaged over g and objects. An object that
   is missing from the transformed rollout scores 0 (no omission credit).
"""

import os
import re
from typing import Dict, List

import numpy as np

try:
    from training import motion_core as mc
except ImportError:  # launched from inside training/
    import motion_core as mc

MOTION_TASKS = {
    "temporal-spatial free-form QA",
    "General video QA Free-form",
    "General video QA MCQ",
}

SCHEMA_BETA = float(os.environ.get("MCOT_SCHEMA_BETA", "0.5"))
MATCH_TIME_THRESHOLD = 2.0  # seconds, same as v3
MATCH_IOU_THRESHOLD = 0.3  # name-free fallback when the rollout's object name differs from the annotation


def _task(kwargs) -> str:
    t = kwargs.get("task", "")
    return t[0] if isinstance(t, list) and t else t


def _at(kwargs, key, i, default=None):
    v = kwargs.get(key)
    if isinstance(v, list) and i < len(v):
        return v[i]
    return default


def _content(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else completion


def last_tag_per_object(think: str) -> Dict[str, Dict]:
    """Last well-formed tag per object (v3 semantics: the last tag wins)."""
    out = {}
    for tag in mc.parse_tags(think):
        if tag["well_formed"]:
            out[tag["obj"]] = tag
    return out


# ============================================================================
# 1. Self-consistency  r_self
# ============================================================================

def self_consistency_details(think: str, image_size=None) -> List[Dict]:
    """Per-tag record: the tag, M(T_hat) over its evidence window, and scores."""
    rows = []
    for tag, win in mc.tag_evidence_windows(think):
        n_ts = len({round(c["t"], 3) for c in win})
        row = {"tag": tag, "n_timestamps": n_ts, "implied": None,
               "score": 0.0, "consistent": False, "schema_ok": n_ts >= 2}
        if tag["well_formed"] and n_ts >= 2:
            implied = mc.motion_descriptor(mc.claims_to_track(win, image_size))
            row["implied"] = implied
            row["score"] = mc.tag_score(tag, implied)
            row["consistent"] = mc.tag_equal(tag, implied)
        rows.append(row)
    return rows


def motion_self_consistency_reward(completions, **kwargs):
    """r_self = mean over tags of tag_score(tag, M(T_hat_o)); tags without two
    grounded timestamps score 0 (and are separately penalized in r_fmt)."""
    task = _task(kwargs)
    rewards = []
    for i, completion in enumerate(completions):
        think = mc.extract_think(_content(completion))
        if task not in MOTION_TASKS or think is None:
            rewards.append(0.0)
            continue
        rows = self_consistency_details(think, _at(kwargs, "image_size", i))
        rewards.append(float(np.mean([r["score"] for r in rows])) if rows else 0.0)
    return rewards


# ============================================================================
# 2. Format with schema check
# ============================================================================

def schema_violations(think: str) -> int:
    """Number of tags whose object has < 2 grounded timestamps in its window."""
    return sum(1 for _, win in mc.tag_evidence_windows(think)
               if len({round(c["t"], 3) for c in win}) < 2)


def _structurally_valid(text: str) -> bool:
    """The hard checks of v3 ``format_reward`` (0 reward if any fails)."""
    think = mc.extract_think(text)
    if think is None or not re.search(r"<answer>.*?</answer>", text, re.DOTALL):
        return False
    if text.count("<think>") != text.count("</think>") or text.count("<answer>") != text.count("</answer>"):
        return False
    return all(think.count(f"<{t}>") == think.count(f"</{t}>") for t in ("obj", "t", "box", "motion"))


def format_reward_v4(completions, require_motion: bool = True, **kwargs):
    """v3 format reward, with two changes:

    * a motion tag is *required* only when the rollout grounds some object at
      >= 2 timestamps (the paper's rule), instead of on every motion task --
      General-QA rollouts without grounding are no longer pushed to emit
      unsupported tags;
    * r_fmt -= beta * 1[any tag is emitted for an object with |T_hat_o| < 2]
      (beta from MCOT_SCHEMA_BETA, default 0.5).
    """
    task = _task(kwargs)
    base = [1.0 if _structurally_valid(_content(c)) else 0.0 for c in completions]
    out = []
    for completion, r in zip(completions, base):
        text = _content(completion)
        think = mc.extract_think(text)
        if think is None or r == 0.0:
            out.append(r)
            continue
        claims = mc.parse_claims(think)
        has_grounding = bool(claims)
        if task in ("temporal QA", "temporal QA (MCQ)"):
            has_grounding = think.count("<t>") >= 2
        if task == "visual QA":
            has_grounding = bool(re.search(r"<obj>(\w+)</obj><box>(\[.*?\])</box>", text))
        multi = {c["obj"] for c in claims
                 if len({round(x["t"], 3) for x in claims if x["obj"] == c["obj"]}) >= 2}
        tagged = {t["obj"] for t in mc.parse_tags(think) if t["well_formed"]}
        if require_motion and task in MOTION_TASKS and multi and not (multi & tagged):
            has_grounding = False
        r = 1.0 if (has_grounding or "General video QA" in task) else 0.5
        if require_motion and schema_violations(think) > 0:
            r -= SCHEMA_BETA
        out.append(r)
    return out


def format_reward_notag(completions, **kwargs):
    """No-tag control: identical structure checks, no <motion/> requirement.

    Used with r_motion = 0 and tag-stripped data (scripts/strip_motion_tags.py)
    so that r_thk = r_t + r_s isolates MCoT from "more RL on denser data".
    """
    return format_reward_v4(completions, require_motion=False, **kwargs)


# ============================================================================
# 3. Trajectory reward against GT (also used on transformed videos)
# ============================================================================

def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_claims_to_gt(claims, gt_tracks_norm, gt_times, image_size=None):
    """Claim -> closest GT keyframe within 2 s -> GT object at that frame.

    The object is the GT object with the same normalized name if there is one,
    otherwise the GT object whose box overlaps the claimed box most (IoU >=
    MATCH_IOU_THRESHOLD). v3's r_s matches by time and IoU only; requiring the
    exact annotation name (e.g. "man in black shirt" vs the rollout's "person")
    left r_traj at 0 for correctly grounded objects.

    Returns ``(matched, aliases)``: GT name -> set of matched GT times, and GT
    name -> set of rollout object names that were matched to it.
    """
    matched, aliases = {}, {}
    for c in claims:
        best, best_dt = None, MATCH_TIME_THRESHOLD
        for t in gt_times:
            dt = abs(t - c["t"])
            if dt < best_dt:
                best, best_dt = t, dt
        if best is None:
            continue
        at_frame = {}
        for name, track in gt_tracks_norm.items():
            for t, b in track:
                if abs(t - best) < 1e-6:
                    at_frame[name] = b
                    break
        hit = next((n for n in at_frame if mc.normalize_obj_name(n) == c["obj"]), None)
        if hit is None and at_frame:
            claim_px = mc.to_pixels(c["box"], image_size)
            ious = {n: _box_iou(claim_px, mc.to_pixels(b, image_size)) for n, b in at_frame.items()}
            n_best = max(ious, key=ious.get)
            if ious[n_best] >= MATCH_IOU_THRESHOLD:
                hit = n_best
        if hit is not None:
            matched.setdefault(hit, set()).add(best)
            aliases.setdefault(hit, set()).add(c["obj"])
    return matched, aliases


def trajectory_details(think: str, key_items, key_frames, image_size=None) -> List[Dict]:
    """Per GT object that the rollout grounded at >= 2 distinct GT frames."""
    gt_tracks = mc.tracks_from_key_items(key_items, key_frames)
    gt_times = sorted({float(f["time"]) for f in key_frames or []})
    matched, aliases = _match_claims_to_gt(mc.parse_claims(think), gt_tracks, gt_times, image_size)
    all_tags = [t for t in mc.parse_tags(think) if t["well_formed"]]
    rows = []
    for name, frames in matched.items():
        if len(frames) < 2 or len(gt_tracks[name]) < 2:
            continue
        keys = {mc.normalize_obj_name(name)} | aliases.get(name, set())
        track_px = [(t, mc.to_pixels(b, image_size)) for t, b in gt_tracks[name]]
        target = mc.motion_descriptor(track_px)
        obj_tags = [t for t in all_tags if t["obj"] in keys and t["ref"] is None]
        piecewise = [t for t in obj_tags if t["from"] is not None and t["to"] is not None]
        if piecewise:
            score = mc.piecewise_score(piecewise, track_px)
            pred = piecewise
        elif obj_tags:
            pred = obj_tags[-1]
            score = mc.tag_score(pred, target)
        else:
            pred, score = None, 0.0  # grounded twice but no tag: no credit
        rows.append({"obj": name, "target": target, "pred": pred, "score": score})
    return rows


def motion_trajectory_reward_v4(completions, **kwargs):
    task = _task(kwargs)
    rewards = []
    for i, completion in enumerate(completions):
        think = mc.extract_think(_content(completion))
        key_items = _at(kwargs, "key_items", i)
        if task not in MOTION_TASKS or think is None or not key_items:
            rewards.append(0.0)
            continue
        try:
            rows = trajectory_details(think, key_items, _at(kwargs, "key_frames", i, []),
                                      _at(kwargs, "image_size", i))
        except Exception as e:  # malformed annotations must not kill a step
            print(f"[motion_trajectory_reward_v4] {e}")
            rows = []
        rewards.append(float(np.mean([r["score"] for r in rows])) if rows else 0.0)
    return rewards


# ============================================================================
# 4. Label-free equivariance consistency
# ============================================================================

def equivariance_consistency(text_v: str, transformed: Dict[str, List[str]]) -> float:
    """Mean over g, transformed samples and objects of rho_consistency."""
    think = mc.extract_think(text_v)
    tags_v = last_tag_per_object(think) if think else {}
    if not tags_v:
        return 0.0
    scores = []
    for g, texts in (transformed or {}).items():
        for t in texts or []:
            th = mc.extract_think(t or "")
            tags_g = last_tag_per_object(th) if th else {}
            for obj, tv in tags_v.items():
                tg = tags_g.get(obj)
                scores.append(mc.rho_consistency(g, tv, tg) if tg else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def motion_equivariance_consistency_reward(completions, **kwargs):
    """Needs ``transformed_completions``: one {g: [texts]} dict per completion
    (the trainer passes the same dict for every rollout of a prompt)."""
    task = _task(kwargs)
    rewards = []
    for i, completion in enumerate(completions):
        tr = _at(kwargs, "transformed_completions", i)
        if task not in MOTION_TASKS or not tr:
            rewards.append(0.0)
            continue
        rewards.append(equivariance_consistency(_content(completion), tr))
    return rewards
