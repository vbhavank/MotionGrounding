"""
Shared motion primitives for MCoT v4.

Single source of truth for:
  * the track -> tag map  M(T)  (identical to scripts/augment_discrete_motion.py,
    which produced the SFT labels and the precomputed ``gt_motion`` field),
  * adjacency-aware bin scoring (identical to motion_reward_v3),
  * parsing of grounded claims and <motion/> tags *with their positions*,
  * the per-tag evidence window used for self-consistency,
  * track transformations g and their label-level action rho_g,
  * extended representations: piecewise segments, camera-compensated
    displacements, relational motion and depth-based scale.

Everything here is pure Python (numpy only for the homography helpers) so it can
be unit tested without a GPU.

Conventions
-----------
A *track* is a time-sorted list of ``(t, box)`` with ``box = [x1, y1, x2, y2]``.
Boxes may be normalized or pixel coordinates, but all boxes of one track must use
the same convention. Image y grows downward; ``N`` means "up on screen".
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================================
# Bins (must match scripts/augment_discrete_motion.py and motion_reward_v3.py)
# ============================================================================

COMPASS_ORDER = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DIR_SET = set(COMPASS_ORDER) | {"STAT"}
SPEED_ORDER = ["stationary", "slow", "moderate", "fast"]
SCALE_ORDER = ["receding", "stable", "approaching"]

SPEED_THRESHOLDS = [(0.02, "stationary"), (0.10, "slow"), (0.30, "moderate")]
SCALE_LOG_THRESHOLD = 0.15          # on log(a_N / a_1)
STAT_FRACTION_OF_DIAG = 0.02        # direction STAT threshold

# log(a_N/a_1) ~= -2 log(z_N/z_1) for a rigid object under a pinhole camera, so
# the depth threshold is half the area threshold.
DEPTH_LOG_THRESHOLD = SCALE_LOG_THRESHOLD / 2.0

# Reward weights (same as motion_reward_v3 so r_traj stays comparable)
W_DIR, W_SPEED, W_SCALE = 0.40, 0.30, 0.30

STATIONARY_TAG = {"dir": "STAT", "speed": "stationary", "scale": "stable"}


# ============================================================================
# Geometry
# ============================================================================

def centroid(b: Sequence[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def diagonal(b: Sequence[float]) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def is_normalized(b: Sequence[float]) -> bool:
    return all(0.0 <= float(c) <= 1.0 for c in b)


def to_pixels(b: Sequence[float], image_size: Optional[Tuple[float, float]]) -> List[float]:
    """Convert a normalized box to pixels; pixel boxes pass through."""
    if image_size is None or not is_normalized(b):
        return [float(c) for c in b]
    w, h = image_size
    return [b[0] * w, b[1] * h, b[2] * w, b[3] * h]


def angle_to_compass(dx: float, dy_up: float) -> str:
    deg = math.degrees(math.atan2(dy_up, dx)) % 360
    return COMPASS_ORDER[int((deg + 22.5) / 45.0) % 8]


# ============================================================================
# M(T): track -> (dir, speed, scale)
# ============================================================================

def speed_bin(norm_speed: float) -> str:
    for thr, label in SPEED_THRESHOLDS:
        if norm_speed < thr:
            return label
    return "fast"


def scale_bin(log_ratio: float, threshold: float = SCALE_LOG_THRESHOLD) -> str:
    if log_ratio > threshold:
        return "approaching"
    if log_ratio < -threshold:
        return "receding"
    return "stable"


def motion_descriptor(track: Sequence[Tuple[float, Sequence[float]]],
                      displacements: Optional[Sequence[Tuple[float, float]]] = None,
                      scale_log_ratio: Optional[float] = None,
                      scale_threshold: float = SCALE_LOG_THRESHOLD) -> Dict[str, str]:
    """M(T). Reproduces ``augment_discrete_motion.compute_motion_descriptor``.

    Direction is a magnitude-weighted *vote* over per-step compass bins (not the
    bin of the summed vector), speed is path length / duration / mean diagonal,
    scale is log(a_N / a_1). ``displacements`` (image-coordinate dx, dy per step)
    and ``scale_log_ratio`` let callers substitute camera-compensated motion or
    depth-based scale while keeping the same binning.
    """
    n = len(track)
    if n < 2:
        return dict(STATIONARY_TAG)
    times = [float(t) for t, _ in track]
    boxes = [b for _, b in track]
    if displacements is None:
        cents = [centroid(b) for b in boxes]
        displacements = [(cents[i + 1][0] - cents[i][0], cents[i + 1][1] - cents[i][1])
                         for i in range(n - 1)]
    mags = [math.hypot(dx, dy) for dx, dy in displacements]

    avg_diag = sum(diagonal(b) for b in boxes) / n
    stat_threshold = STAT_FRACTION_OF_DIAG * avg_diag

    votes: Dict[str, float] = defaultdict(float)
    total = 0.0
    for (dx, dy), m in zip(displacements, mags):
        if m < 1e-9:
            continue
        votes[angle_to_compass(dx, -dy)] += m
        total += m
    d = "STAT" if total < stat_threshold or not votes else max(votes, key=votes.get)

    duration = times[-1] - times[0]
    norm_speed = (sum(mags) / duration) / avg_diag if duration > 0 and avg_diag > 1e-9 else 0.0
    s = speed_bin(norm_speed)

    if scale_log_ratio is None:
        a1, an = area(boxes[0]), area(boxes[-1])
        c = "stable" if a1 < 1e-9 or an < 1e-9 else scale_bin(math.log(an / a1), scale_threshold)
    else:
        c = scale_bin(scale_log_ratio, scale_threshold)

    if d == "STAT":
        s = "stationary"
    elif s == "stationary":
        d = "STAT"
    return {"dir": d, "speed": s, "scale": c}


def tracks_from_key_items(key_items: Dict, key_frames: List[Dict]) -> Dict[str, List[Tuple[float, List[float]]]]:
    """Group STGR ``key_items`` into per-object tracks (same as the augmenter)."""
    idx_to_time = {str(f["idx"]): float(f["time"]) for f in key_frames or []}
    tracks: Dict[str, list] = defaultdict(list)
    for frame_idx, objects in (key_items or {}).items():
        t = idx_to_time.get(str(frame_idx))
        if t is None or not objects:
            continue
        for name, boxes in objects.items():
            if boxes:
                tracks[name].append((t, list(boxes[0])))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in tracks.items()}


def gt_motion_from_key_items(key_items: Dict, key_frames: List[Dict],
                             image_size: Optional[Tuple[float, float]] = None,
                             min_observations: int = 2) -> Dict[str, Dict[str, str]]:
    """M(T_gt) for every object with at least ``min_observations`` timestamps.

    Unlike the augmenter's default (``tag_single_frame=True``), single-frame
    objects are *not* labeled STAT here: a track with one observation carries no
    motion evidence.
    """
    out = {}
    for name, track in tracks_from_key_items(key_items, key_frames).items():
        if len(track) < min_observations:
            continue
        px = [(t, to_pixels(b, image_size)) for t, b in track]
        out[name] = motion_descriptor(px)
    return out


# ============================================================================
# Scoring
# ============================================================================

def direction_score(pred: str, gt: str) -> float:
    if pred == gt:
        return 1.0
    if pred not in COMPASS_ORDER or gt not in COMPASS_ORDER:
        return 0.0
    i, j = COMPASS_ORDER.index(pred), COMPASS_ORDER.index(gt)
    return 0.5 if min(abs(i - j), 8 - abs(i - j)) == 1 else 0.0


def ordinal_score(pred: str, gt: str, order: List[str]) -> float:
    if pred == gt:
        return 1.0
    if pred not in order or gt not in order:
        return 0.0
    return 0.5 if abs(order.index(pred) - order.index(gt)) == 1 else 0.0


def tag_score(pred: Dict[str, str], target: Dict[str, str]) -> float:
    """Adjacency-aware bin matching used by r_traj, r_self and r_ground'."""
    return (W_DIR * direction_score(pred.get("dir", ""), target["dir"])
            + W_SPEED * ordinal_score(pred.get("speed", ""), target["speed"], SPEED_ORDER)
            + W_SCALE * ordinal_score(pred.get("scale", ""), target["scale"], SCALE_ORDER))


def tag_equal(a: Dict[str, str], b: Dict[str, str]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("dir", "speed", "scale"))


def direction_adjacent(pred: str, gt: str) -> bool:
    """|angle difference| <= 45 deg (STAT only matches STAT)."""
    return direction_score(pred, gt) > 0


def ordinal_adjacent(pred: str, gt: str, order: List[str]) -> bool:
    return ordinal_score(pred, gt, order) > 0


# ============================================================================
# Parsing (with positions)
# ============================================================================

CLAIM_RE = re.compile(r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)at<t>(.*?)</t>s", re.DOTALL)
TAG_RE = re.compile(r"<motion\s+([^<>]*?)/>")
ATTR_RE = re.compile(r'(\w+)\s*=\s*\\?"([^"\\]*)\\?"')
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def normalize_obj_name(name: str) -> str:
    name = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return re.sub(r"\s+", " ", name).strip()


def extract_think(text: str) -> Optional[str]:
    m = THINK_RE.search(text or "")
    return m.group(1) if m else None


def parse_claims(think: str) -> List[Dict]:
    """Grounded claims ``<obj>o</obj><box>b</box>at<t>t</t>s`` in order."""
    claims = []
    for m in CLAIM_RE.finditer(think or ""):
        try:
            t = float(m.group(3).strip())
            boxes = [[float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", b)]
                     for b in re.findall(r"\[(.*?)\]", m.group(2))]
            boxes = [b for b in boxes if len(b) == 4]
            if not boxes:
                continue
        except ValueError:
            continue
        claims.append({"obj": normalize_obj_name(m.group(1)), "raw_obj": m.group(1).strip(),
                       "t": t, "box": boxes[0], "start": m.start(), "end": m.end()})
    return claims


def parse_tags(think: str) -> List[Dict]:
    """All <motion .../> tags with every attribute and their character span.

    Recognised optional attributes (v4): ``from``/``to`` (segment window in
    seconds), ``frame`` ("image" | "scene"), ``ref`` (reference object for
    relational motion). ``well_formed`` is True when obj/dir/speed/scale are
    present and in-vocabulary.
    """
    tags = []
    for m in TAG_RE.finditer(think or ""):
        attrs = {k: v.strip() for k, v in ATTR_RE.findall(m.group(1))}
        tag = {
            "obj": normalize_obj_name(attrs.get("obj", "")),
            "dir": attrs.get("dir", ""),
            "speed": attrs.get("speed", ""),
            "scale": attrs.get("scale", ""),
            "frame": attrs.get("frame", "image"),
            "ref": normalize_obj_name(attrs["ref"]) if "ref" in attrs else None,
            "from": _to_float(attrs.get("from")),
            "to": _to_float(attrs.get("to")),
            "start": m.start(), "end": m.end(), "raw": m.group(0),
        }
        tag["well_formed"] = bool(tag["obj"]) and tag["dir"] in DIR_SET \
            and tag["speed"] in SPEED_ORDER and tag["scale"] in SCALE_ORDER
        tags.append(tag)
    return tags


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def tag_evidence_windows(think: str) -> List[Tuple[Dict, List[Dict]]]:
    """Pair every tag with the claims it summarizes (the model's own T_hat_o).

    * If the tag has ``from``/``to``, the window is the claims of ``obj`` with
      ``from <= t <= to``.
    * Otherwise it is the claims of ``obj`` that precede the tag and follow the
      previous tag of the same object, plus the last claim covered by that
      previous tag as an anchor (so consecutive tags describe contiguous
      segments, e.g. the W-then-E baby in Fig. S9).
    Returns ``[(tag, claims_in_window), ...]`` in document order.
    """
    claims = parse_claims(think)
    out = []
    last_tag_end: Dict[str, int] = {}
    for tag in parse_tags(think):
        o = tag["obj"]
        own = [c for c in claims if c["obj"] == o]
        if tag["from"] is not None and tag["to"] is not None:
            win = [c for c in own if tag["from"] - 1e-6 <= c["t"] <= tag["to"] + 1e-6]
        else:
            prev_end = last_tag_end.get(o, -1)
            before = [c for c in own if c["end"] <= tag["start"]]
            win = [c for c in before if c["start"] > prev_end]
            anchor = [c for c in before if c["start"] <= prev_end]
            if anchor and prev_end >= 0:
                win = [anchor[-1]] + win
        last_tag_end[o] = tag["start"]
        out.append((tag, win))
    return out


def claims_to_track(claims: List[Dict], image_size=None) -> List[Tuple[float, List[float]]]:
    """Time-sorted track; duplicate timestamps keep the last box."""
    by_t = {}
    for c in claims:
        by_t[round(c["t"], 3)] = to_pixels(c["box"], image_size)
    return sorted(by_t.items())


# ============================================================================
# Transformations g and their label action rho_g
# ============================================================================

OPPOSITE = {d: COMPASS_ORDER[(i + 4) % 8] for i, d in enumerate(COMPASS_ORDER)}
OPPOSITE["STAT"] = "STAT"
MIRROR_X = {d: COMPASS_ORDER[(4 - i) % 8] for i, d in enumerate(COMPASS_ORDER)}   # E<->W, NE<->NW
MIRROR_X["STAT"] = "STAT"
SCALE_FLIP = {"approaching": "receding", "receding": "approaching", "stable": "stable"}


def rho(g: str, tag: Dict[str, str]) -> Dict[str, str]:
    """Label-level action rho_g on (dir, speed, scale).

    Exact for ``reverse`` and ``hflip``. For ``speedup`` only dir/scale are
    determined (speed moves up monotonically, see ``rho_consistency``), and for
    a full ``freeze`` the result is STAT. Prefer ``motion_descriptor(g(T_gt))``
    as the training target: it is exact for all g, including partial freezes
    and speed-ups that push a sub-threshold "stationary" track to "slow".
    """
    d, s, c = tag.get("dir", ""), tag.get("speed", ""), tag.get("scale", "")
    if g == "reverse":
        return {"dir": OPPOSITE.get(d, d), "speed": s, "scale": SCALE_FLIP.get(c, c)}
    if g == "hflip":
        return {"dir": MIRROR_X.get(d, d), "speed": s, "scale": c}
    if g == "speedup":
        return {"dir": d, "speed": s, "scale": c}
    if g == "freeze":
        return dict(STATIONARY_TAG)
    raise ValueError(f"unknown transformation {g}")


def rho_consistency(g: str, tag_v: Dict[str, str], tag_gv: Dict[str, str]) -> float:
    """Label-free equivariance check 1[M_hat(gV) = rho_g(M_hat(V))], per attribute.

    Returns the mean over (dir, speed, scale) so partial agreement is visible.
    For ``speedup`` the speed attribute is correct iff its rank did not drop.
    """
    exp = rho(g, tag_v)
    ok_d = tag_gv.get("dir") == exp["dir"]
    ok_c = tag_gv.get("scale") == exp["scale"]
    if g == "speedup":
        sv, sg = tag_v.get("speed"), tag_gv.get("speed")
        ok_s = sv in SPEED_ORDER and sg in SPEED_ORDER and SPEED_ORDER.index(sg) >= SPEED_ORDER.index(sv)
    else:
        ok_s = tag_gv.get("speed") == exp["speed"]
    return (ok_d + ok_s + ok_c) / 3.0


def interpolate_box(track, t: float) -> List[float]:
    """Linear interpolation of the box at time t (clamped at the ends)."""
    if t <= track[0][0]:
        return list(track[0][1])
    if t >= track[-1][0]:
        return list(track[-1][1])
    for (t0, b0), (t1, b1) in zip(track, track[1:]):
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return [a + w * (b - a) for a, b in zip(b0, b1)]
    return list(track[-1][1])


def transform_track(track, g: str, *, duration: float = None, width: float = 1.0,
                    k: float = 2.0, span: Tuple[float, float] = None):
    """Apply g to a track (so that M(g T) is the exact equivariant target).

    reverse : t -> duration - t (requires ``duration``)
    hflip   : [x1,y1,x2,y2] -> [W-x2, y1, W-x1, y2]   (W = 1 for normalized)
    speedup : t -> t / k
    freeze  : boxes with t in span=[a, b] are replaced by the box at a
    """
    if g == "reverse":
        if duration is None:
            raise ValueError("reverse needs duration")
        return sorted(((duration - t, list(b)) for t, b in track), key=lambda x: x[0])
    if g == "hflip":
        return [(t, [width - b[2], b[1], width - b[0], b[3]]) for t, b in track]
    if g == "speedup":
        return [(t / k, list(b)) for t, b in track]
    if g == "freeze":
        a, b_end = span if span is not None else (track[0][0], float("inf"))
        frozen = interpolate_box(track, a)
        return [(t, list(frozen) if a <= t <= b_end else list(b)) for t, b in track]
    raise ValueError(f"unknown transformation {g}")


def transform_time(t: float, g: str, *, duration: float = None, k: float = 2.0) -> float:
    if g == "reverse":
        return duration - t
    if g == "speedup":
        return t / k
    return t


def transform_key_annotations(key_items: Dict, key_frames: List[Dict], g: str, *,
                              duration: float = None, k: float = 2.0,
                              span: Tuple[float, float] = None, width: float = 1.0):
    """Transform STGR ``key_items``/``key_frames`` so every existing reward
    (thk_spatial, thk_temporal_point, r_traj) can score a rollout on gV.

    key_items boxes are normalized (width=1) in STGR.
    """
    frames = []
    for f in key_frames or []:
        f2 = dict(f)
        f2["time"] = transform_time(float(f["time"]), g, duration=duration, k=k)
        frames.append(f2)
    frames.sort(key=lambda f: f["time"])

    items = {}
    if g == "freeze":
        idx_to_time = {str(f["idx"]): float(f["time"]) for f in key_frames or []}
        tracks = tracks_from_key_items(key_items, key_frames)
        a, b_end = span if span is not None else (min(idx_to_time.values(), default=0.0), float("inf"))
    for fidx, objects in (key_items or {}).items():
        new_objs = {}
        for name, boxes in (objects or {}).items():
            if not boxes:
                new_objs[name] = boxes
                continue
            if g == "hflip":
                new_objs[name] = [[width - b[2], b[1], width - b[0], b[3]] for b in boxes]
            elif g == "freeze":
                t = idx_to_time.get(str(fidx))
                if t is not None and a <= t <= b_end and name in tracks:
                    new_objs[name] = [interpolate_box(tracks[name], a)]
                else:
                    new_objs[name] = [list(b) for b in boxes]
            else:
                new_objs[name] = [list(b) for b in boxes]
        items[fidx] = new_objs
    return items, frames


# ---- video-level versions of g (frames + displayed timestamps) -------------

def frame_prompt(times: List[float], total_seconds: float) -> str:
    s = "".join(f"Frame {i + 1} at {round(t, 1)}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                for i, t in enumerate(times))
    return s + f"The video is in total {int(total_seconds)} seconds.\n"


def transform_frames(frames, times: List[float], g: str, *, duration: float,
                     k: float = 2.0, span=None):
    """Apply g to a (T, C, H, W) frame stack and its displayed timestamps.

    Works for torch tensors and numpy arrays (index-list based, no negative
    strides). Returns (frames', times', total_seconds').
    """
    n = len(times)
    if g == "reverse":
        idx = list(range(n - 1, -1, -1))
        return frames[idx], [duration - times[i] for i in idx], duration
    if g == "hflip":
        w = frames.shape[-1]
        return frames[..., list(range(w - 1, -1, -1))], list(times), duration
    if g == "speedup":
        return frames, [t / k for t in times], duration / k
    if g == "freeze":
        a, b = span if span is not None else (times[0], float("inf"))
        src = max([i for i, t in enumerate(times) if t <= a] or [0])
        idx = [src if a <= t <= b else i for i, t in enumerate(times)]
        return frames[idx], list(times), duration
    raise ValueError(g)


# ============================================================================
# 5a. Piecewise segmentation
# ============================================================================

def segment_track(track, min_turn_bins: int = 3, image_size=None) -> List[Tuple[float, float]]:
    """Split a track where the heading turns by >= ``min_turn_bins`` * 45 deg.

    Default 3 bins (>= 135 deg) splits reversals (back-and-forth, the baby in
    Fig. S9) but not gentle curves. Sub-threshold (jitter) steps never start a
    segment. Returns ``[(t_from, t_to), ...]`` covering the track.
    """
    px = [(t, to_pixels(b, image_size)) for t, b in track]
    if len(px) < 3:
        return [(px[0][0], px[-1][0])] if px else []
    avg_diag = sum(diagonal(b) for _, b in px) / len(px)
    thr = STAT_FRACTION_OF_DIAG * avg_diag
    bounds = [0]
    prev_bin = None
    for i in range(len(px) - 1):
        (c0x, c0y), (c1x, c1y) = centroid(px[i][1]), centroid(px[i + 1][1])
        dx, dy = c1x - c0x, c1y - c0y
        if math.hypot(dx, dy) < thr:
            continue
        b = COMPASS_ORDER.index(angle_to_compass(dx, -dy))
        if prev_bin is not None:
            turn = min(abs(b - prev_bin), 8 - abs(b - prev_bin))
            if turn >= min_turn_bins and i > bounds[-1]:
                bounds.append(i)
        prev_bin = b
    bounds.append(len(px) - 1)
    return [(px[s][0], px[e][0]) for s, e in zip(bounds, bounds[1:])]


def restrict_track(track, t_from: float, t_to: float):
    """Track restricted to [t_from, t_to] with interpolated endpoints."""
    inner = [(t, list(b)) for t, b in track if t_from < t < t_to]
    return [(t_from, interpolate_box(track, t_from))] + inner + [(t_to, interpolate_box(track, t_to))]


def piecewise_score(pred_tags: List[Dict], gt_track, image_size=None) -> float:
    """Duration-weighted r_traj over predicted segments.

    Each predicted segment [from, to] (clipped to the GT span, overlaps removed
    in order) is scored against M(T_gt restricted to that window) and weighted by
    its share of the GT span. Tagging only an easy sub-window therefore cannot
    reach full credit.
    """
    if len(gt_track) < 2:
        return 0.0
    t0, t1 = gt_track[0][0], gt_track[-1][0]
    span = t1 - t0
    if span <= 0:
        return 0.0
    px = [(t, to_pixels(b, image_size)) for t, b in gt_track]
    # GT reversal boundaries: a window that spans a reversal is scored per GT
    # piece, so one tag cannot collect full credit for back-and-forth motion
    # (M of the whole window is the vote winner, i.e. only one of the legs).
    cuts = sorted({s for s, _ in segment_track(px)[1:]})
    covered_until = t0
    total = 0.0
    for tag in sorted(pred_tags, key=lambda x: (x["from"] if x["from"] is not None else t0)):
        a = max(tag["from"] if tag["from"] is not None else t0, covered_until)
        b = min(tag["to"] if tag["to"] is not None else t1, t1)
        if b - a <= 1e-6:
            continue
        edges = [a] + [c for c in cuts if a < c < b] + [b]
        for lo, hi in zip(edges, edges[1:]):
            target = motion_descriptor(restrict_track(px, lo, hi))
            total += (hi - lo) / span * tag_score(tag, target)
        covered_until = b
    return total


# ============================================================================
# 5b. Camera compensation (image vs scene frame)
# ============================================================================

def apply_homography(H, pt: Tuple[float, float]) -> Tuple[float, float]:
    x, y = pt
    u = H[0][0] * x + H[0][1] * y + H[0][2]
    v = H[1][0] * x + H[1][1] * y + H[1][2]
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    return (u / w, v / w) if abs(w) > 1e-12 else (u, v)


def scene_displacements(track, homographies) -> List[Tuple[float, float]]:
    """Delta c_i^scene = c_{i+1} - H_{i->i+1}(c_i).

    ``homographies[i]`` maps background pixels of observation i to observation
    i+1 (chain frame-to-frame estimates between observations with
    ``chain_homographies``). Feed the result to ``motion_descriptor(...,
    displacements=...)`` and emit ``frame="scene"``.
    """
    out = []
    for i in range(len(track) - 1):
        c0, c1 = centroid(track[i][1]), centroid(track[i + 1][1])
        w0 = apply_homography(homographies[i], c0)
        out.append((c1[0] - w0[0], c1[1] - w0[1]))
    return out


def chain_homographies(hs):
    """Compose H_{a->a+1}, ..., H_{b-1->b} into H_{a->b} (numpy)."""
    import numpy as np
    H = np.eye(3)
    for h in hs:
        H = np.asarray(h) @ H
    return H / H[2, 2]


def estimate_background_homography(frame_a, frame_b, exclude_boxes=(), max_features: int = 2000):
    """Background homography between two uint8 RGB/gray frames (H x W [x C]).

    Features inside ``exclude_boxes`` (pixel boxes of moving objects) are masked
    out so the estimate follows the camera, not the object. Returns a 3x3 numpy
    array or the identity if too few matches are found. Requires OpenCV.
    """
    import cv2
    import numpy as np

    def gray(f):
        f = np.asarray(f)
        return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f

    ga, gb = gray(frame_a), gray(frame_b)
    mask = np.full(ga.shape, 255, np.uint8)
    for x1, y1, x2, y2 in exclude_boxes:
        mask[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0
    orb = cv2.ORB_create(max_features)
    ka, da = orb.detectAndCompute(ga, mask)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.eye(3)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 8:
        return np.eye(3)
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    return H if H is not None else np.eye(3)


# ============================================================================
# 5c. Relational motion
# ============================================================================

def relational_descriptor(track_a, track_b, image_size=None) -> Dict[str, str]:
    """M applied to the displacement of A relative to B: Delta(c^A - c^B).

    B is interpolated at A's timestamps (within B's observed span). Speed is
    normalized by A's size; scale is A's area change relative to B's, so a
    camera zoom that scales both objects reads as "stable".
    """
    a = [(t, to_pixels(b, image_size)) for t, b in track_a]
    b = [(t, to_pixels(x, image_size)) for t, x in track_b]
    if len(a) < 2 or len(b) < 1:
        return dict(STATIONARY_TAG)
    lo, hi = b[0][0], b[-1][0]
    a = [(t, x) for t, x in a if lo - 1e-6 <= t <= hi + 1e-6] if len(b) > 1 else a
    if len(a) < 2:
        return dict(STATIONARY_TAG)
    rel = []
    for t, box in a:
        cb = centroid(interpolate_box(b, t))
        ca = centroid(box)
        rel.append((ca[0] - cb[0], ca[1] - cb[1]))
    disp = [(rel[i + 1][0] - rel[i][0], rel[i + 1][1] - rel[i][1]) for i in range(len(rel) - 1)]
    area_a = [area(x) for _, x in a]
    area_b = [area(interpolate_box(b, t)) for t, _ in a]
    if min(area_a[0], area_a[-1], area_b[0], area_b[-1]) > 1e-9:
        lr = math.log(area_a[-1] / area_a[0]) - math.log(area_b[-1] / area_b[0])
    else:
        lr = 0.0
    return motion_descriptor(a, displacements=disp, scale_log_ratio=lr)


# ============================================================================
# 5d. Depth-based scale
# ============================================================================

def box_median_depth(depth_map, box) -> Optional[float]:
    """Median of a depth map (H x W, numpy) inside a pixel box."""
    import numpy as np
    d = np.asarray(depth_map)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    patch = d[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


def depth_scale_bin(track, depth_maps, is_disparity: bool = False) -> str:
    """c = bin(log(z_1 / z_N)) with z_i the median depth inside b_i.

    ``depth_maps[i]`` must correspond to ``track[i]``. Set ``is_disparity`` for
    inverse-depth models (e.g. MiDaS/DPT relative output). Threshold is half the
    area threshold since log(a_N/a_1) ~= 2 log(z_1/z_N).
    """
    z1 = box_median_depth(depth_maps[0], track[0][1])
    zn = box_median_depth(depth_maps[-1], track[-1][1])
    if not z1 or not zn:
        return "stable"
    if is_disparity:
        z1, zn = 1.0 / z1, 1.0 / zn
    return scale_bin(math.log(z1 / zn), DEPTH_LOG_THRESHOLD)
