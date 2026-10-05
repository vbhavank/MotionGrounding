"""
Metrics for MCoT evaluation (no model / GPU needed).

* per-attribute exact, adjacent (+-45 deg / +-1 rank), balanced accuracy,
  macro-F1, and the majority-class baseline on the *same* evaluation labels
* self-consistency rate SC and schema-violation rate
* localization / motion-reasoning error decomposition (oracle-box diagnostic)
* tag presence rho_tag and accuracy split by tag presence
* tag sensitivity TS and tag-following rate (intervention test)
* paired accuracy PA for reversal-contrast pairs
* mean +- std over seeds and Welch's t-test
"""

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import motion_core as mc  # noqa: E402

ATTRS = ("dir", "speed", "scale")
CLASSES = {"dir": ["STAT"] + mc.COMPASS_ORDER, "speed": mc.SPEED_ORDER, "scale": mc.SCALE_ORDER}

# Fig. S5 counts (augmented training set as reported in the paper)
PAPER_FIG_S5 = {
    "dir": {"STAT": 5582, "N": 166, "NE": 246, "E": 1244, "SE": 377, "S": 325, "SW": 321, "W": 1171, "NW": 260},
    "speed": {"stationary": 5582, "slow": 1396, "moderate": 1266, "fast": 1448},
    "scale": {"stable": 5751, "approaching": 2322, "receding": 1619},
}


# ============================================================================
# Classification metrics
# ============================================================================

def adjacent(attr: str, pred: str, gt: str) -> bool:
    if attr == "dir":
        return mc.direction_adjacent(pred, gt)
    return mc.ordinal_adjacent(pred, gt, CLASSES[attr])


def classification_report(pairs: Sequence[Tuple[str, str]], attr: str) -> Dict:
    """pairs = [(pred, gt), ...]. Unmatched predictions should be passed as
    pred=None so they count as wrong (and against recall of the GT class)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    classes = CLASSES[attr]
    exact = sum(p == g for p, g in pairs)
    adj = sum(p is not None and adjacent(attr, p, g) for p, g in pairs)
    gt_counts = Counter(g for _, g in pairs)
    pred_counts = Counter(p for p, _ in pairs if p is not None)
    tp = Counter(g for p, g in pairs if p == g)
    present = [c for c in classes if gt_counts[c] > 0]
    recalls = {c: tp[c] / gt_counts[c] for c in present}
    f1s = {}
    for c in present:
        prec = tp[c] / pred_counts[c] if pred_counts[c] else 0.0
        rec = recalls[c]
        f1s[c] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    maj_class, maj_count = gt_counts.most_common(1)[0]
    return {
        "n": n,
        "exact": exact / n,
        "adjacent": adj / n,
        "balanced_acc": sum(recalls.values()) / len(recalls),
        "macro_f1": sum(f1s.values()) / len(f1s),
        "majority_class": maj_class,
        "majority_baseline_acc": maj_count / n,
        "majority_baseline_balanced_acc": 1.0 / len(recalls),
        "per_class_recall": recalls,
        "confusion": {g: dict(Counter(p for p, gg in pairs if gg == g)) for g in present},
    }


def majority_baseline(counts: Dict[str, int]) -> Tuple[str, float]:
    k, v = max(counts.items(), key=lambda x: x[1])
    return k, v / sum(counts.values())


# ============================================================================
# Self-consistency and error decomposition
# ============================================================================

def self_consistency(texts: Iterable[str], image_size=None) -> Dict:
    """SC = fraction of tags equal to M(T_hat_o) over tags with >= 2 grounded
    timestamps; schema_violation_rate over all tags."""
    try:
        from motion_reward_v4 import self_consistency_details
    except ImportError:
        from training.motion_reward_v4 import self_consistency_details
    rows = []
    for t in texts:
        th = mc.extract_think(t or "") or (t or "")
        rows.extend(self_consistency_details(th, image_size))
    evaluable = [r for r in rows if r["schema_ok"] and r["tag"]["well_formed"]]
    stat_when_static = [r for r in evaluable if r["implied"]["dir"] == "STAT"]
    return {
        "n_tags": len(rows),
        "n_evaluable": len(evaluable),
        "SC": sum(r["consistent"] for r in evaluable) / len(evaluable) if evaluable else None,
        "SC_per_attr": {a: (sum(r["tag"][a] == r["implied"][a] for r in evaluable) / len(evaluable))
                        if evaluable else None for a in ATTRS},
        "schema_violation_rate": sum(not r["schema_ok"] for r in rows) / len(rows) if rows else None,
        "frac_tags_on_static_boxes": len(stat_when_static) / len(evaluable) if evaluable else None,
        "moving_tag_on_static_boxes": sum(r["tag"]["dir"] != "STAT" for r in stat_when_static),
        # boxes identical at every timestamp: "static" because copied, not because observed
        "frac_tags_on_copied_boxes": sum(r["copied"] for r in evaluable) / len(evaluable) if evaluable else None,
    }


def error_decomposition(records: List[Dict]) -> Dict:
    """records: {"gt": tag, "pred": tag|None, "implied": M(T_hat)|None,
    "oracle_pred": tag|None} per GT object.

    total        = Err(M_hat, d*)
    localization = Err(M(T_hat), M(T_gt))
    reasoning    = Err(M_hat | T_gt)   (from the oracle-box run)
    """
    out = {}
    for a in ATTRS:
        def err(key):
            xs = [r for r in records if r.get(key) is not None]
            return (sum(r[key][a] != r["gt"][a] for r in xs) / len(xs), len(xs)) if xs else (None, 0)
        out[a] = {"total": err("pred"), "localization": err("implied"), "reasoning_oracle": err("oracle_pred")}
    return out


# ============================================================================
# Tag presence, intervention, paired accuracy
# ============================================================================

def has_wellformed_tag(text: str) -> bool:
    return any(t["well_formed"] for t in mc.parse_tags(text or ""))


def two_proportion_z(k1, n1, k2, n2) -> Optional[float]:
    if min(n1, n2) == 0:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))  # two-sided p


def tag_presence(preds: List[Dict], text_key="pred_text", correct_key="correct") -> Dict:
    with_tag = [p for p in preds if has_wellformed_tag(p.get(text_key, ""))]
    without = [p for p in preds if not has_wellformed_tag(p.get(text_key, ""))]
    k1 = sum(int(p[correct_key]) for p in with_tag)
    k2 = sum(int(p[correct_key]) for p in without)
    return {
        "n": len(preds),
        "rho_tag": len(with_tag) / len(preds) if preds else None,
        "acc_given_tag": k1 / len(with_tag) if with_tag else None,
        "acc_given_no_tag": k2 / len(without) if without else None,
        "n_tag": len(with_tag), "n_no_tag": len(without),
        "p_value_diff": two_proportion_z(k1, len(with_tag), k2, len(without)),
        "has_think": sum("<think>" in (p.get(text_key) or "") for p in preds) / len(preds) if preds else None,
    }


def paired_accuracy(pairs: List[Dict]) -> Dict:
    """pairs: {"pred": a, "gt": A, "pred_rev": b, "gt_rev": A_rev}."""
    n = len(pairs)
    if not n:
        return {"n": 0}
    both = sum(p["pred"] == p["gt"] and p["pred_rev"] == p["gt_rev"] for p in pairs)
    same = sum(p["pred"] == p["pred_rev"] for p in pairs)
    return {"n": n, "PA": both / n,
            "acc_forward": sum(p["pred"] == p["gt"] for p in pairs) / n,
            "acc_reversed": sum(p["pred_rev"] == p["gt_rev"] for p in pairs) / n,
            "same_answer_rate": same / n}


def tag_sensitivity(records: List[Dict]) -> Dict:
    """records: {"answer": A, "cf_answer": A~, "follows": True|False|None}."""
    n = len(records)
    det = [r for r in records if r.get("follows") is not None]
    return {"n": n,
            "TS": sum(r["answer"] != r["cf_answer"] for r in records) / n if n else None,
            "tag_following_rate": sum(bool(r["follows"]) for r in det) / len(det) if det else None,
            "n_following_determinable": len(det)}


# ============================================================================
# Direction words in answers (for tag-following and reversal templates)
# ============================================================================

_WORDS = [
    (r"\b(left|leftward|leftwards|west)\b", ("dir", "W")),
    (r"\b(right|rightward|rightwards|east)\b", ("dir", "E")),
    (r"\b(up|upward|upwards|rising|rises|north)\b", ("dir", "N")),
    (r"\b(down|downward|downwards|falling|falls|south)\b", ("dir", "S")),
    (r"\b(toward|towards) the camera\b|\b(closer|approach\w*|nearer)\b", ("scale", "approaching")),
    (r"\baway from the camera\b|\b(farther|further away|reced\w*)\b", ("scale", "receding")),
    (r"\b(stationary|still|not moving|motionless|static)\b", ("dir", "STAT")),
]


def motion_claims_in_text(text: str) -> Dict[str, set]:
    out = defaultdict(set)
    for pat, (attr, val) in _WORDS:
        if re.search(pat, (text or "").lower()):
            out[attr].add(val)
    return out


def dir_components(d: str) -> set:
    return {"E": {"E"}, "W": {"W"}, "N": {"N"}, "S": {"S"}, "NE": {"N", "E"}, "NW": {"N", "W"},
            "SE": {"S", "E"}, "SW": {"S", "W"}, "STAT": {"STAT"}}.get(d, set())


def answer_follows_tag(answer_text: str, tag: Dict[str, str]) -> Optional[bool]:
    """True/False when the answer text makes a direction/scale claim that can be
    checked against the tag; None when it makes no such claim."""
    claims = motion_claims_in_text(answer_text)
    if not claims:
        return None
    ok = True
    if claims.get("dir"):
        ok &= bool(claims["dir"] & dir_components(tag.get("dir", "")))
    if claims.get("scale"):
        ok &= tag.get("scale") in claims["scale"]
    return ok


# ============================================================================
# Seeds and significance
# ============================================================================

def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return m, sd


def welch_t(m1, s1, n1, m2, s2, n2) -> Dict:
    """Welch's t-test from summary statistics (two-sided)."""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:  # normal approximation, fine for df > 30
        p = math.erfc(abs(t) / math.sqrt(2))
    return {"t": t, "df": df, "p": p}
