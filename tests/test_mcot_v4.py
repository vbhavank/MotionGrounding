"""CPU-only tests for MCoT v4 (run: python -m pytest tests -q)."""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

import motion_core as mc  # noqa: E402
import motion_reward_v4 as mr  # noqa: E402
import mcot_metrics as mm  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_track(rng, n=None, step=20.0):
    n = n or rng.randint(2, 8)
    ts = sorted(rng.uniform(0, 30) for _ in range(n))
    x, y = rng.uniform(0, 500), rng.uniform(0, 500)
    out = []
    for t in ts:
        w, h = rng.uniform(5, 200), rng.uniform(5, 200)
        x += rng.gauss(0, rng.choice([0.1, 2.0, step]))
        y += rng.gauss(0, rng.choice([0.1, 2.0, step]))
        out.append((t, [x, y, x + w, y + h]))
    return out


def completion(text):
    return [{"role": "assistant", "content": text}]


DUCK = ("<think>At 16.0s, the <obj>duck</obj><box>[160,100,224,140]</box>at<t>16.0</t>s is behind. "
        "<obj>duck</obj><box>[160,100,224,140]</box>at<t>18.0</t>s remains. "
        "<obj>duck</obj><box>[160,100,224,140]</box>at<t>19.0</t>s still. "
        "<obj>duck</obj><box>[160,100,224,140]</box>at<t>21.0</t>s"
        '<motion obj="duck" dir="E" speed="moderate" scale="stable"/> seen.</think><answer>x</answer>')
SEASHELLS = ("<think>A <obj>hand</obj><box>[234,156,420,287]</box>at<t>0</t>s holds the bottle; "
             "<obj>seashells</obj><box>[0,209,420,364]</box> are visible. Later the "
             "<obj>seashells</obj><box>[0,209,420,364]</box>at<t>20</t>s"
             '<motion obj="seashells" dir="STAT" speed="stationary" scale="stable"/> remain.</think>'
             "<answer>C</answer>")


# ---------------------------------------------------------------------------
# M(T)
# ---------------------------------------------------------------------------

def test_descriptor_matches_sft_augmenter():
    aug = _load_script("augment_discrete_motion")
    rng = random.Random(0)
    for _ in range(3000):
        tr = random_track(rng, rng.randint(1, 8))
        a = aug.compute_motion_descriptor([b for _, b in tr], [t for t, _ in tr])
        assert mc.tag_equal(a, mc.motion_descriptor(tr))


@pytest.mark.parametrize("g", ["reverse", "hflip"])
def test_exact_equivariance(g):
    rng = random.Random(1)
    for _ in range(3000):
        tr = random_track(rng)
        m = mc.motion_descriptor(tr)
        mg = mc.motion_descriptor(mc.transform_track(tr, g, duration=30.0, width=1000.0))
        assert mc.tag_equal(mg, mc.rho(g, m))


def test_speedup_is_monotone_and_can_change_dir():
    rng = random.Random(2)
    changed_dir = 0
    for _ in range(3000):
        tr = random_track(rng)
        m, mg = mc.motion_descriptor(tr), mc.motion_descriptor(mc.transform_track(tr, "speedup", k=3.0))
        assert mc.SPEED_ORDER.index(mg["speed"]) >= mc.SPEED_ORDER.index(m["speed"])
        assert mg["scale"] == m["scale"]
        changed_dir += mg["dir"] != m["dir"]
    # the STAT<->stationary coupling means a speed-up can turn STAT into a direction,
    # which is why the reward target is M(g T_gt) rather than a fixed rho on labels
    assert changed_dir > 0


def test_full_freeze_is_stationary():
    rng = random.Random(3)
    for _ in range(500):
        tr = random_track(rng)
        assert mc.tag_equal(mc.motion_descriptor(mc.transform_track(tr, "freeze")), mc.STATIONARY_TAG)


@pytest.mark.parametrize("g", ["reverse", "hflip", "freeze"])
def test_transform_key_annotations_matches_track_transform(g):
    key_frames = [{"idx": i, "time": float(t)} for i, t in enumerate([1.0, 2.0, 4.0, 7.0])]
    key_items = {
        "0": {"car": [[0.10, 0.40, 0.20, 0.50]], "tree": [[0.7, 0.1, 0.8, 0.6]]},
        "1": {"car": [[0.20, 0.40, 0.30, 0.52]], "tree": [[0.7, 0.1, 0.8, 0.6]]},
        "2": {"car": [[0.40, 0.38, 0.52, 0.52]]},
        "3": {"car": [[0.70, 0.35, 0.85, 0.55]], "dog": [[0.1, 0.1, 0.2, 0.2]]},
    }
    orig = mc.gt_motion_from_key_items(key_items, key_frames)
    assert "dog" not in orig  # single-frame objects are not motion targets
    items2, frames2 = mc.transform_key_annotations(key_items, key_frames, g, duration=10.0)
    new = mc.gt_motion_from_key_items(items2, frames2)
    for obj in orig:
        assert mc.tag_equal(new[obj], mc.rho(g, orig[obj])), (g, obj, orig[obj], new[obj])


def test_transform_frames_numpy():
    frames = np.arange(5 * 1 * 2 * 3).reshape(5, 1, 2, 3)
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    f, t, d = mc.transform_frames(frames, times, "reverse", duration=5.0)
    assert (f[0] == frames[4]).all() and t == [1.0, 2.0, 3.0, 4.0, 5.0] and d == 5.0
    f, t, _ = mc.transform_frames(frames, times, "hflip", duration=5.0)
    assert (f[..., 0] == frames[..., 2]).all() and t == times
    f, t, d = mc.transform_frames(frames, times, "speedup", duration=5.0, k=2.0)
    assert t == [0.0, 0.5, 1.0, 1.5, 2.0] and d == 2.5
    f, _, _ = mc.transform_frames(frames, times, "freeze", duration=5.0, span=(1.5, 3.0))
    assert (f[2] == frames[1]).all() and (f[3] == frames[1]).all() and (f[4] == frames[4]).all()
    assert "Frame 2 at 0.5s" in mc.frame_prompt([0.0, 0.5], 1.0)


# ---------------------------------------------------------------------------
# 1. self-consistency + schema
# ---------------------------------------------------------------------------

def test_self_consistency_flags_duck_example():
    rows = mr.self_consistency_details(mc.extract_think(DUCK))
    assert len(rows) == 1 and rows[0]["implied"] == mc.STATIONARY_TAG and not rows[0]["consistent"]
    [r] = mr.motion_self_consistency_reward([completion(DUCK)], task=["temporal-spatial free-form QA"])
    assert r == pytest.approx(mc.W_SCALE)  # only scale=stable agrees
    fixed = DUCK.replace('dir="E" speed="moderate"', 'dir="STAT" speed="stationary"')
    [r] = mr.motion_self_consistency_reward([completion(fixed)], task=["temporal-spatial free-form QA"])
    assert r == pytest.approx(1.0)


def test_consecutive_tags_use_contiguous_windows():
    baby = ("<obj>baby</obj><box>[0,0,10,10]</box>at<t>1.5</t>s <obj>baby</obj><box>[5,0,15,10]</box>at<t>3.0</t>s"
            '<motion obj="baby" dir="E" speed="moderate" scale="stable"/> <obj>baby</obj><box>[0,0,10,10]</box>'
            'at<t>12.0</t>s<motion obj="baby" dir="W" speed="slow" scale="stable"/>')
    wins = mc.tag_evidence_windows(baby)
    assert [c["t"] for c in wins[0][1]] == [1.5, 3.0]
    assert [c["t"] for c in wins[1][1]] == [3.0, 12.0]
    assert mc.motion_descriptor(mc.claims_to_track(wins[1][1]))["dir"] == "W"


def test_schema_penalty_catches_single_timestamp_tag():
    assert mr.schema_violations(mc.extract_think(SEASHELLS)) == 1
    assert mr.schema_violations(mc.extract_think(DUCK)) == 0
    task = ["temporal-spatial free-form QA"]
    [bad] = mr.format_reward_v4([completion(SEASHELLS)], task=task)
    [good] = mr.format_reward_v4([completion(DUCK)], task=task)
    assert good == 1.0 and bad == pytest.approx(1.0 - mr.SCHEMA_BETA)
    [ctl] = mr.format_reward_notag([completion(SEASHELLS)], task=task)
    assert ctl == 1.0


def test_format_requires_tag_only_for_multi_timestamp_objects():
    task = ["temporal-spatial free-form QA"]
    untagged = DUCK.replace('<motion obj="duck" dir="E" speed="moderate" scale="stable"/>', "")
    [r] = mr.format_reward_v4([completion(untagged)], task=task)
    assert r == 0.5
    [r] = mr.format_reward_v4([completion("<think>no grounding</think><answer>B</answer>")],
                              task=["General video QA MCQ"])
    assert r == 1.0
    [r] = mr.format_reward_v4([completion("<think><obj>a</obj><box>[1,2,3,4]</think><answer>B</answer>")], task=task)
    assert r == 0.0  # unbalanced <obj> is still a hard failure


# ---------------------------------------------------------------------------
# 2. trajectory / equivariance rewards
# ---------------------------------------------------------------------------

KEY_FRAMES = [{"idx": 0, "time": 0.0}, {"idx": 1, "time": 2.0}, {"idx": 2, "time": 4.0}]
KEY_ITEMS = {"0": {"car": [[0.1, 0.4, 0.2, 0.5]]},
             "1": {"car": [[0.3, 0.4, 0.4, 0.5]]},
             "2": {"car": [[0.5, 0.4, 0.6, 0.5]]}}


def car_rollout(tag):
    claims = "".join(f"<obj>car</obj><box>[{0.1 + 0.2 * i:.1f},0.4,{0.2 + 0.2 * i:.1f},0.5]</box>at<t>{2.0 * i}</t>s "
                     for i in range(3))
    return f"<think>{claims}{tag}</think><answer>a</answer>"


def test_trajectory_reward_v4():
    target = mc.gt_motion_from_key_items(KEY_ITEMS, KEY_FRAMES, image_size=(640, 480))["car"]
    good = f'<motion obj="car" dir="{target["dir"]}" speed="{target["speed"]}" scale="{target["scale"]}"/>'
    kw = dict(task=["temporal-spatial free-form QA"] * 3, key_items=[KEY_ITEMS] * 3,
              key_frames=[KEY_FRAMES] * 3, image_size=[(640, 480)] * 3)
    r = mr.motion_trajectory_reward_v4([completion(car_rollout(good)), completion(car_rollout("")),
                                        completion(car_rollout(good.replace(target["dir"], "W")))], **kw)
    assert r[0] == pytest.approx(1.0) and r[1] == 0.0 and r[2] < 1.0


def test_trajectory_reward_matches_renamed_object_by_iou():
    """Rollout says "car", annotation says "red sedan": matched by box overlap, not name."""
    items = {k: {"red sedan": v["car"]} for k, v in KEY_ITEMS.items()}
    target = mc.gt_motion_from_key_items(items, KEY_FRAMES, image_size=(640, 480))["red sedan"]
    good = f'<motion obj="car" dir="{target["dir"]}" speed="{target["speed"]}" scale="{target["scale"]}"/>'
    kw = dict(task=["temporal-spatial free-form QA"] * 2, key_items=[items] * 2,
              key_frames=[KEY_FRAMES] * 2, image_size=[(640, 480)] * 2)
    far = "".join(f"<obj>car</obj><box>[0.8,0.8,0.9,0.9]</box>at<t>{2.0 * i}</t>s " for i in range(3))
    r = mr.motion_trajectory_reward_v4(
        [completion(car_rollout(good)), completion(f"<think>{far}{good}</think><answer>a</answer>")], **kw)
    assert r[0] == pytest.approx(1.0) and r[1] == 0.0
    # rollout boxes in pixels (Qwen2.5-VL absolute coords) against normalized annotations
    px = "".join(f"<obj>car</obj><box>[{(0.1 + 0.2 * i) * 640:.0f},192,{(0.2 + 0.2 * i) * 640:.0f},240]</box>"
                 f"at<t>{2.0 * i}</t>s " for i in range(3))
    [r] = mr.motion_trajectory_reward_v4([completion(f"<think>{px}{good}</think><answer>a</answer>")],
                                         **{k: v[:1] for k, v in kw.items()})
    assert r == pytest.approx(1.0)


def test_self_consistency_needs_grounded_boxes():
    """Copying one box and tagging STAT is self-consistent but not grounded."""
    task = ["temporal-spatial free-form QA"]
    kw = dict(task=task, key_items=[KEY_ITEMS], key_frames=[KEY_FRAMES], image_size=[(640, 480)])
    copied = "".join(f"<obj>car</obj><box>[0.1,0.4,0.2,0.5]</box>at<t>{2.0 * i}</t>s " for i in range(3))
    stat = '<motion obj="car" dir="STAT" speed="stationary" scale="stable"/>'
    [r_copy] = mr.motion_self_consistency_reward(
        [completion(f"<think>{copied}{stat}</think><answer>a</answer>")], **kw)
    [r_nogt] = mr.motion_self_consistency_reward(
        [completion(f"<think>{copied}{stat}</think><answer>a</answer>")], task=task)
    assert r_copy == pytest.approx(0.0) and r_nogt == pytest.approx(1.0)
    target = mc.gt_motion_from_key_items(KEY_ITEMS, KEY_FRAMES, image_size=(640, 480))["car"]
    good = f'<motion obj="car" dir="{target["dir"]}" speed="{target["speed"]}" scale="{target["scale"]}"/>'
    [r_good] = mr.motion_self_consistency_reward([completion(car_rollout(good))], **kw)
    assert r_good == pytest.approx(1.0)
    # a static object, correctly boxed at the same place every time, keeps full credit
    still = {k: {"car": [[0.1, 0.4, 0.2, 0.5]]} for k in KEY_ITEMS}
    [r_still] = mr.motion_self_consistency_reward(
        [completion(f"<think>{copied}{stat}</think><answer>a</answer>")],
        task=task, key_items=[still], key_frames=[KEY_FRAMES], image_size=[(640, 480)])
    assert r_still == pytest.approx(1.0)
    assert mr.copied_box_rate(copied) == 1.0 and mr.copied_box_rate(mc.extract_think(car_rollout(""))) == 0.0


def test_trajectory_reward_on_transformed_annotations():
    items2, frames2 = mc.transform_key_annotations(KEY_ITEMS, KEY_FRAMES, "reverse", duration=4.0)
    target = mc.gt_motion_from_key_items(items2, frames2, image_size=(640, 480))["car"]
    assert target["dir"] == "W"
    claims = "".join(f"<obj>car</obj><box>[{0.5 - 0.2 * i:.1f},0.4,{0.6 - 0.2 * i:.1f},0.5]</box>at<t>{2.0 * i}</t>s "
                     for i in range(3))
    text = f'<think>{claims}<motion obj="car" dir="W" speed="{target["speed"]}" scale="stable"/></think><answer>a</answer>'
    [r] = mr.motion_trajectory_reward_v4([completion(text)], task=["temporal-spatial free-form QA"],
                                         key_items=[items2], key_frames=[frames2], image_size=[(640, 480)])
    assert r == pytest.approx(1.0)


def test_label_free_equivariance_no_omission_credit():
    v = car_rollout('<motion obj="car" dir="E" speed="fast" scale="stable"/>')
    rev_ok = car_rollout('<motion obj="car" dir="W" speed="fast" scale="stable"/>')
    rev_same = v
    rev_missing = car_rollout("")
    task = ["temporal-spatial free-form QA"]
    for tr, expected in (({"reverse": [rev_ok]}, 1.0), ({"reverse": [rev_same]}, 2 / 3),
                         ({"reverse": [rev_missing]}, 0.0), ({"hflip": [rev_ok]}, 1.0)):
        [r] = mr.motion_equivariance_consistency_reward([completion(v)], task=task, transformed_completions=[tr])
        assert r == pytest.approx(expected), tr


# ---------------------------------------------------------------------------
# 5. representations
# ---------------------------------------------------------------------------

def back_and_forth():
    xs = [0, 40, 80, 120, 80, 40, 0]
    return [(float(i), [x, 100, x + 50, 150]) for i, x in enumerate(xs)]


def test_segment_track_splits_reversal():
    tr = back_and_forth()
    segs = mc.segment_track(tr)
    assert segs == [(0.0, 3.0), (3.0, 6.0)]
    assert [mc.motion_descriptor(mc.restrict_track(tr, *s))["dir"] for s in segs] == ["E", "W"]
    curve = [(float(i), [100 * math.cos(i / 4), 100 * math.sin(i / 4), 100 * math.cos(i / 4) + 30,
                         100 * math.sin(i / 4) + 30]) for i in range(5)]
    assert len(mc.segment_track(curve)) == 1


def test_piecewise_score_rewards_segmented_tags():
    tr = back_and_forth()
    seg_tags = [t for t in mc.parse_tags(
        '<motion obj="x" from="0" to="3" dir="E" speed="fast" scale="stable"/>'
        '<motion obj="x" from="3" to="6" dir="W" speed="fast" scale="stable"/>')]
    one_tag = [t for t in mc.parse_tags('<motion obj="x" from="0" to="6" dir="E" speed="fast" scale="stable"/>')]
    half = seg_tags[:1]
    s_seg, s_one, s_half = (mc.piecewise_score(x, tr) for x in (seg_tags, one_tag, half))
    assert s_seg == pytest.approx(1.0) and s_seg > s_one and s_half == pytest.approx(0.5)


def test_relational_motion():
    a = [(float(t), [10 * t, 0, 10 * t + 20, 20]) for t in range(4)]
    b = [(float(t), [10 * t + 50, 0, 10 * t + 70, 20]) for t in range(4)]
    assert mc.relational_descriptor(a, b)["dir"] == "STAT"   # moving together
    c = [(float(t), [100.0, 0, 120.0, 20]) for t in range(4)]
    assert mc.relational_descriptor(a, c)["dir"] == "E"


def test_scene_displacements_remove_camera_pan():
    # object is static in the scene; camera pans so the object slides W in the image
    track = [(float(t), [200 - 10 * t, 50, 240 - 10 * t, 90]) for t in range(5)]
    H = [[[1, 0, -10], [0, 1, 0], [0, 0, 1]]] * 4
    assert mc.motion_descriptor(track)["dir"] == "W"
    scene = mc.motion_descriptor(track, displacements=mc.scene_displacements(track, H))
    assert scene["dir"] == "STAT"
    assert np.allclose(mc.chain_homographies(H[:2]), [[1, 0, -20], [0, 1, 0], [0, 0, 1]])


def test_estimate_background_homography_translation():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    img = (rng.random((240, 320)) * 255).astype(np.uint8)
    import cv2
    img = cv2.GaussianBlur(img, (5, 5), 0)
    shifted = np.roll(img, 7, axis=1)
    H = mc.estimate_background_homography(img, shifted)
    assert abs(H[0][2] - 7) < 1.0 and abs(H[1][2]) < 1.0


def test_depth_scale():
    track = [(0.0, [10, 10, 50, 50]), (1.0, [10, 10, 50, 50])]
    near_then_far = [np.full((100, 100), 2.0), np.full((100, 100), 4.0)]
    assert mc.depth_scale_bin(track, near_then_far) == "receding"
    assert mc.depth_scale_bin(track, near_then_far[::-1]) == "approaching"
    assert mc.depth_scale_bin(track, near_then_far, is_disparity=True) == "approaching"
    assert mc.depth_scale_bin(track, [np.full((100, 100), 2.0), np.full((100, 100), 2.1)]) == "stable"


def test_parse_tag_attributes():
    [t] = mc.parse_tags('<motion obj="Duck 2" dir="E" speed="slow" scale="stable" from="1.5" to="3" '
                        'frame="scene" ref="duck"/>')
    assert t["obj"] == "duck 2" and t["from"] == 1.5 and t["to"] == 3.0
    assert t["frame"] == "scene" and t["ref"] == "duck" and t["well_formed"]
    [t] = mc.parse_tags(r'<motion obj=\"man\" dir=\"E\" speed=\"slow\" scale=\"receding\"/>')
    assert t["well_formed"]


# ---------------------------------------------------------------------------
# 3/4/6. metrics
# ---------------------------------------------------------------------------

def test_majority_baseline_from_paper_counts():
    k, acc = mm.majority_baseline(mm.PAPER_FIG_S5["dir"])
    assert k == "STAT" and acc == pytest.approx(0.576, abs=1e-3)
    assert mm.majority_baseline(mm.PAPER_FIG_S5["scale"])[1] == pytest.approx(0.593, abs=1e-3)


def test_classification_report_always_stat():
    gts = ["STAT"] * 6 + ["E"] * 2 + ["W"] * 2
    rep = mm.classification_report([("STAT", g) for g in gts], "dir")
    assert rep["exact"] == pytest.approx(0.6) and rep["balanced_acc"] == pytest.approx(1 / 3)
    assert rep["majority_baseline_acc"] == pytest.approx(0.6)
    rep = mm.classification_report([("NE", "E"), ("N", "E"), (None, "E")], "dir")
    assert rep["adjacent"] == pytest.approx(1 / 3) and rep["exact"] == 0


def test_welch_token_claim_not_significant():
    r = mm.welch_t(123, 90, 100, 138, 90, 100)
    assert r["p"] > 0.2


def test_paired_accuracy_static_model_scores_zero():
    pairs = [{"pred": "A", "gt": "A", "pred_rev": "A", "gt_rev": "B"}] * 5
    assert mm.paired_accuracy(pairs)["PA"] == 0.0
    assert mm.paired_accuracy(pairs)["acc_forward"] == 1.0


def test_tag_presence_and_following():
    preds = [{"pred_text": DUCK, "correct": 1}, {"pred_text": "B", "correct": 1}, {"pred_text": "C", "correct": 0}]
    r = mm.tag_presence(preds)
    assert r["rho_tag"] == pytest.approx(1 / 3) and r["acc_given_no_tag"] == 0.5
    assert mm.answer_follows_tag("It moves to the left", {"dir": "W", "scale": "stable"}) is True
    assert mm.answer_follows_tag("It moves to the left", {"dir": "E", "scale": "stable"}) is False
    assert mm.answer_follows_tag("A red car", {"dir": "E"}) is None


def test_self_consistency_metric():
    r = mm.self_consistency([DUCK, SEASHELLS])
    assert r["n_tags"] == 2 and r["n_evaluable"] == 1 and r["SC"] == 0.0
    assert r["schema_violation_rate"] == 0.5 and r["moving_tag_on_static_boxes"] == 1


# ---------------------------------------------------------------------------
# data scripts
# ---------------------------------------------------------------------------

def test_strip_and_relabel(tmp_path):
    strip = _load_script("strip_motion_tags")
    assert strip.strip_tags('a<obj>x</obj> <motion obj="x" dir="E" speed="slow" scale="stable"/> b') == "a<obj>x</obj> b"
    relabel = _load_script("relabel_motion_v4")
    sample = {"task": "temporal-spatial free-form QA", "key_frames": KEY_FRAMES,
              "key_items": {**KEY_ITEMS, "2": {**KEY_ITEMS["2"], "cup": [[0.1, 0.1, 0.2, 0.2]]}},
              "reasoning_process": "<obj>car</obj><box>[1,2,3,4]</box>at<t>0.0</t>s then "
                                   "<obj>car</obj><box>[5,2,7,4]</box>at<t>4.0</t>s. "
                                   "<obj>cup</obj><box>[1,1,2,2]</box>at<t>4.0</t>s"
                                   '<motion obj="cup" dir="STAT" speed="stationary" scale="stable"/>'}
    out, emitted = relabel.relabel(sample)
    assert "cup" not in out["gt_motion"] and 'obj="cup"' not in out["reasoning_process"]
    assert out["reasoning_process"].count('<motion obj="car"') == 1 and len(emitted) == 1


def test_reversal_pair_builder(tmp_path):
    build = _load_script("build_reversal_pairs")
    d, cands = build.candidates(mc.tracks_from_key_items(KEY_ITEMS, KEY_FRAMES)["car"])
    assert ("horizontal", "right") in cands
    data = [{"id": "v1", "video_path_full": "v.mp4", "key_items": KEY_ITEMS, "key_frames": KEY_FRAMES}]
    (tmp_path / "d.json").write_text(json.dumps(data))
    sys.argv = ["x", "--dataset_json", str(tmp_path / "d.json"), "--output", str(tmp_path / "p.json")]
    build.main()
    [p] = json.loads((tmp_path / "p.json").read_text())
    assert p["answer"] != p["answer_rev"]
