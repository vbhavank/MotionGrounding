#!/usr/bin/env python3
"""Side-by-side qualitative GIF: video, boxes, trajectories, tags and reasoning.

One column per model. Each frame shows
  * the ground-truth boxes (green, thin) and GT track centres,
  * the model's grounded boxes ``#k obj`` near the current time and the path of
    its box centres so far (one colour per object),
  * below: the answer, every <motion/> tag checked against M(model's own boxes)
    (self-consistency) and against M(GT track), and the reasoning with the
    current claim highlighted.
The header shows the question, the GT answer and the GT motion labels.

Generate with models (one GPU; models are loaded one after the other):

    python scripts/visualize_qualitative.py --dataset_json $J/splits_v4/eval_heldout.json \\
        --model "Paper protocol (SFT, video input)" outputs/sft_v4_s42/merged video \\
        --model "Ours (v4 RL, training-format input)" outputs/grpo_v4_mcot_s42_<job>/merged images \\
        --seed 3 --out_dir results/qualitative

Or reuse saved generations from eval_motion_tags_v2.py outputs (no GPU):

    python scripts/visualize_qualitative.py --dataset_json $J/splits_v4/eval_heldout.json \\
        --from_results "SFT" results/sft_v4_s42_tags_img.json \\
        --from_results "v4 RL" results/grpo_v4_mcot_s42_<job>_tags_img.json --n 5

FORMAT is ``images`` (training input: timestamped frames, 128*28*28 px/frame) or
``video`` (the paper's eval_motion_tags.py input: one <video>, 2097152 px/frame).
Model boxes are pixels of the frames the model saw; for ``video`` input the
processor may resize again, so those boxes are drawn approximately.
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))
import motion_core as mc  # noqa: E402

CLAIM_RE = re.compile(r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)\s*at\s*<t>\s*([\d.]+)\s*</t>\s*s", re.S)
TAG_RE = re.compile(r"<motion\s+([^>]*?)/?>")
GT_COLOR = (60, 220, 60)
PALETTE = [(255, 80, 80), (80, 160, 255), (255, 190, 40), (220, 90, 255), (40, 220, 220), (255, 130, 200)]
NBSP = " "


def font(size, bold=False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Parsing one generation
# ----------------------------------------------------------------------------
def name_sim(a, b):
    from difflib import SequenceMatcher
    a, b = mc.normalize_obj_name(a), mc.normalize_obj_name(b)
    if a == b:
        return 1.0
    if a and b and (a in b or b in a):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def fmt_m(m):
    return "/".join(m[k] for k in ("dir", "speed", "scale")) if m else "-"


def analyse(text, gt_motion):
    think = mc.extract_think(text or "") or (text or "")
    claims = mc.parse_claims(think)
    for k, c in enumerate(claims, 1):
        c["k"] = k
    tags = [t for t in mc.parse_tags(think) if t["well_formed"]]
    rows = []
    for t in tags:
        own = [c for c in claims if c["obj"] == t["obj"]]
        implied = mc.motion_descriptor(mc.claims_to_track(own)) if len({c["t"] for c in own}) >= 2 else None
        gt_name = max(gt_motion, key=lambda g: name_sim(g, t["obj"]), default=None)
        gt = gt_motion.get(gt_name) if gt_name and name_sim(gt_name, t["obj"]) > 0.5 else None
        rows.append({"tag": t, "implied": implied, "gt_name": gt_name if gt else None, "gt": gt,
                     "self_ok": mc.tag_equal(t, implied) if implied else None,
                     "gt_ok": mc.tag_equal(t, gt) if gt else None})
    m = re.search(r"<answer>(.*?)</answer>", text or "", re.S)
    answer = (m.group(1) if m else "").strip()

    # plain reasoning: claims -> "[#k obj @t s]", tags -> "{obj: dir/speed/scale}"
    k_iter = iter(range(1, 10 ** 6))

    def claim_tok(mt):
        k = next(k_iter)
        return f"[#{k}{NBSP}{mt.group(1).strip().replace(' ', NBSP)}{NBSP}@{float(mt.group(3)):g}s]"

    def tag_tok(mt):
        a = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', mt.group(1)))
        return "{" + f"{a.get('obj', '?')}:{NBSP}{a.get('dir', '?')}/{a.get('speed', '?')}/{a.get('scale', '?')}" \
            .replace(" ", NBSP) + "}"

    plain = CLAIM_RE.sub(claim_tok, think)
    plain = TAG_RE.sub(tag_tok, plain)
    plain = re.sub(r"</?think>", "", plain)
    plain = re.sub(r"[ \t\r\n\f\v]+", " ", plain).strip()  # keep NBSP inside tokens
    return {"claims": claims, "tag_rows": rows, "answer": answer, "plain": plain}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_box(d, box, color, width, label=None, fnt=None):
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if label:
        tw = d.textlength(label, font=fnt)
        d.rectangle([x0, max(0, y0 - 16), x0 + tw + 6, max(16, y0)], fill=color)
        d.text((x0 + 3, max(0, y0 - 16)), label, fill=(0, 0, 0), font=fnt)


def scale_box(b, sx, sy, W, H):
    x0, y0, x1, y1 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
    x0, x1 = sorted((min(max(x0, 0), W - 1), min(max(x1, 0), W - 1)))
    y0, y1 = sorted((min(max(y0, 0), H - 1), min(max(y1, 0), H - 1)))
    return [x0, y0, x1, y1]


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def render_frame(img, t, gen, model_size, gt_tracks, tol, fnt):
    """img: PIL display frame. Model boxes are in model_size pixels; GT normalized."""
    W, H = img.size
    d = ImageDraw.Draw(img)
    # GT: track centres so far + box near t
    def gt_px(b):  # GT boxes are normalized in STGR; pass pixel boxes through
        return scale_box(b, W, H, W, H) if mc.is_normalized(b) else scale_box(b, 1, 1, W, H)

    for name, tr in gt_tracks.items():
        pts = [centre(gt_px(b)) for tt, b in tr if tt <= t + 1e-6]
        if len(pts) > 1:
            d.line(pts, fill=GT_COLOR, width=1)
        for tt, b in tr:
            if abs(tt - t) <= tol:
                draw_box(d, gt_px(b), GT_COLOR, 1, f"GT {name}", fnt)
    # model
    sx, sy = W / model_size[0], H / model_size[1]
    objs = list(dict.fromkeys(c["obj"] for c in gen["claims"]))
    for oi, obj in enumerate(objs):
        col = PALETTE[oi % len(PALETTE)]
        own = sorted((c for c in gen["claims"] if c["obj"] == obj), key=lambda c: c["t"])
        pts = [centre(scale_box(c["box"], sx, sy, W, H)) for c in own if c["t"] <= t + 1e-6]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)
        for c in own:
            if abs(c["t"] - t) <= tol:
                draw_box(d, scale_box(c["box"], sx, sy, W, H), col, 3, f"#{c['k']} {c['raw_obj']}", fnt)
    d.rectangle([0, 0, 92, 20], fill=(0, 0, 0))
    d.text((5, 2), f"t = {t:4.1f} s", fill=(255, 255, 255), font=fnt)
    return img


def wrap(text, width):
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]


def text_panel(width, height, label, color, gen, t, fonts, chars):
    f, fb, fs = fonts
    img = Image.new("RGB", (width, height), (24, 24, 28))
    d = ImageDraw.Draw(img)
    y = 6
    d.text((8, y), label, fill=color, font=fb)
    y += 22
    for line in wrap("Answer: " + (gen["answer"] or "(none)"), chars)[:3]:
        d.text((8, y), line, fill=(235, 235, 235), font=f)
        y += 16
    y += 4
    d.text((8, y), "Motion tags  (tag | M(own boxes) | M(GT track))", fill=(170, 170, 170), font=fs)
    y += 15
    if not gen["tag_rows"]:
        d.text((8, y), "no <motion/> tag", fill=(200, 120, 120), font=f)
        y += 16
    for r in gen["tag_rows"][:4]:
        mark = lambda ok: "" if ok is None else (" ✓" if ok else " ✗")  # noqa: E731
        line = (f"{r['tag']['obj']}: {fmt_m(r['tag'])} | own {fmt_m(r['implied'])}{mark(r['self_ok'])}"
                f" | GT {fmt_m(r['gt'])}{mark(r['gt_ok'])}")
        for i, part in enumerate(wrap(line, chars)[:2]):
            ok = r["gt_ok"] if r["gt_ok"] is not None else r["self_ok"]
            col = (235, 235, 235) if ok is None else ((120, 230, 120) if ok else (240, 120, 120))
            d.text((8 + 12 * (i > 0), y), part, fill=col, font=f)
            y += 16
    y += 4
    d.text((8, y), "Reasoning  ([#k obj @t] = grounded box, {obj: tag})", fill=(170, 170, 170), font=fs)
    y += 15
    lines = wrap(gen["plain"], chars)
    past = [c for c in gen["claims"] if c["t"] <= t + 1e-6]
    cur = max(past, key=lambda c: (c["t"], c["k"]))["k"] if past else None
    hit = next((i for i, ln in enumerate(lines) if cur and f"[#{cur}{NBSP}" in ln), 0)
    n_fit = max(1, (height - y - 4) // 16)
    start = max(0, min(hit - n_fit // 3, len(lines) - n_fit))
    for i in range(start, min(len(lines), start + n_fit)):
        if i == hit and cur:
            d.rectangle([4, y - 1, width - 4, y + 15], fill=(80, 70, 20))
        d.text((8, y), lines[i].replace(NBSP, " "), fill=(225, 225, 225), font=f)
        y += 16
    return img


def header_panel(width, sample, gt_motion, fonts):
    f, fb, _ = fonts
    chars = max(40, width // 8)
    lines = wrap("Q: " + sample.get("question", ""), chars)[:2] + \
        wrap("GT answer: " + str(sample.get("answer", "")), chars)[:2] + \
        wrap("GT motion (M of GT boxes): " + ("; ".join(f"{k}: {fmt_m(v)}" for k, v in gt_motion.items()) or "-"),
             chars)[:2]
    img = Image.new("RGB", (width, 10 + 17 * len(lines) + 6), (12, 12, 14))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((8, 6 + 17 * i), ln, fill=GT_COLOR if ln.startswith("GT motion") else (240, 240, 240),
               font=fb if i == 0 else f)
    return img


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def clip_args(fmt, **kw):
    a = types.SimpleNamespace(frame_format=fmt, video_max_frames=16, insert_keyframes=False, kf_roots=None,
                              video_max_pixels=128 * 28 * 28 if fmt == "images" else 2097152)
    for k, v in kw.items():
        if v is not None:
            setattr(a, k, v)
    return a


def model_frame_size(path, a):
    from mcot_eval_common import clip_size, load_clip
    return clip_size(load_clip(a, path))


def generate(model_specs, samples, args):
    import torch
    from mcot_eval_common import MOTION_SYSTEM_PROMPT, HFEngine, build_prompt, clip_size, load_clip
    out = {}
    for label, path, fmt in model_specs:
        a = clip_args(fmt, insert_keyframes=args.insert_keyframes, kf_roots=args.kf_roots)
        a.model_path, a.temperature, a.max_tokens = path, 0.0, args.max_tokens
        engine = HFEngine(a)
        for s in samples:
            clip = load_clip(a, s["video_path_full"], s)
            text = engine.generate_one(build_prompt(engine.processor, MOTION_SYSTEM_PROMPT, s["question"], clip=clip),
                                       clip)
            out[(label, s["id"])] = {"text": text, "size": clip_size(clip), "format": fmt}
            print(f"[{label}] {s['id']}: {text[:160]!r}")
        del engine
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--model", nargs=3, action="append", default=[], metavar=("LABEL", "PATH", "FORMAT"))
    ap.add_argument("--from_results", nargs=2, action="append", default=[], metavar=("LABEL", "EVAL_JSON"))
    ap.add_argument("--id", action="append", default=[], help="sample id(s); default: random")
    ap.add_argument("--n", type=int, default=1, help="number of random samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moving_only", action="store_true", help="only samples with a non-STAT GT object")
    ap.add_argument("--insert_keyframes", action="store_true")
    ap.add_argument("--kf_roots", nargs=2, default=None)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--display_width", type=int, default=480)
    ap.add_argument("--display_frames", type=int, default=32)
    ap.add_argument("--panel_height", type=int, default=380)
    ap.add_argument("--ms_per_frame", type=int, default=600)
    ap.add_argument("--out_dir", default="results/qualitative")
    args = ap.parse_args()
    if not args.model and not args.from_results:
        ap.error("give --model LABEL PATH FORMAT and/or --from_results LABEL EVAL_JSON")
    if any(m[2] not in ("images", "video") for m in args.model):
        ap.error("FORMAT must be images or video")

    from mcot_eval_common import load_video

    data = json.load(open(args.dataset_json))
    saved = {}
    for label, f in args.from_results:
        res = json.load(open(f))
        ra = res.get("args", {})
        fmt = ra.get("frame_format", "video")  # files written before frame_format existed used video input
        saved[label] = ({s["id"]: s["text"] for s in res["per_sample"]}, fmt, ra.get("video_max_pixels"))

    def usable(s):
        if not os.path.isfile(s.get("video_path_full", "")):
            return False
        gt = mc.gt_motion_from_key_items(s.get("key_items") or {}, s.get("key_frames") or [])
        if not gt or (args.moving_only and all(v["dir"] == "STAT" for v in gt.values())):
            return False
        return all(s.get("id") in texts for texts, _, _ in saved.values())

    if args.id:
        samples = [s for s in data if str(s.get("id")) in set(args.id)]
    else:
        pool = [s for s in data if usable(s)]
        if not pool:
            sys.exit("no usable sample (video present, GT motion, in every results file)")
        samples = random.Random(args.seed).sample(pool, min(args.n, len(pool)))

    gens = generate(args.model, samples, args) if args.model else {}
    for label, (texts, fmt, mp) in saved.items():
        for s in samples:
            size = model_frame_size(s["video_path_full"], clip_args(fmt, video_max_pixels=mp))
            gens[(label, s["id"])] = {"text": texts[s["id"]], "size": size, "format": fmt}
    labels = [m[0] for m in args.model] + list(saved)

    fonts = (font(13), font(15, bold=True), font(11))
    os.makedirs(args.out_dir, exist_ok=True)
    for s in samples:
        gt_tracks = {k: v for k, v in mc.tracks_from_key_items(s.get("key_items") or {},
                                                               s.get("key_frames") or []).items()}
        gt_motion = mc.gt_motion_from_key_items(s.get("key_items") or {}, s.get("key_frames") or [])
        frames, fps = load_video(s["video_path_full"], args.display_width * args.display_width * 3 // 4,
                                 args.display_frames)
        frames = np.transpose(np.clip(frames, 0, 255).astype(np.uint8), (0, 2, 3, 1))
        times = [i / fps for i in range(len(frames))]
        tol = max(0.5, 0.75 / fps)
        H0, W0 = frames.shape[1:3]
        W = args.display_width
        H = int(round(H0 * W / W0))
        chars = max(30, W // 7)
        analysed = {lb: analyse(gens[(lb, s["id"])]["text"], gt_motion) for lb in labels}
        header = header_panel(W * len(labels), s, gt_motion, fonts)
        gif = []
        for fr, t in zip(frames, times):
            canvas = Image.new("RGB", (W * len(labels), header.height + H + args.panel_height), (0, 0, 0))
            canvas.paste(header, (0, 0))
            for j, lb in enumerate(labels):
                g = gens[(lb, s["id"])]
                img = Image.fromarray(fr).resize((W, H), Image.BILINEAR)
                img = render_frame(img, t, analysed[lb], g["size"], gt_tracks, tol, fonts[2])
                canvas.paste(img, (j * W, header.height))
                canvas.paste(text_panel(W, args.panel_height, f"{lb}  [{g['format']} input]",
                                        PALETTE[j % len(PALETTE)], analysed[lb], t, fonts, chars),
                             (j * W, header.height + H))
            gif.append(canvas)
        stem = os.path.join(args.out_dir, f"qual_{s['id']}")
        gif[0].save(stem + ".gif", save_all=True, append_images=gif[1:], duration=args.ms_per_frame, loop=0,
                    optimize=True)
        gif[-1].save(stem + "_last.png")
        json.dump({"sample": {k: s.get(k) for k in ("id", "question", "answer", "video_path_full", "source")},
                   "gt_motion": gt_motion,
                   "generations": {lb: gens[(lb, s["id"])] | {"tags": [
                       {"tag": fmt_m(r["tag"]), "obj": r["tag"]["obj"], "own_boxes": fmt_m(r["implied"]),
                        "gt": fmt_m(r["gt"]), "self_ok": r["self_ok"], "gt_ok": r["gt_ok"]}
                       for r in analysed[lb]["tag_rows"]]} for lb in labels}},
                  open(stem + ".json", "w"), indent=2, default=str)
        print(f"wrote {stem}.gif ({len(gif)} frames), {stem}_last.png, {stem}.json")


if __name__ == "__main__":
    main()
