#!/usr/bin/env python3
"""
Does the answer A depend on the <motion/> tag m?  (counterfactual intervention)

For each sample: generate R, A. Split R = R_pre + m + R_post at a well-formed
tag, replace m by a counterfactual m~ and regenerate the rest:
    A~ ~ pi(. | x, R_pre, m~)
Reports on motion-dependent questions
    TS  = P(A~ != A | m~ != m)                    tag sensitivity
    TF  = P(A~ consistent with m~)                tag-following (when checkable)
    TS0 = P(A' != A | same m, same prefix)        regeneration control
TS - TS0 is the effect attributable to the tag. TS ~ TS0 ~ 0 means the answer
ignores the tag (MCoT is decorative).

Input json: list of {"id", "video_path_full", "question", ["options"], ["answer"]}
(STGR-style dataset json works as is).

    python scripts/eval_tag_intervention.py --model_path M --input_json Q.json \
        --intervention dir --output_file ts.json [--motion_questions_only]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))
import motion_core as mc  # noqa: E402
import mcot_metrics as mm  # noqa: E402

MOTION_Q = re.compile(r"\b(mov\w*|direction|left|right|toward\w*|away|approach\w*|enter\w*|leav\w*|"
                      r"walk\w*|run\w*|turn\w*|rotat\w*|up|down|closer|farther|still|stationary|speed|fast|slow)\b",
                      re.IGNORECASE)


def counterfactual(tag, kind):
    d, s, c = tag["dir"], tag["speed"], tag["scale"]
    if kind == "dir":
        new = {"dir": mc.OPPOSITE[d], "speed": s, "scale": c} if d != "STAT" \
            else {"dir": "E", "speed": "fast", "scale": c}
    elif kind == "motion":
        new = {"dir": "E", "speed": "fast", "scale": c} if d == "STAT" \
            else {"dir": "STAT", "speed": "stationary", "scale": "stable"}
    elif kind == "scale":
        new = {"dir": d, "speed": s, "scale": {"approaching": "receding", "receding": "approaching",
                                                "stable": "approaching"}[c]}
    else:
        raise ValueError(kind)
    raw = f'<motion obj="{tag["obj"]}" dir="{new["dir"]}" speed="{new["speed"]}" scale="{new["scale"]}"/>'
    return new, raw


def format_question(s):
    q = s["question"]
    if s.get("options"):
        q += "\nOptions:\n" + "\n".join(s["options"]) + \
             "\nOutput only the option letter inside <answer> </answer>."
    return q


def answer_key(text, s):
    from mcot_eval_common import extract_answer, extract_letter
    if s.get("options"):
        return extract_letter(text, len(s["options"]))
    return re.sub(r"\W+", " ", extract_answer(text).lower()).strip()


def answer_text_for_following(text, s):
    from mcot_eval_common import extract_answer, extract_letter
    if s.get("options"):
        L = extract_letter(text, len(s["options"]))
        if L is None:
            return ""
        i = ord(L) - ord("A")
        return s["options"][i] if 0 <= i < len(s["options"]) else ""
    return extract_answer(text)


def main():
    ap = argparse.ArgumentParser()
    from mcot_eval_common import add_model_args
    add_model_args(ap)
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--intervention", choices=["dir", "motion", "scale"], default="dir")
    ap.add_argument("--which_tag", choices=["first", "last"], default="last")
    ap.add_argument("--motion_questions_only", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    from mcot_eval_common import MOTION_SYSTEM_PROMPT, build_prompt, generate, load_video, make_llm
    data = json.load(open(args.input_json))
    if args.motion_questions_only:
        data = [s for s in data if MOTION_Q.search(s["question"])]
    data = data[: args.max_samples] if args.max_samples else data
    llm, sp, processor = make_llm(args)

    videos, prompts, kept = [], [], []
    for s in data:
        try:
            v, _ = load_video(s["video_path_full"], args.video_max_pixels, args.video_max_frames)
        except Exception as e:
            print(f"skip {s.get('id')}: {e}")
            continue
        kept.append(s)
        videos.append(v)
        prompts.append(build_prompt(processor, MOTION_SYSTEM_PROMPT, format_question(s)))
    first = generate(llm, sp, list(zip(prompts, videos)), args.batch_size)

    cf_jobs, ctl_jobs, meta = [], [], []
    n_with_tag = 0
    for s, p, v, text in zip(kept, prompts, videos, first):
        tags = [t for t in mc.parse_tags(text) if t["well_formed"]]
        if not tags:
            continue
        n_with_tag += 1
        tag = tags[0] if args.which_tag == "first" else tags[-1]
        new, raw = counterfactual(tag, args.intervention)
        pre = text[: tag["start"]]
        cf_jobs.append((p + pre + raw, v))
        ctl_jobs.append((p + pre + tag["raw"], v))
        meta.append((s, text, tag, new, pre))
    cf = generate(llm, sp, cf_jobs, args.batch_size)
    ctl = generate(llm, sp, ctl_jobs, args.batch_size)

    records, control, rows = [], [], []
    for (s, text, tag, new, pre), cf_cont, ctl_cont in zip(meta, cf, ctl):
        cf_text = pre + f'<motion obj="{tag["obj"]}" dir="{new["dir"]}" speed="{new["speed"]}" scale="{new["scale"]}"/>' + cf_cont
        ctl_text = pre + tag["raw"] + ctl_cont
        a, a_cf, a_ctl = answer_key(text, s), answer_key(cf_text, s), answer_key(ctl_text, s)
        follows = mm.answer_follows_tag(answer_text_for_following(cf_text, s), new)
        records.append({"answer": a, "cf_answer": a_cf, "follows": follows})
        control.append({"answer": a, "cf_answer": a_ctl, "follows": None})
        rows.append({"id": s.get("id"), "tag": {k: tag[k] for k in ("obj", "dir", "speed", "scale")},
                     "cf_tag": new, "answer": a, "cf_answer": a_cf, "control_answer": a_ctl,
                     "follows": follows, "text": text, "cf_text": cf_text})

    ts = mm.tag_sensitivity(records)
    ts0 = mm.tag_sensitivity(control)
    summary = {"n_samples": len(kept), "rho_tag_under_prompt": n_with_tag / len(kept) if kept else None,
               "intervention": args.intervention, "TS": ts["TS"], "TS_control": ts0["TS"],
               "TS_minus_control": (ts["TS"] - ts0["TS"]) if ts["TS"] is not None else None,
               "tag_following_rate": ts["tag_following_rate"],
               "n_following_determinable": ts["n_following_determinable"], "n_intervened": ts["n"]}
    json.dump({"summary": summary, "args": vars(args), "rows": rows}, open(args.output_file, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
