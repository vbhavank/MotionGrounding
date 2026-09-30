#!/usr/bin/env python3
"""
Evaluate a video QA model on MotionBench (https://huggingface.co/datasets/zai-org/MotionBench).

MotionBench is a fine-grained video motion understanding benchmark with ~5000
multiple-choice questions across 6 categories:
  Action Order, Camera Motion, Location-related Motion,
  Motion Recognition, Motion-related Objects, Repetition Count

Each JSONL entry has this structure:
  {
    "question_type": "Action Order",
    "key": "37e1b635be3544d5a45106ea71c3b97c",
    "qa": [{"uid": "...", "answer": "C",
            "question": "Please describe ...\\nA. ...\\nB. ...\\nC. ..."}],
    "video_path": "37e1b635be3544d5a45106ea71c3b97c.mp4",
    "video_info": {"duration": 8.38, "fps": 60.0, ...}
  }

The question text embeds the options inline. The answer is a letter (A/B/C/D).
DEV samples have "answer" filled in; TEST samples may have it empty/null.

Usage:
    python scripts/eval_motionbench.py \
        --model_path outputs/.../merged \
        --video_dir /path/to/MotionBench/MotionBench \
        --meta_file /path/to/MotionBench/video_info.meta.jsonl \
        [--categories "Action Order" "Camera Motion"] \
        [--max_samples 50] \
        --output_file evaluation/logs/motionbench_results.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# All 6 MotionBench question categories
MOTIONBENCH_CATEGORIES = [
    "Action Order",
    "Camera Motion",
    "Location-related Motion",
    "Motion Recognition",
    "Motion-related Objects",
    "Repetition Count",
]

# Short aliases for convenience on CLI
CATEGORY_ALIASES = {
    "AO": "Action Order",
    "CM": "Camera Motion",
    "LM": "Location-related Motion",
    "MR": "Motion Recognition",
    "MO": "Motion-related Objects",
    "RC": "Repetition Count",
}

MOTIONBENCH_SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the fine-grained motion details, "
    "including the movement of objects, the actions and poses of people, and camera motion. "
    "Based on your observations, select the best option that accurately addresses the question. "
    "Answer with only the letter of your choice."
)


def resolve_category(name: str) -> str:
    """Resolve a category name or alias to full name."""
    if name in MOTIONBENCH_CATEGORIES:
        return name
    if name.upper() in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[name.upper()]
    # Fuzzy match
    name_lower = name.lower()
    for cat in MOTIONBENCH_CATEGORIES:
        if name_lower in cat.lower():
            return cat
    return name


def parse_question_and_options(question_text: str):
    """
    Parse the combined question+options string from MotionBench.

    Example input:
      "Please describe the detailed breakdown of the action in the video.\nA. Jump, Lay down, Stand\nB. Jump, Stand, Lay down\nC. Lay down, Stand, Jump"

    Returns: (question_str, [(letter, text), ...])
    """
    lines = question_text.strip().split("\n")

    question_lines = []
    options = []
    option_pattern = re.compile(r"^([A-Z])\.\s*(.*)")

    for line in lines:
        m = option_pattern.match(line.strip())
        if m:
            options.append((m.group(1), m.group(2)))
        else:
            question_lines.append(line)

    question = "\n".join(question_lines).strip()
    return question, options


def load_meta_jsonl(meta_file: str, categories: list = None):
    """
    Load video_info.meta.jsonl.
    Returns list of sample dicts.
    Only includes entries that have a ground truth answer (DEV set).
    """
    samples = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            q_type = entry.get("question_type", "")
            if categories and q_type not in categories:
                continue

            key = entry.get("key", "")
            video_path_name = entry.get("video_path", f"{key}.mp4")

            qa_list = entry.get("qa", [])
            for qa in qa_list:
                answer = qa.get("answer", "")
                if not answer or answer.upper() == "NA":
                    # Skip TEST set entries without ground truth
                    continue

                question_text = qa.get("question", "")
                if not question_text:
                    continue

                uid = qa.get("uid", "")

                # Parse question + options
                question, options = parse_question_and_options(question_text)

                samples.append({
                    "key": key,
                    "uid": uid,
                    "question_type": q_type,
                    "question": question,
                    "question_raw": question_text,
                    "options": options,
                    "answer": answer.strip().upper(),
                    "video_filename": video_path_name,
                    "video_info": entry.get("video_info", {}),
                })

    return samples


def resolve_video_path(video_dir: str, video_filename: str) -> str:
    """
    Resolve video path by searching subdirectories.
    Layout: video_dir/public-dataset/<file>.mp4 or video_dir/self-collected/<file>.mp4
    """
    candidates = [
        os.path.join(video_dir, video_filename),
        os.path.join(video_dir, "public-dataset", video_filename),
        os.path.join(video_dir, "self-collected", video_filename),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def build_qa_prompt(question_raw: str) -> str:
    """
    Build the prompt from the raw question text (already contains options).
    We just add the instruction to answer with a letter.
    """
    prompt = question_raw.strip()
    prompt += "\nAnswer with only the letter of your choice."
    return prompt


def extract_pred_letter(text: str, valid_letters: list) -> str:
    """
    Extract predicted option letter from model output.
    Returns the letter (e.g. "A") or "" if unclear.
    """
    text = text.strip().split("\n")[0]
    text_upper = text.upper()

    for letter in valid_letters:
        if f"({letter})" in text_upper:
            return letter
        if re.search(rf"\b{letter}\s*[\.\)\:]", text, re.IGNORECASE):
            return letter
        if re.search(rf"^{letter}\b", text.strip(), re.IGNORECASE):
            return letter
    return ""


def build_vllm_input(processor, video_path: str, prompt: str, video_max_pixels: int, video_max_frames: int):
    """Build one vLLM input dict: video + text prompt."""
    messages = [
        {"role": "system", "content": MOTIONBENCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": video_max_pixels,
                    "max_frames": video_max_frames,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    full_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    mm_data = {"video": video_inputs[0].numpy()} if video_inputs else {}
    return {"prompt": full_prompt, "multi_modal_data": mm_data}


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MotionBench")
    parser.add_argument("--model_path", required=True, help="Path to merged model (full HF-style dir)")
    parser.add_argument("--video_dir", required=True,
                        help="Root dir for MotionBench videos (contains public-dataset/ and self-collected/)")
    parser.add_argument("--meta_file", required=True,
                        help="Path to video_info.meta.jsonl")
    parser.add_argument("--output_file", default=None, help="JSON output path")
    parser.add_argument("--categories", nargs="*", default=None,
                        help=f"Question categories to run (default: all). "
                             f"Accepts full names or aliases: AO, CM, LM, MR, MO, RC")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total number of samples")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--video_max_pixels", type=int, default=2097152)
    parser.add_argument("--video_max_frames", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override max model sequence length (e.g. 32768 to save memory)")
    args = parser.parse_args()

    # Resolve category names
    categories = None
    if args.categories:
        categories = [resolve_category(c) for c in args.categories]
        print(f"Filtering to categories: {categories}")

    if not os.path.isfile(args.meta_file):
        print(f"ERROR: meta_file not found: {args.meta_file}")
        return 1
    if not os.path.isdir(args.video_dir):
        print(f"ERROR: video_dir not found: {args.video_dir}")
        return 1

    # ── Load samples ──
    print(f"Loading MotionBench metadata from {args.meta_file}...")
    samples = load_meta_jsonl(args.meta_file, categories=categories)
    print(f"  Loaded {len(samples)} QA samples (with ground truth)")

    # Resolve video paths
    valid = []
    missing_count = 0
    for s in samples:
        vp = resolve_video_path(args.video_dir, s["video_filename"])
        if vp:
            s["video_path"] = vp
            valid.append(s)
        else:
            missing_count += 1
    samples = valid
    print(f"  {len(samples)} samples with valid video paths ({missing_count} missing)")

    if not samples:
        print("ERROR: No valid samples found. Check --video_dir and --meta_file.")
        print(f"  Expected videos in: {args.video_dir}/public-dataset/ or {args.video_dir}/self-collected/")
        return 1

    # Cap samples
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
        print(f"  Capped to {args.max_samples} samples")

    # Print category distribution
    cat_counts = defaultdict(int)
    for s in samples:
        cat_counts[s["question_type"]] += 1
    print("\nSample distribution by category:")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]}")

    # ── Init model ──
    print(f"\nLoading model: {args.model_path}")
    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 32, "video": 10},
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    # ── Run inference ──
    print(f"\nRunning inference on {len(samples)} samples (batch_size={args.batch_size})...")
    results_by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds = []
    t0 = time.time()
    done = 0

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start: start + args.batch_size]
        inputs_meta = []
        for s in batch:
            prompt = build_qa_prompt(s["question_raw"])
            try:
                inp = build_vllm_input(
                    processor, s["video_path"], prompt,
                    args.video_max_pixels, args.video_max_frames,
                )
                inputs_meta.append((inp, s))
            except Exception as e:
                print(f"  WARN: Skip {s['key']}: {e}")
                continue

        if not inputs_meta:
            continue

        inputs_only = [x[0] for x in inputs_meta]
        outputs = llm.generate(inputs_only, sampling_params=sampling_params)

        for out, (inp, s) in zip(outputs, inputs_meta):
            pred_text = out.outputs[0].text
            valid_letters = [o[0] for o in s["options"]] if s["options"] else ["A", "B", "C", "D"]
            pred_letter = extract_pred_letter(pred_text, valid_letters)
            correct = int(pred_letter == s["answer"])

            cat = s["question_type"]
            results_by_cat[cat]["total"] += 1
            results_by_cat[cat]["correct"] += correct

            all_preds.append({
                "key": s["key"],
                "uid": s["uid"],
                "question_type": cat,
                "gt": s["answer"],
                "pred": pred_letter,
                "pred_raw": pred_text[:200],
                "pred_text": pred_text,
                "correct": correct,
            })

        done += len(inputs_meta)
        elapsed = time.time() - t0
        print(f"  {done}/{len(samples)} ({100 * done / len(samples):.0f}%) — {elapsed:.0f}s")

    total_time = time.time() - t0

    # ── Report ──
    total_correct = sum(r["correct"] for r in results_by_cat.values())
    total_count = sum(r["total"] for r in results_by_cat.values())
    overall_acc = 100.0 * total_correct / total_count if total_count else 0.0

    print("\n" + "=" * 60)
    print("MOTIONBENCH RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {total_correct}/{total_count} = {overall_acc:.2f}%")
    print(f"Time: {total_time:.0f}s")
    print("\nPer-category accuracy:")
    for cat in sorted(results_by_cat.keys()):
        r = results_by_cat[cat]
        acc = 100.0 * r["correct"] / r["total"] if r["total"] else 0.0
        print(f"  {cat}: {r['correct']}/{r['total']} = {acc:.2f}%")

    out = {
        "model_path": args.model_path,
        "video_dir": args.video_dir,
        "overall_accuracy_pct": round(overall_acc, 2),
        "total_correct": total_correct,
        "total_count": total_count,
        "per_category": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy_pct": round(100.0 * v["correct"] / v["total"], 2) if v["total"] else 0,
            }
            for k, v in results_by_cat.items()
        },
        "time_seconds": round(total_time, 1),
        "predictions_sample": all_preds[:100],
        "predictions": all_preds,  # full text: scripts/analyze_tag_presence.py
    }
    output_file = args.output_file or "evaluation/logs/motionbench_logs/motionbench_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())