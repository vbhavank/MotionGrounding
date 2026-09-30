#!/usr/bin/env python3
"""
Evaluate a video QA model on TVBench (https://huggingface.co/datasets/FunAILab/TVBench).

TVBench is a temporal-understanding video QA benchmark with 10 subsets (~2,525 samples).
It shares the same column schema as MVBench (video, question, answer, candidates)
but is specifically designed to require genuine temporal reasoning.

Videos must be available under --video_dir. Clone the HF repo to get them:
    git lfs install
    git clone https://huggingface.co/datasets/FunAILab/TVBench
    # Videos are under TVBench/video/<subset_name>/
    # Exception: action_antonym requires NTU RGB+D videos (manual download).

Usage:
    python scripts/eval_tvbench.py \
        --model_path outputs/.../merged \
        --video_dir /path/to/TVBench/video \
        [--subsets action_count moving_direction] \
        [--max_samples_per_subset 20] \
        --output_file evaluation/logs/tvbench_results.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# All 10 TVBench subsets
TVBENCH_SUBSETS = [
    "action_antonym",
    "action_count",
    "action_localization",
    "action_sequence",
    "egocentric_sequence",
    "moving_direction",
    "object_count",
    "object_shuffle",
    "scene_transition",
    "unexpected_action",
]

TVBENCH_SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question."
)


def resolve_video_path(video_dir: str, subset: str, video_filename: str) -> str:
    """
    Resolve video path. TVBench stores videos under video/<subset_name>/<filename>.
    Try subset folder first, then root, then common fallbacks.
    """
    for path in [
        os.path.join(video_dir, subset, video_filename),
        os.path.join(video_dir, video_filename),
        os.path.join(video_dir, subset, "video", video_filename),
    ]:
        if path and (os.path.isfile(path) or os.path.isdir(path)):
            return path
    return ""


def build_qa_prompt(question: str, candidates: list) -> str:
    """Build multiple-choice prompt with dynamic number of options."""
    prompt = f"Question: {question}\nOptions:\n"
    for i, c in enumerate(candidates):
        prompt += f"({chr(ord('A') + i)}) {c}\n"
    prompt = prompt.rstrip()
    option_letters = ", ".join(chr(ord('A') + i) for i in range(len(candidates)))
    prompt += f"\nAnswer with only the letter ({option_letters})."
    return prompt


def get_gt_index(candidates: list, answer: str) -> int:
    """Ground truth: index of answer in candidates."""
    try:
        return candidates.index(answer)
    except ValueError:
        answer_lower = answer.strip().lower()
        for i, c in enumerate(candidates):
            if c.strip().lower() == answer_lower:
                return i
    return -1


def extract_pred_letter(text: str, num_options: int) -> int:
    """
    Extract predicted option index from model output.
    Handles (A), A), A., standalone A, etc. Returns -1 if unclear.
    """
    text = text.strip().split("\n")[0]  # first line only
    for i in range(min(num_options, 26)):
        letter = chr(ord("A") + i)
        if f"({letter})" in text.upper():
            return i
        if re.search(rf"\b{letter}\s*[\.\)\:]", text, re.IGNORECASE):
            return i
        if re.search(rf"^{letter}\b", text.strip(), re.IGNORECASE):
            return i
    return -1


def build_vllm_input(processor, video_path: str, prompt: str, video_max_pixels: int, video_max_frames: int):
    """Build one vLLM input dict: video + text prompt."""
    messages = [
        {"role": "system", "content": TVBENCH_SYSTEM_PROMPT},
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
    parser = argparse.ArgumentParser(description="Evaluate on TVBench (FunAILab/TVBench)")
    parser.add_argument("--model_path", required=True, help="Path to merged model (full HF-style dir)")
    parser.add_argument("--video_dir", required=True, help="Root dir for TVBench videos (e.g. .../TVBench/video)")
    parser.add_argument("--output_file", default=None, help="JSON output path")
    parser.add_argument("--subsets", nargs="*", default=None, help="Subset names to run (default: all 10)")
    parser.add_argument("--max_samples_per_subset", type=int, default=None, help="Cap samples per subset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--video_max_pixels", type=int, default=2097152)
    parser.add_argument("--video_max_frames", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    args = parser.parse_args()

    subsets = args.subsets or TVBENCH_SUBSETS
    video_dir = os.path.abspath(args.video_dir)
    if not os.path.isdir(video_dir):
        print(f"ERROR: video_dir not found: {video_dir}")
        return 1

    # ── Load dataset per subset ──
    print("Loading TVBench from Hugging Face (FunAILab/TVBench)...")
    all_samples = []
    for subset in subsets:
        try:
            ds = load_dataset("FunAILab/TVBench", subset, split="train")
        except Exception as e:
            print(f"  WARN: Could not load subset '{subset}': {e}")
            continue
        subset_count = 0
        for idx, row in enumerate(ds):
            video_filename = row.get("video") or row.get("video_id", "")
            if isinstance(video_filename, list):
                video_filename = video_filename[0] if video_filename else ""
            video_path = resolve_video_path(video_dir, subset, video_filename)
            if not video_path:
                continue
            question = row.get("question", "")
            candidates = list(row.get("candidates", []))
            answer = row.get("answer", "")
            if len(candidates) < 2:
                continue
            gt_index = get_gt_index(candidates, answer)
            if gt_index < 0:
                continue
            prompt = build_qa_prompt(question, candidates)
            all_samples.append((subset, idx, row, video_path, prompt, gt_index))
            subset_count += 1
        print(f"  {subset}: {subset_count}/{len(ds)} samples with valid video path")

    if not all_samples:
        print("ERROR: No samples with valid video paths. Check --video_dir and dataset layout.")
        return 1

    # Optional cap per subset
    if args.max_samples_per_subset:
        by_subset = defaultdict(list)
        for t in all_samples:
            by_subset[t[0]].append(t)
        all_samples = []
        for sub in subsets:
            all_samples.extend(by_subset[sub][: args.max_samples_per_subset])
        print(f"  Capped to {args.max_samples_per_subset} per subset -> {len(all_samples)} total")

    # ── Init model ──
    print(f"\nLoading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 32, "video": 10},
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"

    # ── Run inference ──
    print(f"\nRunning inference on {len(all_samples)} samples (batch_size={args.batch_size})...")
    results_by_subset = defaultdict(lambda: {"correct": 0, "total": 0})
    all_preds = []
    t0 = time.time()
    done_count = 0
    for start in range(0, len(all_samples), args.batch_size):
        batch = all_samples[start : start + args.batch_size]
        inputs_and_meta = []
        for t in batch:
            subset, _, row, video_path, prompt, gt_index = t
            try:
                inp = build_vllm_input(
                    processor, video_path, prompt, args.video_max_pixels, args.video_max_frames
                )
                inputs_and_meta.append((inp, t))
            except Exception as e:
                print(f"  WARN: Skip sample: {e}")
                continue
        if not inputs_and_meta:
            continue
        inputs_only = [x[0] for x in inputs_and_meta]
        outputs = llm.generate(inputs_only, sampling_params=sampling_params)
        for out, (inp, (subset, _, row, _, prompt, gt_index)) in zip(outputs, inputs_and_meta):
            pred_text = out.outputs[0].text
            num_opts = len(row.get("candidates", []))
            pred_index = extract_pred_letter(pred_text, num_opts)
            correct = int(pred_index == gt_index)
            results_by_subset[subset]["total"] += 1
            results_by_subset[subset]["correct"] += correct
            all_preds.append({
                "subset": subset,
                "gt": gt_index,
                "pred": pred_index,
                "correct": correct,
                "pred_text": pred_text,
            })
        done_count += len(inputs_and_meta)
        elapsed = time.time() - t0
        print(f"  {done_count}/{len(all_samples)} ({100*done_count/len(all_samples):.0f}%) — {elapsed:.0f}s")
    total_time = time.time() - t0

    # ── Report ──
    total_correct = sum(r["correct"] for r in results_by_subset.values())
    total_count = sum(r["total"] for r in results_by_subset.values())
    overall_acc = 100.0 * total_correct / total_count if total_count else 0.0

    print("\n" + "=" * 60)
    print("TVBENCH RESULTS")
    print("=" * 60)
    print(f"Overall accuracy: {total_correct}/{total_count} = {overall_acc:.2f}%")
    print(f"Time: {total_time:.0f}s")
    print("\nPer-subset accuracy:")
    for sub in subsets:
        if sub not in results_by_subset:
            continue
        r = results_by_subset[sub]
        acc = 100.0 * r["correct"] / r["total"] if r["total"] else 0.0
        print(f"  {sub}: {r['correct']}/{r['total']} = {acc:.2f}%")

    out = {
        "benchmark": "TVBench",
        "model_path": args.model_path,
        "video_dir": video_dir,
        "overall_accuracy_pct": round(overall_acc, 2),
        "total_correct": total_correct,
        "total_count": total_count,
        "per_subset": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy_pct": round(100.0 * v["correct"] / v["total"], 2) if v["total"] else 0,
            }
            for k, v in results_by_subset.items()
        },
        "time_seconds": round(total_time, 1),
        "predictions_sample": all_preds[:100],
        "predictions": all_preds,  # full text: scripts/analyze_tag_presence.py
    }
    output_file = args.output_file or "evaluation/logs/tvbench_logs/tvbench_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())