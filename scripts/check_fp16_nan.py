"""Locate the first module that produces inf/NaN in a half-precision forward pass.

Use when RL sampling fails with "probability tensor contains either inf, nan or
element < 0". Checks the weights, runs one forward pass on a video prompt with a
hook on every module, reports the first non-finite output, then tries a short
greedy generation.

    python scripts/check_fp16_nan.py --model_path outputs/sft_v4_s42/merged \
        --video /path/to/video.mp4 --question "What is the person doing?" --attn sdpa
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from mcot_eval_common import MOTION_SYSTEM_PROMPT, build_prompt, load_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--question", default="Describe how the main object moves.")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if torch.cuda.get_device_capability(0) < (8, 0):
        torch.backends.cudnn.enabled = False
    dtype = getattr(torch, args.dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation=args.attn)
    model = model.to(dtype).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    print(f"[weights] non-finite tensors: {len(bad)}" + (f" e.g. {bad[:5]}" if bad else ""))

    video, _ = load_video(args.video, args.max_pixels, args.max_frames)
    prompt = build_prompt(processor, MOTION_SYSTEM_PROMPT, args.question)
    inputs = processor(text=[prompt], videos=[torch.from_numpy(video)], return_tensors="pt").to("cuda")

    first, maxabs = [], {}

    def hook(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t) or not t.is_floating_point():
                return
            maxabs[name] = t.detach().abs().float().nan_to_num(posinf=float("inf")).max().item()
            if not first and not torch.isfinite(t).all():
                first.append(name)
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    with torch.inference_mode():
        out = model(**inputs)
    for h in handles:
        h.remove()
    logits_ok = torch.isfinite(out.logits).all().item()
    print(f"[forward] logits finite: {logits_ok}; first non-finite module: {first[0] if first else None}")
    top = sorted(maxabs.items(), key=lambda kv: -kv[1])[:8]
    print("[forward] largest activations (fp16 max is 65504):")
    for n, v in top:
        print(f"   {v:12.1f}  {n}")

    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[generate] {text!r}")


if __name__ == "__main__":
    main()
