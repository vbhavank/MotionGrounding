"""Shared helpers for the MCoT evaluation scripts (lazy heavy imports).

Input format. SFT and RL never show the model a ``<video>``: they pass the
sampled frames as separate images, each preceded by ``Frame i at t s:``, and
insert the annotated keyframes for grounded tasks (train_sft_v2.collate_fn,
grpo_trainer_v4._prepare_sample). ``--frame_format images`` (default)
reproduces that; ``video`` is the v1 eval_motion_tags.py protocol, where the
model gets no timestamps and cannot anchor ``<t>`` (claims collapse to 0-5 s
and boxes are copied across times). Frame size defaults to the training value
(vision_process VIDEO_MAX_PIXELS = 128*28*28 per frame).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same prompt as scripts/eval_motion_tags.py (trajectory-grounded, Appendix F.3 style)
MOTION_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a video and asks a question, "
    "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
    "and then provide the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual "
    "evidence from the video. When you mention any related object, person, or specific visual element "
    "in the reasoning process, you must strictly follow the following format: "
    "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. "
    "After the last observation of each object, you MUST describe its motion trajectory using a self-closing "
    "motion tag with discrete attributes: "
    '`<motion obj="object_name" dir="DIRECTION" speed="SPEED" scale="SCALE"/>` '
    "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
    "SPEED is one of {stationary, slow, moderate, fast}, "
    "and SCALE is one of {approaching, stable, receding}. "
    "The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
)

# Appendix F.4 letter-only benchmark prompt
LETTER_SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to every detail. "
    "Based on your observations, select the option that best answers "
    "the question. Answer with only the letter of your choice."
)


def add_model_args(ap):
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--video_max_pixels", type=int, default=128 * 28 * 28,
                    help="per-frame pixels; SFT/RL use 128*28*28 (v1 eval used 2097152)")
    ap.add_argument("--video_max_frames", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto",
                    help="auto: vLLM when importable on a compute>=8.0 GPU, else transformers generate")
    ap.add_argument("--frame_format", choices=["images", "video"], default="images",
                    help="images: training format (timestamped frames); video: v1 protocol")
    ap.add_argument("--insert_keyframes", action="store_true",
                    help="insert the annotated keyframes as in training (uses GT timing: a diagnostic, "
                         "not the deployment setting)")
    ap.add_argument("--kf_roots", nargs=2, default=None, metavar=("PLM_KFS", "TG_KFS"),
                    help="keyframe folders; default DATA_ROOT/videos/stgr/{plm,temporal_grounding}/kfs")


def _use_vllm(args):
    if args.backend != "auto":
        return args.backend == "vllm"
    try:
        import torch
        import vllm  # noqa: F401
        return torch.cuda.get_device_capability(0) >= (8, 0)
    except Exception:
        return False


class HFEngine:
    """transformers.generate backend (no vLLM). Works on V100s: fp16 instead of bf16,
    cuDNN disabled for the Conv3d patch embedding, SDPA attention."""

    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
        self.torch, self.args = torch, args
        pre_ampere = torch.cuda.get_device_capability(0) < (8, 0)
        if pre_ampere:
            torch.backends.cudnn.enabled = False
        dtype = torch.float16 if pre_ampere else torch.bfloat16
        cls = Qwen2VLForConditionalGeneration if "Qwen2-VL" in args.model_path else Qwen2_5_VLForConditionalGeneration
        model = cls.from_pretrained(args.model_path, attn_implementation="sdpa")
        self.model = model.to(dtype).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        print(f"[eval] transformers backend, dtype={self.model.dtype}")

    def generate_one(self, prompt, media):
        torch = self.torch
        if isinstance(media, dict) and media.get("format") == "images":
            frames = torch.from_numpy(np.ascontiguousarray(media["frames"]))
            inputs = self.processor(text=[prompt], images=[frames], videos=None, return_tensors="pt",
                                    padding=True, do_resize=False).to("cuda")
        else:
            video = media["frames"] if isinstance(media, dict) else media
            v = torch.from_numpy(np.ascontiguousarray(video)) if not torch.is_tensor(video) else video
            inputs = self.processor(text=[prompt], videos=[v], return_tensors="pt", padding=True).to("cuda")
        sample = self.args.temperature > 0
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.args.max_tokens, do_sample=sample,
                                      temperature=self.args.temperature if sample else None,
                                      repetition_penalty=1.05)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


import numpy as np  # noqa: E402

VIDEO_PAD = "<|vision_start|><|video_pad|><|vision_end|>"


def make_llm(args):
    from transformers import AutoProcessor
    if not _use_vllm(args):
        engine = HFEngine(args)
        return engine, None, engine.processor
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
              max_num_seqs=args.batch_size, gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 32, "video": 10}, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, repetition_penalty=1.05,
                        max_tokens=args.max_tokens, seed=args.seed)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    return llm, sp, processor


def load_video(path, max_pixels, max_frames):
    """(T, C, H, W) uint8/float numpy array, as eval_motion_tags.py feeds vLLM."""
    try:  # repo copy: decord -> OpenCV -> torchvision fallback (new torchvision has no read_video)
        from vision_process import process_vision_info
    except ImportError:
        from qwen_vl_utils import process_vision_info
    msgs = [{"role": "user", "content": [{"type": "video", "video": path, "max_pixels": max_pixels,
                                          "max_frames": max_frames}]}]
    _, video_inputs, video_kwargs = process_vision_info(msgs, return_video_kwargs=True)
    fps = (video_kwargs or {}).get("fps", [2.0])[0]
    return video_inputs[0].numpy(), fps


def _kf_roots(args):
    if getattr(args, "kf_roots", None):
        return args.kf_roots
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from configs.data_root import DATA_ROOT
    except ImportError:
        raise SystemExit("--insert_keyframes needs --kf_roots PLM_KFS TG_KFS (configs.data_root not importable)")
    base = os.path.join(DATA_ROOT, "videos", "stgr")
    return os.path.join(base, "plm", "kfs"), os.path.join(base, "temporal_grounding", "kfs")


def _insert_keyframes(frames, fps, sample, args):
    """train_sft_v2.collate_fn: a keyframe replaces the slot at the first frame whose
    integer second reaches its time, and is labeled with its own time."""
    from PIL import Image
    plm_root, tg_root = _kf_roots(args)
    root = plm_root if "plm" in (sample.get("source") or "").lower() else tg_root
    h, w = frames.shape[2], frames.shape[3]
    kfs = []
    for kf in sorted(sample.get("key_frames") or [], key=lambda k: float(k["time"])):
        im = Image.open(os.path.join(root, kf["path"])).convert("RGB").resize((w, h))
        kfs.append((float(kf["time"]), np.transpose(np.asarray(im), (2, 0, 1)).astype(frames.dtype)))
    out, times, k, i = [], [], 0, 0
    while i < len(frames):
        if k < len(kfs) and int(i / fps) >= kfs[k][0]:
            out.append(kfs[k][1])
            times.append(kfs[k][0])
            k += 1
        else:
            out.append(frames[i])
            times.append(round(i / fps, 1))
            i += 1
    return np.stack(out), times


def load_clip(args, path, sample=None):
    """Frames + displayed timestamps in the format chosen by --frame_format."""
    frames, fps = load_video(path, args.video_max_pixels, args.video_max_frames)
    clip = {"format": getattr(args, "frame_format", "video"), "frames": frames, "fps": fps,
            "times": [round(i / fps, 1) for i in range(len(frames))], "total": len(frames) / fps}
    if clip["format"] == "images" and getattr(args, "insert_keyframes", False) and sample \
            and sample.get("key_frames"):
        clip["frames"], clip["times"] = _insert_keyframes(frames, fps, sample, args)
    return clip


def reverse_clip(clip):
    """Play the clip backwards; displayed timestamps stay ascending."""
    return dict(clip, frames=clip["frames"][::-1].copy())


def clip_size(clip):
    """(W, H) of the frames the model sees: boxes are in these pixel coordinates."""
    return clip["frames"].shape[3], clip["frames"].shape[2]


def build_prompt(processor, system, question, prefix="", clip=None):
    """Chat prompt ending in the assistant turn, optionally pre-filled with
    ``prefix`` (used for oracle boxes and counterfactual tag continuation).
    With an images-format ``clip`` the video placeholder becomes the training
    frame list (``Frame i at t s: <image>`` ... ``The video is in total N seconds.``)."""
    import motion_core as mc
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if clip is not None and clip.get("format") == "images":
        text = text.replace(VIDEO_PAD, mc.frame_prompt(clip["times"], clip["total"]))
    return text + prefix


def generate(llm, sp, prompts_and_videos, batch_size=8):
    """prompts_and_videos: list of (prompt, clip or video_array). Returns texts."""
    if hasattr(llm, "generate_one"):
        out = []
        for i, (p, v) in enumerate(prompts_and_videos):
            out.append(llm.generate_one(p, v))
            if (i + 1) % 20 == 0 or i + 1 == len(prompts_and_videos):
                print(f"  generated {i + 1}/{len(prompts_and_videos)}", flush=True)
        return out
    def mm(media):
        if isinstance(media, dict) and media.get("format") == "images":
            from PIL import Image
            return {"image": [Image.fromarray(np.transpose(f, (1, 2, 0)).clip(0, 255).astype(np.uint8))
                              for f in media["frames"]]}
        return {"video": media["frames"] if isinstance(media, dict) else media}

    out = []
    for i in range(0, len(prompts_and_videos), batch_size):
        chunk = prompts_and_videos[i:i + batch_size]
        res = llm.generate([{"prompt": p, "multi_modal_data": mm(v)} for p, v in chunk], sampling_params=sp)
        out.extend(r.outputs[0].text for r in res)
    return out


def extract_answer(text):
    import re
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()


def extract_letter(text, n_options=4):
    import re
    ans = extract_answer(text)
    for i in range(n_options):
        L = chr(ord("A") + i)
        if re.match(rf"^\(?{L}\b", ans.strip(), re.IGNORECASE):
            return L
    m = re.search(r"\b([A-H])\b", ans)
    return m.group(1).upper() if m else None
