"""
GRPO/GSPO trainer v4: equivariance views instead of a single motion-masked chain.

Differences from grpo_trainer_v3.Qwen2VLGRPOTrainer
---------------------------------------------------
* v3 set ``reward_kwargs['masked_completion']`` *after* each reward call on a
  dict that is rebuilt every iteration, so ``motion_grounding_reward`` never
  received the masked chain and always returned 0 (see revision/README.md).
  v4 builds reward kwargs once and passes auxiliary rollouts explicitly.
* r_ground' is implemented as extra GRPO groups: for a transformation g sampled
  from ``equiv_transforms`` (reverse | hflip | speedup | freeze) the policy is
  sampled ``equiv_num_generations`` times on gV, and those rollouts are scored
  by ``equiv_reward_funcs`` against the *transformed* annotations
  (``motion_core.transform_key_annotations``), i.e. against M(g T_gt). Their
  GSPO loss is added with weight ``equiv_weight``. A reward that depends only
  on the gV rollout cannot be credited to the original rollouts: it is
  constant within their group, so its advantage would be 0.
* The label-free term (``motion_equivariance_consistency``) *does* depend on
  the original rollout, so it is an ordinary reward on the original group; it
  receives the gV texts through ``transformed_completions``.
* ``reward_weights`` (e.g. lambda for r_self) are applied before summation.
* Masked spans for ``freeze`` come from ground-truth keyframes, never from the
  policy's own output.
"""

import copy
import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image
import numpy as np
from transformers import (GenerationConfig, PreTrainedModel, Qwen2VLForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration, Trainer)
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation

from grpo_trainer_v3 import (
    Qwen2VLGRPOTrainer,
    VIDEO_ESPRESSO_ROOT, VIDEO_ESPRESSO_KF_ROOT, TIMERFT_ROOT, GQA_ROOT,
    STR_PLM_DATA, STR_DATA, STR_PLM_KF_ROOT, STR_KF_ROOT, TVG_ROOT, GENERAL_VIDEO_ROOT,
)
from vision_process import process_vision_info

try:
    from training import motion_core as mc
    from training import motion_reward_v4 as mr4
except ImportError:
    import motion_core as mc
    import motion_reward_v4 as mr4

VISION_SPECIAL_IDS = [151652, 151653, 151654, 151655, 151656]
MOTION_TASKS = {"temporal-spatial free-form QA", "General video QA Free-form", "General video QA MCQ"}
VIDEO_PAD = "<|vision_start|><|video_pad|><|vision_end|>"


class Qwen2VLGRPOTrainerV4(Qwen2VLGRPOTrainer):

    def __init__(self, *args,
                 equiv_transforms=("reverse", "hflip", "freeze"),
                 equiv_reward_funcs=None,
                 equiv_weight: float = 1.0,
                 equiv_num_generations: int = 2,
                 equiv_per_step: int = 1,
                 equiv_speedup_k: float = 2.0,
                 freeze_span: str = "first_keyframe",
                 reward_weights: Optional[List[float]] = None,
                 **kwargs):
        # Pre-Ampere GPUs (V100): no bf16, and cuDNN lacks a half-precision Conv3d engine for
        # the vision patch embedding. MCOT_DTYPE=float16 preloads and casts the model here
        # (transformers 5.x may ignore torch_dtype); cuDNN is disabled below compute 8.0.
        flag = os.environ.get("MCOT_DISABLE_CUDNN")
        if flag == "1" or (flag is None and torch.cuda.is_available()
                           and torch.cuda.get_device_capability(0) < (8, 0)):
            torch.backends.cudnn.enabled = False
            print("[v4] cuDNN disabled (pre-Ampere GPU or MCOT_DISABLE_CUDNN=1)")
        dtype_name = os.environ.get("MCOT_DTYPE")
        if dtype_name and isinstance(kwargs.get("model"), str):
            path, dtype = kwargs["model"], getattr(torch, dtype_name)
            cls = Qwen2VLForConditionalGeneration if "Qwen2-VL" in path else Qwen2_5_VLForConditionalGeneration
            attn = kwargs.get("attn_implementation") or "sdpa"
            if dtype == torch.float16 and attn == "eager":
                # eager attention overflows in fp16 (unscaled Q.K^T) -> NaN logits in sampling
                print("[v4] fp16: eager attention overflows, using sdpa")
                attn = "sdpa"
            kwargs["attn_implementation"] = attn
            model = cls.from_pretrained(path, attn_implementation=attn)
            model = model.to(dtype)
            if kwargs.get("args") is not None:
                model.config.use_cache = not kwargs["args"].gradient_checkpointing
                kwargs["args"].model_init_kwargs = None
            kwargs["model"] = model
            print(f"[v4] model preloaded as {model.dtype}, attention={attn}")
        super().__init__(*args, **kwargs)
        self.equiv_transforms = list(equiv_transforms or [])
        self.equiv_reward_funcs = list(equiv_reward_funcs or [])
        self.equiv_weight = equiv_weight
        self.equiv_num_generations = equiv_num_generations
        self.equiv_per_step = equiv_per_step
        self.equiv_speedup_k = equiv_speedup_k
        self.freeze_span = freeze_span
        if reward_weights is None:
            reward_weights = [1.0] * len(self.reward_funcs)
        if len(reward_weights) != len(self.reward_funcs):
            raise ValueError("reward_weights must match reward_funcs")
        self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Input construction (v3 logic, plus the displayed frame timestamps)
    # ------------------------------------------------------------------
    def _prepare_sample(self, inputs):
        prompts_text = [maybe_apply_chat_template(ex, self.processing_class)["prompt"] for ex in inputs]
        input_copy = [copy.deepcopy(inputs[0]["prompt"][1])]
        src = inputs[0]["source"]
        if src == "videoespresso_train_video":
            input_copy[0]["content"][0]["video"] = os.path.join(VIDEO_ESPRESSO_ROOT, inputs[0]["video_path"])
        elif src == "timerft":
            input_copy[0]["content"][0]["video"] = os.path.join(TIMERFT_ROOT, inputs[0]["video_path"])
        elif src == "gqa":
            input_copy[0]["content"][0]["image"] = os.path.join(GQA_ROOT, inputs[0]["image_path"])
        elif "STR" in src:
            root = STR_PLM_DATA if "STR_plm" in src else STR_DATA
            input_copy[0]["content"][0]["video"] = os.path.join(root, inputs[0]["video_path"])
        elif "TVG" in src:
            input_copy[0]["content"][0]["video"] = os.path.join(TVG_ROOT, inputs[0]["video_path"])
        elif "videor1" in src:
            input_copy[0]["content"][0]["video"] = os.path.join(GENERAL_VIDEO_ROOT, inputs[0]["video_path"])
        else:
            raise ValueError(f"Invalid source: {src}")
        input_copy = self.remove_none_from_data(input_copy)

        if "key_items" in inputs[0] and inputs[0]["key_items"]:
            for key in [k for k, v in inputs[0]["key_items"].items() if v is None]:
                del inputs[0]["key_items"][key]
            for item in inputs[0]["key_items"].values():
                if isinstance(item, dict):
                    for k in [k for k, v in item.items() if v is None]:
                        del item[k]

        image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        if image_inputs is not None:
            inputs[0]["image_size_refine"] = (image_inputs[0].size[0], image_inputs[0].size[1])
        if video_inputs is not None:
            fps = video_kwargs["fps"][0]
            inputs[0]["video_sample_fps"] = fps
            inputs[0]["video_duration"] = video_inputs[0].size(0) / fps
            inputs[0]["image_size"] = (video_inputs[0].size(3), video_inputs[0].size(2))
        inputs[0]["step_percent"] = (self.state.global_step + 1) / self.state.max_steps

        ctx = {"base_prompt": prompts_text[0], "multi_image": video_inputs is not None,
               "frames": None, "times": None, "duration": None}
        if video_inputs is None:
            pi = self.processing_class(text=copy.deepcopy(prompts_text), images=image_inputs, videos=None,
                                       return_tensors="pt", padding=True, padding_side="left",
                                       add_special_tokens=False)
            inputs[0]["prompt_text_final"] = prompts_text[0]
            ctx["prompt_inputs"] = pi
            return ctx

        fps = video_kwargs["fps"][0]
        vid = video_inputs[0]
        if inputs[0]["task"] != "temporal-spatial free-form QA":
            frames = vid
            times = [round(i / fps, 1) for i in range(len(vid))]
        else:
            if src == "videoespresso_train_video":
                kf_root = VIDEO_ESPRESSO_KF_ROOT
            elif "STR_plm" in src:
                kf_root = STR_PLM_KF_ROOT
            else:
                kf_root = STR_KF_ROOT
            image_size = (vid.size(3), vid.size(2))
            key_frames = []
            for kf in inputs[0]["key_frames"]:
                im = Image.open(os.path.join(kf_root, kf["path"])).convert("RGB").resize(image_size)
                key_frames.append((round(kf["time"]), torch.from_numpy(np.transpose(np.array(im), (2, 0, 1)))))
            refined, times = [], []
            kf_idx = ori_idx = 0
            while ori_idx < len(vid):
                time_now = int(ori_idx / fps)
                if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                    refined.append(key_frames[kf_idx][1])
                    times.append(round(key_frames[kf_idx][0], 1))
                    kf_idx += 1
                else:
                    refined.append(vid[ori_idx])
                    times.append(round(ori_idx / fps, 1))
                    ori_idx += 1
            frames = torch.stack(refined)
        duration = len(vid) / fps
        ctx.update(frames=frames, times=times, duration=duration)
        ctx["prompt_inputs"] = self._frames_to_inputs(ctx["base_prompt"], frames, times, duration)
        inputs[0]["prompt_text_final"] = ctx["prompt_inputs"].pop("_text")
        return ctx

    def _frames_to_inputs(self, base_prompt, frames, times, total_seconds):
        text = base_prompt.replace(VIDEO_PAD, mc.frame_prompt(times, total_seconds))
        pi = self.processing_class(text=[text], images=[frames], videos=None, return_tensors="pt",
                                   padding=True, padding_side="left", add_special_tokens=False)
        pi["_text"] = text
        return pi

    # ------------------------------------------------------------------
    # Generation / scoring / loss for one view (one GRPO group)
    # ------------------------------------------------------------------
    def _generate(self, model, prompt_inputs, n: int):
        prompt_inputs.pop("_text", None)
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        if self.max_prompt_length is not None:
            # per-token arrays (input_ids, attention_mask, mm_token_type_ids on transformers 5.x)
            seq_keys = [k for k, v in prompt_inputs.items() if torch.is_tensor(v) and v.dim() == 2
                        and v.shape == prompt_inputs["input_ids"].shape]
            for k in seq_keys:
                prompt_inputs[k] = prompt_inputs[k][:, -self.max_prompt_length:]
        gen_cfg = copy.deepcopy(self.generation_config)
        gen_cfg.num_return_sequences = n
        # gradient checkpointing sets config.use_cache=False, which would re-encode the whole
        # video prompt for every new token; sample in eval mode with the KV cache instead
        gen_cfg.use_cache = True
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            was_training = unwrapped.training
            unwrapped.eval()
            try:
                ids = unwrapped.generate(**prompt_inputs, generation_config=gen_cfg)
            finally:
                unwrapped.train(was_training)
        prompt_len = prompt_inputs["input_ids"].size(1)
        completion = ids[:, prompt_len:]
        pad = self.processing_class.pad_token_id
        for vid in VISION_SPECIAL_IDS:
            completion = completion.masked_fill(completion == vid, pad)
        ids = torch.cat([ids[:, :prompt_len], completion], dim=1)

        is_eos = completion == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        mask = (seq <= eos_idx.unsqueeze(1)).int()

        vis = {k: v for k, v in prompt_inputs.items() if k not in ("input_ids", "attention_mask")}
        # transformers 5.x derives the 3D RoPE positions from mm_token_type_ids, which the
        # processor builds for the prompt only: extend it over the completion (0 = text)
        for k in ("mm_token_type_ids", "token_type_ids"):
            if torch.is_tensor(vis.get(k)):
                t = vis[k]
                t = torch.cat([t, t.new_zeros(t.size(0), ids.size(1) - t.size(1))], dim=1)
                vis[k] = t.repeat_interleave(ids.size(0) // t.size(0), dim=0)
        # keep the video timing so log-probs use the same temporal positions as sampling
        spg = vis.get("second_per_grid_ts")
        if spg is not None:
            vis["second_per_grid_ts"] = spg.repeat(n) if torch.is_tensor(spg) else list(spg) * n
        if "pixel_values" in vis:
            vis["pixel_values"] = vis["pixel_values"].repeat(n, 1)
            vis["image_grid_thw"] = vis["image_grid_thw"].repeat(n, 1)
        if "pixel_values_videos" in vis:
            vis["pixel_values_videos"] = vis["pixel_values_videos"].repeat(n, 1)
            vis["video_grid_thw"] = vis["video_grid_thw"].repeat(n, 1)
        texts = self.processing_class.batch_decode(completion, skip_special_tokens=True)
        return {"ids": ids, "prompt_len": prompt_len, "mask": mask, "vis": vis, "texts": texts, "n": n}

    def _score(self, gen, inputs, reward_funcs, weights, sample_overrides=None, extra=None):
        example = dict(inputs[0])
        example.update(sample_overrides or {})
        n = gen["n"]
        kwargs = {k: [v] * n for k, v in example.items() if k not in ("prompt", "completion")}
        kwargs.update(extra or {})
        completions = [[{"role": "assistant", "content": t}] for t in gen["texts"]] \
            if is_conversational(inputs[0]) else gen["texts"]
        prompts = [inputs[0]["prompt"]] * n
        device = self.accelerator.device
        per_func = torch.zeros(n, len(reward_funcs), device=device)
        for i, f in enumerate(reward_funcs):
            per_func[:, i] = torch.tensor(f(prompts=prompts, completions=completions, **kwargs),
                                          dtype=torch.float32, device=device)
        return per_func, (per_func * weights.to(device)).sum(dim=1)

    @staticmethod
    def _completion_logps(model, ids, prompt_len, vis):
        """log p(token) for the completion tokens only.

        Same values as v3's ``_get_per_token_logps(...)[:, prompt_len - 1:]``, but the LM
        head runs on the completion positions only (``logits_to_keep``) and the
        log-softmax is a per-row fp32 logsumexp, instead of materialising
        (B, prompt+completion, 152k) logits (OOM on a 32 GB V100 with video prompts).
        """
        k = ids.size(1) - prompt_len
        logits = model(ids, logits_to_keep=k + 1, **vis).logits[:, :-1]  # predicts ids[:, -k:]
        target = ids[:, -k:]
        out = []
        for lg, tgt in zip(logits, target):
            lg = lg.float()
            out.append(lg.gather(1, tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(lg, dim=-1))
        return torch.stack(out)

    def _gspo_loss(self, model, gen, rewards):
        ids, pl, mask = gen["ids"], gen["prompt_len"], gen["mask"]
        # reference first, so its activations are freed before the policy graph is built
        with torch.inference_mode():
            if self.ref_model is not None:
                ref = self._completion_logps(self.ref_model, ids, pl, gen["vis"])
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref = self._completion_logps(model, ids, pl, gen["vis"])
        ref = ref.clone()  # inference tensors cannot enter autograd ops
        logps = self._completion_logps(model, ids, pl, gen["vis"])
        x = torch.clamp(ref - logps, min=-10, max=10)
        kl = torch.exp(x) - x - 1
        std = rewards.std() if rewards.numel() > 1 else torch.zeros((), device=rewards.device)
        adv = (rewards - rewards.mean()) / (std + 1e-4)
        log_ratio = logps - logps.detach()
        if self.gspo:
            lw = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
        else:
            lw = log_ratio
        c1 = torch.exp(lw)
        c2 = torch.clamp(c1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_tok = -torch.min(c1 * adv.unsqueeze(1), c2 * adv.unsqueeze(1)) + self.beta * kl
        loss = ((per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        mean_kl = ((kl * mask).sum(1) / mask.sum(1).clamp(min=1.0)).mean()
        return loss, std, mean_kl

    # ------------------------------------------------------------------
    def _freeze_span(self, inputs, times):
        kfs = sorted(float(f["time"]) for f in inputs[0].get("key_frames") or [])
        start = kfs[0] if kfs else times[0]
        if self.freeze_span == "gt_track" and kfs:
            return (start, kfs[-1])
        return (start, float("inf"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        try:
            ctx = self._prepare_sample(inputs)
        except Exception as e:
            print(f"process_vision_info error, skipping sample: {e}")
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        gen = self._generate(model, ctx["prompt_inputs"], self.num_generations)

        # ---- transformed views (r_ground') ----
        task = inputs[0].get("task", "")
        views = []
        can_transform = (task in MOTION_TASKS and ctx["multi_image"] and inputs[0].get("key_items")
                         and self.equiv_transforms and self.equiv_num_generations > 0)
        if can_transform:
            gs = random.sample(self.equiv_transforms, min(self.equiv_per_step, len(self.equiv_transforms)))
            for g in gs:
                try:
                    span = self._freeze_span(inputs, ctx["times"]) if g == "freeze" else None
                    f2, t2, total2 = mc.transform_frames(ctx["frames"], ctx["times"], g, duration=ctx["duration"],
                                                      k=self.equiv_speedup_k, span=span)
                    items2, frames2 = mc.transform_key_annotations(
                        inputs[0]["key_items"], inputs[0].get("key_frames", []), g,
                        duration=ctx["duration"], k=self.equiv_speedup_k, span=span)
                    pi2 = self._frames_to_inputs(ctx["base_prompt"], f2, t2, total2)
                    text2 = pi2["_text"]
                    gen2 = self._generate(model, pi2, self.equiv_num_generations)
                    views.append((g, gen2, {"key_items": items2, "key_frames": frames2,
                                            "gt_motion": None, "prompt_text_final": text2}))
                except Exception as e:
                    print(f"[v4 equivariance] transform {g} failed: {e}")

        transformed = {g: gen2["texts"] for g, gen2, _ in views}
        extra = {"transformed_completions": [transformed] * gen["n"]} if transformed else None
        per_func, rewards = self._score(gen, inputs, self.reward_funcs, self.reward_weights, extra=extra)
        loss, std, mean_kl = self._gspo_loss(model, gen, rewards)

        equiv_losses = []
        if views and self.equiv_reward_funcs:
            w = torch.ones(len(self.equiv_reward_funcs))
            for g, gen2, overrides in views:
                pf2, r2 = self._score(gen2, inputs, self.equiv_reward_funcs, w, sample_overrides=overrides)
                l2, _, _ = self._gspo_loss(model, gen2, r2)
                equiv_losses.append(l2)
                for i, f in enumerate(self.equiv_reward_funcs):
                    self._metrics[f"equiv/{g}/{f.__name__}"].append(
                        self.accelerator.gather_for_metrics(pf2[:, i]).mean().item())
            loss = loss + self.equiv_weight * torch.stack(equiv_losses).mean()

        # ---- metrics (same keys as v3 where applicable) ----
        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(gen["mask"].sum(1)).float().mean().item())
        mean_per_func = self.accelerator.gather_for_metrics(per_func).mean(0)
        for i, f in enumerate(self.reward_funcs):
            name = f.config._name_or_path.split("/")[-1] if isinstance(f, PreTrainedModel) else f.__name__
            self._metrics[f"rewards/{name}"].append(mean_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std.reshape(1)).mean().item())
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl.reshape(1)).mean().item())
        self._metrics["equiv/num_views"].append(float(len(views)))
        # share of objects grounded >= 2 times whose boxes are all identical (copying)
        rates = [r for r in (mr4.copied_box_rate(mc.extract_think(t) or "") for t in gen["texts"]) if r is not None]
        if rates:
            self._metrics["grounding/copied_box_rate"].append(float(np.mean(rates)))
        return loss
