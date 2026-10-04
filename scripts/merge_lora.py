#!/usr/bin/env python3
"""
Properly merge a PEFT LoRA adapter into a Qwen2.5-VL base model.

Handles known pitfalls:
  - Preserves vocab_size and all vision-token IDs in config.json
  - Copies processor/tokenizer files from the base model
  - Validates embedding shapes after merge

Usage:
    python scripts/merge_lora.py \
        --base  Open-o3-Video/Qwen2.5-VL-open-o3 \
        --adapter outputs/open-o3_motion_sft_4666166 \
        --output  outputs/open-o3_motion_sft_4666166/merged_v2 \
        [--checkpoint checkpoint-3920]
"""

import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


def _get_vocab(config):
    v = getattr(config, "vocab_size", None)
    if v is None and getattr(config, "text_config", None) is not None:
        v = getattr(config.text_config, "vocab_size", None)
    return v


def _set_vocab(config, vocab):
    if getattr(config, "text_config", None) is not None:
        config.text_config.vocab_size = vocab
    try:
        config.vocab_size = vocab
    except AttributeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen2.5-VL base")
    parser.add_argument("--base", required=True, help="Path to the base model (before SFT)")
    parser.add_argument("--adapter", required=True,
                        help="Path to adapter dir (contains adapter_config.json & adapter_model.safetensors)")
    parser.add_argument("--output", required=True, help="Where to save the merged model")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional: sub-directory checkpoint name (e.g. checkpoint-3920)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    adapter_path = args.adapter
    if args.checkpoint:
        adapter_path = os.path.join(adapter_path, args.checkpoint)

    assert os.path.isfile(os.path.join(adapter_path, "adapter_config.json")), \
        f"adapter_config.json not found in {adapter_path}"
    assert os.path.isfile(os.path.join(args.base, "config.json")), \
        f"config.json not found in {args.base}"

    os.makedirs(args.output, exist_ok=True)
    torch_dtype = getattr(torch, args.dtype)

    print(f"Base model:  {args.base}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {args.output}")
    print(f"dtype:       {args.dtype}")

    # --- Load base model ---
    print("\n[1/5] Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    # transformers 5.x may ignore torch_dtype: enforce it (V100s need float16)
    base_model = base_model.to(torch_dtype)
    print(f"  Base dtype:         {base_model.dtype}")
    # transformers 5.x keeps vocab_size in config.text_config
    base_vocab = _get_vocab(base_model.config)
    embed_layer = getattr(base_model.model, 'embed_tokens', None) or getattr(base_model.model.language_model, 'embed_tokens', None)
    base_embed_shape = embed_layer.weight.shape if embed_layer else (base_vocab, "unknown")
    print(f"  Base vocab_size:    {base_vocab}")
    print(f"  Base embed shape:   {list(base_embed_shape)}")

    # --- Load adapter on top ---
    print("\n[2/5] Loading LoRA adapter...")

    # Patch adapter_config to point to the correct base path on THIS machine
    ac_path = os.path.join(adapter_path, "adapter_config.json")
    with open(ac_path) as f:
        ac = json.load(f)
    original_base = ac.get("base_model_name_or_path", "")
    if not os.path.isdir(original_base):
        print(f"  Patching base_model_name_or_path: {original_base} -> {os.path.abspath(args.base)}")
        ac["base_model_name_or_path"] = os.path.abspath(args.base)
        with open(ac_path, "w") as f:
            json.dump(ac, f, indent=2)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"  LoRA rank:          {ac.get('r')}")
    print(f"  Target modules:     {ac.get('target_modules')}")

    # --- Merge ---
    print("\n[3/5] Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    embed_layer = getattr(merged_model.model, 'embed_tokens', None) or getattr(merged_model.model.language_model, 'embed_tokens', None)
    lm_head = getattr(merged_model, 'lm_head', None) or getattr(merged_model.model.language_model, 'lm_head', None)
    merged_embed_shape = embed_layer.weight.shape if embed_layer else "unknown"
    merged_lm_head_shape = lm_head.weight.shape if lm_head else "unknown"
    print(f"  Merged embed shape:   {list(merged_embed_shape) if isinstance(merged_embed_shape, torch.Size) else merged_embed_shape}")
    print(f"  Merged lm_head shape: {list(merged_lm_head_shape) if isinstance(merged_lm_head_shape, torch.Size) else merged_lm_head_shape}")

    if isinstance(merged_embed_shape, torch.Size) and isinstance(base_embed_shape, torch.Size):
        if merged_embed_shape[0] != base_embed_shape[0]:
            print(f"  WARNING: Embedding size changed from {base_embed_shape[0]} to {merged_embed_shape[0]}!")

    # --- Fix config ---
    print("\n[4/5] Fixing config and saving...")

    # Ensure vocab_size is set correctly (PEFT merge bug often sets it to None)
    _set_vocab(merged_model.config, base_vocab)
    # Restore vision token IDs from base config if the merge dropped them (scalars only;
    # vision_config / rope_scaling are sub-configs in transformers 5.x and are left as loaded)
    base_config = json.loads(open(os.path.join(args.base, "config.json")).read())
    for key in ["image_token_id", "video_token_id", "vision_token_id",
                "vision_start_token_id", "vision_end_token_id"]:
        if key in base_config and getattr(merged_model.config, key, None) is None:
            setattr(merged_model.config, key, base_config[key])
    merged_model = merged_model.to(torch_dtype)

    merged_model.config.use_cache = True
    merged_model.save_pretrained(args.output, safe_serialization=True)
    print(f"  Model saved to {args.output}")

    # Verify saved config
    saved_cfg = json.loads(open(os.path.join(args.output, "config.json")).read())
    saved_vocab = saved_cfg.get("vocab_size") or (saved_cfg.get("text_config") or {}).get("vocab_size")
    assert saved_vocab == base_vocab, f"Saved config vocab_size={saved_vocab}, expected {base_vocab}"
    print(f"  Verified: saved config.json has vocab_size={saved_vocab}")

    # --- Copy processor/tokenizer from base ---
    print("\n[5/5] Copying processor & tokenizer from base...")
    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.output)

    # Also copy generation_config.json if it exists
    gen_cfg = os.path.join(args.base, "generation_config.json")
    if os.path.isfile(gen_cfg):
        shutil.copy2(gen_cfg, os.path.join(args.output, "generation_config.json"))
        print(f"  Copied generation_config.json")

    print(f"\nDone! Merged model saved to: {args.output}")
    print(f"You can now use this path as --model_name_or_path for GRPO training.")


if __name__ == "__main__":
    main()
