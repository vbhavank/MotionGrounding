import os
os.environ["WANDB_MODE"] = "offline" 

import os
from configs.data_root import DATA_ROOT

ROOT = os.path.join(DATA_ROOT, "videos")
TREEVGR_ROOT = os.path.join(ROOT, "treevgr")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")
    
### for debug ###
# os.environ["MASTER_PORT"] = "29501"
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["LOCAL_RANK"] = "0"

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    # AutoModelForVision2Seq,  # Not available in transformers 5.x, using specific model classes instead
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from accelerate import Accelerator
from training.vision_process import process_vision_info

from datasets import Dataset, DatasetDict

import wandb
from PIL import Image
import numpy as np
from typing import List, Dict, Any

import re

def load_json_dataset(path):
    """Load a .json/.jsonl list of samples in memory.

    Dataset.from_json writes a cache copy first and aborts with "Not enough disk
    space" when the filesystem reports 0 free bytes (common on NFS with quotas).
    Columns are the union of keys over all rows (from_list would use only the
    first row's keys).
    """
    import json as _json
    from datasets import Dataset as _Dataset
    with open(path) as f:
        rows = [_json.loads(l) for l in f if l.strip()] if path.endswith(".jsonl") else _json.load(f)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return _Dataset.from_dict({k: [r.get(k) for r in rows] for k in keys})


_MOTION_INSTRUCTION_RE = re.compile(
    r"(After the last observation of each object|When reasoning about objects in the video, describe their motion using:)"
    r".*?\{approaching, stable, receding\}\.\s*",
    re.DOTALL,
)


def strip_motion_instructions(prepared: Dict[str, Any]) -> Dict[str, Any]:
    """No-tag control (MCOT_NO_MOTION_PROMPT=1): drop <motion/> instructions
    from the system prompt. Pair with scripts/strip_motion_tags.py data."""
    for msg in prepared["messages"]:
        if msg["role"] == "system":
            for part in msg["content"]:
                if part.get("type") == "text":
                    part["text"] = _MOTION_INSTRUCTION_RE.sub("", part["text"])
    return prepared


def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    if example['task'] == 'visual QA':
        system_message = "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant MUST localize the object with bounding box coordinates between <box> and </box>. You MUST strictly follow the format."
        image_root = TREEVGR_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(image_root,example['image_path'])
                    },
                    {
                        "type": "text",
                        "text": question,
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "image_size": example["image_size"], "task": "visual QA", "source": example["source"], "key_frames":[]}
    
    elif example['task'] == 'temporal-spatial free-form QA':
        system_message = (
            "A conversation between user and assistant. The user provides a video and asks a question, "
            "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
            "and then provide the user with the answer. The reasoning process and answer are enclosed within "
            "<think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual "
            "evidence from the video. When you mention any related object, person, or specific visual element, "
            "you must strictly follow the following format: "
            "`<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. "
            "After the last observation of each object, you MUST describe its motion trajectory using a self-closing "
            "motion tag with discrete attributes: "
            "`<motion obj=\"object_name\" dir=\"DIRECTION\" speed=\"SPEED\" scale=\"SCALE\"/>` "
            "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
            "SPEED is one of {stationary, slow, moderate, fast}, "
            "and SCALE is one of {approaching, stable, receding}."
        )
        question = example['question']
        video_root = STR_DATA
        if example['source'] == "STR_plm_rdcap":
            video_root = STR_PLM_DATA
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]    
        return {"messages": messages, "key_frames": example["key_frames"], "task": "temporal-spatial free-form QA", "source": example["source"], "image_size":[]}
    
    elif example['task'] == 'temporal QA':
        system_message = "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'"
        video_root = TVG_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "temporal QA", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA MCQ":
        system_message = (
            "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, "
            "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
            "and then provide the user with the answer. The reasoning process and answer are enclosed within "
            "<think> </think> and <answer> </answer> tags, respectively. Only output the correct option in the "
            "<answer> </answer> section. "
            "When reasoning about objects in the video, describe their motion using: "
            "`<motion obj=\"object_name\" dir=\"DIRECTION\" speed=\"SPEED\" scale=\"SCALE\"/>` "
            "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
            "SPEED is one of {stationary, slow, moderate, fast}, "
            "and SCALE is one of {approaching, stable, receding}."
        )
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "General video QA MCQ", "source": example["source"], "key_frames":[], "image_size":[]}
    elif example["task"] == "General video QA Free-form":
        system_message = (
            "A conversation between user and assistant. The user provides a video and asks a question, "
            "and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind "
            "and then provide the user with the answer. The reasoning process and answer are enclosed within "
            "<think> </think> and <answer> </answer> tags, respectively. "
            "When reasoning about objects in the video, describe their motion using: "
            "`<motion obj=\"object_name\" dir=\"DIRECTION\" speed=\"SPEED\" scale=\"SCALE\"/>` "
            "where DIRECTION is one of {N, NE, E, SE, S, SW, W, NW, STAT}, "
            "SPEED is one of {stationary, slow, moderate, fast}, "
            "and SCALE is one of {approaching, stable, receding}."
        )
        video_root = GENERAL_VIDEO_ROOT
        question = example['question']
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": os.path.join(video_root, example['video_path'])
                    },
                    {
                        "type": "text",
                        "text": "Question: " + question
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<think>" + example["reasoning_process"] + "</think>\n<answer>" + example["answer"]+"</answer>"}]
            }
        ]
        return {"messages": messages, "task": "General video QA Free-form", "source": example["source"], "key_frames":[], "image_size":[]}
    
    raise ValueError(f"Unknown task: {example['task']}")


def convert_coord_format_espressso(bbox, image_size):
    # for videoespresso
    # image_size: (W, H)
    nx, ny, nw, nh = [coord / 1000.0 for coord in bbox]
    x_center = nx * image_size[0]
    y_center = ny * image_size[1]
    width = nw * image_size[0]
    height = nh * image_size[1]

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_size[0], x_max)
    y_max = min(image_size[1], y_max)

    return [x_min, y_min, x_max, y_max]

def convert_coord_format_gemini(coords, image_size):
    # for gemini annotated data
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = coords
    width, height = image_size
    real_x_min = norm_x_min * width
    real_y_min = norm_y_min * height
    real_x_max = norm_x_max * width
    real_y_max = norm_y_max * height
    return [real_x_min, real_y_min, real_x_max, real_y_max]

import re
def resize_bounding_boxes_for_image(text: str, old_image_size: tuple, new_image_size: tuple) -> str:

    old_w, old_h = old_image_size
    new_w, new_h = new_image_size
    ratios = (new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h)

    def resizer(match: re.Match) -> str:
        coords = [int(c) for c in match.group(1).strip('[]').split(',')]
        new_coords = [int(round(c * r)) for c, r in zip(coords, ratios)]
        return f"<box>[{','.join(map(str, new_coords))}]</box>"

    return re.sub(r"<box>(\[.*?\])</box>", resizer, text)

def replace_boxes_for_videoespresso(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_espresso(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)


def replace_boxes_for_gemini_data(text, image_size):
    import re
    pattern = re.compile(r'<box>\[([^]]+)\]</box>')
    
    def replacer(match):
        box_str = match.group(1)
        coords = list(map(float, box_str.split(',')))
        new_coords = convert_coord_format_gemini(coords, image_size)
        new_coords = str([round(coord) for coord in new_coords])
        new_coords = new_coords.replace(" ","")
        return '<box>' + new_coords + '</box>'
    
    return pattern.sub(replacer, text)

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []

    for i, example in enumerate(examples):
        try:
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")
        
    # batch size must be 1
    assert len(texts) == 1

    if example["task"] == "visual QA":
        old_image_size = example["image_size"]
        new_image_size = [image_inputs[0].size[0], image_inputs[0].size[1]] # W * H
        texts[0] = resize_bounding_boxes_for_image(texts[0], old_image_size, new_image_size)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            return_tensors="pt",
            padding=True
        )
        
    elif example["task"] == "temporal-spatial free-form QA":

        width, height = video_inputs[0].size(3), video_inputs[0].size(2)
        image_size = (width, height)

        # Here, we need to add key frames.
        key_frame_root = STR_KF_ROOT
        if example['source'] == "STR_plm_rdcap":
            key_frame_root = STR_PLM_KF_ROOT

        key_frames = []

        for key_frame in example["key_frames"]:
            kf_path = os.path.join(key_frame_root, key_frame["path"])
            kf = Image.open(kf_path)
            kf = kf.convert('RGB')
            resized_kf = kf.resize(image_size)
            resized_kf= np.array(resized_kf)
            resized_kf = np.transpose(resized_kf, (2, 0, 1))
            resized_kf = torch.from_numpy(resized_kf)
            key_frames.append((key_frame["time"], resized_kf))
        
        frame_prompt = ""
        refined_image_inputs = []
        kf_idx = 0
        ori_idx = 0
        frame_idx = 1
        while ori_idx < len(video_inputs[0]):
            time_now = int(ori_idx / video_kwargs['fps'][0])
            if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                refined_image_inputs.append(key_frames[kf_idx][1])
                time_now = key_frames[kf_idx][0]
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"   
                kf_idx += 1
            else:
                refined_image_inputs.append(video_inputs[0][ori_idx])
                time_now = round(ori_idx / video_kwargs['fps'][0],1)
                frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"       
                ori_idx += 1
            frame_idx += 1

        refined_image_inputs = torch.stack(refined_image_inputs)
        texts[0] = texts[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
        texts[0] = replace_boxes_for_gemini_data(texts[0], image_size)
        # print(refined_image_inputs.shape, texts[0])

        inputs = processor(
            text=texts,
            images=[refined_image_inputs],   # (16+k)*3*h*w
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )

    elif example["task"] == "temporal QA" or "General video QA" in example["task"]:
        frame_prompt = ""
        ori_idx = 0
        while ori_idx < len(video_inputs[0]):
            time_now = round(ori_idx / video_kwargs['fps'][0], 1)
            frame_prompt += f"Frame {ori_idx + 1} at {time_now}: <|vision_start|><|image_pad|><|vision_end|>\n"    
            ori_idx += 1
        frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
        texts[0] = texts[0].replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)

        # print(texts[0])

        inputs = processor(
            text=texts,
            images=video_inputs,
            videos=None,
            return_tensors="pt",
            padding=True,
            do_resize=False
        )
    else:
        raise ValueError(f"Unknown task: {example['task']}")

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch) 
        # we can add more loss here
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset = DatasetDict({"train": load_json_dataset(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Quick test mode - limit dataset size for faster debugging
    import os
    if os.environ.get("QUICK_TEST", "false").lower() == "true":
        max_samples = int(os.environ.get("MAX_SAMPLES", "10"))
        print(f"\n{'='*50}")
        print(f"🚀 QUICK TEST MODE: Using only {max_samples} samples")
        print(f"{'='*50}\n")
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
        training_args.num_train_epochs = 1
        training_args.save_steps = max(1, max_samples // 2)  # Save checkpoint midway

    # Setup model
    # TRL renamed ModelConfig.torch_dtype to ModelConfig.dtype (pass --dtype on new TRL)
    dtype_name = getattr(model_config, "torch_dtype", None) or getattr(model_config, "dtype", None)
    torch_dtype = dtype_name if dtype_name in ["auto", None] else getattr(torch, dtype_name)

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    # NOTE: For DeepSpeed ZeRO-2, load model on CPU first to avoid OOM during initialization
    # Then DeepSpeed will handle sharding and device placement
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=getattr(model_config, "trust_remote_code", False),
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(),  # Not needed for DeepSpeed
        # quantization_config=bnb_config,
    )
    
    # Load model - it will be on CPU initially, DeepSpeed moves it to GPU
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        # For other vision models, use the specific model class directly
        # AutoModelForVision2Seq not available in transformers 5.x
        raise ValueError(f"Unsupported model: {model_config.model_name_or_path}. Only Qwen2-VL and Qwen2.5-VL are supported.")
    
    # CRITICAL: For ZeRO-2 with CPU offloading, move model to GPU to avoid device mismatch
    # But keep it in fp32/bf16 to let DeepSpeed manage memory efficiently
    import os
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        # Move model to GPU - DeepSpeed will handle optimizer offloading
        model = model.to(device)
        print(f"[Rank {local_rank}] Model moved to {device}")
        # Free up some memory
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=getattr(model_config, "trust_remote_code", False)
    )

    # Prepare dataset
    from tqdm import tqdm 
    prepared_dataset = [prepare_dataset(example) for example in tqdm(dataset['train'], desc="Preparing dataset")]
    if os.environ.get("MCOT_NO_MOTION_PROMPT", "0") == "1":
        prepared_dataset = [strip_motion_instructions(p) for p in prepared_dataset]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = MySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()