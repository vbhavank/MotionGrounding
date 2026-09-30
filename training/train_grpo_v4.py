"""
GRPO/GSPO v4 entrypoint: self-consistency, schema check, equivariance views.

Default reward set (Motion-o v4):
    r = r_acc + r_t + r_s + r_traj + lambda * r_self + r_fmt(schema)
    + equivariance groups on gV scored by r_traj/r_s/r_t/r_fmt against M(g T_gt)

No-tag control (isolates MCoT from "more RL on denser data"):
    --motion_prompt false
    --reward_funcs ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial format_notag
    --equiv_transforms ""            (no transformed groups)
    with tag-stripped data from scripts/strip_motion_tags.py
"""

import os
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from data_loader_v4 import get_data
from reward_func_v3 import (
    ans_acc_reward, ans_tiou_reward, ans_viou_reward,
    thk_temporal_point_reward, thk_temporal_segment_reward,
    thk_spatial_reward, format_reward,
)
from motion_reward_v3 import motion_trajectory_reward, motion_grounding_reward
from motion_reward_v4 import (
    motion_trajectory_reward_v4, motion_self_consistency_reward,
    motion_equivariance_consistency_reward, format_reward_v4, format_reward_notag,
)
from grpo_trainer_v4 import Qwen2VLGRPOTrainerV4

reward_funcs_registry = {
    "ans_acc": ans_acc_reward,
    "ans_tiou": ans_tiou_reward,
    "ans_viou": ans_viou_reward,
    "thk_temporal_point": thk_temporal_point_reward,
    "thk_temporal_segment": thk_temporal_segment_reward,
    "thk_spatial": thk_spatial_reward,
    "format": format_reward,
    # v3 (kept for ablations)
    "motion_trajectory": motion_trajectory_reward,
    "motion_grounding": motion_grounding_reward,
    # v4
    "motion_trajectory_v4": motion_trajectory_reward_v4,
    "motion_self": motion_self_consistency_reward,
    "motion_equiv_consistency": motion_equivariance_consistency_reward,
    "format_v4": format_reward_v4,
    "format_notag": format_reward_notag,
}


@dataclass
class GRPOScriptArgumentsV4(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["ans_acc", "ans_tiou", "ans_viou", "thk_temporal_point",
                                 "thk_temporal_segment", "thk_spatial", "motion_trajectory_v4",
                                 "motion_self", "format_v4"],
        metadata={"help": "Rewards on the original video."},
    )
    motion_reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={"help": "One weight per reward func (e.g. lambda for motion_self). Default all 1."},
    )
    equiv_transforms: list[str] = field(
        default_factory=lambda: ["reverse", "hflip", "freeze"],
        metadata={"help": "Transformations g for r_ground' (reverse, hflip, speedup, freeze). Empty disables."},
    )
    equiv_reward_funcs: list[str] = field(
        default_factory=lambda: ["thk_temporal_point", "thk_spatial", "motion_trajectory_v4", "format_v4"],
        metadata={"help": "Rewards for rollouts on gV. ans_acc is excluded: answers may not be g-invariant."},
    )
    equiv_weight: float = field(default=1.0, metadata={"help": "Weight of the gV GSPO loss."})
    equiv_num_generations: int = field(default=2, metadata={"help": "Rollouts per transformed view."})
    equiv_per_step: int = field(default=1, metadata={"help": "Transformations sampled per step (estimates E_g)."})
    equiv_speedup_k: float = field(default=2.0)
    freeze_span: str = field(default="first_keyframe", metadata={"help": "first_keyframe (v3) | gt_track"})
    motion_prompt: bool = field(default=True, metadata={"help": "False for the no-tag control."})
    piecewise_prompt: bool = field(default=False)
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    temporal: Optional[bool] = field(default=True)
    len_control: Optional[bool] = field(default=True)
    gen_temperature: Optional[float] = field(default=0.7)


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[f] for f in script_args.reward_funcs]
    equiv_transforms = [g for g in script_args.equiv_transforms if g]
    equiv_funcs = [reward_funcs_registry[f] for f in script_args.equiv_reward_funcs if f]
    dataset = get_data(script_args)

    if os.environ.get("QUICK_TEST", "false").lower() == "true":
        n = int(os.environ.get("MAX_SAMPLES", "10"))
        split = script_args.dataset_train_split
        dataset[split] = dataset[split].select(range(min(n, len(dataset[split]))))
        training_args.num_train_epochs = 1
        training_args.save_steps = max(1, n // 2)

    trainer = Qwen2VLGRPOTrainerV4(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        temperature=script_args.gen_temperature,
        equiv_transforms=equiv_transforms,
        equiv_reward_funcs=equiv_funcs,
        equiv_weight=script_args.equiv_weight,
        equiv_num_generations=script_args.equiv_num_generations,
        equiv_per_step=script_args.equiv_per_step,
        equiv_speedup_k=script_args.equiv_speedup_k,
        freeze_span=script_args.freeze_span,
        reward_weights=script_args.motion_reward_weights,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArgumentsV4, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
