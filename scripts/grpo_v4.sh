#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2
#SBATCH --time=23:59:59
#SBATCH --job-name=grpo_v4
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/grpo_v4_%j.out
#SBATCH --error=logs/grpo_v4_%j.err
#
# GRPO/GSPO v4. Configure with environment variables:
#
#   VARIANT     mcot  (default) : r_traj_v4 + lambda*r_self + schema r_fmt + equivariance groups
#               mcot_labelfree  : mcot + label-free equivariance consistency on the original group
#               notag           : no-tag control, r_motion = 0, tags stripped, no transformed groups
#               mcot_v3rewards  : v3 reward set on the v4 trainer (r_ground now actually receives
#                                 the masked chain) -- for attributing v3 -> v4 changes
#   MODEL_PATH  SFT checkpoint (the no-tag control needs an SFT model trained with
#               MCOT_NO_MOTION_PROMPT=1 on scripts/strip_motion_tags.py output)
#   DATASET_JSON RL json (notag: stripped json)
#   SEED        default 42; run >= 3 seeds and report mean +- std
#   LAMBDA_SELF weight of r_self (default 0.5)
#   EQUIV       transformations, default "reverse hflip freeze"
#   NPROC       GPUs on this node to use (default 2)
#   OUT_DIR     default outputs/grpo_v4_<variant>_s<seed>_<job>; reruns resume from its latest
#               checkpoint. QUICK_TEST=true uses a separate *_quicktest dir, wiped on each run.
#   extra args  appended last, so they override defaults, e.g. --num_generations 2
#   PRECISION   bf16 (default, A100/H100) | fp16 (V100: also sets MCOT_DTYPE=float16)
#   ATTN        attention implementation; default eager (bf16) / sdpa (fp16). Eager attention
#               forms the unscaled Q.K^T in fp16, which overflows on Qwen2.5 and makes
#               sampling fail with "probability tensor contains either inf, nan or element < 0"
#
# Example (3 seeds x 2 variants):
#   for s in 42 43 44; do
#     SEED=$s VARIANT=mcot  MODEL_PATH=... DATASET_JSON=...        sbatch scripts/grpo_v4.sh
#     SEED=$s VARIANT=notag MODEL_PATH=... DATASET_JSON=...-notag  sbatch scripts/grpo_v4.sh
#   done

set -euo pipefail

VARIANT="${VARIANT:-mcot}"
SEED="${SEED:-42}"
LAMBDA_SELF="${LAMBDA_SELF:-0.5}"
EQUIV="${EQUIV:-reverse hflip freeze}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint}"
DATASET_JSON="${DATASET_JSON:?set DATASET_JSON to the RL json}"
MASTER_PORT="${MASTER_PORT:-12331}"
NPROC="${NPROC:-2}"
PRECISION="${PRECISION:-bf16}"
if [ "$PRECISION" = "fp16" ]; then
  PREC_ARGS=(--fp16 true --bf16 false)
  export MCOT_DTYPE=float16
  ATTN="${ATTN:-sdpa}"
else
  PREC_ARGS=(--bf16 true)
  ATTN="${ATTN:-eager}"
fi
EXP_NAME="grpo_v4_${VARIANT}_s${SEED}_${SLURM_JOB_ID:-local}"
QUICK_TEST="${QUICK_TEST:-false}"
[ "$QUICK_TEST" = "true" ] && EXP_NAME="${EXP_NAME}_quicktest"
OUT_DIR="${OUT_DIR:-outputs/${EXP_NAME}}"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd):$(pwd)/training"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_EOF_RETRY_MAX=20480
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p logs

BASE_REWARDS="ans_acc ans_tiou ans_viou thk_temporal_point thk_temporal_segment thk_spatial"
case "$VARIANT" in
  mcot)
    REWARD_ARGS=(--reward_funcs $BASE_REWARDS motion_trajectory_v4 motion_self format_v4
                 --motion_reward_weights 1 1 1 1 1 1 1 "$LAMBDA_SELF" 1
                 --equiv_transforms $EQUIV
                 --equiv_reward_funcs thk_temporal_point thk_spatial motion_trajectory_v4 format_v4)
    ;;
  mcot_labelfree)
    # adds 1[M_hat(gV) = rho_g(M_hat(V))] as a reward on the original rollouts
    REWARD_ARGS=(--reward_funcs $BASE_REWARDS motion_trajectory_v4 motion_self motion_equiv_consistency format_v4
                 --motion_reward_weights 1 1 1 1 1 1 1 "$LAMBDA_SELF" 1 1
                 --equiv_transforms $EQUIV
                 --equiv_reward_funcs thk_temporal_point thk_spatial motion_trajectory_v4 format_v4)
    ;;
  notag)
    REWARD_ARGS=(--reward_funcs $BASE_REWARDS format_notag
                 --motion_prompt false
                 --equiv_transforms "")
    ;;
  mcot_v3rewards)
    REWARD_ARGS=(--reward_funcs $BASE_REWARDS motion_trajectory motion_grounding format
                 --equiv_transforms "")
    ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

# Auto-resume from the latest checkpoint in OUT_DIR (never for QUICK_TEST: a finished
# smoke test would otherwise "resume" at the last step and exit without training)
RESUME_ARG=()
if [ "$QUICK_TEST" = "true" ]; then
  case "$OUT_DIR" in *_quicktest) rm -rf "$OUT_DIR" ;; esac
else
  LATEST_CKPT=$(ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [ -n "$LATEST_CKPT" ] && RESUME_ARG=(--resume_from_checkpoint "$LATEST_CKPT")
fi

echo "variant=$VARIANT seed=$SEED model=$MODEL_PATH data=$DATASET_JSON out=$OUT_DIR"

torchrun --nproc_per_node="$NPROC" --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    training/train_grpo_v4.py \
    --output_dir "$OUT_DIR" \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_JSON" \
    --use_peft true --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --num_generations 4 --generation_batch_size 4 \
    --equiv_num_generations 2 --equiv_per_step 1 \
    --max_prompt_length 16384 --max_completion_length 768 --max_pixels 401408 \
    --learning_rate 5e-7 --lr_scheduler_type cosine --weight_decay 0.01 \
    "${PREC_ARGS[@]}" --gradient_checkpointing true --attn_implementation "$ATTN" \
    --num_train_epochs 1 --beta 0.04 --max_grad_norm 5 \
    --logging_steps 25 --save_steps 200 --save_only_model true \
    --report_to wandb --run_name "$EXP_NAME" \
    --seed "$SEED" --data_seed "$SEED" \
    --gen_temperature 0.7 \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"} \
    "${REWARD_ARGS[@]}" \
    "$@"
