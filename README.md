# Motion-o: Trajectory-Grounded Video Reasoning

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4B3B9A.svg)](https://neurips.cc/)
[![Project Page](https://img.shields.io/badge/Project%20Page-motion--o.github.io-blue)](https://ostadabbas.github.io/motion-o.github.io/)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/bishoygaloaa/motion-o)
[![Dataset on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/bishoygaloaa/Motion-o-MCoT-PLM-motion-keyframes)
[![arXiv](https://img.shields.io/badge/arXiv-2603.18856-b31b1b.svg)](https://arxiv.org/abs/2603.18856)

**Training VLMs to reason about video motion through verifiable, motion-aware evidence chains with trajectory-level geometric rewards.**

See our IO page: https://ostadabbas.github.io/motion-o.github.io/

## Revision in progress: MCoT v4

[`revision/README.md`](revision/README.md) documents the v4 additions: tag/box self-consistency reward and schema check, equivariance-based grounding (time reversal, flip, speed-up, freeze), tag-intervention and tag-presence tests, majority/no-tag/oracle-box controls, piecewise, camera-compensated, relational and depth-based tags, and reversal-contrast pairs. It also records findings from the v3 code that affect the reported numbers and the list of paper corrections. CPU tests: `python -m pytest tests -q`.

## Core Contribution

**Motion-aware trajectory reward** — evaluating not just *where* objects are, but *how they moved*, using geometric motion metrics derived from predicted bbox sequences.

| Aspect | Baseline | This Work |
|--------|--------------|-----------|
| **Spatial Reward** | Per-frame IoU (static) | Trajectory-level motion |
| **Motion Modeling** | Implicit in text | Explicit (direction, speed, smoothness) |
| **Evaluation** | "Is bbox in right place?" | "Did object move correctly?" |
| **Key Question** | "What happened and where?" | "How did it move and why?" |

### Motion-Aware Reward Components

**R_motion (λ=0.35)**: Trajectory-level motion matching
- **Direction similarity** (0.4): Cosine similarity of displacement vectors
- **Speed fidelity** (0.4): Velocity magnitude matching
- **Trajectory smoothness** (0.2): Acceleration penalty for physically implausible motion

Combined with baseline reward components:
- R_spatial (λ=0.25): Bbox IoU via Hungarian matching
- R_temporal (λ=0.15): Temporal interval IoU  
- R_caption (λ=0.20): Text similarity
- R_format (gate): Parseability validation

---

## Installation

```bash
conda create -n motion_vlm python=3.10
conda activate motion_vlm

# Install dependencies
pip install -r requirements.txt

# For DeepSpeed (multi-GPU training)
pip install deepspeed
```

---

## Quick Start

### 1. Setup Data

Download STGR dataset and update `configs/data_root.py`:

```python
DATA_ROOT = "/path/to/your/stgr/dataset"
```

Expected structure:
```
${DATA_ROOT}/
├── json_data/
│   ├── STGR-SFT.json (30k samples)
│   └── STGR-RL.json (36k samples)
└── videos/
    ├── gqa/, stgr/, timerft/, etc.
```

---

### 2. Pretrained Models

We release several Motion-o variants on Hugging Face:

- **Motion-o (no visual grounding)** – main model at repo root  
  [`bishoygaloaa/motion-o`](https://huggingface.co/bishoygaloaa/motion-o)

- **Open-o3 + MCoT (with visual grounding)** – subfolder  
  `bishoygaloaa/motion-o` with `subfolder="open-o3-mcot"`

- **Open-o3 + MCoT (no visual grounding)** – subfolder  
  `bishoygaloaa/motion-o` with `subfolder="open-o3-mcot-no-vg"`

- **PLM motion keyframes (subset)** – STGR PLM rows with `<motion/>` tags and sharded keyframes  
  [`bishoygaloaa/Motion-o-MCoT-PLM-motion-keyframes`](https://huggingface.co/datasets/bishoygaloaa/Motion-o-MCoT-PLM-motion-keyframes)

Example:

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("bishoygaloaa/motion-o")
processor = AutoProcessor.from_pretrained("bishoygaloaa/motion-o")
```

### 3. Train Baseline (without motion reward)

```bash
# SFT cold-start
bash scripts/run_sft.sh

# RL training (baseline)
bash scripts/run_grpo_baseline.sh
```

### 4. Train with Motion Reward

```bash
# RL training with motion-aware trajectory reward
bash scripts/run_grpo_motion.sh
```

### 5. Evaluate

All evaluation commands below assume you are in the project root (`vlmm-mcot/`).

#### 5.1 Full evaluation pipeline (V-STaR + Video-MME + VideoMMMU + WorldSense)

Run the SLURM wrapper on a merged or LoRA checkpoint directory:

```bash
sbatch scripts/eval_all_h200.sh /path/to/checkpoint-dir 100   # 100 samples per benchmark
# or
sbatch scripts/eval_all_h200.sh /path/to/checkpoint-dir       # full benchmarks
```

This will:
- Merge the LoRA adapter into a base model (if needed),
- Run V-STaR, Video-MME, VideoMMMU, and WorldSense,
- Write logs/metrics under `logs/` and `evaluation/logs/`.

#### 5.2 V-STaR only (with LLM-as-judge)

```bash
sbatch scripts/eval_vstar.sh /path/to/checkpoint-dir 100   # optional 2nd arg = max_samples
```

This script:
- Merges LoRA → `/path/to/checkpoint-dir/merged`,
- Runs V-STaR inference,
- Runs LLM-as-judge scoring using a large judge model.

#### 5.3 Visualizing V-STaR chains (frames + video per sample)

```bash
python evaluation/visualize_results.py \
  --result_json /path/to/vstar_results.json \
  --video_dir /path/to/V-STAR/videos \
  --output_dir /path/to/output_vstar_frames \
  --task vqa \
  --format frames \
  --max_samples 20
```

Each sample gets:
- Keyframes with and without boxes,
- `entry.json` (raw result entry),
- A copy of the original `.mp4` video.

#### 5.4 Visualizing Video-MME reasoning (correct answers only)

First, run the Video-MME eval (already done in `eval_all_h200.sh`). Then, for any `metrics_*.json`:

```bash
python evaluation/visualize_videomme_results.py \
  --metrics_json /path/to/metrics_model_xxx_mme.json \
  --video_dir /path/to/Video-MME/data \
  --output_dir /path/to/output_videomme_vis_model_xxx \
  --format frames \
  --max_samples 50
```

The script:
- Uses `reasoning_process` from the metrics JSON,
- Only visualizes samples where `pred_answer == answer`,
- Saves frames, `entry.json`, and a copy of the original video per sample.

---

## Project Structure

```
vlmm-mcot/
├── src/                        # Core motion reasoning modules
│   ├── motion_metrics.py       # ⭐ Trajectory-level geometric metrics
│   ├── geometric_reward.py     # Multi-dimensional reward composition
│   ├── evidence_parser.py      # Evidence chain parsing
│   └── [supporting modules]
│
├── training/                   # GRPO-based training infrastructure
│   ├── grpo_trainer.py         # GRPO trainer (808 lines, DeepSpeed)
│   ├── reward_func.py          # Modular reward functions
│   ├── motion_reward.py        # ⭐ Our motion trajectory reward
│   ├── train_grpo.py           # GRPO training script
│   ├── train_sft.py            # SFT training script
│   └── [data loader, vision processing]
│
├── evaluation/                 # Evaluation suite
│   ├── test/                   # V-STaR, Video-MME, VideoMMMU, WorldSense
│   ├── visualize_results.py    # V-STaR visualization (frames / GIF)
│   └── visualize_videomme_results.py  # Video-MME visualization (frames / GIF)
│
├── configs/                    # Configuration files
│   ├── data_root.py            # Dataset path configuration
│   └── [DeepSpeed configs]
│
└── scripts/                    # Training launchers
    ├── run_sft.sh              # SFT cold-start
    ├── run_grpo_baseline.sh    # Baseline (no motion reward)
    └── run_grpo_motion.sh      # With motion reward ⭐
```

---

## Key Metrics

### Trajectory-Level Motion Metrics

**Direction Matching** (`src/motion_metrics.py`):
```python
cos_sim = dot(pred_direction, gt_direction) / (||pred|| * ||gt||)
```

**Speed Fidelity**:
```python
speed_score = exp(-|pred_speed - gt_speed| / (gt_speed + eps))
```

**Trajectory Smoothness**:
```python
accel = [v[t+1] - v[t] for t in range(T-1)]
smoothness = 1 / (1 + std(accel))
```

---

## Expected Results

### Quantitative Improvements

| Benchmark | Baseline | +Motion Reward | Δ |
|-----------|----------|----------------|---|
| V-STAR mAM | 35.5% | 37-40% | +2-5% |
| V-STAR mLGM | 49.0% | 52-56% | +3-7% |
| Motion-heavy tasks | - | - | +5-10% |

**Justification:** The baseline spatial reward is frame-independent. Our reward optimizes trajectory-level motion properties.

---

## Training Tips

### Multi-GPU Setup

```bash
# 8x GPUs with DeepSpeed ZeRO-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_grpo_motion.sh
```

### Memory Optimization

- **DeepSpeed ZeRO-3**: Enabled in `configs/zero3.json`
- **Gradient checkpointing**: `--gradient_checkpointing true`
- **Flash Attention 2**: `--attn_implementation flash_attention_2`
- **4-bit quantization**: Add `--load_in_4bit` flag

### Monitoring Training

```bash
# Check reward components
tail -f /path/to/checkpoints/rl_motion/log.txt | grep "rewards/"
```

Expected output:
```
rewards/ans_acc: 0.75
rewards/thk_spatial: 0.62
rewards/motion_trajectory: 0.48  ← Motion-aware trajectory score
rewards/format: 1.0
reward: 5.23                      ← Total (sum of all rewards)
```

---

## Ablation Studies

Run experiments with different motion components:

```bash
# Edit training/motion_reward.py, line ~235:
# Full motion reward
motion_reward = 0.4*direction + 0.4*speed + 0.2*smoothness

# Direction only
motion_reward = 1.0*direction

# Direction + Speed
motion_reward = 0.5*direction + 0.5*speed
```

---


**Smaller setups:**
- 4x GPUs: Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`
- Single GPU: Use 4-bit quantization + LoRA

---


## Citation

If you use Motion-o in your work, please cite:

```bibtex
@article{galoaa2026motion,
  title   = {Motion-o: Trajectory-Grounded Video Reasoning},
  author  = {Galoaa, Bishoy* and Moezzi, Shayda* and Bai, Xiangyu and Ostadabbas, Sarah},
  journal = {arXiv preprint arXiv:2603.18856},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.18856}
}
```

## License

MIT License

---

## Acknowledgments

This work builds on the STGR dataset and training / evaluation infrastructure released as part of [Open-o3 Video](https://github.com/marinero4972/Open-o3-Video), and makes extensive use of public video benchmarks such as V-STaR, Video-MME, VideoMMMU, and WorldSense for evaluation.

```bibtex
@article{meng2025openo3,
  title={Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence},
  author={Meng, Jiahao and Li, Xiangtai and Wang, Haochen and others},
  journal={arXiv preprint arXiv:2510.20579},
  year={2025}
}
```
