# Motion-o revision: MCoT v4

This directory tracks the revision of *Motion-o: Trajectory-Grounded Video
Reasoning* (arXiv 2603.18856v2). It maps each proposed addition to code,
records what the repository shows about the current method, and lists the
paper corrections.

Nothing in this branch has been trained or evaluated yet. The pure-Python parts
(motion labels, rewards, transformations, metrics, data scripts) are covered by
`tests/test_mcot_v4.py` (31 tests, CPU only: `python -m pytest tests -q`). The
v4 trainer (`training/grpo_trainer_v4.py`) and the vLLM evaluation scripts
compile, but they have not been run, because that needs GPUs, torch and the
data.

---

## 0. Findings from the code

These come from reading the repository. Several of them change how the paper's
numbers should be read, so check them before rerunning anything.

| # | Finding | Where | Consequence |
|---|---|---|---|
| F1 | **`motion_grounding_reward` never receives the masked chain.** `reward_kwargs` is rebuilt for every reward function, and `reward_kwargs['masked_completion']` is assigned *after* the call. The reward reads `kwargs.get('masked_completion', [None])`, gets `None`, and returns 0 for every rollout. The `grounding/has_masked_chain` metric still logs 1.0, which hides this. | `training/grpo_trainer_v3.py:851-863`, `training/motion_reward_v3.py:576-580` | If the paper's r_ground runs used this trainer, r_ground contributed nothing. The "with vs. without r_ground" rows of Tables 1, 2 and S4 would then differ only by run-to-run variation or by other config differences. Check which commit produced those rows. v4 passes auxiliary rollouts explicitly. |
| F2 | **The masked span comes from the ground truth, not the policy, but it is not limited to the "sections with `<motion/>` tags".** The code freezes every frame from the first GT keyframe to the end of the clip. It also runs one greedy masked decode per prompt and shares it across all K rollouts. | `grpo_trainer_v3.py:783-846` | This settles the circularity question in point 7 (no circularity), but Sec. 3.3 describes the masking incorrectly. |
| F3 | **The omission exploit is concrete.** Any object in the rollout's tags that is missing from the shared masked chain scores 1.0 (matched by normalized name). A rollout that names the object differently ("duck" vs. "the duck on the left") collects full credit. | `motion_reward_v3.py:614-620` | This is point 2's exploit, reachable purely through naming. |
| F4 | **SFT labels tag every single-frame object as STAT/stationary/stable.** The augmenter's default is `tag_single_frame=True` ("CRITICAL for solving the sparsity problem, coverage 26% to 100%"). | `scripts/augment_discrete_motion.py:358,395-404` | This contradicts the paper's own rule (a tag only for objects seen at ≥ 2 timestamps, Appendix F.2). It directly teaches the seashell behavior (Fig. S8), and it likely inflates the STAT class in Fig. S5, which in turn inflates the majority baseline. |
| F5 | **The v3 system prompts differ from Appendix F.2.** The code asks for a tag "after the last observation of *each* object", with no ≥ 2 condition. The General-QA prompts ask for tags without any `<obj><box><t>` grounding. The v3 format reward also *requires* at least one tag on General-QA MCQ tasks. | `training/data_loader_v3.py`, `training/reward_func_v3.py:format_reward` | Tags without evidence are rewarded during RL, and Appendix F.2 does not match the prompts actually used. |
| F6 | **Table S4 is computed on training data.** `eval_motion_tags.py` evaluates on the SFT json (`STGR-SFT-motion-mixed.json`) and reports exact match only. The adjacent match described in the appendix is never computed. | `scripts/eval_motion_tags.py` | Table S4 needs a held-out split. `eval_motion_tags_v2.py --exclude_json` provides one. |
| F7 | **The direction label is a magnitude-weighted *vote* over per-step compass bins**, not "the dominant displacement vector" (Sec. 3.3). The reward's on-the-fly fallback uses a *third* definition: first/last endpoints, with speed = net displacement / dt. | `augment_discrete_motion.py:compute_direction`, `motion_reward_v3.py:compute_gt_*` | For back-and-forth motion, the labeler does **not** output STAT. It outputs the longer leg (the first leg on a tie), with speed computed from path length, e.g. `E`/`fast` for E-then-W. The reversal disappears rather than cancelling to zero. Only the endpoint fallback collapses it to STAT. See §5a. |
| F8 | **Reward scales differ from the equations.** r_traj is weighted 0.4/0.3/0.3 (range [0, 1]) and r_ground 0.5/0.3/0.2, while Eqs. 6 and 7 are unweighted sums (range [0, 3]). | `motion_reward_v3.py` | State the weights in the paper, or change the code. v4 keeps 0.4/0.3/0.3. |
| F9 | **The "TVGBench" column is evaluated with a TVBench script.** `scripts/eval_tvbench.py` evaluates FunAILab **TVBench** (multiple choice, letter accuracy). The paper's column says TVGBench [24] (temporal grounding, IoU). | `scripts/eval_tvbench.py` | This is more than a protocol typo. The column most likely reports a different benchmark. |
| F10 | **Frame counts do not match Appendix F.1** (32 train / 64 eval). Training uses `FPS_MAX_FRAMES = 16`. The MVBench/MotionBench/TVBench scripts default to 16 frames, V-STAR uses 16, and only VideoMME/WorldSense use 64. | `training/vision_process.py:37`, `scripts/eval_*.py`, `evaluation/config/*.yaml` | Fix F.1 or the configs. |
| F11 | **The benchmark scripts discarded the generated text** (only the first 100 predictions were kept, truncated to 200 characters), so ρ_tag could not be computed. Fixed: every prediction now stores `pred_text` under `"predictions"`. | `scripts/eval_{mvbench,motionbench,tvbench}.py` | Rerun the benchmarks, then run `scripts/analyze_tag_presence.py`. |
| F12 | **Fig. S9 (the baby) fails self-consistency too.** The three baby boxes (1.5 s, 3.0 s, 12.0 s) are identical, so M(T̂) is STAT for both segments, yet the tags say W and then E. The figure presents this as a success. | paper, Fig. S9 | This is a second example for r_self, alongside the duck. |
| F13 | **In the released dense data, the motion tags come from the original sparse boxes only.** On `bishoygaloaa/Motion-o-MCoT-PLM-motion-keyframes` (2,933 PLM tracks), the released tag equals M(original STGR boxes) on 100% of tracks, M(injected mask boxes) on 7.5%, and M(all boxes in `key_items`) on 63.5%. In 21% of tracks the injected and original boxes differ by more than 4× in median area. | released json vs. `STGR-SFT.json` key_frames | Dense injection (Sec. 3.4) adds grounded boxes and keyframes to the reasoning but changes no motion label. The density ablation (Table 3, 31.2 → 35.5 mAM) must therefore come from the extra box and keyframe supervision, not from better trajectory labels. In the dense SFT text, the tag disagrees with the boxes shown about 37% of the time, which is what r_self penalizes. `build_v4_splits.py` keeps dense rows only when both box sources agree in scale and labels them from all boxes. |
| F14 | **For the grounded sources, the STGR SFT and RL sets are the same questions.** 6,792 of the ~6.8k available SFT rows share (video, question) with an RL row under a different id. | `STGR-SFT.json`, `STGR-RL.json` | RL on these sources revisits SFT samples rather than adding new ones. Say so in Appendix A, and split held-out data by video (`build_v4_splits.py`). |
| F15 | **STGR-RL rows list one keyframe.** Each grounded RL row has a single `key_frames` entry (e.g. idx 2 at 3.5 s), while `key_items` keeps the boxes of the earlier keyframes ("0", "1", "2") with no timestamps. Every GT track therefore has one observation. In the 799-row RL subset, 0 of 714 grounded rows had any object at ≥ 2 keyframes, and r_traj v4 logged exactly 0 in RL. | `STGR-RL.json` `key_frames` vs `key_items` | The v3 r_traj builds tracks the same way (`group_boxes_by_object` with `key_frames`), so check whether the paper's RL runs had any GT motion signal. `scripts/repair_rl_keyframes.py` recovers the timestamps from the keyframe filenames (`<sample>_<idx>_time_<s>.jpg`) and recomputes `gt_motion`. |

---

## 1. Self-consistency between the tag and the model's own boxes

**Reward.** For each tag, the *evidence window* is the claims it summarizes (`motion_core.tag_evidence_windows`):

* the claims of `obj` with `from ≤ t ≤ to` if the tag carries a window;
* otherwise, the claims of `obj` after the previous tag of `obj`, plus the last claim that tag covered, used as an anchor. This makes consecutive tags describe contiguous segments; for the baby, the windows are {1.5, 3.0} and {3.0, 12.0}.

$$r_{self}=\frac{1}{|\text{tags}|}\sum_{\text{tags}} \big[w_d\,r_{dir}(\hat d, d(\hat{\mathcal T}_o)) + w_s\,r_{speed}(\hat s, s(\hat{\mathcal T}_o)) + w_c\,r_{scale}(\hat c, c(\hat{\mathcal T}_o))\big]$$

This is the same adjacency-aware matcher as r_traj (`motion_core.tag_score`). A tag whose window has fewer than 2 distinct timestamps scores 0. It is weighted by λ via `--motion_reward_weights` (default λ = 0.5 in `scripts/grpo_v4.sh`).

**Schema check.** $r_{fmt} \leftarrow r_{fmt} - \beta\,\mathbb 1[\exists\ \text{tag with } |\hat{\mathcal T}_o|<2]$, with β = `MCOT_SCHEMA_BETA` (default 0.5). v4 also changes the requirement side: a tag is required only when the rollout grounds some object at ≥ 2 timestamps (F5). On the paper's examples:

| Example | M(T̂) | Tag | r_self | Schema |
|---|---|---|---|---|
| Duck (Fig. 4) | STAT/stationary/stable | E/moderate/stable | 0.30 (scale only) | ok |
| Baby (Fig. S9) | STAT, STAT | W, E | 0.30 each | ok |
| Seashells (Fig. S8) | (1 timestamp) | STAT | 0 | **violation**, −β |

**Gaming.** A policy that copies boxes and always says STAT satisfies r_self, but r_s (IoU against the moving GT box) and r_traj (against M(T_gt)) penalize it. The three terms have to be used together.

**Metric.** SC is the fraction of evaluable tags equal to M(T̂). It is reported with the per-attribute SC, the schema-violation rate, and the number of moving tags on static boxes (`mcot_metrics.self_consistency`).

Code: `training/motion_reward_v4.py` (`motion_self_consistency_reward`, `format_reward_v4`, `schema_violations`), `training/motion_core.py`.

---

## 2. Equivariance instead of "any change"

**Exact targets.** For a transformation g, the target is $\mathcal M(g\,\mathcal T^{gt})$, computed by transforming the GT track itself (`motion_core.transform_track`, `transform_key_annotations`). For the labeler actually used to build the SFT data, this is *exactly* $\rho_g(\mathcal M(\mathcal T^{gt}))$ for reversal and horizontal flip (tested on 3k random tracks each):

| g | on the track | ρ_g |
|---|---|---|
| `reverse` | t → T − t | d + 180°, s unchanged, approaching ↔ receding |
| `hflip` | [x1,y1,x2,y2] → [W−x2, y1, W−x1, y2] | E↔W, NE↔NW, SE↔SW |
| `speedup` | t → t/k | s moves up monotonically; **d can change**: the STAT↔stationary coupling turns a sub-threshold "stationary" track into slow + a direction (the test finds such cases) |
| `freeze` (span) | boxes in the span := box at span start | STAT for a full freeze; exact M(gT) for a partial one |

Because of the speed-up and partial-freeze cases, the code uses M(gT_gt) rather than a fixed label map.

**The reward as written has zero gradient.** $r_{ground}'=\mathbb E_g[r_{traj}(\hat{\mathcal M}(gV), \rho_g(\cdot))]$ depends only on the rollout on gV. If it is added to the reward of the K original rollouts, it is the same for every member of the group, so its GRPO/GSPO advantage is 0. v4 therefore turns each sampled g into its **own group**. It draws `equiv_num_generations` rollouts on gV, scores them with the usual rewards (`thk_temporal_point`, `thk_spatial`, `motion_trajectory_v4`, `format_v4`) against the transformed annotations, and adds their GSPO loss with weight `equiv_weight`. `ans_acc` is excluded because free-form answers are not g-invariant ("moves to the right"). E_g is estimated stochastically, with `equiv_per_step` transformations per step (default 1).

This fixes the three issues:

* correctness against a known target is scored, not change;
* STAT under freezing is the correct target, not a zero;
* a missing tag scores 0. `motion_trajectory_reward_v4` scores every GT object the rollout grounded at ≥ 2 GT frames, so omitting the tag gets no credit, and there is no shared masked chain to exploit.

The freeze span comes from GT keyframes: `--freeze_span first_keyframe` (the v3 behavior) or `gt_track`.

**Label-free variant.** $\mathbb 1[\hat{\mathcal M}(gV)=\rho_g(\hat{\mathcal M}(V))]$ depends on the original rollout, so it *can* be a reward on the original group (`motion_equivariance_consistency_reward`, fed the gV texts via `transformed_completions`). An object missing from the gV rollout scores 0. For speed-up, the speed attribute passes if its rank did not drop. Use `VARIANT=mcot_labelfree`.

Code: `training/grpo_trainer_v4.py`, `training/train_grpo_v4.py`, `scripts/grpo_v4.sh`, `training/motion_core.py` (`transform_frames`, `transform_key_annotations`, `rho`, `rho_consistency`).

---

## 3. Does A depend on the tag?

`scripts/eval_tag_intervention.py` splits $R = R_{pre}\oplus m\oplus R_{post}$ at a well-formed tag, pre-fills $R_{pre}\oplus\tilde m$, and regenerates. Interventions:

* `dir`: d → d + 180°, and STAT → E/fast;
* `motion`: STAT ↔ E/fast;
* `scale`: approaching ↔ receding, and stable → approaching.

It reports:

* **TS** = P(Ã ≠ A \| m̃ ≠ m).
* **TS₀**, a **regeneration control**: the same prefix with the *original* m, so decoding noise is not counted as sensitivity. Report TS − TS₀.
* **Tag-following rate**: whether the chosen option or free-form answer makes a direction or depth claim consistent with m̃, measured on the subset where this can be checked (`mcot_metrics.answer_follows_tag`).
* ρ_tag under the MCoT prompt.

`--motion_questions_only` keeps motion-dependent questions (keyword filter).

**Tag presence under benchmark prompts.** Run `scripts/analyze_tag_presence.py` on the new benchmark outputs. It reports ρ_tag, P(correct \| tag) vs. P(correct \| no tag) with a two-proportion test, and the fraction of outputs containing `<think>` at all. With the letter-only prompts of Appendix F.4, expect ρ_tag ≈ 0. If so, the benchmark gains come from training, not from MCoT at inference, and the paper should say so.

---

## 4. Controls and baselines

**Majority class** (Fig. S5 counts; `mcot_metrics.PAPER_FIG_S5`):

| Attribute | Majority | Count | Acc_maj | Table S4 SFT-only (Qwen2.5) |
|---|---|---|---|---|
| Direction | STAT | 5,582 / 9,692 | 57.6% | **51.3%** (below always-STAT) |
| Speed | stationary | 5,582 / 9,692 | 57.6% | 68.2% |
| Scale | stable | 5,751 / 9,692 | 59.3% | 71.4% |

Given F4, recompute these on the held-out evaluation labels. `eval_motion_tags_v2.py` reports `majority_baseline_acc` next to every accuracy, on the same labels.

**Metrics.** Per attribute, `eval_motion_tags_v2.py` reports:

* exact, adjacent (±45° / ±1 rank), balanced accuracy, macro-F1 and the per-class confusion matrix;
* each of these both over all GT objects (unmatched counts as wrong) and over matched tags only;
* GT defaults to M(T_gt) recomputed from key_items for objects with ≥ 2 observations (`--gt_source tags` reproduces v1);
* `--exclude_json` removes training ids.

**No-tag control.** `VARIANT=notag` in `scripts/grpo_v4.sh`: r_motion = 0, `format_notag`, no transformed groups, `--motion_prompt false`. The data is `scripts/strip_motion_tags.py` applied to *both* the SFT and RL json, and the SFT stage uses `MCOT_NO_MOTION_PROMPT=1 python training/train_sft_v2.py ...`. Everything else (samples, dense keyframes, schedule) is identical. Run the same control on Open-o3: continued SFT + RL on the stripped mixture from the Open-o3 checkpoint.

**Oracle boxes.** `eval_motion_tags_v2.py --oracle_boxes` pre-fills `<think>` with the GT grounded claims of every tracked object and lets the model emit the tags. The report splits error into:

* `total` = Err(M̂, d*);
* `localization` = Err(M(T̂), M(T_gt)), from the model's own boxes in the normal run;
* `reasoning_oracle` = Err(M̂ \| T_gt).

This directly tests the claim in Sec. 4.1 and Appendix B that localization is the bottleneck.

**Variance.** `scripts/grpo_v4.sh` takes `SEED`. Run ≥ 3 seeds per row and aggregate with `scripts/summarize_seeds.py agg`. For Fig. S6, `python scripts/summarize_seeds.py welch-stats 123 90 100 138 90 100` gives t = −1.18, df = 198, **p = 0.24**. With std ≈ 90 and n = 100, 123 vs. 138 tokens is not a significant difference. Use the real per-sample lengths to confirm.

---

## 5. Representation

**5a. Piecewise tags.** `motion_core.segment_track` splits a track where the heading turns by ≥ 135° (3 compass bins), ignoring jitter below the STAT threshold. `scripts/relabel_motion_v4.py --piecewise` emits `<motion obj=".." from="t1" to="t2" .../>` per segment. Tags that carry `from`/`to` are scored by `motion_core.piecewise_score`: a duration-weighted r_traj over the GT span, where each predicted window is further split at GT reversal points, so tagging an easy sub-window or covering a reversal with one tag cannot reach full credit.

As noted in F7, the current labeler turns an exact E-then-W track into `E/fast`, not STAT. A single `E` tag scores 1.0 under the v3 reward and 0.8 under the piecewise scorer; the two segment tags score 1.0.

**5b. Camera compensation.** `estimate_background_homography` uses ORB + RANSAC with object boxes masked out. `chain_homographies` composes frame-to-frame estimates between observations. `scene_displacements` computes $\Delta c_i^{scene}=c_{i+1}-H_{i\to i+1}(c_i)$ and feeds it to `motion_descriptor(..., displacements=...)`. Record which frame a tag describes with `frame="scene"` (default `"image"`); it is parsed into the tag dict.

**5c. Relational motion.** `relational_descriptor(A, B)` applies M to $\Delta(c^A-c^B)$, with B interpolated at A's timestamps. Scale is A's area change relative to B's, so a camera zoom reads as stable. Tag attribute: `ref="B"`. v4 r_traj skips `ref` tags; score them against `relational_descriptor` once GT pairs exist.

**5d. Depth-based scale.** $c=\mathrm{bin}(\log(z_1/z_N))$, where z is the median depth inside the box (`depth_scale_bin`). Because $\log(a_N/a_1)\approx 2\log(z_1/z_N)$, the threshold is 0.075, half the area threshold. Pass `is_disparity=True` for inverse-depth models such as MiDaS or DPT.

`piecewise_prompt` in `data_loader_v4` documents the optional attributes in the system prompt.

---

## 6. Reversal-contrast pairs

`scripts/build_reversal_pairs.py` generates pairs from dense GT tracks. It covers left/right, up/down, closer/farther and entering/leaving (a box at the frame edge vs. inside), and only for objects whose answer flips under reversal. Option order is shuffled per pair, and the question window is rewritten as [T − t1, T − t0] for the reversed clip. `scripts/eval_reversal_pairs.py` reports:

* PA overall and per kind;
* forward and reversed accuracy, and the same-answer rate (1.0 for an appearance-only model, whose PA = 0);
* the tag-level check $\hat d(V^{rev})=\hat d(V)+180°$, $\hat c(V^{rev})=\mathrm{flip}(\hat c(V))$ via `rho_consistency("reverse", ...)`.

Chance PA is 25% on binary pairs. Run it with both `--prompt mcot` and `--prompt letter`.

---

## 7. Paper corrections

Proposed text is in *italics*.

1. **Eq. 4 wording.** "the thinking reward as the sum of temporal and spatial grounding terms" becomes *"the sum of temporal, spatial, and motion terms"*.
2. **Eq. 7 and STAT.** With the v4 formulation, replace Eq. 7 by
   *$r_{ground}=\mathbb E_{g\in G}\,r_{traj}\big(\hat{\mathcal M}(gV),\,\mathcal M(g\,\mathcal T^{gt})\big)$, optimized on rollouts of gV*. STAT under freezing is then the correct target and needs no special case. If Eq. 7 is kept, it must include the case: $r_{ground}=3$ if $d^*=\text{STAT}$ and $d=d'=\text{STAT}$ (and likewise for s and c).
3. **Missing objects.** Delete "objects absent from the motion-masked output are treated as fully grounded". A missing object scores 0.
4. **Masking definition.** *"We freeze all frames from the first ground-truth keyframe onward"* (this is what `grpo_trainer_v3.py` does), and state that the span comes from $\mathcal T^{gt}$ (F2). Also say that one greedy masked chain is shared by the K rollouts.
5. **F1.** Re-run or re-attribute every "w/o r_ground vs. w/ r_ground" comparison after checking which trainer produced them.
6. **Numeric-tag ablation.** 14.4 on VideoMME is below the 25% chance level of a 4-way multiple-choice benchmark, which points to answer-parsing or r_fmt failure. Report the parse/format failure rate, accuracy on the parsable subset, and the letter distribution, or remove the representational claim.
7. **Where/Chain2** 6.0 → 38.1 (+32.1) vs. +8.2 on Chain1. Even without r_ground it is 31.4 (vs. 12.3 for Motion-o). Give the per-sample distribution and check that Chain2 is not scored on a different parse (e.g. boxes now in pixel vs. normalized coordinates) or on fewer answered items.
8. **Evaluation protocols.** TVGBench vs. TVBench (F9): state which benchmark and which metric. Fix F.4's list. Replace "GRPO" with GSPO in Table S4. MVBench: the appendix says "69.2 vs. 67.9 strongest open-source", but Table 2 has Qwen3-VL-8B at 69.0, a 0.2-point margin. Fix the frame counts (F10).
9. **Citations.** V-STAR is cited as [25] (VSTAR, a 2023 dialogue dataset); cite the V-STAR spatio-temporal reasoning benchmark instead. Time-R1 and TVG-R1 both point to [24]; add the TVG-R1 reference.
10. **Data source.** Sec. 3.2 derives tags from STGR boxes and Sec. 3.4 from PLM dense masks. State which $\mathcal T^{gt}$ is used for the SFT labels, for r_traj (the code prefers the precomputed `gt_motion` from key_items) and for Table S4 (the code uses the SFT reasoning tags, F6). Also state the single-frame STAT labelling (F4). State that the released dense tags equal the sparse-box labels (F13).
11. **Label definition.** Sec. 3.3 says "dominant displacement vector". The code uses a magnitude-weighted vote over per-step bins (F7), and the reward fallback uses endpoints. Describe the one actually used, and use a single definition throughout.
12. **Reward weights.** Give the 0.4/0.3/0.3 and 0.5/0.3/0.2 weights (F8), or drop them.
13. **Appendix F.2 prompts** are presented as verbatim but differ from `data_loader_v3.py` (F5).
14. **Table S4** is on training data and has no adjacent-match column (F6).

---

## Run order

```bash
# data
python scripts/relabel_motion_v4.py --input SFT.json --output SFT-v4.json --report          # >= 2 rule
python scripts/strip_motion_tags.py --input SFT-v4.json --output SFT-notag.json             # control
python scripts/strip_motion_tags.py --input RL.json --output RL-notag.json

# SFT (MCoT and control)
python training/train_sft_v2.py --dataset_name SFT-v4.json ...
MCOT_NO_MOTION_PROMPT=1 python training/train_sft_v2.py --dataset_name SFT-notag.json ...

# RL, 3 seeds each
for s in 42 43 44; do
  SEED=$s VARIANT=mcot  MODEL_PATH=<sft> DATASET_JSON=RL.json       sbatch scripts/grpo_v4.sh
  SEED=$s VARIANT=notag MODEL_PATH=<sft-notag> DATASET_JSON=RL-notag.json sbatch scripts/grpo_v4.sh
done

# evaluation
python scripts/eval_motion_tags_v2.py --model_path M --dataset_json EVAL.json --exclude_json SFT.json \
    --oracle_boxes --output_file tags.json
python scripts/eval_tag_intervention.py --model_path M --input_json EVAL.json --intervention dir \
    --motion_questions_only --output_file ts.json
python scripts/build_reversal_pairs.py --dataset_json EVAL.json --exclude_json SFT.json --output pairs.json
python scripts/eval_reversal_pairs.py --model_path M --pairs_json pairs.json --output_file pa.json
python scripts/eval_mvbench.py ... && python scripts/analyze_tag_presence.py mvbench.json --by subset
python scripts/summarize_seeds.py agg run_s42.json run_s43.json run_s44.json
```

## File map

| Point | Files |
|---|---|
| shared | `training/motion_core.py` (M(T), scoring, parsing, g, ρ_g, segments, homography, relational, depth) |
| 1 | `training/motion_reward_v4.py`: `motion_self_consistency_reward`, `format_reward_v4` |
| 2 | `training/grpo_trainer_v4.py`, `training/train_grpo_v4.py`, `training/data_loader_v4.py`, `scripts/grpo_v4.sh`, `motion_reward_v4.motion_trajectory_reward_v4`, `motion_equivariance_consistency_reward` |
| 3 | `scripts/eval_tag_intervention.py`, `scripts/analyze_tag_presence.py`, patched `scripts/eval_{mvbench,motionbench,tvbench}.py` |
| 4 | `evaluation/mcot_metrics.py`, `scripts/eval_motion_tags_v2.py`, `scripts/strip_motion_tags.py`, `format_reward_notag`, `MCOT_NO_MOTION_PROMPT` in `training/train_sft_v2.py`, `scripts/summarize_seeds.py` |
| 5 | `scripts/relabel_motion_v4.py`, `motion_core.{segment_track,piecewise_score,scene_displacements,relational_descriptor,depth_scale_bin}` |
| 6 | `scripts/build_reversal_pairs.py`, `scripts/eval_reversal_pairs.py` |
| tests | `tests/test_mcot_v4.py` |

v3 files are unchanged except for the full-text saving in the three benchmark scripts and the opt-in `MCOT_NO_MOTION_PROMPT` switch in `train_sft_v2.py`, so the paper's runs stay reproducible.
