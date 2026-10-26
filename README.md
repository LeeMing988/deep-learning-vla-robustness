# deep-learning-vla-robustness

![Python](https://img.shields.io/badge/python-3.12.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![LeRobot](https://img.shields.io/badge/LeRobot-0.6.1-yellow)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.1-red)
![Benchmark](https://img.shields.io/badge/benchmark-LIBERO--Spatial-purple)
![Study](https://img.shields.io/badge/study-VLA%20robustness-blueviolet)
![Core study](https://img.shields.io/badge/core%20study-N00%E2%80%93N08%20Complete-brightgreen)
![M3 extension](https://img.shields.io/badge/M3%20extension-N09%E2%80%93N10%20Evaluated-brightgreen)
![License](https://img.shields.io/badge/license-Academic-orange)

A Deep Learning research project on **robustness-enhanced Vision-Language-Action (VLA) learning** for robotic manipulation under **linguistic and environmental distribution shifts**.

The **original study through Notebook 08 is complete**, including its definitive N07 analysis and supplementary N07b/E5 stress test. Its eight core checkpoints, protocols and results remain preserved. The separate **post-hoc M3-RVA architectural extension (Notebooks 09–10) is also completed**: three fixed-budget training runs and a three-seed exploratory closed-loop comparison against M2-LE. No consistent additional robustness benefit was demonstrated for the tested adapter. The **M3-RVA+C consistency-loss extension was not conducted** and remains future work. The active milestone is the consolidated academic report and submission QA.

> **Reading this README:** N00–N08 are the **completed original core study**; N07b/E5 is a **supplementary exploratory stress test**; and N09/N10 document a **completed, explicitly post-hoc exploratory architectural extension**. Core and extension results are not pooled into one hypothesis test or silently combined into one headline success rate. The consistency-learning idea (M3-RVA+C) is **not an executed experiment**.

---

### Submission status and verified evidence — 26 October 2026

This dated update supplements the complete research history below; the earlier N00–N10 study descriptions remain intact. **N11 is now the last executed report-finalisation notebook**, and it is not a new training or robot-evaluation experiment. The consolidated academic report has a revised Word draft; repository documentation, submission packaging, final tag creation and corresponding Appendix A cross-references are still being finalised.

- **Verified `main` evidence baseline before this README update:** [`42e74eefa1aa5ff40c29a668d1c901a2fdcbecf7`](https://github.com/LeeMing988/deep-learning-vla-robustness/commit/42e74eefa1aa5ff40c29a668d1c901a2fdcbecf7). This commit preserves the three M3-RVA seeds' language assignments and model/processor metadata.
- **N11 original merge:** [`8b36afb7223f049def24088866f326cc005da516`](https://github.com/LeeMing988/deep-learning-vla-robustness/commit/8b36afb7223f049def24088866f326cc005da516); [executed Notebook 11](notebooks/11_report_finalization_and_evidence_audit.ipynb) and [report-finalisation evidence](results/metrics/report_finalization/20261025T124022_898151Z/) are available on `main`.
- **N11 audit outcome:** `COMPLETE`, 119 saved-evidence checks passed, six figures, fourteen tables, 25 source records; **0 new training runs, 0 robot episodes and 0 new significance tests**. This validates saved counts, source integrity and accounting, **not** a fresh simulation replay or every statistical assumption.
- **Release state:** the final annotated **submission tag is pending**, not yet an established Git reference. The exact tagged commit SHA and release URL must be recorded only after verification and tagging; do not use an earlier evidence commit as a substitute for that final SHA.

## Project metadata

| Field | Details |
|---|---|
| **Author** | YEOH LEE MING |
| **Course** | Deep Learning — Master in Artificial Intelligence |
| **Module Code** | `CT100-3-M-DL` |
| **Institution** | Asia Pacific University of Technology & Innovation (APU) |
| **Assessment Structure** | Assignment 1 — 40% · Assignment 2 — 60% |
| **Project Type** | Single continuous Deep Learning research project |
| **Primary Model** | SmolVLA |
| **Primary Benchmark** | LIBERO-Spatial |
| **Robustness Evaluation** | Controlled linguistic, photometric and Gaussian sensor-noise shifts |
| **Framework** | Hugging Face LeRobot |
| **Simulation** | MuJoCo |
| **Core-study Training Budget** | 25,000 optimizer steps |
| **Primary Training Seeds** | 42, 123, 2026 |
| **Frozen Core-study Checkpoints** | 8 |
| **Definitive Robustness Episodes** | 2,050 |
| **Supplementary E5 Episodes** | 300 |
| **Core-study Robustness Evaluation Episodes** | 2,350 |
| **Evaluated Architectural Extension** | N09: M3-RVA adapter training; N10: M3-RVA versus M2-LE closed-loop comparison |
| **M3-RVA Training** | Three independently initialized 25,000-step checkpoints: seeds 42, 123, 2026 |
| **M3 Evaluation Role** | Post-hoc exploratory; E0 ID, E1 L3, E2 seen photometric, E3 held-out Gaussian noise |
| **M3 Scheduled Evaluation Slots** | 1,200 across three training seeds and two evaluated policies |
| **M3 Scored Episodes / Valid Pairs** | 1,196 scored episodes / 596 complete valid pairs |
| **M3 Technical Exclusions** | 4 image-only initial-pairing mismatches; unscored, not task failures |
| **M3 Performance Finding** | No consistent additional robustness advantage demonstrated over M2-LE |
| **Deferred Extension** | M3-RVA+C matched-two-view consistency objective; **not implemented/evaluated** |
| **Year** | 2026 |
| **Dataset download** | [LeRobot LIBERO on Hugging Face](https://huggingface.co/datasets/lerobot/libero); pinned dataset revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| **N11 report audit** | Executed read-only evidence notebook; 119 checks passed; six figures and fourteen tables |
| **N11 evidence path** | `results/metrics/report_finalization/20261025T124022_898151Z/` |
| **M3 seed reproducibility metadata** | Three training seeds with 346 saved episode-language assignments per seed, plus model and processor JSON metadata |
| **Repository visibility** | Public GitHub repository; large local weights and captures are not part of the public Git tree |

---

# Project overview

Vision-Language-Action models combine:

```text
vision
+
language
+
robot state
        ↓
       VLA
        ↓
robot action / action sequence
```

A VLA policy can therefore receive a natural-language task instruction together with visual and proprioceptive observations and output robot-control actions.

Strong benchmark performance, however, does not guarantee that the same policy will remain reliable when deployment conditions differ from those seen during training. A robotic policy may encounter:

- semantically equivalent but differently worded instructions,
- camera or sensor corruption,
- lighting / appearance variation,
- different object configurations,
- different robot initial states,
- combined linguistic and environmental shifts.

This project studies whether **controlled robustness-oriented fine-tuning** can improve SmolVLA generalization while preserving useful in-distribution manipulation competence.

The completed research uses **LIBERO-Spatial** as the controlled manipulation benchmark, evaluates defined **linguistic, photometric and Gaussian-noise distribution shifts**, and then tests a lightweight visual-adapter modification to the same SmolVLA family. The architectural test extends rather than replaces the core fine-tuning study.

> This repository does **not** claim complete LIBERO-Plus benchmark coverage.

---

## Central research problem

> **Can robustness-oriented fine-tuning improve VLA robotic manipulation performance under linguistic and environmental distribution shifts without substantially degrading standard in-distribution performance?**

---

# Assessment alignment

This repository supports both Deep Learning assignments as one continuous research project.

| Component | Weight | Repository focus |
|---|---:|---|
| **Assignment 1** | **40%** | Domain background, VLA literature, recent architectures, research problem, methodology, benchmark and model-selection justification |
| **Assignment 2 — Model Implementation, Tuning & Validation** | **42%** | SmolVLA implementation, robustness adaptation, hyperparameter tuning, validation, training-budget study and frozen final models |
| **Assignment 2 — Visualization & Critical Analysis** | **18%** | Closed-loop evaluation, multi-seed statistics, ablations, robustness analysis, failure analysis and critical interpretation |

The final academic deliverable is a **single consolidated Deep Learning report** supported by reproducible Jupyter notebooks and saved experimental artifacts.

---

# Research aim

To investigate whether robustness-oriented fine-tuning improves the generalization of an efficient Vision-Language-Action policy for robotic manipulation under controlled linguistic and environmental distribution shifts.

---

# Research objectives

1. Establish a reproducible SmolVLA + LeRobot + LIBERO experimental environment.
2. Reproduce and audit an existing LIBERO-fine-tuned SmolVLA checkpoint.
3. Establish a standard SmolVLA fine-tuning baseline.
4. Measure performance degradation under controlled linguistic and environmental perturbations.
5. Develop language-aware and environment-aware robustness fine-tuning strategies.
6. Evaluate whether robustness learned from selected perturbations transfers to held-out perturbation categories.
7. Quantify the trade-off between in-distribution performance and out-of-distribution robustness.
8. Analyse failure patterns, training variability, statistical uncertainty, generalization gaps and study limitations.
9. In an explicitly exploratory follow-up, design, train and compare a lightweight residual visual adapter against M2-LE under the frozen four-condition extension evaluation.

---

# Research questions

- **RQ1.** To what extent does robustness-oriented fine-tuning improve seen-perturbation robustness and generalization to unseen perturbation categories in VLA robotic manipulation?
- **RQ2.** At what level of linguistic variation does SmolVLA manipulation performance degrade, and can language-aware fine-tuning reduce this degradation?
- **RQ3.** Which evaluated environmental perturbations most strongly affect VLA manipulation performance?
- **RQ4.** Does robustness training on selected perturbation categories transfer to categories absent during training?
- **RQ5.** Does improved out-of-distribution robustness reduce standard in-distribution performance?

Notebook 07 provides the **definitive RQ1–RQ5 evidence**.

Notebook 07b provides **supplementary exploratory evidence** for transfer under a stricter double-held-out condition. It does not replace or redefine Notebook 07's confirmatory analysis.

**Exploratory extension question (RQ6; not part of the original predeclared RQ1–RQ5 family):** Does adding a lightweight residual visual-token adapter to the augmentation-trained SmolVLA policy improve manipulation robustness beyond M2-LE under the selected standard, language, photometric and held-out-noise evaluations? N09/N10 address this retrospectively motivated architectural question; they do not redefine the original hypotheses.

---

# Research hypotheses

| Hypothesis | Description |
|---|---|
| **H1** | Language-aware fine-tuning will reduce degradation under held-out but semantically equivalent task instructions. |
| **H2** | Robustness-oriented training may improve performance under environmental perturbations represented during training. |
| **H3** | Robustness training will transfer to at least some perturbation categories absent during training. |
| **H4** | The combined robustness model will produce a smaller ID→OOD generalization gap than standard fine-tuning on at least some OOD conditions. |
| **H5** | Robustness improvements may preserve or slightly reduce standard performance in exchange for improved OOD generalization. |

Negative or null results are treated as scientifically meaningful rather than as failed experiments.

---

# Core-study model variants

The project separates checkpoint reproduction, standard fine-tuning and robustness-oriented fine-tuning.

| Model | Role |
|---|---|
| **M0** | Released LIBERO-fine-tuned SmolVLA checkpoint — evaluator / reproducibility control |
| **M1** | Standard SmolVLA fine-tuning baseline |
| **M2-L** | Linguistic robustness fine-tuning |
| **M2-E** | Environmental robustness fine-tuning |
| **M2-LE** | Combined linguistic + environmental robustness fine-tuning |

The primary final comparison is:

```text
M1 — Standard Fine-Tuning
        vs
M2-LE — Combined Robustness Fine-Tuning
```

using three matched independent training seeds:

```text
42
123
2026
```

M2-L and M2-E are retained as **single-seed mechanism ablations**.

---

# Frozen core-study training configuration

The original model-development protocol was frozen before the definitive Notebook 07 evaluation. The following configuration records that completed phase; it does not imply that the proposed M3 protocol has already been finalized.

| Parameter | Final value |
|---|---:|
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Batch size | `4` |
| Warm-up | `2,500` optimizer steps |
| Final training budget | `25,000` optimizer steps |
| Primary seeds | `42`, `123`, `2026` |
| Final trainability scope | Standard restricted scope |
| Post-OOD model replacement | **None** |

The 25k training budget is **not claimed to be globally optimal**. It was retained because it was already adequate under the bounded predeclared protocol and neither 35k nor 50k satisfied every replacement criterion required to supersede it.
**Recorded adequacy-gate qualification:** The original repeatability/adequacy threshold did **not** pass unchanged: the repeated seed-42 25k result was below the prior threshold. The hard stop was amended to `VARIABILITY OBSERVED`. Retaining 25k was therefore a bounded resource and replacement-rule decision, **not** a claim that all original acceptance gates passed or that longer-run schedules were independently optimised. See N05b and the [N11 budget amendment table](results/metrics/report_finalization/20261025T124022_898151Z/tables/06_budget_repeatability_amendment.csv).

---

# Completed core-study evidence chain

```text
Notebook 03
Hyperparameter tuning
        ↓
Notebook 04
Pilot robustness evaluation
→ severe floor effect discovered
        ↓
Notebook 05
M0 evaluator control + training adequacy
        ↓
Notebook 05b
25k / 35k / 50k multi-seed E0 validation
        ↓
25k frozen
        ↓
Notebook 06
8 final frozen checkpoints
        ↓
Notebook 07
Definitive E0–E4 robustness evaluation
2,050 episodes
        ↓
Notebook 07b
Supplementary E5 double-held-out stress test
300 episodes
        ↓
Notebook 08
Final results / assignment evidence / report figures
```

For the completed N00–N08 **core-study phase**, no further tuning or replacement of those eight final core checkpoints occurred after the definitive OOD outcomes were observed. **M3 is a separate, explicitly post-hoc research extension trained after those outcomes were known.** Its three additional checkpoints do not replace the eight core checkpoints, and N10's fresh M2-LE control rollouts are not pooled into N07/N07b.

---

# Completed custom-model extension — Notebooks 09 and 10

**Status: COMPLETED for the bounded M3-RVA architecture question.** Three fixed-budget M3-RVA checkpoints were trained and the selected three-seed exploratory comparison finished with four explicitly unscored technical attempts. The planned **M3-RVA+C consistency-training arm was not implemented**. Experimental configurations and trained-model choices are frozen; the project has moved to report writing.

This extension was formulated **after inspecting the N07/N07b results**. It is **post-hoc exploratory** and may not be described as a predeclared/confirmatory replication of the original robustness hypotheses.

## Motivation from the completed core study

The original study showed strong held-out linguistic robustness effects and mixed environmental effects. We therefore asked whether a small trainable visual-representation correction would improve robustness beyond the combined language-and-photometric augmentation of M2-LE. The new model was tested against that directly relevant augmentation-trained comparator, **not** against a conveniently weaker baseline alone.

An adapter is not presumed beneficial merely because it introduces trainable parameters. The evaluation includes negative and null outcomes. The implementation does not claim a novel VLA foundation architecture or a previously unknown adapter principle.

## N09 — M3-RVA: implemented robust visual adapter

**Executed notebook:** [`notebooks/09_custom_robust_visual_adapter.ipynb`](notebooks/09_custom_robust_visual_adapter.ipynb)

**Question:** Does a lightweight residual adapter placed in SmolVLA's visual-token pathway produce additional robustness when the pretrained initialization and robustness training are otherwise matched to M2-LE?

**Actual adapter placement (verified against LeRobot v0.6.1):**

```text
LIBERO camera images
         ↓
Existing SmolVLA / SmolVLM vision encoder
         ↓
Existing visual connector / token projection and resampling
         ↓
M3-RVA: token-wise residual visual adapter  ← project-authored module
         ↓
Existing image-token normalization/scaling
         ↓
Inherited multimodal context and flow-matching action expert
         ↓
50-step action chunk / 7-D LIBERO action interface
```

The adapter applies normalization, a 960→64 down-projection, a GELU nonlinearity, a 64→960 up-projection, and a learned residual scale. The up-projection is initially zero to preserve the unmodified image-token path at insertion. These are design choices of this **SmolVLA adaptation**, not claims that residual networks or adapters were invented in this study.

| Architectural measurement | Verified engineering configuration |
|---|---|
| Input/output feature width | 960 visual-token features |
| Adapter bottleneck | 64 features |
| Residual architecture | Token-wise normalized bottleneck + nonlinear projection + learned residual scale |
| Added adapter parameters | **125,825** |
| Added share of original ≈450M-parameter policy | **Approximately 0.028%** |
| Inherited model family | SmolVLA; no complete backbone replacement |
| Frozen source revision | LeRobot `7e241bd630a3719a56157a497ce5d08f244784f1` |

### N09 engineering, qualification and training gates

1. **Stage A (synthetic engineering preflight):** connector-output shape/dtype, initial identity, gradient path through frozen components, two diagnostic adapter-only updates, and adapter reconstruction were checked. These checks are **not** robot-performance results.
2. **Stage B (real-data qualification):** original LIBERO demonstration pipeline, batch size 4, 10 warm-up plus 100 timed qualification updates, finite losses/gradients, and fresh-process full policy/processor restoration passed. Qualification updates did **not** become extra steps for the final model.
3. **Stage C (fixed training):** independent runs for seeds **42, 123, 2026**, each at **25,000 optimizer steps**, starting from the pinned SmolVLA base rather than continuing an M2-LE checkpoint. Same inherited trainability scope and robustness augmentation strategy as the selected comparator, plus the new visual adapter. Each fixed-final export underwent offline validation and full reconstruction checks.

The seed-42 worker recorded **63.39 minutes** elapsed and **62.29 minutes** excluding checkpoint I/O, an offline validation loss of **0.586101**, and approximately **2.107 GiB peak allocated GPU memory** during that measured run. These are **seed-42 measurements**, not invented averages for all three seeds; training loss is **not** robot success rate.

**Evidence locations:**

```text
configs/experiments/m3_rva_training_protocol_v1.json
src/m3_rva_training.py
results/metrics/m3_design_audit/m3_preflight_latest.json
results/metrics/m3_rva_extension/v1/qualification.json
results/metrics/m3_rva_extension/v1/training_summary.json
results/metrics/m3_rva_extension/v1/m3_model_registry.csv
experiments/m3_rva_extension/v1/m3rva_seed{42,123,2026}_25000/
    result.json
    run_manifest.json
    train_metrics.csv
    model/m3_manifest.json
    model/model.safetensors                 # local large checkpoint; not ordinary Git
    recovery.pt                             # local optimizer recovery; not ordinary Git
results/figures/m3_rva_extension/
```

### Preserved M3 training-seed provenance (added to `main`)

The final M3-RVA runs retain compact, independently inspectable inputs for seeds **42, 123 and 2026**, separate from the model-weight files. The following four paths are present for each `m3rva_seed{seed}_25000` directory:

```text
experiments/m3_rva_extension/v1/m3rva_seed{seed}_25000/
    language_assignment.csv           # 346 recorded training-episode assignments per seed
    model/config.json                 # recorded policy configuration
    model/policy_preprocessor.json    # saved processor configuration
    model/policy_postprocessor.json   # saved postprocessor configuration
```

All 12 files were verified as parseable CSV/JSON and committed in [`42e74ee`](https://github.com/LeeMing988/deep-learning-vla-robustness/commit/42e74eefa1aa5ff40c29a668d1c901a2fdcbecf7). This improves the training configuration's auditability; it **does not** supply omitted `.safetensors` weights, prove the model's semantic validity, or make complete simulator replay possible from GitHub alone.

## N10 — Completed three-seed exploratory evaluation

**Executed notebook:** [`notebooks/10_m3_rva_evaluation_and_comparison.ipynb`](notebooks/10_m3_rva_evaluation_and_comparison.ipynb)

**Comparison:** newly trained **M3-RVA versus the existing frozen M2-LE** checkpoint for the *same training seed*, with fresh M2-LE rollouts under the extension evaluator. The conditions reuse the selected N07 perturbation definitions, but their new pairing/reset protocol and post-hoc role require reporting the N10 results **separately** from N07/N07b.

| Extension condition | Role | Planned pairs across 3 seeds |
|---|---|---:|
| `E0_ID` | Original instruction and images | 150 |
| `E1_L3` | Held-out compositional language | 150 |
| `E2_SEEN_PHOTO` | Specified photometric variation | 150 |
| `E3_HELDOUT_NOISE` | Gaussian image noise, σ = 0.05 | 150 |

Each condition scheduled **3 training seeds × 10 LIBERO-Spatial tasks × 5 replicas × 2 policies = 300 model-episode slots**; across four conditions the plan comprised **1,200 slots**.

### Technical execution and audit status

| N10 completion field | Final recorded value |
|---|---:|
| Execution status | `FINISHED` |
| Final status | `FINISHED_WITH_UNSCORED` |
| Scheduled model-episode slots | **1,200** |
| Scored episode records | **1,196** |
| Valid complete comparison pairs | **596 of 600** |
| Episodes in paired analysis | **1,192** |
| Scored, unpaired baseline episodes | **4** |
| Image-only technical unscored attempts | **4** |
| Pending slots | **0** |
| Three selected training seeds | **42, 123, 2026** |
| Fully scored 600-pair matrix | **No** — four technical exclusions |

Four **M3-RVA** attempts in **E1_L3, task 8** were unscored because the initial wrist-camera image violated the fixed pairing rule despite matching selected benchmark state and exposed robot-state variables:

| Seed | Task | Replica | Status |
|---:|---:|---:|---|
| 42 | 8 | 2 | `UNSCORED_INITIAL_IMAGE_MISMATCH` |
| 123 | 8 | 2 | `UNSCORED_INITIAL_IMAGE_MISMATCH` |
| 123 | 8 | 4 | `UNSCORED_INITIAL_IMAGE_MISMATCH` |
| 2026 | 8 | 2 | `UNSCORED_INITIAL_IMAGE_MISMATCH` |

These attempts had **no valid robotic outcome**; they were neither counted as manipulation failures nor retried to obtain a passing reset. The baseline-only outcomes remain in the evidence inventory but are excluded from the **matched-denominator** paired analysis. The 1,200 scheduled slots are therefore **not** equivalent to 1,200 scored episodes.

The initial strict byte-comparison exposed sparse wrist-camera variation. A reviewed V2 reset/indexed-state pairing protocol was introduced, with exact exposed-state matching and a narrowly specified image rule. A subsequent model-free repeatability probe showed that camera variations can occur under the same recorded benchmark and simulator states. A separate completion amendment therefore accounted for **image-only pairing rejections as technical exclusions without widening the image tolerance or silently converting them into task failures**. This repairs progress accounting, **not** renderer determinism or behavior equivalence.

The earlier V1 and original V2 evidence remains separately preserved. The final completion analysis uses compatible V2 scored episodes, valid pairs and explicit exclusions. Existing original model weights, frozen core-study protocols and LeRobot source remained unchanged according to the completed N10 verification.

### Final M3-RVA versus M2-LE success results

The estimates below use only **valid complete model pairs**; positive Δ favours M3. “pp” denotes percentage points. This is an **exploratory three-seed** analysis, not part of N07's original confirmatory test family.

| Condition | Valid pairs | M2-LE paired success | M3-RVA paired success | Δ M3 − M2-LE | Inferential status |
|---|---:|---:|---:|---:|---|
| **E0 ID** | **150/150** | 43.33% | 42.00% | **−1.33 pp** | Balanced; interval includes zero |
| **E1 L3 held-out language** | **146/150** | 23.29% | 19.86% | **−3.42 pp** | Four missing pairs; bootstrap/p withheld |
| **E2 seen photometric** | **150/150** | 42.00% | 43.33% | **+1.33 pp** | Balanced; no demonstrated advantage |
| **E3 held-out Gaussian noise** | **150/150** | 50.67% | 48.67% | **−2.00 pp** | Balanced; interval includes zero |

A small photometric improvement was observed in aggregate, but it was **not consistent across training seeds** and did not establish statistical superiority. Other tested conditions showed small negative effects or near-equality. The experiment therefore **does not demonstrate a consistent additional robustness advantage** of this specific residual visual adapter beyond M2-LE under the tested protocol. Matching observed rates does not establish model equivalence.

**Missing-outcome sensitivity for E1:** across the **150 scheduled pairs**, all 150 baseline outcomes are observed, while four M3 outcomes are unknown. The possible scheduled-sample M3 − M2-LE difference is **−3.33 to −0.67 pp** under best-/worst-case binary completion. Those are **bounds, not a confidence interval**, and missingness need not be random. Bootstrap and p-value estimates for this incomplete condition are withheld in the final extension summary.

The N10 comparison is a **post-hoc architectural study** prompted by the core findings; its controls and exploratory multiple comparisons are not retroactively added to the N07 predeclared inference family. Three seeds still provide limited information about training variability, and one task suite, selected perturbations and simulated manipulation do not justify universal robustness or real-robot claims.

**Final analysis artifacts:**

```text
configs/experiments/m3_rva_evaluation_protocol_v1.json   # archived initial protocol
configs/experiments/m3_rva_evaluation_protocol_v2.json   # reset/pairing protocol
configs/experiments/m3_rva_evaluation_completion_v1.json # fixed exclusion-accounting amendment
src/m3_rva_evaluation.py                                # unchanged V2 evaluator
src/m3_rva_completion.py                                # completion runner
results/metrics/m3_rva_evaluation/v2_completion_v1/
    notebook10_summary.json
    m3_vs_m2le_effects.csv
    paired_episode_results.csv
    per_seed_paired_success.csv
    pair_coverage.csv
    initial_pairing_audit.csv
    unscored_attempts.csv
    scheduled_matrix_bounds.csv
    coverage_by_task.csv
    descriptive_paired_efficiency.csv
    scored_episode_inventory.csv
results/figures/m3_rva_evaluation/v2_completion_v1/
    m3_success_with_coverage.png
    m3_missing_outcome_bounds.png
results/qualitative/m3_rva_evaluation/               # curated illustrative frames
```

## M3 interpretation and closeout boundary

The architectural contribution is **a tested adaptation**, not a performance win: a small residual visual token module was specified, implemented, trained under a matched fixed budget, and evaluated against M2-LE. Its outcome was **mixed and inconclusive regarding additional benefit**. Negative/null findings are retained rather than hidden or followed by outcome-driven model redesign.

**This bounded M3-RVA experiment is closed.** No additional architecture, checkpoint search, perturbation severity search or post-result tuning is introduced for submission. The original eight core checkpoints remain unchanged. There are **three additional M3-RVA checkpoints**, giving **11 evaluated/trained core-plus-extension final checkpoints by inventory**, of which the original study's definitive inference still concerns its original eight; this is **not** an 11-checkpoint pooled confirmatory study.

## Deferred M3-RVA+C consistency-learning arm — NOT conducted

Earlier design discussions proposed using two semantically action-equivalent supervised views and adding a velocity-prediction consistency penalty to flow matching:

```text
L_total = 0.5 × (L_FM_view_A + L_FM_view_B) + lambda × L_action_consistency
```

The necessary future control is the same architecture and paired-view supervision with **`lambda = 0`**, using matched action targets, flow time/noise and valid-action masks. The planned comparison would isolate the added loss from extra view exposure and compute. **No M3-RVA+C training, no matched lambda=0 training arm, and no separate E6 held-out corruption evaluation were conducted.** These are appropriately attributed **future work**, not N10 results. Notebook 10's actual role is *M3-RVA evaluation and comparison*, not consistency training.

---

# Benchmark scope

## LIBERO-Spatial

**LIBERO-Spatial** is the primary manipulation benchmark used in the final experiments.

The validated installation exposes **10 manipulation tasks** involving closely related goals under different spatial configurations.

Example task:

```text
pick up the black bowl between the plate and the ramekin
and place it on the plate
```

The final study evaluates all 10 LIBERO-Spatial tasks.

---

## Robustness evaluation scope

The project does **not** claim full LIBERO-Plus implementation.

Instead, the final experiment uses controlled perturbations selected to investigate specific robustness dimensions while preserving experimental validity.

The implemented robustness dimensions are:

- language rewording,
- brightness / contrast / saturation photometric changes,
- Gaussian sensor noise,
- combined language + visual perturbations.

The following robustness dimensions were **not** part of the final implemented evaluation:

- full camera-viewpoint perturbation suite,
- full object-layout perturbation suite,
- full robot-initial-state perturbation suite,
- full background perturbation suite,
- complete LIBERO-Plus benchmark reproduction.

These remain future-work opportunities.

---

# Linguistic perturbation levels

Language robustness is evaluated using semantically equivalent instructions of increasing difficulty.

| Level | Description | Training role |
|---|---|---|
| **L0** | Original benchmark instruction | Standard |
| **L1** | Lexical / synonym substitution | Seen during robustness training |
| **L2** | Syntactic restructuring | Seen during robustness training |
| **L3** | Compositional rewording | **Held out from robustness training** |

Example:

```text
L0: Pick up the black bowl and place it on the plate.

L1: Grab the black bowl and put it on the plate.

L2: Place the black bowl onto the plate after picking it up.

L3: Move the black bowl from its current position so that
    it ends up resting on the plate.
```

The final robustness language table contains 40 deterministic rows:

```text
10 tasks × 4 language levels = 40 instructions
```

L3 is excluded from robustness training and reserved for held-out evaluation.

---

# Final seen vs held-out design

| Perturbation | Final role |
|---|---|
| Original L0 instruction | ID |
| L1 lexical variation | Seen language family |
| L2 syntactic variation | Seen language family |
| **L3 compositional variation** | **Held-out language category** |
| Brightness / contrast / saturation | Seen environmental augmentation family |
| **Gaussian noise σ = 0.05** | **Held-out visual corruption category** |
| **L3 + Gaussian noise** | **Double-held-out composition; supplementary exploratory** |

---

# Evaluation conditions

| Condition | Final definition | Methodological status |
|---|---|---|
| **E0_ID** | Original L0 instruction, no perturbation | Definitive |
| **E1_L1** | Lexical language variation | Definitive |
| **E1_L2** | Syntactic language variation | Definitive |
| **E1_L3** | Held-out compositional language variation | Definitive |
| **E2_SEEN_PHOTO** | Fixed seen-family photometric shift | Definitive |
| **E3_HELDOUT_NOISE** | Gaussian sensor noise, `σ=0.05` | Definitive |
| **E4_COMBINED** | L3 + seen-family photometric shift | Definitive |
| **E5_DOUBLE_HELDOUT** | L3 + held-out Gaussian noise | **Supplementary exploratory** |

### E5 methodological note

E5 was introduced **after Notebook 07 completed its predeclared definitive evaluation**.

Therefore E5 is explicitly labelled:

```text
SUPPLEMENTARY_EXPLORATORY
```

It was not used for:

- model selection,
- hyperparameter selection,
- training-budget selection,
- checkpoint replacement,
- post-OOD retraining.

Its raw exact McNemar test is reported separately and is **not retroactively added** to Notebook 07's original Holm-corrected confirmatory hypothesis family.

---

# Primary metrics

## Closed-loop task success rate

```text
                  Successful Episodes
Task Success = ─────────────────────────
                    Total Episodes
```

## Generalization gap

```text
G = SR_ID - SR_OOD
```

where:

- `SR_ID` = in-distribution success rate,
- `SR_OOD` = out-of-distribution success rate.

A smaller gap indicates better robustness retention.

## Secondary measurements

- per-task success rate,
- per-condition success rate,
- episode completion steps,
- wall-clock time,
- action-selection latency,
- simulator latency,
- GPU memory usage during training,
- training loss,
- validation loss,
- failure proxy / review outputs.

---

# Statistical analysis

The primary M1 vs M2-LE comparison uses three independently trained models:

```text
seed 42
seed 123
seed 2026
```

For each primary model-condition cell:

```text
10 LIBERO-Spatial tasks
× 5 replicas per task
= 50 closed-loop episodes
```

For the primary three-seed comparison:

```text
3 seeds
× 10 tasks
× 5 replicas
= 150 episodes per model per primary condition
```

The final analysis includes:

- mean ± SD across training seeds,
- hierarchical paired-bootstrap confidence intervals,
- paired exact McNemar tests,
- Holm correction across the predeclared Notebook 07 primary-condition family,
- per-seed direction / consistency analysis,
- ID→OOD generalization gaps,
- mechanism ablations,
- failure and efficiency summaries.

Evaluation uncertainty and training-seed variability are treated as distinct sources of uncertainty.

---

# Completed core-study headline results

**Published-baseline context for readers:** The local M1 benchmark result of **51.3%** is a controlled, matched comparator for M2-LE under this study's protocol, **not** a reproduction of SmolVLA's published **90.0% LIBERO-Spatial** result. The published setting reports substantially different training/evaluation choices (including 100k updates, batch 64, and replanning after individual actions), whereas this study uses 25k updates, batch 4 and 50-action chunks. These differences preclude attributing the gap to one isolated factor or treating the two percentages as directly comparable.

## Definitive Notebook 07 results

| Condition | M1 | M2-LE | Δ M2-LE − M1 | 95% paired bootstrap CI | Evidence |
|---|---:|---:|---:|---:|---|
| **E0 ID** | 51.3% | 44.7% | −6.7 pp | [−26.7, +14.0] | Directional evidence against |
| **E1 L3 held-out language** | 6.0% | **32.7%** | **+26.7 pp** | **[+14.7, +38.7]** | **Strong multi-seed support** |
| **E2 seen photometric** | **59.3%** | 49.3% | −10.0 pp | [−25.3, +6.0] | Directional evidence against |
| **E3 held-out Gaussian noise** | 48.0% | **58.0%** | +10.0 pp | [−4.7, +24.7] | Directional support |
| **E4 L3 + photometric** | 9.3% | **26.7%** | **+17.3 pp** | **[+2.0, +32.7]** | **Strong multi-seed support** |

### Primary interpretation

The strongest reproducible robustness gain is **linguistic generalization**.

M2-LE improves held-out L3 performance substantially and the advantage remains under the combined L3 + photometric condition.

The method does **not** uniformly improve every visual condition.

The ID result also suggests a possible robustness / standard-performance trade-off, but that effect is not statistically conclusive.

---

# Supplementary E5 double-held-out stress test

E5 combines:

```text
held-out L3 language
+
held-out Gaussian sensor noise
```

Both perturbation categories were absent from robustness training.

## E5 results

| Model | Mean success across 3 seeds |
|---|---:|
| **M1** | 4.7% |
| **M2-LE** | **21.3%** |

Effect:

```text
M2-LE − M1 = +16.7 percentage points
```

Hierarchical paired-bootstrap 95% CI:

```text
[+4.7, +31.3] pp
```

Seed-level effects:

| Seed | M1 | M2-LE | Difference |
|---:|---:|---:|---:|
| 42 | 4% | 32% | +28 pp |
| 123 | 6% | 14% | +8 pp |
| 2026 | 4% | 18% | +14 pp |

All 3/3 seeds improved.

Notebook 07b classifies this as:

```text
EXPLORATORY_STRONG_SUPPORT
```

This is **supplementary exploratory evidence**, not retroactive confirmatory evidence.

---

# Generalization-gap interpretation

For E5:

```text
M1:
ID 51.3% → E5 4.7%
mean gap = 46.7 pp

M2-LE:
ID 44.7% → E5 21.3%
mean gap = 23.3 pp
```

M2-LE therefore shows a much smaller degradation under the double-held-out stress condition.

This supports a cautious conclusion that robustness-oriented fine-tuning can improve resilience to compounded distribution shift, especially when language variation is involved.

---

# Mechanism ablations

## Language ablation — seed 42

| Condition | M1 | M2-L | M2-LE |
|---|---:|---:|---:|
| L0 | 62% | 52% | 40% |
| L1 | 16% | **50%** | 48% |
| L2 | 14% | **52%** | 42% |
| L3 | 10% | 26% | **36%** |

The language-only M2-L ablation strongly supports the interpretation that the main robustness improvement is driven by **language augmentation**.

## Environmental ablation — seed 42

| Condition | M1 | M2-E | M2-LE |
|---|---:|---:|---:|
| E0 ID | **62%** | 46% | 40% |
| Seen photometric | **60%** | 56% | 46% |
| Held-out noise | 54% | 54% | **60%** |

The simple photometric augmentation does not independently provide a clear robustness advantage under the evaluated photometric condition.

---

# Critical interpretation

The final result is deliberately not summarized as:

> “M2-LE is better at everything.”

Instead, the evidence supports a more specific conclusion:

```text
                              M2-LE effect

ID                            possible ↓ / inconclusive
Seen photometric              no demonstrated advantage
Held-out sensor noise         ↑ directional
Held-out L3 language          ↑↑↑ strong
L3 + photometric              ↑↑ strong
L3 + held-out noise           ↑↑ strong exploratory
```

Therefore robustness in this study is **perturbation dependent**, not a single universal property.

The strongest evidence concerns **linguistic robustness and compounded shifts involving linguistic variation**.

---

# Training-budget validation

The initial 1.5k training budget produced a severe floor effect.

Notebook 05 then evaluated a standard training trajectory and established that substantially more training was required.

Notebook 05b performed the bounded final-budget comparison:

```text
25k
35k
50k
```

across three independent training seeds using **E0 only**.

Pooled mean E0 results were approximately:

| Training budget | Mean E0 success |
|---:|---:|
| 25k | 47.3% |
| 35k | 48.7% |
| 50k | 53.3% |

Although 50k had a higher raw mean, its gain:

- was not statistically conclusive,
- was concentrated in seed 2026,
- reversed for seeds 42 and 123,
- failed the predeclared seed-consistency criterion.

Therefore 25k remained the final budget.

The correct interpretation is:

> **25k is empirically frozen under the bounded predeclared protocol; it is not claimed to be globally optimal.**

---

# Recorded completion status — original N00–N08 study

| Component | Status |
|---|---|
| WSL2 Ubuntu environment | ✅ PASS |
| PyTorch / CUDA | ✅ PASS |
| RTX 5090 computation | ✅ PASS |
| LeRobot | ✅ PASS |
| MuJoCo | ✅ PASS |
| LIBERO-Spatial | ✅ PASS |
| SmolVLA policy integration | ✅ PASS |
| M0 reference reproduction | ✅ PASS |
| M1 training | ✅ PASS |
| M2-L training | ✅ PASS |
| M2-E training | ✅ PASS |
| M2-LE training | ✅ PASS |
| Hyperparameter tuning | ✅ PASS |
| Training-regime diagnosis | ✅ PASS |
| 25k/35k/50k budget validation | ✅ PASS |
| Eight frozen final checkpoints | ✅ PASS |
| Notebook 07 definitive evaluation | ✅ 2050 / 2050 |
| Notebook 07b supplementary E5 | ✅ 300 / 300 |
| Notebook 08 final assignment evidence | ✅ PASS |
| Post-OOD model replacement | **NONE** |

---

# Experimental environment

| Component | Validated configuration |
|---|---|
| Host | Windows 11 |
| Linux environment | WSL2 — Ubuntu 24.04 |
| Python | 3.12.14 |
| Conda environment | `vla-libero` |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU compute capability | 12.0 |
| LeRobot | 0.6.1 |
| MuJoCo | 3.8.1 |
| LIBERO | `hf_libero==0.1.4` |
| Rendering | WSLg + Mesa D3D12 + GLFW |

### LeRobot revision

```text
Tag:    v0.6.1
Commit: 7e241bd630a3719a56157a497ce5d08f244784f1
```

---

# Repository structure

```text
deep-learning-vla-robustness/
│
├── configs/
│   └── experiment / training / evaluation configurations
│
├── docs/
│   ├── SETUP.md
│   ├── environment snapshots / project notes
│   └── M3_source_audit_and_design_decision.md
│
├── experiments/
│   ├── original training runs, frozen checkpoints and metadata
│   └── m3_rva_extension/v1/               # three independent M3 trained seeds
│
├── notebooks/
│   ├── 00_environment_and_dataset_eda.ipynb
│   ├── 01_smolvla_baseline_reproduction.ipynb
│   ├── 02_robustness_training.ipynb
│   ├── 03_tuning_and_final_training.ipynb
│   ├── 04_evaluation_and_critical_analysis.ipynb
│   ├── 05_reference_control_and_training_budget_validation.ipynb
│   ├── 05b_final_training_budget_validation.ipynb
│   ├── 06_validated_final_retraining.ipynb
│   ├── 07_definitive_robustness_evaluation.ipynb
│   ├── 07b_supplementary_double_heldout_stress_test.ipynb
│   ├── 08_final_results_and_assignment_evidence.ipynb
│   ├── 09_custom_robust_visual_adapter.ipynb
│   └── 10_m3_rva_evaluation_and_comparison.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── qualitative/
│
├── scripts/
│   └── environment and reproducibility utilities
│
├── src/
│   ├── reusable core-study implementation
│   ├── m3_rva_training.py
│   ├── m3_rva_evaluation.py
│   └── m3_rva_completion.py
│
├── .gitignore
├── environment.yml
├── conda-linux-64.lock.txt
├── requirements-lock.txt
└── README.md
```

---

### Repository tree supplement for N11 (additive; earlier tree retained)

The historical directory diagram above ends at N10. The current `main` branch **also includes** the following tracked additions, without replacing any earlier notebook or final result:

```text
notebooks/
    11_report_finalization_and_evidence_audit.ipynb
results/metrics/report_finalization/
    20261025T124022_898151Z/
        REPORT_INTEGRATION_GUIDE.md
        notebook11_summary.json
        evidence_audit_checks.csv
        source_manifest.csv
        figure_catalog.csv
        table_catalog.csv
        rubric_evidence_map.csv
        figures/                    # F01–F06, PNG and SVG
        tables/                     # 01–14, CSV
experiments/m3_rva_extension/v1/
    m3rva_seed{42,123,2026}_25000/
        language_assignment.csv
        model/{config,policy_preprocessor,policy_postprocessor}.json
```

# Notebook workflow

The workflow below preserves the original N00–N08 evidence chain and adds the separately completed N09/N10 custom-model study. N09/N10 are **executed**, but N10 is **exploratory** and should not be used to reinterpret the predeclared N07 results.

## `00_environment_and_dataset_eda.ipynb`

Environment, dataset and benchmark validation:

- Python / package versions,
- CUDA verification,
- LIBERO task inspection,
- dataset audit,
- train / validation split,
- simulator compatibility,
- camera observations,
- task mapping.

## `01_smolvla_baseline_reproduction.ipynb`

Released SmolVLA checkpoint reproduction:

- M0 reference checkpoint,
- model revision audit,
- closed-loop rollout,
- evaluator validation,
- system-level PASS gate.

## `02_robustness_training.ipynb`

Robustness-training implementation:

- standard M1 training path,
- M2-L linguistic augmentation,
- M2-E environmental augmentation,
- M2-LE combined augmentation,
- forward/backward smoke checks,
- reload / checkpoint validation.

## `03_tuning_and_final_training.ipynb`

Initial tuning and pilot final training:

- six M1 optimizer configurations,
- frozen validation subset,
- LR / weight-decay selection,
- initial 1.5k pilot final models.

## `04_evaluation_and_critical_analysis.ipynb`

Pilot diagnostic robustness evaluation:

- 1,230 closed-loop episodes,
- E0 / language / visual / combined conditions,
- severe floor-effect diagnosis,
- pilot-only interpretation.

## `05_reference_control_and_training_budget_validation.ipynb`

Training adequacy and evaluator-control study:

- M0 evaluator reference,
- M1 progression from short to longer training,
- standard vs broad trainability scope,
- 25k candidate selection.

## `05b_final_training_budget_validation.ipynb`

Bounded final training-budget validation:

- 25k / 35k / 50k,
- three independent training seeds,
- 450 E0 closed-loop episodes,
- hierarchical paired bootstrap,
- paired McNemar testing,
- predeclared replacement criteria,
- final 25k freeze.

## `06_validated_final_retraining.ipynb`

Frozen final-model construction:

- M1 seeds 42 / 123 / 2026,
- M2-L seed 42,
- M2-E seed 42,
- M2-LE seeds 42 / 123 / 2026,
- eight final checkpoints,
- frozen 25k budget,
- no further tuning.

## `07_definitive_robustness_evaluation.ipynb`

Definitive evaluation:

- 2,050 closed-loop episodes,
- E0–E4,
- three-seed primary M1 vs M2-LE comparison,
- language and environmental ablations,
- hierarchical bootstrap,
- McNemar + Holm correction,
- seed consistency,
- generalization gaps,
- failure / efficiency summaries,
- RQ1–RQ5 evidence.

## `07b_supplementary_double_heldout_stress_test.ipynb`

Supplementary exploratory E5:

```text
held-out L3 language
+
held-out Gaussian sensor noise
```

- 300 additional closed-loop episodes,
- M1 vs M2-LE,
- seeds 42 / 123 / 2026,
- paired hierarchical bootstrap,
- raw exact McNemar test,
- exploratory evidence classification,
- no training / tuning / checkpoint changes.

## `08_final_results_and_assignment_evidence.ipynb`

Final evidence consolidation:

- no model training,
- no robot evaluation,
- report-ready figures,
- final headline tables,
- tuning evidence,
- 25k budget evidence,
- RQ mapping,
- critical-analysis claim ledger,
- limitations / threats to validity,
- Assignment 2 rubric evidence map,
- report-results narrative skeleton.

Notebook 08 consolidates the **original** study and supplementary E5. The later M3 findings are documented in N09/N10 and their own extension metrics; Notebook 08's frozen core-study analysis is not retroactively rewritten.

## `09_custom_robust_visual_adapter.ipynb`

Completed **M3-RVA adapter engineering and training**:

- source-grounded insertion after the SmolVLA visual connector, before existing image-token scaling,
- residual 960→64→960 bottleneck with 125,825 additional parameters,
- synthetic identity, gradient-flow and reconstruction preflight,
- real-data batch-4 qualification and measured runtime,
- three independent 25k-step training runs (42, 123, 2026),
- offline validation and complete model/processor reload verification,
- isolated M3 checkpoints and metadata, without replacing original-study models.

## `10_m3_rva_evaluation_and_comparison.ipynb`

Completed **post-hoc M3-RVA versus M2-LE exploratory evaluation**:

- fresh M2-LE matched-seed comparator rollouts,
- E0 ID, E1 held-out L3, E2 photometric, E3 Gaussian-noise conditions,
- 1,200 scheduled slots across three seeds,
- 1,196 scored records, 596 valid pairs, four image-only unscored attempts,
- matched-pair success analysis, seed-level results and coverage,
- fixed-state/pairing audits and a separate completion-accounting amendment,
- conservative scheduled-outcome bounds for missing L3 outcomes,
- explicitly inconclusive/custom-adapter comparative findings; no consistency-loss training.

---

## `11_report_finalization_and_evidence_audit.ipynb`

**Executed, completed, post-hoc and read-only evidence audit:** [`notebooks/11_report_finalization_and_evidence_audit.ipynb`](notebooks/11_report_finalization_and_evidence_audit.ipynb). This is a **reporting/evidence** notebook, not another training or closed-loop evaluation notebook. It reads the frozen N00–N10 records and creates a new timestamped output directory rather than changing previous experimental data.

- **Run:** `20261025T124022_898151Z`, recorded as `POST_HOC_REPORT_ANALYSIS_ONLY` and `COMPLETE`.
- **Audit:** 119 passed checks against 25 source records; original inputs unchanged.
- **Outputs:** six figures (both PNG and SVG), fourteen CSV tables, a source manifest, audit-check CSV, rubric map, output catalogs and report-integration guide.
- **N10 reconciliation:** 1,200 planned model-episode slots, 1,196 scored outcomes, 596 complete valid pairs and four technical exclusions; the missing pairs are not silently imputed or called failures.
- **Scientific boundary:** no new training, robot trials, checkpoint selection, inferential tests or M3-RVA+C consistency training; saved confidence intervals are reused rather than independently revalidated.

**Primary sources:** [N11 summary](results/metrics/report_finalization/20261025T124022_898151Z/notebook11_summary.json), [audit checks](results/metrics/report_finalization/20261025T124022_898151Z/evidence_audit_checks.csv), [source manifest](results/metrics/report_finalization/20261025T124022_898151Z/source_manifest.csv), [integration guide](results/metrics/report_finalization/20261025T124022_898151Z/REPORT_INTEGRATION_GUIDE.md).

# How to run

## Completed core-study workflow

```text
00_environment_and_dataset_eda
        ↓
01_smolvla_baseline_reproduction
        ↓
02_robustness_training
        ↓
03_tuning_and_final_training
        ↓
04_evaluation_and_critical_analysis
        ↓
05_reference_control_and_training_budget_validation
        ↓
05b_final_training_budget_validation
        ↓
06_validated_final_retraining
        ↓
07_definitive_robustness_evaluation
        ↓
07b_supplementary_double_heldout_stress_test
        ↓
08_final_results_and_assignment_evidence
```

## Result-inspection workflow

If the saved experimental artifacts are already present and the objective is to inspect the final conclusions / report figures:

```text
08_final_results_and_assignment_evidence.ipynb
```

Notebook 08 reads the saved core-study evidence and regenerates its assignment tables and figures without retraining the models. It does not yet contain M3 extension results.

## Completed post-hoc M3 extension workflow

```text
N00–N08 completed core-study evidence (retained unchanged)
        ↓
M3 source/design audit and fixed training protocol
        ↓
09_custom_robust_visual_adapter.ipynb
    ├── Stage A: engineering preflight
    ├── Stage B: real-data/batch-4 qualification
    └── Stage C: three independent 25k-step M3-RVA models
        ↓
10_m3_rva_evaluation_and_comparison.ipynb
    ├── frozen four-condition paired evaluation
    ├── documented image-only technical exclusions
    └── coverage-aware three-seed exploratory analysis
        ↓
Consolidated Assignment 1 + Assignment 2 report
```

**Result inspection does not require rerunning 25k-step training or 1,200 simulator slots.** Review the final N09/N10 summaries, CSV tables, notebook outputs and figures. Full simulation replay additionally requires locally retained model weights, simulator/benchmark assets and original raw initial-observation evidence. The **M3-RVA+C consistency experiment is not part of this executed workflow**.

---

## N11 final report inspection (no GPU or simulator replay)

To inspect the final report figures and arithmetic, follow the read-only evidence path after the completed N00–N10 workflow:

```text
N07/N07b original evaluation + N08 core consolidation
               +
N09 training and N10 exploratory finalisation
               ↓
11_report_finalization_and_evidence_audit.ipynb
               ↓
results/metrics/report_finalization/20261025T124022_898151Z/
               ↓
Consolidated Assignment 1 + Assignment 2 report (submitted separately)
```

The saved notebook outputs and the corresponding CSV/JSON/figure files can be inspected directly without rerunning the 25k-step training or the simulator. Re-executing N11 is optional and still requires its referenced source artifacts; inspecting its stored outputs is **not** equivalent to replaying trained policies or validating all statistical premises.

# Reproduction tiers

| Goal | Requirement | Expected cost |
|---|---|---|
| **Inspect final results** | Repository + saved core and M3 metrics / figures | Low |
| **Regenerate core-study report tables / figures** | Notebook 08 + original saved artifacts | Low |
| **Inspect M3 architectural comparison** | N10 outputs + `v2_completion_v1` final CSV/JSON/figures | Low |
| **Inspect N11 final report audit** | Executed N11 notebook and frozen `20261025T124022_898151Z` source, tables and figures | Low |
| **Reproduce evaluation** | Frozen checkpoints + LIBERO assets + validated environment | High |
| **Reproduce final training** | Full environment + data + GPU | High |
| **Rebuild complete study** | Environment + all model training + all closed-loop evaluation | Highest |

---

# Reproducibility

| Item | Location |
|---|---|
| Base environment | `environment.yml` |
| Exact Conda snapshot | `conda-linux-64.lock.txt` |
| Python dependency snapshot | `requirements-lock.txt` |
| Setup procedure | `docs/SETUP.md` |
| Experiment configs | `configs/` |
| Training / model runs | `experiments/` |
| Metrics | `results/metrics/` |
| Figures | `results/figures/` |
| **Final assignment evidence (core)** | `results/metrics/final_assignment_evidence/` |
| **M3 architecture/training source** | `src/m3_rva_training.py`; `notebooks/09_custom_robust_visual_adapter.ipynb` |
| **M3 evaluation/completion source** | `src/m3_rva_evaluation.py`; `src/m3_rva_completion.py`; `notebooks/10_m3_rva_evaluation_and_comparison.ipynb` |
| **M3 training summary and registry** | `results/metrics/m3_rva_extension/v1/` |
| **M3 final extension statistics** | `results/metrics/m3_rva_evaluation/v2_completion_v1/` |
| **M3 comparison figures** | `results/figures/m3_rva_evaluation/v2_completion_v1/` |
| **N11 executed report audit** | `notebooks/11_report_finalization_and_evidence_audit.ipynb` |
| **N11 immutable run artifacts** | `results/metrics/report_finalization/20261025T124022_898151Z/` |
| **M3 per-seed language assignments** | `experiments/m3_rva_extension/v1/m3rva_seed{42,123,2026}_25000/language_assignment.csv` |
| **M3 per-seed processor/model metadata** | `experiments/m3_rva_extension/v1/m3rva_seed{42,123,2026}_25000/model/` (`config.json`, `policy_preprocessor.json`, `policy_postprocessor.json`) |
| **M3 training / evaluation protocols** | `configs/experiments/m3_rva_*.json` |
| **Large M3 checkpoints and raw camera observations** | Retained locally; not included in ordinary Git; required for full rollout replay |

Formal experiment records include:

```text
experiment_id
git_commit
model_checkpoint_revision
dataset_revision
benchmark_revision
random_seed
hyperparameters
training_steps
perturbation_configuration
evaluation_configuration
metrics
checkpoint_path
```

### Committed research evidence versus large local replay artifacts

The ordinary Git repository includes notebooks, original project source, fixed protocol configurations, curated qualitative images, final figures, compact training summaries, episode-level analysis CSVs, final JSON summaries, and pairing/coverage records needed to inspect the reported results.

The repository **does not claim to bundle a complete simulator-replay environment**. Large `.safetensors` model weights, `.pt` optimizer recovery, raw `.npz`/`.npy` camera/simulator captures and temporary qualification models are retained outside ordinary Git; they are not deleted merely because they are excluded from the commit. A complete independent rerun requires compatible checkpoint weights, pinned package environment, external LIBERO data/assets and relevant raw artifacts. Tracked manifests/hashes describe those artifacts, but paths alone do not make excluded bytes available to a marker.

**Provenance policy:** retain the original N07/N07b definitive/supplementary evidence separately from N10's fresh M2-LE versus M3 evaluations. Never pool N10 episodes into the core-study primary hypothesis family. All analysis denominators and excluded technical slots must be stated.

---

# Completed core-study experimental scale

| Component | Final value |
|---|---:|
| LIBERO-Spatial tasks | 10 |
| Primary training seeds | 3 |
| Final frozen checkpoints | 8 |
| Final training budget | 25,000 steps |
| Notebook 04 pilot episodes | 1,230 |
| Notebook 05b budget-validation episodes | 450 |
| Notebook 07 definitive robustness episodes | **2,050** |
| Notebook 07b supplementary E5 episodes | **300** |
| Notebook 07 + 07b final robustness episodes | **2,350** |
| Post-OOD model replacements | **0** |

## Separate completed M3 extension scale

| Component | M3-RVA extension value |
|---|---:|
| New M3-RVA final checkpoints | **3** (seeds 42, 123, 2026) |
| Optimizer steps per M3 training seed | **25,000** |
| New adapter trainable parameters | **125,825** |
| N10 evaluated comparator variants | **M2-LE and M3-RVA** |
| N10 conditions | **E0, E1_L3, E2, E3** |
| N10 scheduled model-episode slots | **1,200** |
| N10 scored model-episode outcomes | **1,196** |
| N10 valid complete paired observations | **596 of 600** |
| N10 excluded technical attempts | **4**, all in E1 L3 |
| N10 pending evaluation slots | **0** |
| Consistency-training arm and E6 | **Not conducted** |

The **2,350** core robustness episodes and **1,200** M3 scheduled slots refer to **different scientific phases**; the latter include 4 unscored technical attempts and repeated fresh M2-LE control evaluation. These numbers are not silently added to create a single confirmatory sample size.

---

# Project roadmap

| Stage | Milestone | Status |
|---|---|---|
| **00** | Environment + dataset + LIBERO validation | ✅ Complete |
| **01** | M0 baseline reproduction | ✅ Complete |
| **02** | Robustness implementation | ✅ Complete |
| **03** | Hyperparameter tuning | ✅ Complete |
| **04** | Pilot diagnostic evaluation | ✅ Complete |
| **05** | Training-regime adequacy | ✅ Complete |
| **05b** | Final training-budget validation | ✅ Complete |
| **06** | Frozen final models | ✅ Complete |
| **07** | Definitive robustness evaluation | ✅ Complete |
| **07b** | Supplementary E5 stress test | ✅ Complete |
| **08** | Final results / assignment evidence | ✅ Complete |
| **09** | M3-RVA custom visual adapter: source audit, engineering preflight, real-data qualification, 3-seed 25k training, reload validation | ✅ Complete |
| **10** | M3-RVA versus M2-LE exploratory closed-loop evaluation, 3 seeds and coverage-aware exclusion accounting | ✅ Finished with four unscored technical attempts |
| **11** | Post-hoc, read-only report finalisation audit, 119 passed checks, six figures and fourteen tables | ✅ Completed and merged into `main` |
| **M3-RVA+C** | Matched-two-view consistency-learning study and `lambda=0` control | Deferred — **not conducted** |
| **Report** | Assignment 1 + Assignment 2 consolidated report | 🔄 In progress |
| **Report status update (26 Oct 2026)** | Revised 31-page consolidated report prepared; complete submission QA and append the future release tag and exact commit SHA to Appendix A | ✅ Revised draft prepared; tag and submission verification pending |
| **Evidence provenance freeze** | Executed N11 and three-seed metadata on `main` | ✅ Verified through commits `8b36afb` and `42e74ee` |

---

# Current progress

```text
Environment / CUDA              ██████████ 100%
LeRobot / SmolVLA               ██████████ 100%
MuJoCo / LIBERO                 ██████████ 100%
Dataset / Task Mapping          ██████████ 100%
Baseline Reproduction           ██████████ 100%
Robustness Implementation       ██████████ 100%
Hyperparameter Tuning           ██████████ 100%
Training Adequacy               ██████████ 100%
Training Budget Validation      ██████████ 100%
Final Model Construction        ██████████ 100%
Definitive Evaluation           ██████████ 100%
Supplementary E5                ██████████ 100%
Final Evidence Notebook         ██████████ 100%

M3-RVA / Notebook 09            ██████████ 100% — 3 trained/verified seeds
M3 comparison / Notebook 10     ██████████ 100% — execution accounted for, 4 unscored
M3-RVA+C consistency arm        NOT CONDUCTED — future work
Extension evaluation protocol  FROZEN/RECORDED — post-hoc exploratory, amended execution

Academic Report                In Progress — completion percentage not audited
```

---

**Progress update (26 October 2026):** The earlier `Academic Report — In Progress` entry above is retained as a historical project snapshot. The revised consolidated report now exists; final delivery still requires README/submission QA, confirmation of the assessment packaging requirements, creation of the submission tag and updating report Appendix A with that tag's **actual** commit SHA. N11's report-only audit is **complete**, while the M3-RVA+C arm remains **not conducted**.

# Key limitations

The completed N00–N08 results should be interpreted within the following boundaries. The separately completed M3 extension has additional limitations and must not be treated as an untouched confirmatory test.

- **Simulation domain** — evidence is based on MuJoCo / LIBERO simulation rather than physical robot deployment.
- **Benchmark scope** — LIBERO-Spatial is used rather than the complete LIBERO-Plus benchmark.
- **Visual augmentation scope** — environmental robustness training is limited to brightness, contrast and saturation.
- **Held-out visual category** — the primary unseen visual corruption is Gaussian sensor noise.
- **Single-seed mechanism ablations** — M2-L and M2-E are seed-42 mechanism studies.
- **Primary seed count** — the headline M1 vs M2-LE analysis uses three training seeds.
- **Failure taxonomy** — automated failure labels are proxies and do not constitute full semantic diagnosis of every trajectory.
- **No full geometry shift study** — camera viewpoint, object layout and robot initial-state robustness are not final claims of this project.
- **E5 status** — E5 is a post-hoc supplementary exploratory test rather than a preregistered confirmatory condition.
- **Training-budget claim** — 25k is empirically frozen under the bounded protocol and is not claimed as a universal optimum.
- **M3 post-hoc motivation** — N09/N10 were designed after N07/N07b outcomes were observed; all M3 effect claims are exploratory.
- **M3 architecture scope** — only one rank-64 token-wise residual adapter configuration was evaluated; this does not establish results for other adapter placements, ranks, architectures or regularizers.
- **M3 technical missingness** — N10 accounted for 1,200 scheduled slots, but four image-only pairing failures were unscored; 596 pairs, not 600, entered comparison estimates.
- **M3 camera/reset repeatability** — small wrist-camera differences occurred under repeated model-free simulator resets; the completion amendment did not fix renderer determinism.
- **M3 statistical scope** — only three independently trained extension seeds and ten LIBERO-Spatial tasks; bootstrap/p estimates are conditional on their design and withheld for incomplete E1.
- **M3 comparative result** — no consistent additional adapter advantage was demonstrated, but that does not prove adapter methods are ineffective or that models are equivalent.
- **Not performed** — no M3-RVA+C paired-view consistency training, no lambda-zero matched two-view control, no E6 visual holdout and no physical robot deployment.
- **Preprocessing-statistics provenance** — the original state/action normalisation used inherited dataset metadata statistics, not exclusively statistics fitted to the training-episode split; this limits strict training-only preprocessing claims, and the size of the effect is unmeasured.
- **Development-stage evaluation exposure** — pilot OOD outcomes were visible during methodological development, so the complete development history should not be characterised as untouched final-test selection.
- **Short hyperparameter screen** — six learning-rate/weight-decay candidates were compared after 200 optimisation updates; the batch size and scheduler were fixed design settings, not additional tuned dimensions. This initial screen does not rank full 25k runs.
- **Validation-history limitation** — M3 training logs record batch-loss trajectories and final offline validation losses but not a repeated M3 validation curve; separate M1 budget checkpoints do not fill that historical gap.
- **Language diversity** — L3 uses ten distinct held-out task instructions, each evaluated in repeated rollouts. More episodes do not imply more unique paraphrases; no independent inter-rater consistency study is claimed.
- **Pairing controls differ by phase** — N07 matched task/seed/replica identifiers with deterministic reset/noise seeds, while N10 introduced additional indexed initial-state, exposed-state and observation checks. The stronger N10 audit should not be retroactively attributed to N07.
- **N11 audit scope** — passing arithmetic and source-integrity checks does not independently verify physical robot validity, eliminate test-selection limitations, or establish model superiority. The original experiment record and any technical exclusions remain unchanged.

---

# Research contribution

The contribution is **not** a new large-scale VLA foundation model.

The study adapts an efficient existing VLA model, evaluates controlled robustness-oriented training under a reproducible closed-loop protocol, and subsequently tests a lightweight custom architectural modification. Its claims distinguish **measured outcomes** from hypothesized mechanisms and do not claim that SmolVLA or residual adapters were invented in this project.

The completed core-study contribution is:

> **A controlled multi-seed empirical investigation of robustness-oriented SmolVLA fine-tuning under linguistic and environmental distribution shifts, demonstrating strong improvement in held-out linguistic generalization, mixed visual robustness effects, and supplementary evidence of improved resilience under a double-held-out language-plus-sensor-noise stress condition.**

The completed **post-hoc M3 architectural contribution** is separate:

> **A source-grounded, 125,825-parameter residual visual adapter was implemented, trained across three fixed seeds, and compared with the augmentation-trained M2-LE model in closed-loop LIBERO-Spatial conditions. The study recorded 596 valid model pairs, four technical exclusions and small mixed effects, without demonstrating an additional consistent robustness benefit.**

This contributes an *evaluated design and an informative negative/inconclusive comparison*, not a claim that a new foundation VLA or universally superior policy was created. The proposed consistency regularization is a potential future study, not a completed result.

The study deliberately distinguishes:

```text
standard benchmark performance
```

from:

```text
seen perturbation robustness
```

and:

```text
held-out perturbation generalization
```

as well as:

```text
definitive / confirmatory evidence
```

from:

```text
supplementary / exploratory evidence
```

---

# Assignment deliverables

## Assignment 1

- abstract,
- background and domain,
- research problem,
- aim and objectives,
- significance,
- recent VLA / robotics deep-learning literature,
- architecture survey,
- related work,
- benchmark / dataset methodology,
- model-selection justification,
- proposed methodology,
- evaluation metrics.

## Assignment 2

- preprocessing,
- model implementation,
- standard baseline,
- robustness adaptation,
- training,
- hyperparameter tuning,
- validation,
- training-budget validation,
- final model construction,
- definitive closed-loop evaluation,
- statistical analysis,
- performance visualization,
- mechanism ablations,
- supplementary E5 stress test,
- critical analysis,
- limitations and threats to validity,
- baseline vs final-model comparison,
- **custom M3-RVA architecture, trainable parameter accounting, source-grounded insertion and checkpoint-reload checks**,
- **N10 three-seed exploratory M3-RVA versus M2-LE performance comparison**,
- **technical exclusion/paired-denominator audit and missing-outcome sensitivity bounds**,
- distinction between completed M3-RVA architecture testing and **unimplemented** M3-RVA+C consistency future work.

Python implementation and representative generated outputs are maintained in Jupyter notebooks. N09/N10's extension metrics complement the frozen original Notebook 08 evidence and are **not** retrospectively inserted into it.

---

# Current milestone

> **Completed original milestone:** N00–N08, with Notebook 08 recorded as PASS and the core findings preserved.

> **Preserved core-study scale:** 2,050 definitive + 300 supplementary episodes and eight frozen checkpoints. No M3 run is included in these totals.

> **Completed M3-RVA extension:** N09 trained seeds 42/123/2026, and N10 accounted for 1,200 planned slots (1,196 scored episodes, 596 paired comparisons, four unscored initial-camera mismatches). The evaluated adapter showed **no consistent additional robustness advantage**. M3-RVA+C and E6 were **not conducted**.

> **Next:** keep all training/evaluation protocols and results frozen, publish the curated final research evidence and figures, finish the consolidated Assignment 1 + Assignment 2 report, verify literature references, and perform submission QA. No further model or experiment tuning is required for the completed study.

---

# Marker navigation, reproducibility and submission freeze

This appendix to the README is **additive** to the complete original research narrative. It makes the final evidence easy to locate without changing earlier experiments, their denominators or the scientific claims.

## Evidence-by-rubric reading guide

| Assignment criterion | Read in the consolidated report | Repository evidence to inspect | Important interpretation |
|---|---|---|---|
| **CLO1 (40%): background, architecture, literature and justification** | Sections 1–3 and architecture description in Section 4 | [N00](notebooks/00_environment_and_dataset_eda.ipynb), [N02](notebooks/02_robustness_training.ipynb), [N09](notebooks/09_custom_robust_visual_adapter.ipynb), `configs/` | Foundation-model and benchmark work are attributed; the local contribution is adaptation/evaluation, not invention of SmolVLA. |
| **CLO2 (42%): implementation, tuning and validation** | Sections 4–5 and the supporting appendices | [N03](notebooks/03_tuning_and_final_training.ipynb), [N05b](notebooks/05b_final_training_budget_validation.ipynb), [N06](notebooks/06_validated_final_retraining.ipynb), [N09](notebooks/09_custom_robust_visual_adapter.ipynb), `results/metrics/m3_rva_extension/v1/` | The 200-update initial search, 25k budget choice, amended adequacy gate and limited validation trajectory must all be interpreted as documented. |
| **CLO3 (18%): results, visualisation and critical analysis** | Sections 6–7 and the evidence appendices | [N07](notebooks/07_definitive_robustness_evaluation.ipynb), [N07b](notebooks/07b_supplementary_double_heldout_stress_test.ipynb), [N08](notebooks/08_final_results_and_assignment_evidence.ipynb), [N10](notebooks/10_m3_rva_evaluation_and_comparison.ipynb), [N11](notebooks/11_report_finalization_and_evidence_audit.ipynb) | N07 is the core study; E5 and M3 are exploratory; N11 performs report-only reanalysis. |

## Stable evidence links and release policy

- **Public repository:** <https://github.com/LeeMing988/deep-learning-vla-robustness>.
- **N11 merge commit:** [`8b36afb7223f049def24088866f326cc005da516`](https://github.com/LeeMing988/deep-learning-vla-robustness/commit/8b36afb7223f049def24088866f326cc005da516).
- **M3 seed-metadata commit:** [`42e74eefa1aa5ff40c29a668d1c901a2fdcbecf7`](https://github.com/LeeMing988/deep-learning-vla-robustness/commit/42e74eefa1aa5ff40c29a668d1c901a2fdcbecf7).
- **Authoritative report-audit run:** [`20261025T124022_898151Z`](results/metrics/report_finalization/20261025T124022_898151Z/) and its [summary](results/metrics/report_finalization/20261025T124022_898151Z/notebook11_summary.json).
- **Dataset:** <https://huggingface.co/datasets/lerobot/libero>. The actual dataset files are external and **not** included in the submission repository.
- **Report:** the consolidated Word/PDF report is supplied separately from the GitHub research-evidence repository; the reader should use its Appendix A to locate the frozen source revision and audit path.
- **Final tag:** **not created at the time of this README addendum**. Create one annotated tag only **after** this README update, final link/evidence checks, and any approved supporting changes have been committed to `main`. Then insert the **actual** tag name, target commit SHA and GitHub tag URL into the report's Appendix A. Do not tag a moving branch name or guess the SHA in advance.

## Practical submission checklist

- [x] Preserve original N00–N08 experimental evidence and N09/N10 exploratory extension separately.
- [x] Merge the originally executed N11 notebook and associated 119-check evidence into `main`.
- [x] Preserve three-seed M3-RVA language assignments and the three model/processor JSON files per seed.
- [ ] Review and commit this additive README documentation update.
- [ ] Confirm report-format, word-count and appendix-handling rules with the module's current submission instructions; the available briefing does not explicitly guarantee that appendix text is exempt from word-count assessment.
- [ ] Audit external reference links and ensure the submitted notebook files contain their saved outputs.
- [ ] Create and push the verified annotated final-submission Git tag; verify its target commit.
- [ ] Insert the exact Git tag name, commit SHA and evidence URLs into report Appendix A and perform final document QA.

The 4,093 untracked local files observed during repository cleanup contained historical diagnostic/evaluation data, temporary runtime files and selected small training metadata. **Only the twelve selected M3 seed metadata files were committed in the documented cleanup.** Do not interpret excluded local files as deleted, and do not imply that the ordinary Git checkout contains the omitted large model weights or raw replay observations.

# License

Academic project provided for educational and research purposes.

Third-party software, pretrained models, benchmark assets and datasets remain subject to their respective licences and terms of use.

This repository does not redistribute LIBERO datasets, Hugging Face model weights or other externally licensed assets unless explicitly permitted by their respective licences.

---

# Citation

```bibtex
@misc{yeoh2026vlarobustness,
  author      = {Yeoh, Lee Ming},
  title       = {Robustness-Enhanced Vision-Language-Action Learning for
                 Generalizable Robotic Manipulation under Linguistic and
                 Environmental Distribution Shifts},
  year        = {2026},
  institution = {Asia Pacific University of Technology and Innovation},
  note        = {Deep Learning, CT100-3-M-DL},
  url         = {https://github.com/LeeMing988/deep-learning-vla-robustness}
}
```

---

# Acknowledgements

- Hugging Face **LeRobot** project and contributors.
- **SmolVLA** authors and contributors.
- **LIBERO** benchmark authors and maintainers.
- **MuJoCo** by Google DeepMind.
- Asia Pacific University of Technology & Innovation (APU) course staff for academic guidance.
