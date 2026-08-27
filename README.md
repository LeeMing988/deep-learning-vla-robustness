# deep-learning-vla-robustness

![Python](https://img.shields.io/badge/python-3.12.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![LeRobot](https://img.shields.io/badge/LeRobot-0.6.1-yellow)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.1-red)
![Benchmark](https://img.shields.io/badge/benchmark-LIBERO--Spatial-purple)
![Study](https://img.shields.io/badge/study-VLA%20robustness-blueviolet)
![Status](https://img.shields.io/badge/status-Experiments%20Complete-brightgreen)
![License](https://img.shields.io/badge/license-Academic-orange)

A Deep Learning research project on **robustness-enhanced Vision-Language-Action (VLA) learning** for robotic manipulation under **linguistic and environmental distribution shifts**.

The experimental cycle is complete. The current project phase is **final report preparation, literature comparison, figure selection and submission QA**.

---

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
| **Final Training Budget** | 25,000 optimizer steps |
| **Primary Training Seeds** | 42, 123, 2026 |
| **Frozen Final Checkpoints** | 8 |
| **Definitive Robustness Episodes** | 2,050 |
| **Supplementary E5 Episodes** | 300 |
| **Final Robustness Evaluation Episodes** | 2,350 |
| **Year** | 2026 |

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

The final implemented study uses **LIBERO-Spatial** as the controlled manipulation benchmark and evaluates a specific set of **linguistic, photometric and Gaussian-noise distribution shifts**.

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

---

# Research questions

- **RQ1.** To what extent does robustness-oriented fine-tuning improve seen-perturbation robustness and generalization to unseen perturbation categories in VLA robotic manipulation?
- **RQ2.** At what level of linguistic variation does SmolVLA manipulation performance degrade, and can language-aware fine-tuning reduce this degradation?
- **RQ3.** Which evaluated environmental perturbations most strongly affect VLA manipulation performance?
- **RQ4.** Does robustness training on selected perturbation categories transfer to categories absent during training?
- **RQ5.** Does improved out-of-distribution robustness reduce standard in-distribution performance?

Notebook 07 provides the **definitive RQ1–RQ5 evidence**.

Notebook 07b provides **supplementary exploratory evidence** for transfer under a stricter double-held-out condition. It does not replace or redefine Notebook 07's confirmatory analysis.

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

# Model variants

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

# Final training configuration

The final model-development protocol was frozen before definitive OOD evaluation.

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

---

# Experimental evidence chain

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

No training, hyperparameter tuning or checkpoint replacement was performed after the definitive OOD outcomes were observed.

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

# Final headline results

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

# Final verified experimental status

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
│   └── environment snapshots / project notes
│
├── experiments/
│   └── training runs, frozen checkpoints and run metadata
│
├── notebooks/
│   ├── 00_environment_and_dataset.ipynb
│   ├── 01_baseline_reproduction.ipynb
│   ├── 02_robustness_training.ipynb
│   ├── 03_tuning_and_final_training.ipynb
│   ├── 04_evaluation_and_critical_analysis.ipynb
│   ├── 05_training_regime_validation.ipynb
│   ├── 05b_final_training_budget_validation.ipynb
│   ├── 06_validated_final_retraining.ipynb
│   ├── 07_definitive_robustness_evaluation.ipynb
│   ├── 07b_supplementary_double_heldout_stress_test.ipynb
│   └── 08_final_results_and_assignment_evidence.ipynb
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
│   └── reusable experiment implementation
│
├── .gitignore
├── environment.yml
├── conda-linux-64.lock.txt
├── requirements-lock.txt
└── README.md
```

---

# Notebook workflow

## `00_environment_and_dataset.ipynb`

Environment, dataset and benchmark validation:

- Python / package versions,
- CUDA verification,
- LIBERO task inspection,
- dataset audit,
- train / validation split,
- simulator compatibility,
- camera observations,
- task mapping.

## `01_baseline_reproduction.ipynb`

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

## `05_training_regime_validation.ipynb`

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

---

# How to run

## Full research workflow

```text
00_environment_and_dataset
        ↓
01_baseline_reproduction
        ↓
02_robustness_training
        ↓
03_tuning_and_final_training
        ↓
04_evaluation_and_critical_analysis
        ↓
05_training_regime_validation
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

Notebook 08 reads the saved evidence and regenerates the final assignment tables and figures without retraining the models.

---

# Reproduction tiers

| Goal | Requirement | Expected cost |
|---|---|---|
| **Inspect final results** | Repository + saved metrics / figures | Low |
| **Regenerate report tables / figures** | Notebook 08 + saved artifacts | Low |
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
| Final assignment evidence | `results/metrics/final_assignment_evidence/` |

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

---

# Final experimental scale

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
| **Report** | Assignment 1 + Assignment 2 consolidated report | 🔄 In progress |

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

Academic Report                 ███░░░░░░░  In Progress
```

---

# Key limitations

The final project should be interpreted within the following boundaries.

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

---

# Research contribution

The contribution is **not** a new large-scale VLA foundation model.

The study adapts an efficient existing VLA model and evaluates controlled robustness-oriented training under a reproducible closed-loop protocol.

The final contribution is:

> **A controlled multi-seed empirical investigation of robustness-oriented SmolVLA fine-tuning under linguistic and environmental distribution shifts, demonstrating strong improvement in held-out linguistic generalization, mixed visual robustness effects, and supplementary evidence of improved resilience under a double-held-out language-plus-sensor-noise stress condition.**

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
- baseline vs final-model comparison.

Python implementation and representative generated outputs are maintained in Jupyter notebooks.

---

# Current milestone

> **Experimental cycle complete — Notebook 08 PASS.**

> **Final robustness scale:** 2,050 definitive + 300 supplementary episodes, eight frozen final checkpoints, and no post-OOD retraining or model replacement.

> **Next:** consolidated Assignment 1 + Assignment 2 report writing, published-results comparison, final figure selection, critical analysis and submission QA.

---

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
