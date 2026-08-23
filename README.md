# deep-learning-vla-robustness

![Python](https://img.shields.io/badge/python-3.12.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![LeRobot](https://img.shields.io/badge/LeRobot-0.6.1-yellow)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.1-red)
![Benchmark](https://img.shields.io/badge/benchmark-LIBERO--Spatial-purple)
![Study](https://img.shields.io/badge/study-VLA%20robustness-blueviolet)
![License](https://img.shields.io/badge/license-Academic-orange)
![Status](https://img.shields.io/badge/status-In%20Progress-blue)

A Deep Learning study of **robustness-enhanced Vision-Language-Action (VLA) learning** for generalizable robotic manipulation under **linguistic and environmental distribution shifts**.

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
| **Robustness Benchmark** | LIBERO-Plus — subject to methodology gate |
| **Framework** | Hugging Face LeRobot |
| **Simulation** | MuJoCo |
| **Year** | 2026 |

---

## Project overview

Vision-Language-Action (VLA) models combine **visual perception, natural-language understanding and robotic control** within a unified policy that predicts robot actions from multimodal observations.

Recent VLA models demonstrate strong performance on standard robotic manipulation benchmarks. However, strong in-distribution benchmark performance does not necessarily imply that a policy remains reliable when its deployment conditions differ from those observed during training.

A robotic manipulation policy may encounter changes such as:

- semantically equivalent but differently worded instructions,
- altered object layouts,
- changed camera viewpoints,
- different robot initial states,
- lighting variation,
- background changes,
- sensor noise.

This project investigates whether **controlled robustness-oriented fine-tuning** can improve the ability of an efficient VLA policy to generalize across these distribution shifts while maintaining its standard manipulation performance.

The primary model is **SmolVLA**, an efficient VLA architecture integrated through Hugging Face **LeRobot**. The primary experimental benchmark is **LIBERO-Spatial**, with **LIBERO-Plus** being evaluated as a standardized source of robustness perturbations.

### Central research problem

> **Can robustness-oriented fine-tuning improve VLA robotic manipulation performance under linguistic and environmental distribution shifts without substantially degrading standard in-distribution performance?**

---

## Assessment alignment

This repository supports both components of the Deep Learning module as one continuous research project.

| Component | Weight | Repository focus |
|---|---:|---|
| **Assignment 1** | **40%** | Domain background, VLA literature, deep-learning architectures, research problem, methodology, benchmark selection and experimental design |
| **Assignment 2 — Model Implementation, Tuning & Validation** | **42%** | SmolVLA implementation, baseline reproduction, fine-tuning, hyperparameter tuning, validation and final model building |
| **Assignment 2 — Visualization & Critical Analysis** | **18%** | Performance visualization, robustness analysis, ablation studies, failure analysis and critical interpretation |

The final academic deliverable is a **single consolidated Deep Learning report** supported by reproducible Python notebooks and experimental outputs.

---

## Research aim

To investigate whether robustness-oriented fine-tuning improves the generalization of efficient Vision-Language-Action policies for robotic manipulation under controlled linguistic and environmental distribution shifts.

---

## Research objectives

1. Establish a reproducible SmolVLA + LeRobot + LIBERO experimental environment.
2. Reproduce and audit the performance of an existing LIBERO-fine-tuned SmolVLA checkpoint.
3. Establish a standard SmolVLA fine-tuning baseline.
4. Measure performance degradation under controlled linguistic and environmental perturbations.
5. Develop language-aware and environment-aware robustness fine-tuning strategies.
6. Evaluate whether robustness learned from selected perturbations transfers to unseen perturbation categories.
7. Quantify the trade-off between in-distribution performance and out-of-distribution robustness.
8. Analyse failures, uncertainty and limitations of the resulting models.

---

## Research questions

- **RQ1.** To what extent does robustness-oriented fine-tuning improve both seen-perturbation robustness and generalization to unseen perturbation categories in VLA robotic manipulation?

- **RQ2.** At what level of linguistic variation does SmolVLA manipulation performance begin to degrade, and can language-aware fine-tuning reduce this degradation?

- **RQ3.** Which environmental perturbation dimensions most strongly reduce VLA manipulation performance?

- **RQ4.** Does robustness training on selected perturbation categories transfer to categories absent during training?

- **RQ5.** Does improved out-of-distribution robustness reduce standard in-distribution performance?

---

## Research hypotheses

| Hypothesis | Description |
|---|---|
| **H1** | Language-aware fine-tuning will reduce degradation under unseen but semantically equivalent task instructions. |
| **H2** | Robustness-oriented training will improve performance under environmental perturbations represented during training. |
| **H3** | Robustness training will transfer to at least some perturbation categories absent during training. |
| **H4** | The combined robustness model will produce a smaller in-distribution → out-of-distribution generalization gap than standard fine-tuning. |
| **H5** | Robustness improvements may preserve or slightly reduce standard performance in exchange for improved OOD generalization. |

A null linguistic result will also be considered meaningful if the pretrained VLA representation already provides substantial linguistic invariance.

---

## Experimental design

The project separates **checkpoint reproducibility**, **standard fine-tuning**, and **robustness-oriented fine-tuning**.

| Model | Role |
|---|---|
| **M0** | Released LIBERO-fine-tuned SmolVLA checkpoint — reproducibility / evaluation audit |
| **M1** | Standard SmolVLA fine-tuning baseline |
| **M2-L** | Linguistic robustness fine-tuning |
| **M2-E** | Environmental robustness fine-tuning |
| **M2-LE** | Combined linguistic + environmental robustness fine-tuning |

The main final comparison is planned as:

```text
M1 — Standard Fine-Tuning
          vs
M2-LE — Combined Robustness Fine-Tuning
```

with the following supporting ablation:

```text
                 M1
                  │
        ┌─────────┼─────────┐
        │         │         │
      M2-L      M2-E      M2-LE
    Language  Environment  Combined
```

---

## Experimental framework

```text
                         SmolVLA
                            │
                            ▼
                     LIBERO-Spatial
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      Standard Fine-Tuning       Robustness Fine-Tuning
              M1              M2-L / M2-E / M2-LE
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                   Controlled Evaluation
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
       E0                  E1                E2 / E3
 In-Distribution        Language           Environmental
                         Shift                Shift
                            │
                            ▼
             Success · Robustness · Failure Analysis
```

---

## Benchmark

### LIBERO-Spatial

**LIBERO-Spatial** is the primary benchmark suite for the initial experiments.

The validated installation exposes **10 manipulation tasks** involving closely related goals under different spatial configurations.

Examples include:

```text
pick up the black bowl between the plate and the ramekin
and place it on the plate
```

```text
pick up the black bowl next to the ramekin
and place it on the plate
```

```text
pick up the black bowl from table center
and place it on the plate
```

```text
pick up the black bowl on the stove
and place it on the plate
```

The suite therefore provides a controlled environment for investigating spatial sensitivity and manipulation generalization.

### Validated benchmark suites

The current LIBERO installation successfully discovers:

```text
libero_10
libero_100
libero_90
libero_goal
libero_object
libero_spatial
```

---

## LIBERO-Plus robustness benchmark

LIBERO-Plus is being evaluated as the standardized benchmark for distribution-shift testing.

Its perturbation dimensions are relevant to this research:

| Perturbation dimension | Planned role |
|---|---|
| Language | Linguistic robustness |
| Lighting | Seen environmental variation |
| Background | Seen environmental variation |
| Camera viewpoint | Held-out generalization |
| Robot initial state | Held-out generalization |
| Object layout | Held-out generalization |
| Sensor noise | Held-out generalization |

The final adoption strategy will be frozen after a dedicated methodology and compatibility gate.

Possible outcomes are:

1. full LIBERO-Plus training + evaluation,
2. LIBERO-Plus evaluation with controlled training augmentations, or
3. partial use where individual perturbations satisfy experimental validity requirements.

---

## Seen robustness vs held-out generalization

The experimental methodology explicitly distinguishes between:

```text
Seen robustness
```

and:

```text
Held-out generalization
```

A model performing well on the same perturbation types used during training does **not** automatically demonstrate generalization.

The current provisional split is:

| Category | Role |
|---|---|
| Language | Seen / training |
| Lighting | Seen / training |
| Background | Seen / training |
| Camera viewpoint | Held-out |
| Robot initial state | Held-out |
| Object layout | Held-out |
| Sensor noise | Held-out |

This split remains provisional until the robustness methodology is frozen.

---

## Linguistic perturbation levels

Language robustness will be evaluated using semantically equivalent instructions of increasing difficulty.

| Level | Description |
|---|---|
| **L0** | Original benchmark instruction |
| **L1** | Lexical / synonym substitution |
| **L2** | Syntactic restructuring |
| **L3** | Compositional rewording |

Example:

```text
L0: Pick up the black bowl and place it on the plate.

L1: Grab the black bowl and put it on the plate.

L2: Place the black bowl onto the plate after picking it up.

L3: Move the black bowl from its current position so that
    it ends up resting on the plate.
```

All generated variants must preserve the semantics of the original manipulation task.

---

## Evaluation conditions

| Condition | Purpose |
|---|---|
| **E0** | Standard in-distribution LIBERO performance |
| **E1** | Linguistic distribution shift |
| **E2** | Seen environmental perturbations |
| **E3** | Held-out environmental perturbation categories |
| **E4** | Combined perturbations — optional depending on compute budget |

Where possible, competing models will be evaluated using the **same task, initial state and perturbation condition** to enable paired comparison.

---

## Primary metrics

### Task success rate

The primary metric is:

```text
                  Successful Episodes
Task Success = ─────────────────────────
                    Total Episodes
```

### Generalization gap

```text
G = SR_ID - SR_OOD
```

where:

- `SR_ID` = in-distribution success rate
- `SR_OOD` = out-of-distribution success rate

A smaller generalization gap indicates stronger robustness.

### Secondary measurements

- per-task success rate,
- per-perturbation success rate,
- episode length,
- completion steps,
- inference latency,
- simulator latency,
- GPU memory utilization,
- training loss,
- validation loss,
- failure-category frequency.

---

## Statistical strategy

Two different sources of variability will be treated separately.

### Evaluation uncertainty

Bootstrap confidence intervals may be computed across evaluation episodes for an individual trained checkpoint.

### Training uncertainty

Independent training seeds will quantify model-training stochasticity.

Planned headline seeds:

```text
42
123
2026
```

Where computationally feasible, final results will therefore report:

```text
mean ± standard deviation
```

across independently trained models.

Episode-level bootstrap uncertainty will **not** be presented as a substitute for independent training runs.

---

## Failure taxonomy

Failed episodes will be categorized instead of analysing only aggregate success rate.

| Code | Failure |
|---|---|
| **F01** | Object localization failure |
| **F02** | Wrong target selection |
| **F03** | Failed grasp |
| **F04** | Object slip |
| **F05** | Incorrect placement |
| **F06** | Language misunderstanding |
| **F07** | Trajectory / control failure |
| **F08** | Timeout |
| **F09** | Collision / environmental interaction |
| **F10** | Other |

---

## Current verified checkpoint

> The following are **technical validation results**, not final research-performance findings.

### Checkpoint 01 — Environment + LIBERO

| Component | Status |
|---|---|
| WSL2 Ubuntu environment | **PASS** |
| Conda environment | **PASS** |
| PyTorch / CUDA | **PASS** |
| NVIDIA RTX 5090 CUDA computation | **PASS** |
| LeRobot installation | **PASS** |
| MuJoCo installation | **PASS** |
| GPU-accelerated OpenGL | **PASS** |
| MuJoCo RGB rendering | **PASS** |
| LIBERO configuration | **PASS** |
| LIBERO-Spatial discovery | **PASS** |
| Benchmark task loading | **PASS** |
| Fixed initial-state loading | **PASS** |
| Agent-view camera observation | **PASS** |
| Eye-in-hand camera observation | **PASS** |
| 7-D action interface | **PASS** |
| SmolVLA policy integration | **NEXT** |

---

## LIBERO simulator validation

A real **LIBERO-Spatial Task 0** environment has been successfully instantiated.

### Task

```text
pick up the black bowl between the plate and the ramekin
and place it on the plate
```

### Benchmark state

```text
Available fixed initial states: 50
Environment reset: PASS
Fixed initial-state loading: PASS
```

### Agent-view camera

```text
Shape:           (128, 128, 3)
Type:            uint8
Minimum pixel:   6
Maximum pixel:   238
Mean pixel:      119.4102
Non-zero pixels: 49152
Status:          PASS
```

### Eye-in-hand camera

```text
Shape:           (128, 128, 3)
Type:            uint8
Minimum pixel:   3
Maximum pixel:   231
Mean pixel:      126.3576
Non-zero pixels: 49152
Status:          PASS
```

### Action interface

A seven-dimensional control action was successfully accepted:

```text
Action:
[0. 0. 0. 0. 0. 0. 0.]

Reward:
0.0

Done:
False

Simulator step:
PASS
```

The zero action is used strictly as a technical compatibility test and is not expected to complete the manipulation task.

---

## Experimental environment

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
| EGL helper | `hf_egl_probe==1.0.2` |
| Rendering | WSLg + Mesa D3D12 + GLFW |

### LeRobot revision

```text
Tag:    v0.6.1
Commit: 7e241bd630a3719a56157a497ce5d08f244784f1
```

### Patched `egl_probe` revision

```text
Version: 1.0.1
Commit:  3ddf90db69264de2c621af567bfc557849126cff
```

---

## GPU validation

### Deep-learning computation

```text
PyTorch
   ↓
CUDA 12.8
   ↓
RTX 5090
   ↓
GPU Tensor Computation
   ↓
PASS
```

Validated output:

```text
PyTorch:            2.11.0+cu128
CUDA runtime:       12.8
CUDA available:     True
GPU:                NVIDIA GeForce RTX 5090
Compute capability: (12, 0)

RTX 5090 CUDA TEST: PASS
```

### MuJoCo rendering

```text
MuJoCo
   ↓
OpenGL / GLFW
   ↓
Mesa D3D12
   ↓
WSLg
   ↓
RTX 5090
   ↓
RGB Frame
   ↓
PASS
```

Validated standalone rendering:

```text
Image shape:      (256, 256, 3)
Image dtype:      uint8
Minimum pixel:    0
Maximum pixel:    255
Mean pixel:       33.8369
Non-zero pixels:  110960

MuJoCo RENDER TEST: PASS
```

---

## Repository structure

```text
deep-learning-vla-robustness/
│
├── configs/
│   └── experiment / training / evaluation configurations
│
├── docs/
│   ├── SETUP.md
│   └── environment_snapshot_2026-08-23.txt
│
├── experiments/
│   └── experiment manifests and run metadata
│
├── notebooks/
│   ├── 00_environment_and_dataset.ipynb
│   ├── 01_baseline_reproduction.ipynb
│   ├── 02_robustness_training.ipynb
│   ├── 03_tuning_and_final_training.ipynb
│   └── 04_evaluation_and_critical_analysis.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── tables/
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

> Some directories and notebooks above represent the planned final repository structure and will be populated progressively as experiments are completed.

---

## Notebook workflow

The research implementation is intentionally organized into a small number of substantive notebooks.

### `00_environment_and_dataset.ipynb`

Environment and benchmark validation:

- Python / package versions,
- CUDA verification,
- LIBERO benchmark discovery,
- LIBERO-Spatial task inspection,
- simulator initialization,
- observation inspection,
- camera visualization,
- action-interface validation.

### `01_baseline_reproduction.ipynb`

Baseline reproduction:

- released LIBERO-fine-tuned SmolVLA checkpoint,
- checkpoint revision audit,
- observation preprocessing,
- policy inference,
- baseline episode evaluation,
- M0 reproducibility analysis,
- standard M1 baseline.

### `02_robustness_training.ipynb`

Robustness implementation:

- standard fine-tuning,
- linguistic robustness,
- environmental robustness,
- combined robustness,
- M1 / M2-L / M2-E / M2-LE comparison.

### `03_tuning_and_final_training.ipynb`

Model optimization:

- learning-rate tuning,
- weight-decay tuning,
- augmentation-probability tuning,
- validation,
- final hyperparameter selection,
- independent final training seeds.

### `04_evaluation_and_critical_analysis.ipynb`

Final analysis:

- E0 in-distribution evaluation,
- E1 linguistic robustness,
- E2 seen environmental robustness,
- E3 held-out generalization,
- ablations,
- statistics,
- visualization,
- failure analysis,
- critical interpretation.

---

## Hyperparameter strategy

The study uses **staged tuning** rather than an exhaustive Cartesian search.

Candidate ranges include:

| Parameter | Candidate values |
|---|---|
| Learning rate | `1e-5`, `3e-5`, `1e-4` |
| Weight decay | `0.01`, `0.05`, `0.10` |
| Language augmentation probability | `0.25`, `0.50`, `0.75` |

Approximately **8–12 meaningful tuning runs** are planned before final model training.

Exploratory tuning may use a single seed; headline models will use independent seeds where computationally feasible.

---

## Methodology highlights

### Controlled baseline

Robustness gains will be measured against a standard SmolVLA fine-tuning baseline rather than against an unrelated architecture.

### Same-condition evaluation

Where possible, competing models will receive the same:

- manipulation task,
- benchmark initial state,
- perturbation,
- language instruction condition.

### Held-out perturbations

Not all robustness perturbation categories will be visible during training.

This allows the study to distinguish:

```text
robustness to known perturbations
```

from:

```text
generalization to unseen perturbation categories
```

### Action-label validity

Physical states in recorded demonstrations will not be arbitrarily transformed while retaining actions that would become invalid for the modified state.

### Repeated final training

Headline models are planned to use multiple independent training seeds rather than relying only on repeated evaluation of one checkpoint.

### Failure analysis

Success rate will be complemented with structured episode-failure categories.

---

## Experimental gates

The project uses explicit technical and methodological gates before expensive training.

### G0 — Technical compatibility

Required pipeline:

```text
WSL/Linux
   ↓
PyTorch + CUDA
   ↓
LeRobot
   ↓
SmolVLA
   ↓
LIBERO
   ↓
MuJoCo
   ↓
Observation
   ↓
Policy inference
   ↓
Valid action
   ↓
Simulator step
   ↓
Episode termination
```

Current state:

```text
Environment → Observation → Manual Action
PASS

SmolVLA → Policy Action → Complete Episode
NEXT
```

### G1 — Runtime benchmark

After G0, a controlled warm-up and timing study will determine:

- episode wall-clock time,
- inference latency,
- simulator latency,
- GPU utilization,
- peak VRAM,
- practical final evaluation budget.

### G2 — LIBERO-Plus methodology gate

LIBERO-Plus will be audited before its perturbation protocol is frozen.

---

## Setup

### Prerequisites

The validated development environment uses:

- Windows 11,
- WSL2,
- Ubuntu 24.04,
- NVIDIA GPU with current WSL-compatible Windows driver,
- Miniforge / Conda,
- Git.

The current reference machine uses an **NVIDIA GeForce RTX 5090**, but exact hardware equivalence is not required provided that the selected configuration supports the model and simulator.

### Environment files

The repository separates readable configuration from exact environment snapshots:

```text
environment.yml
conda-linux-64.lock.txt
requirements-lock.txt
docs/SETUP.md
```

Detailed environment construction and compatibility instructions are documented in:

```text
docs/SETUP.md
```

### LIBERO external data

LIBERO assets and model checkpoints are **not committed to Git**.

The first LIBERO simulator run downloads the required assets through the benchmark tooling and stores them in the local LIBERO cache.

---

## How to run

The planned notebook order is:

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
```

The notebooks will contain the Python code, comments and representative generated outputs required to reproduce the research workflow.

---

## Reproduction tiers

| Goal | Requirement | Expected cost |
|---|---|---|
| **Inspect methodology / results** | Repository + saved metrics/figures | Low |
| **Reproduce evaluation** | Repository + pinned checkpoint + LIBERO assets | Moderate |
| **Reproduce final training** | Full environment + datasets + model checkpoint + GPU | High |
| **Rebuild complete study** | Environment + training + all model seeds + all evaluation conditions | Highest |

As the study progresses, committed metrics and figures will allow final analysis to be reproduced without retraining every model.

---

## Reproducibility

| Item | Location |
|---|---|
| Base environment | `environment.yml` |
| Exact Conda build snapshot | `conda-linux-64.lock.txt` |
| Python dependency snapshot | `requirements-lock.txt` |
| Full setup procedure | `docs/SETUP.md` |
| Environment snapshot | `docs/environment_snapshot_2026-08-23.txt` |
| Experiment configs | `configs/` |
| Experiment manifests | `experiments/` |
| Metrics | `results/metrics/` |
| Figures | `results/figures/` |
| Tables | `results/tables/` |

Each formal experiment is planned to record:

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

## Current experimental scale

### Verified

| Component | Current verified value |
|---|---:|
| LIBERO-Spatial tasks discovered | 10 |
| Task 0 fixed initial states | 50 |
| Agent-view observation | `128 × 128 × 3` |
| Eye-in-hand observation | `128 × 128 × 3` |
| Robot action dimensionality | 7 |
| CUDA GPU | RTX 5090 |
| CUDA tensor computation | PASS |
| MuJoCo RGB rendering | PASS |
| LIBERO simulator step | PASS |

### Planned formal evaluation

Initial formal evaluation planning uses multiple tasks, fixed benchmark states and repeated episodes per model-condition.

The exact final evaluation budget will be frozen after **G1 runtime benchmarking** rather than assumed in advance.

---

## Project roadmap

| Checkpoint | Milestone | Status |
|---|---|---|
| **01** | Environment + LIBERO simulator validation | ✅ Complete |
| **02** | Notebook 00 + full G0 integration | 🔄 Next |
| **03** | M0 baseline checkpoint reproduction | ⏳ Planned |
| **04** | Standard M1 baseline fine-tuning | ⏳ Planned |
| **05** | Robustness implementation | ⏳ Planned |
| **06** | Hyperparameter tuning | ⏳ Planned |
| **07** | Final independent training seeds | ⏳ Planned |
| **08** | E0–E3 evaluation + ablation | ⏳ Planned |
| **09** | Visualization + critical analysis | ⏳ Planned |
| **10** | Final consolidated report | ⏳ Planned |

---

## Current progress

```text
Environment Setup                 ██████████ 100%
CUDA / RTX 5090 Validation        ██████████ 100%
LeRobot Integration               ██████████ 100%
MuJoCo Integration                ██████████ 100%
LIBERO Integration                ██████████ 100%
LIBERO Simulator Validation       ██████████ 100%

Notebook 00                       ░░░░░░░░░░   0%
SmolVLA Integration               ░░░░░░░░░░   0%
M0 Baseline Reproduction          ░░░░░░░░░░   0%
M1 Baseline Training              ░░░░░░░░░░   0%
Robustness Training               ░░░░░░░░░░   0%
Hyperparameter Tuning             ░░░░░░░░░░   0%
Final Evaluation                  ░░░░░░░░░░   0%
Critical Analysis                 ░░░░░░░░░░   0%
```

---

## Key limitations

At the current stage:

- **Simulation domain:** experiments are based primarily on simulated manipulation rather than physical robot deployment.
- **Benchmark scope:** LIBERO captures controlled manipulation tasks and cannot represent the full complexity of real-world robotics.
- **Robustness scope:** perturbation validity must be carefully controlled so that evaluation does not change the intended task.
- **Compute:** VLA training and repeated rollout evaluation are substantially more computationally expensive than conventional supervised learning experiments.
- **Model scope:** the project investigates adaptation of an efficient existing VLA architecture rather than training a VLA foundation model from scratch.
- **Current status:** research-performance conclusions cannot yet be made because model training and comparative evaluation have not been completed.

These limitations will be revisited after final experiments.

---

## Expected contribution

The intended contribution is **not** a new large-scale VLA foundation model.

Instead, this study investigates whether relatively lightweight robustness-oriented adaptation can improve the reliability of an efficient existing VLA policy.

The intended research contribution is:

> **A controlled empirical analysis of robustness-oriented fine-tuning for efficient Vision-Language-Action robotic manipulation under linguistic and environmental distribution shifts.**

The study specifically distinguishes:

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

---

## Assignment deliverables

The final project will support the required Deep Learning assessment outputs:

### Assignment 1

- abstract,
- domain and background,
- research problem,
- aim and objectives,
- significance,
- deep-learning architecture survey,
- recent related work,
- model-selection justification,
- dataset / benchmark methodology,
- evaluation metrics,
- proposed methodology.

### Assignment 2

- preprocessing,
- baseline implementation,
- model adaptation,
- training,
- hyperparameter tuning,
- validation,
- final model evaluation,
- architecture visualization,
- performance visualization,
- critical analysis,
- comparison between baseline and final models.

Python implementation and generated outputs will be presented through Jupyter notebooks.

---

## License

Academic project provided for educational and research purposes.

Third-party software, pretrained models, benchmark assets and datasets remain subject to their respective licences and terms of use.

This repository does not redistribute LIBERO datasets, Hugging Face model weights or other externally licensed assets unless explicitly permitted by their respective licences.

---

## Citation

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

## Acknowledgements

- Hugging Face **LeRobot** project and contributors.
- **SmolVLA** authors and contributors.
- **LIBERO** benchmark authors and maintainers.
- **MuJoCo** by Google DeepMind.
- Asia Pacific University of Technology & Innovation (APU) course staff for academic guidance.

---

> **Current milestone:** Checkpoint 01 complete — reproducible environment, RTX 5090 compute, MuJoCo rendering and LIBERO-Spatial simulation validated.

> **Next:** Notebook 00 → SmolVLA integration → complete G0 → baseline reproduction.