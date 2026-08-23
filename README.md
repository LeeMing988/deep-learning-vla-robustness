# Robustness-Enhanced Vision-Language-Action Learning

## Generalizable Robotic Manipulation under Linguistic and Environmental Distribution Shifts

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![LeRobot](https://img.shields.io/badge/LeRobot-0.6.1-yellow)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8.1-red)
![LIBERO](https://img.shields.io/badge/LIBERO-Spatial-purple)
![Status](https://img.shields.io/badge/Research-In%20Progress-blueviolet)

> Deep Learning research project investigating robustness and out-of-distribution generalization in Vision-Language-Action policies for robotic manipulation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Motivation](#research-motivation)
- [Research Aim](#research-aim)
- [Research Questions](#research-questions)
- [Research Hypotheses](#research-hypotheses)
- [Research Framework](#research-framework)
- [Model Design](#model-design)
- [Benchmark and Dataset](#benchmark-and-dataset)
- [Perturbation Framework](#perturbation-framework)
- [Evaluation Strategy](#evaluation-strategy)
- [Primary Metrics](#primary-metrics)
- [Failure Analysis](#failure-analysis)
- [Experimental Environment](#experimental-environment)
- [Validated GPU Pipeline](#validated-gpu-pipeline)
- [LIBERO Simulation Validation](#libero-simulation-validation)
- [Technical Compatibility Gate](#technical-compatibility-gate)
- [Repository Structure](#repository-structure)
- [Notebook Plan](#notebook-plan)
- [Reproducibility](#reproducibility)
- [Experiment Tracking](#experiment-tracking)
- [Project Roadmap](#project-roadmap)
- [Current Status](#current-status)
- [Disclaimer](#disclaimer)

---

# Project Overview

Vision-Language-Action (VLA) models combine:

- visual perception,
- natural-language understanding, and
- robotic control

within a unified policy capable of generating robot actions from multimodal observations.

Recent VLA systems demonstrate strong performance on standard robotic manipulation benchmarks. However, high benchmark performance does not necessarily imply robustness when the deployment environment differs from the model's training distribution.

Changes in:

- natural-language instructions,
- object arrangement,
- camera viewpoint,
- lighting,
- background appearance,
- robot initial state,
- sensor observations,

may substantially reduce manipulation performance.

This project investigates whether **controlled robustness-oriented fine-tuning** can improve the generalization of an efficient VLA policy under linguistic and environmental distribution shifts while preserving standard in-distribution performance.

The primary VLA architecture used in this research is **SmolVLA**, integrated through the **LeRobot** framework and evaluated using the **LIBERO robotic manipulation benchmark**.

---

# Research Motivation

A robot deployed outside a controlled benchmark environment cannot assume that every observation will match its training data.

For example, a robot may receive:

```text
Pick up the black bowl and place it on the plate.
```

but later encounter a semantically equivalent instruction such as:

```text
Move the black bowl onto the plate.
```

Similarly, the task may remain unchanged while:

- the object appears at a different location,
- the camera is moved,
- lighting changes,
- the background changes,
- sensor observations become noisy.

A policy that succeeds only under the original benchmark configuration may therefore have limited practical robustness.

The central research problem is:

> **Can robustness-oriented fine-tuning improve VLA manipulation performance under controlled distribution shifts without significantly degrading standard benchmark performance?**

---

# Research Aim

To investigate whether robustness-oriented fine-tuning improves the generalization of efficient Vision-Language-Action policies for robotic manipulation under controlled linguistic and environmental distribution shifts.

---

# Research Questions

## RQ1 — Overall Robustness

To what extent does robustness-oriented fine-tuning improve both seen-perturbation robustness and generalization to unseen perturbation categories in VLA robotic manipulation?

## RQ2 — Linguistic Robustness

At what level of linguistic variation does SmolVLA manipulation performance begin to degrade, and can language-aware fine-tuning reduce this degradation?

## RQ3 — Environmental Sensitivity

Which environmental perturbation dimensions most strongly reduce VLA manipulation performance?

## RQ4 — Held-Out Generalization

Does robustness training on selected perturbation categories transfer to perturbation categories absent during training?

## RQ5 — Robustness–Performance Trade-Off

Does improved out-of-distribution robustness reduce standard in-distribution manipulation performance?

---

# Research Hypotheses

### H1 — Linguistic Robustness

Language-augmented fine-tuning will reduce performance degradation under unseen but semantically equivalent task instructions.

A null result may indicate that the pretrained language/vision representation already provides substantial linguistic invariance.

### H2 — Seen Perturbation Robustness

Robustness-oriented training will improve manipulation success under perturbation categories observed during training.

### H3 — Held-Out Generalization

Robustness training will transfer to at least some environmental perturbation categories absent during training.

### H4 — Generalization Gap

The combined robustness model will exhibit a smaller in-distribution to out-of-distribution generalization gap than the standard fine-tuning baseline.

### H5 — In-Distribution Trade-Off

Robustness improvements may preserve or slightly reduce standard benchmark performance in exchange for improved OOD generalization.

---

# Research Framework

The research follows the general experimental structure below:

```text
                     ┌─────────────────────┐
                     │     SmolVLA VLA     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │    LIBERO-Spatial   │
                     │  Manipulation Tasks │
                     └──────────┬──────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
      Standard Fine-Tuning              Robustness Fine-Tuning
              M1                         M2-L / M2-E / M2-LE
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
                    Controlled Evaluation
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
             E0                E1               E2/E3
       In-Distribution     Linguistic         Environmental
                            Shift               Shift
                                │
                                ▼
                   Success / Robustness Analysis
```

---

# Model Design

The study separates checkpoint reproducibility, standard fine-tuning, and robustness-oriented fine-tuning.

| Model | Description |
|---|---|
| **M0** | Released LIBERO-fine-tuned SmolVLA checkpoint used for reproducibility and evaluation auditing |
| **M1** | Standard SmolVLA fine-tuning baseline |
| **M2-L** | SmolVLA with linguistic robustness-oriented fine-tuning |
| **M2-E** | SmolVLA with environmental robustness-oriented fine-tuning |
| **M2-LE** | Combined linguistic + environmental robustness model |

The primary final comparison is planned as:

```text
M1
Standard Fine-Tuning
        VS
M2-LE
Combined Robustness Fine-Tuning
```

Supporting ablation analysis will compare:

```text
M1
│
├── M2-L
├── M2-E
└── M2-LE
```

---

# Benchmark and Dataset

## LIBERO

LIBERO is used as the primary robotic manipulation benchmark.

The validated installation currently exposes the following suites:

```text
libero_10
libero_100
libero_90
libero_goal
libero_object
libero_spatial
```

---

## LIBERO-Spatial

**LIBERO-Spatial** is the primary benchmark suite selected for the initial study.

The suite contains **10 manipulation tasks** focused on spatial relationships.

The tasks maintain a closely related manipulation objective while varying the initial spatial location of the target object.

Examples include:

```text
Pick up the black bowl between the plate and the ramekin
and place it on the plate.
```

```text
Pick up the black bowl next to the ramekin
and place it on the plate.
```

```text
Pick up the black bowl from table center
and place it on the plate.
```

```text
Pick up the black bowl on the stove
and place it on the plate.
```

This makes LIBERO-Spatial particularly relevant for investigating robustness to spatial variation.

---

## LIBERO-Plus

LIBERO-Plus is planned as a first-class robustness evaluation benchmark.

Perturbation dimensions to be investigated include:

- object layout,
- camera viewpoint,
- robot initial state,
- language,
- lighting,
- background appearance,
- sensor noise.

A dedicated compatibility and methodology gate will determine whether LIBERO-Plus is used:

1. fully for robustness training and evaluation,
2. partially for evaluation only, or
3. selectively alongside custom controlled perturbations.

The final protocol will be frozen only after this audit.

---

# Perturbation Framework

The methodology explicitly separates:

```text
Seen robustness
```

from:

```text
Held-out generalization
```

This prevents evaluation from measuring only memorization of training augmentations.

A provisional design is:

| Category | Usage |
|---|---|
| Language variation | Training / seen robustness |
| Lighting | Training / seen robustness |
| Background appearance | Training / seen robustness |
| Camera viewpoint | Held-out evaluation |
| Robot initial state | Held-out evaluation |
| Object layout | Held-out evaluation |
| Sensor noise | Held-out evaluation |

The exact split will be frozen after the LIBERO-Plus methodology audit.

---

# Linguistic Perturbation Levels

Language variation is planned across four controlled difficulty levels.

| Level | Description |
|---|---|
| **L0** | Original benchmark instruction |
| **L1** | Lexical / synonym substitution |
| **L2** | Syntactic restructuring |
| **L3** | Compositional rewording |

Example:

### L0

```text
Pick up the black bowl and place it on the plate.
```

### L1

```text
Grab the black bowl and place it on the plate.
```

### L2

```text
Place the black bowl onto the plate after picking it up.
```

### L3

```text
Move the black bowl from its current position so that it ends up on the plate.
```

All generated instruction variants must remain semantically equivalent to the original manipulation objective.

---

# Evaluation Strategy

The planned evaluation framework separates standard benchmark performance from robustness and generalization.

| Condition | Description |
|---|---|
| **E0** | Standard in-distribution LIBERO evaluation |
| **E1** | Linguistic distribution shift |
| **E2** | Seen environmental perturbation categories |
| **E3** | Held-out environmental perturbation categories |
| **E4** | Combined perturbations if computational budget permits |

The same task definitions, initial states, and perturbation conditions will be paired across models whenever possible.

---

# Primary Metrics

## Task Success Rate

The primary metric is:

```text
Task Success Rate
```

defined as:

```text
Successful Episodes
────────────────────
 Total Episodes
```

---

## Generalization Gap

The generalization gap is defined as:

```text
G = SR_ID - SR_OOD
```

where:

- `SR_ID` = in-distribution success rate
- `SR_OOD` = out-of-distribution success rate

A smaller value indicates better robustness.

---

## Secondary Metrics

Additional measurements may include:

- per-task success rate,
- per-perturbation success rate,
- episode length,
- completion steps,
- inference latency,
- simulator latency,
- GPU memory utilization,
- validation loss,
- training loss,
- failure-category frequency.

---

# Statistical Analysis

Two sources of uncertainty will be treated separately.

## Evaluation Uncertainty

Evaluation uncertainty for an individual trained checkpoint may be estimated using bootstrap confidence intervals over evaluation episodes.

## Training Uncertainty

Training stochasticity will be measured using independent model-training seeds.

Planned final seeds:

```text
42
123
2026
```

Headline final results are intended to report:

```text
Mean ± Standard Deviation
```

across independently trained models where computationally feasible.

A bootstrap over episodes will not be presented as a substitute for independent training runs.

---

# Failure Analysis

Failed manipulation episodes will be categorized using a structured failure taxonomy.

| Code | Failure Category |
|---|---|
| **F01** | Object localization failure |
| **F02** | Incorrect target selection |
| **F03** | Failed grasp |
| **F04** | Object slip |
| **F05** | Incorrect placement |
| **F06** | Language misunderstanding |
| **F07** | Trajectory / control failure |
| **F08** | Episode timeout |
| **F09** | Collision / environmental interaction failure |
| **F10** | Other |

This enables analysis beyond aggregate success rate.

---

# Experimental Environment

## Validated Software Stack

Checkpoint 01 was validated using:

| Component | Version / Configuration |
|---|---|
| Host OS | Windows 11 |
| Linux Environment | WSL2 Ubuntu 24.04 |
| Python | 3.12.14 |
| Conda Environment | `vla-libero` |
| PyTorch | 2.11.0+cu128 |
| CUDA Runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU Compute Capability | 12.0 |
| LeRobot | 0.6.1 |
| MuJoCo | 3.8.1 |
| LIBERO | `hf_libero 0.1.4` |
| EGL Helper | `hf_egl_probe 1.0.2` |
| Rendering Backend | WSLg + Mesa D3D12 + GLFW |

---

## LeRobot Revision

The LeRobot dependency is pinned to:

```text
Tag:
v0.6.1

Commit:
7e241bd630a3719a56157a497ce5d08f244784f1
```

---

## egl_probe Revision

The compatibility-patched `egl_probe` dependency is based on:

```text
Version:
1.0.1

Git commit:
3ddf90db69264de2c621af567bfc557849126cff
```

The patch and installation procedure are documented separately to ensure reproducibility with modern CMake versions.

---

# Validated GPU Pipeline

Two GPU-dependent execution paths have been validated.

## Deep Learning Compute

```text
PyTorch
    │
    ▼
CUDA 12.8
    │
    ▼
NVIDIA GeForce RTX 5090
    │
    ▼
GPU Tensor Computation
    │
    ▼
PASS
```

Validated output:

```text
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
```

A real CUDA matrix multiplication test completed successfully.

---

## MuJoCo Rendering

The validated WSL graphics path is:

```text
MuJoCo
   │
   ▼
OpenGL / GLFW
   │
   ▼
Mesa D3D12
   │
   ▼
WSLg
   │
   ▼
NVIDIA GeForce RTX 5090
   │
   ▼
RGB Frame
   │
   ▼
PASS
```

Validated MuJoCo test output:

```text
Image shape: (256, 256, 3)
Image dtype: uint8
Min pixel: 0
Max pixel: 255
Mean pixel: 33.8369
Non-zero pixels: 110960

MuJoCo RENDER TEST: PASS
```

---

# LIBERO Simulation Validation

LIBERO-Spatial Task 0 has successfully completed the initial technical compatibility test.

Validated functionality includes:

- benchmark discovery,
- task metadata loading,
- BDDL task definition resolution,
- fixed benchmark initial states,
- MuJoCo environment creation,
- environment reset,
- fixed initial-state loading,
- robot state observations,
- RGB observations,
- action interface,
- simulator stepping,
- clean environment shutdown.

---

## Benchmark Initial States

For LIBERO-Spatial Task 0:

```text
Initial state representation:
numpy.ndarray

Available benchmark initial states:
50
```

---

## Camera Validation

### Agent View

```text
Observation:
agentview_image

Shape:
(128, 128, 3)

Type:
uint8

Minimum pixel:
6

Maximum pixel:
238

Mean:
119.4102

Non-zero pixels:
49152

Status:
PASS
```

### Robot Eye-in-Hand View

```text
Observation:
robot0_eye_in_hand_image

Shape:
(128, 128, 3)

Type:
uint8

Minimum pixel:
3

Maximum pixel:
231

Mean:
126.3576

Non-zero pixels:
49152

Status:
PASS
```

Both camera observations were confirmed to contain valid non-black rendered images.

---

## Robot State Observations

Validated observations include:

```text
robot0_eef_pos
robot0_eef_quat
robot0_gripper_qpos
robot0_gripper_qvel
robot0_joint_pos
robot0_joint_vel
robot0_proprio-state
object-state
agentview_image
robot0_eye_in_hand_image
```

Additional object-specific pose observations are also provided by LIBERO.

---

## Action Interface

The environment successfully accepted a seven-dimensional control action.

```text
Action:
[0. 0. 0. 0. 0. 0. 0.]

Reward:
0.0

Done:
False

Environment step:
PASS
```

The zero action is used only as a technical interface validation and is not expected to solve the manipulation task.

---

# Technical Compatibility Gate

## G0 — Technical Compatibility

Current status:

| Test | Status |
|---|---|
| WSL2 / Ubuntu | ✅ PASS |
| Conda / Python | ✅ PASS |
| RTX 5090 visible inside WSL | ✅ PASS |
| PyTorch CUDA | ✅ PASS |
| Blackwell GPU computation | ✅ PASS |
| LeRobot import | ✅ PASS |
| Dependency integrity | ✅ PASS |
| MuJoCo import | ✅ PASS |
| GPU OpenGL rendering | ✅ PASS |
| MuJoCo RGB rendering | ✅ PASS |
| LIBERO initialization | ✅ PASS |
| LIBERO-Spatial discovery | ✅ PASS |
| BDDL task loading | ✅ PASS |
| Benchmark initial states | ✅ PASS |
| Agent-view RGB observation | ✅ PASS |
| Eye-in-hand RGB observation | ✅ PASS |
| 7-D action step | ✅ PASS |
| SmolVLA checkpoint loading | ⏳ NEXT |
| SmolVLA preprocessing | ⏳ NEXT |
| SmolVLA inference | ⏳ NEXT |
| Model action → LIBERO | ⏳ NEXT |
| Complete policy episode | ⏳ PENDING |

---

# Repository Structure

```text
deep-learning-vla-robustness/
│
├── configs/
│   └── Experiment, training and evaluation configurations
│
├── docs/
│   ├── SETUP.md
│   └── Environment and research documentation
│
├── experiments/
│   └── Experiment manifests and run metadata
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
│   └── Environment and reproducibility utilities
│
├── src/
│   └── Reusable experimental implementation
│
├── .gitignore
├── environment.yml
├── conda-linux-64.lock.txt
├── requirements-lock.txt
└── README.md
```

Some directories and notebooks represent the planned research structure and will be populated progressively as experiments are completed.

---

# Notebook Plan

## 00 — Environment and Dataset

Purpose:

- verify the computational environment,
- inspect software versions,
- verify CUDA,
- discover LIBERO benchmark suites,
- inspect LIBERO-Spatial tasks,
- initialize simulator environments,
- inspect observations,
- visualize camera observations,
- validate action interfaces.

---

## 01 — Baseline Reproduction

Purpose:

- load the released LIBERO-fine-tuned SmolVLA checkpoint,
- freeze checkpoint revision,
- validate preprocessing,
- perform policy inference,
- reproduce baseline evaluation,
- establish M0,
- implement standard M1 baseline evaluation.

---

## 02 — Robustness Training

Purpose:

- implement standard fine-tuning,
- implement linguistic augmentation,
- implement environmental robustness augmentation,
- implement combined robustness training,
- compare M1 / M2-L / M2-E / M2-LE.

---

## 03 — Tuning and Final Training

Purpose:

- controlled hyperparameter tuning,
- learning-rate selection,
- regularization analysis,
- augmentation probability selection,
- final independent training runs,
- final checkpoint generation.

---

## 04 — Evaluation and Critical Analysis

Purpose:

- E0 in-distribution evaluation,
- E1 language robustness,
- E2 seen environmental robustness,
- E3 held-out generalization,
- ablation studies,
- statistical analysis,
- visualization,
- failure analysis,
- critical interpretation of results.

---

# Hyperparameter Strategy

The project will use staged tuning rather than an exhaustive Cartesian grid.

Candidate values include:

### Learning Rate

```text
1e-5
3e-5
1e-4
```

### Weight Decay

```text
0.01
0.05
0.10
```

### Language Augmentation Probability

```text
0.25
0.50
0.75
```

Approximately **8–12 meaningful experimental configurations** are targeted before final model training.

Exploratory tuning may use one random seed.

Headline final experiments are planned to use multiple independent seeds.

---

# Reproducibility

Reproducibility is treated as a first-class component of the research methodology.

The environment is represented through multiple complementary artifacts.

| File | Purpose |
|---|---|
| `environment.yml` | Human-readable Conda environment definition |
| `conda-linux-64.lock.txt` | Exact Conda package/build snapshot |
| `requirements-lock.txt` | Exact Python dependency snapshot |
| `docs/SETUP.md` | Detailed environment setup instructions |
| `docs/environment_snapshot_2026-08-23.txt` | Validated system environment snapshot |

Large datasets, pretrained model weights, experiment checkpoints, video recordings, and external caches are intentionally excluded from source control.

---

# Environment Reproduction Strategy

The project separates:

```text
Host prerequisites
```

from:

```text
Research environment
```

and:

```text
Experiment configuration
```

This avoids embedding operating-system installation procedures directly inside research notebooks.

The intended structure is:

```text
Windows / WSL prerequisites
            │
            ▼
       docs/SETUP.md
            │
            ▼
      environment.yml
            │
            ▼
   requirements-lock.txt
            │
            ▼
      pinned LeRobot
            │
            ▼
   environment checker
            │
            ▼
   experiment notebooks
```

---

# Experiment Tracking

Each formal experiment is planned to generate a structured run manifest containing:

```text
experiment_id
git_commit
model_checkpoint_revision
dataset_revision
benchmark_revision
seed
hyperparameters
training_steps
perturbation_configuration
evaluation_configuration
metrics
checkpoint_path
```

Example:

```json
{
  "experiment_id": "M2LE_seed42",
  "model": "smolvla",
  "seed": 42,
  "training_mode": "linguistic_environmental_robustness",
  "evaluation_conditions": [
    "E0",
    "E1",
    "E2",
    "E3"
  ]
}
```

This enables experiments to be mapped back to:

- source-code revision,
- environment,
- training configuration,
- evaluation configuration,
- generated checkpoints,
- reported results.

---

# Project Roadmap

## Checkpoint 01 — Environment + LIBERO

**Status: ✅ COMPLETE**

Completed:

- WSL2 setup
- Conda environment
- CUDA verification
- RTX 5090 validation
- LeRobot installation
- MuJoCo installation
- GPU rendering
- LIBERO setup
- LIBERO-Spatial task discovery
- simulator initialization
- RGB observation validation
- action interface validation

---

## Checkpoint 02 — Notebook 00 + G0 Completion

**Status: ⏳ NEXT**

Planned:

- create `00_environment_and_dataset.ipynb`,
- reproduce environment validation inside notebook,
- visualize robot camera observations,
- load official SmolVLA checkpoint,
- verify preprocessing,
- execute one model inference,
- execute predicted action in LIBERO,
- complete a policy-controlled episode.

---

## Checkpoint 03 — Baseline Reproduction

Planned:

- reproduce official SmolVLA LIBERO performance,
- establish M0 evaluation protocol,
- establish M1 standard fine-tuning baseline,
- freeze Evaluation Protocol v1.0.

---

## Checkpoint 04 — Robustness Implementation

Planned:

- linguistic robustness pipeline,
- environmental robustness pipeline,
- combined training strategy,
- augmentation validity checks.

---

## Checkpoint 05 — Hyperparameter Tuning

Planned:

- controlled tuning runs,
- validation analysis,
- final hyperparameter selection.

---

## Checkpoint 06 — Final Training

Planned:

- independent final seeds,
- final M1 baseline checkpoints,
- final robustness model checkpoints.

---

## Checkpoint 07 — Final Evaluation

Planned:

- E0 evaluation,
- E1 evaluation,
- E2 evaluation,
- E3 held-out evaluation,
- ablation study,
- failure analysis,
- uncertainty analysis.

---

## Checkpoint 08 — Final Analysis

Planned:

- visualization,
- statistical interpretation,
- robustness trade-off analysis,
- critical discussion,
- final research conclusions.

---

# Current Status

```text
Environment Setup                  ██████████ 100%
CUDA / RTX 5090 Validation         ██████████ 100%
LeRobot Integration                ██████████ 100%
MuJoCo Integration                 ██████████ 100%
LIBERO Integration                 ██████████ 100%
LIBERO Simulator Validation        ██████████ 100%

Notebook 00                        ░░░░░░░░░░   0%
SmolVLA Integration                ░░░░░░░░░░   0%
Baseline Reproduction              ░░░░░░░░░░   0%
Robustness Training                ░░░░░░░░░░   0%
Hyperparameter Tuning              ░░░░░░░░░░   0%
Final Evaluation                   ░░░░░░░░░░   0%
Critical Analysis                  ░░░░░░░░░░   0%
```

---

# Current Research Milestone

## Checkpoint 01

### Reproducible VLA Environment and LIBERO Simulation Validation

The research environment can currently:

1. execute CUDA workloads on the NVIDIA RTX 5090,
2. load PyTorch with CUDA acceleration,
3. load the pinned LeRobot implementation,
4. initialize MuJoCo,
5. render valid RGB simulation observations,
6. discover all LIBERO benchmark suites,
7. instantiate LIBERO-Spatial tasks,
8. load benchmark-defined initial states,
9. return robot proprioceptive observations,
10. return agent-view camera observations,
11. return eye-in-hand camera observations,
12. accept a valid seven-dimensional robot control action,
13. advance the MuJoCo simulation.

The next technical milestone is integration of the released LIBERO-fine-tuned **SmolVLA** checkpoint.

---

# Research Principles

The experimental design follows several controls intended to improve validity and reproducibility.

### Comparable Evaluation

Baseline and robustness models will be evaluated using equivalent task and perturbation conditions.

### Seen vs Unseen Perturbations

Training perturbations will be explicitly separated from held-out perturbation categories.

### Action Validity

Physical object states will not be arbitrarily transformed in recorded demonstrations while retaining actions that are no longer valid for the modified state.

### Independent Sources of Uncertainty

Evaluation uncertainty and training stochasticity will be analysed separately.

### Reproducible Dependencies

Important external dependencies are pinned using explicit versions and Git revisions.

### Minimal Notebook Hidden Logic

Research notebooks will expose substantive preprocessing, training, evaluation, and analysis operations rather than hiding the complete experiment behind opaque utility functions.

---

# Expected Contribution

The intended contribution of this project is not a new large-scale VLA foundation model.

Instead, the project investigates whether relatively lightweight robustness-oriented adaptation can improve the reliability of an efficient existing VLA policy.

The expected contribution is therefore:

> **A controlled empirical analysis of robustness-oriented fine-tuning for efficient VLA robotic manipulation under linguistic and environmental distribution shifts.**

Particular emphasis is placed on distinguishing:

```text
standard benchmark performance
```

from:

```text
seen perturbation robustness
```

and:

```text
generalization to unseen perturbation categories
```

---

# Technology Stack

### Machine Learning

- PyTorch
- Transformers
- Hugging Face
- LeRobot
- SmolVLA

### Robotics

- LIBERO
- LIBERO-Spatial
- LIBERO-Plus
- MuJoCo
- Robosuite
- Robomimic

### Experimentation

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter
- TensorBoard
- Weights & Biases

### Infrastructure

- Windows 11
- WSL2
- Ubuntu 24.04
- Conda / Miniforge
- NVIDIA CUDA
- NVIDIA GeForce RTX 5090

---

# Disclaimer

This repository is an academic Deep Learning research project under active development.

Experimental protocols, model configurations, perturbation definitions, hyperparameters, evaluation conditions, and conclusions may change until the methodology and final evaluation protocol are formally frozen.

Results should therefore not be interpreted as final research conclusions until the corresponding experiment checkpoint has been completed.

---

## Author

**Yeoh Lee Ming**

Deep Learning Research Project  
Vision-Language-Action Learning · Robotics · Embodied AI

---

## Project Status

> **Checkpoint 01 Complete — Reproducible environment and LIBERO simulator validated.**

**Next:** Notebook 00 and SmolVLA baseline integration.