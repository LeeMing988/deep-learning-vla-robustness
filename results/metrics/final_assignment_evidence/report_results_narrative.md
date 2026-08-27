# Assignment 2 Results Evidence Skeleton

## Frozen model-development decisions

- Optimizer selected in Notebook 03: T06, learning rate 0.0001, weight decay 0.01.
- Final training budget: 25,000 optimizer steps.
- Budget selection used E0 only; OOD results were not used for checkpoint selection.
- Final registry: 8 checkpoints, with M1 and M2-LE evaluated across seeds 42, 123 and 2026.
- No post-OOD retraining or model replacement was performed.

## Definitive / confirmatory findings — Notebook 07

- E0 ID: M1 51.3% vs M2-LE 44.7%; paired effect -6.7 pp; DIRECTIONAL_EVIDENCE_AGAINST.
- E1 held-out L3: M1 6.0% vs M2-LE 32.7%; paired effect 26.7 pp, 95% CI [14.7, 38.7] pp; STRONG_MULTI_SEED_SUPPORT.
- E2 seen photometric: M1 59.3% vs M2-LE 49.3%; paired effect -10.0 pp; DIRECTIONAL_EVIDENCE_AGAINST.
- E3 held-out Gaussian noise: M1 48.0% vs M2-LE 58.0%; paired effect 10.0 pp; DIRECTIONAL_SUPPORT.
- E4 L3 + seen photometric: M1 9.3% vs M2-LE 26.7%; paired effect 17.3 pp, 95% CI [2.0, 32.7] pp; STRONG_MULTI_SEED_SUPPORT.

## Supplementary / exploratory finding — Notebook 07b

- E5 L3 + held-out Gaussian noise: paired M2-LE−M1 effect 16.7 pp, 95% CI [4.7, 31.3] pp; 3/3 training seeds improved; EXPLORATORY_STRONG_SUPPORT.
- E5 was post-hoc supplementary evidence and was not part of Notebook 07's original Holm-corrected confirmatory family.

## Central interpretation

The strongest reproducible robustness benefit was linguistic generalization. The combined M2-LE intervention also retained a clear advantage under the predeclared L3+photometric condition and under the supplementary double-held-out L3+noise stress test. However, the intervention did not uniformly improve every visual condition, and a possible ID-performance trade-off remained statistically inconclusive.

## Boundaries

Do not claim:

- full LIBERO-Plus robustness;
- real-world robot robustness;
- camera-viewpoint, object-layout or robot-initial-state robustness;
- universal language understanding;
- that E5 was preregistered;
- that 25k is globally optimal.
