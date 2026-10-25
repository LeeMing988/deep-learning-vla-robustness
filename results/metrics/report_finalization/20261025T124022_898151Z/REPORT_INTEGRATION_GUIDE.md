# N11 report integration guide

This package is a read-only post-hoc analysis of the frozen submitted evidence. It does not change the experiments or establish additional training validity.

## Recommended edits (do not insert every figure)

1. Add F01 in Section 4.4: identify what was inherited, what was added, and what trained.

2. Use F02 instead of the mean-only budget figure when explaining why seed consistency matters. Include table 05 and disclose table 06.

3. Add F03 and a compact selection from table 07 in Section 5. Describe the losses as logged training batches; no validation-over-time curve was recorded here.

4. Replace the M3 mean-only bar chart with F05. Use F06 in the appendix or main text to explain replication. Retain the original N07 effect figure.

5. Use F04 only where the language ablation needs visual support; otherwise retain its table. Table 12 can remain in the appendix.

## Draft analytical observations to verify and adapt

The initial grid screens learning rate and weight decay at a short horizon. It does not independently establish the best full-run configuration. Later budget experiments and final checkpoint validation answer different questions.

The budget trajectories disagree by seed. Retaining 25k is a bounded resource decision under the recorded replacement criteria, not evidence that 50k can never help or that every original adequacy check passed.

The original language result improves relative success but leaves a low absolute completion rate. The language-only ablation supports the intervention's relevance within one seed; it does not prove a general internal mechanism.

In the supplied N10 evidence, the photometric benefit appears only in seed 42. The remaining seeds tie, so the average effect is weaker than the initial single-seed observation suggested. No favourable seed is selected or omitted.

Across all possible outcomes for the four unscored L3 attempts, the scheduled-sample difference ranges from -3.33 to -0.67 percentage points. Assigning success to all four gives the upper endpoint only. These bounds are not CIs or population-level evidence of harm.

## Interpretation safeguards

Keep N07, N07b and N10 results separate. Use fresh N10 M2-LE controls for M3. Differences across conditions can reflect different initial-state schedules, so do not claim that corruption improves performance.

Do not infer failure mechanisms from final screenshots alone. No new attention, trajectory, action-consistency or learning-stability measurement has been performed.

Do not add new p-values, change missing-value rules, select settings or reopen training on the basis of N11. All new task/seed summaries are descriptive report analyses.

Retain the assignment's word allocations. Replace repetition with analysis rather than adding a long technical appendix to the main discussion. Check institutional AI-assistance and attribution requirements.

## Known source limitations

- The N09 training summary's NOT_EVALUATED field is historical training-stage metadata; N10 supplies robot outcomes.

- The N10 summary retains a legacy one-seed limitation string, but its selected seeds and CSVs contain three seeds.

- The N10 all_planned_seeds_evaluated flag remains false because the complete scored matrix has four missing outcomes; execution is finished.

- The N05b original adequacy threshold did not pass unchanged. Its amendment remains a limitation, not an all-gates-pass claim.

- N11 republishes saved CIs and does not revalidate their statistical assumptions; small-seed/clustering limitations remain.

- Archive evidence does not contain every raw initial observation or model weight; this is an arithmetic/source audit, not full replay verification.

## Figure captions and placement

### F01_m3_architecture

The visual-token adapter is inserted after the inherited connector and before image scaling. Its residual branch initially equals zero. Frozen backbone weights still participate in the differentiable computation; original action-side components remain trainable. Diagram derived from the archived source and manifests.

Suggested placement: Section 4.4 — add to main report

Source: `src/m3_rva_training.py; M3 protocol/registry; N06 compute audit`

### F02_budget_by_seed

Each point uses 50 E0 episodes. The same seed trajectory supplies its 25k/35k/50k checkpoints. Higher pooled success at 50k did not satisfy the full replacement rule. The extended trajectories retained the schedule reaching its floor at 25k. See the separate repeatability-amendment table; this is not an optimality claim.

Suggested placement: Section 5.2 — replace or supplement the existing mean-only budget figure

Source: `results/metrics/final_budget_validation/per_seed_budget_success_summary.csv`

### F03_m3_training_losses

Unsmoothed logged batch losses from the three completed M3 runs. Logs contain selected updates, not every batch or epoch averages. No repeated validation trajectory is asserted. A lower training objective does not establish better robot success or OOD robustness.

Suggested placement: Section 5 — add one training-behaviour figure

Source: `experiments/m3_rva_extension/v1/m3rva_seed*/train_metrics.csv`

### F04_language_ablation

Every point contains 50 episodes. Language-only and combined augmentation improve observed language-shift success relative to M1 in this seed. The standard-condition trade-off remains visible. This does not establish the mechanism across all seeds, and differences between conditions can also reflect their evaluation schedules.

Suggested placement: Section 6.1 — useful mechanism-analysis figure, or appendix if space is tight

Source: `results/metrics/definitive_robustness_evaluation/closed_loop_episode_results.csv`

### F05_m3_effects_and_uncertainty

Effects use identical complete pairs for both policies. Horizontal bars reproduce N10's saved 95% bootstrap intervals for complete conditions; they do not remove small-seed or clustering limitations. The L3 diamond is a point estimate without inferential interval. Missing-outcome bounds are reported separately, not substituted for a CI.

Suggested placement: Section 6.3 — replace the current mean-only M3 bar chart

Source: `results/metrics/m3_rva_evaluation/v2_completion_v1/m3_vs_m2le_effects.csv`

### F06_m3_seed_effects

All preselected training seeds are shown. L3 uses 49, 48 and 49 valid pairs; other cells use 50. Seed-specific differences show directional repeatability, not a test of equal model stability. In the supplied results, the photometric gain is confined to seed 42.

Suggested placement: Section 6.3 or appendix — supports the replication discussion

Source: `results/metrics/m3_rva_evaluation/v2_completion_v1/paired_episode_results.csv`
