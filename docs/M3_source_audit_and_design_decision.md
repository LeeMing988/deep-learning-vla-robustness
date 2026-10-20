# M3 source audit and candidate design decision

## Status

- Static source review: complete for the supplied five-file snapshot.
- Candidate implementation: supplied in `09_custom_robust_visual_adapter.ipynb`, Stage A.
- Local validation of the delivered notebook: syntax/schema and 21 CPU assertions passed using a small stand-in network, not a real SmolVLA checkpoint.
- Target RTX 5090 / LeRobot runtime validation: pending execution by the user.
- Full 25k training: not started by this deliverable.
- Scientific robustness improvement: not evaluated.

This document is a design record, not a claim that M3 has been trained or validated experimentally.

## 1. Recorded source-control state

The uploaded `smolvla_source_audit.txt` records:

- Research HEAD: `ae20ee6dc4e66329c576e9ecc0c41ce062e9daf0`.
- Research branch: `experiment/m3-robust-visual-adapter`, tracking its origin counterpart; no changes shown in the recorded status.
- LeRobot HEAD: `7e241bd630a3719a56157a497ce5d08f244784f1`.
- LeRobot is a clean detached checkout in that recorded snapshot.

These are snapshot facts, not a guarantee of a later live filesystem state. The runtime notebook rechecks them as appropriate without changing branches or updating LeRobot.

## 2. What the supplied source supports

### Visual path and insertion point

`SmolVLMWithExpertModel.embed_image` applies the vision model and then `get_vlm_model().connector` before returning image representations (audit lines 1314–1327).

`VLAFlowMatching.embed_prefix` calls `embed_image`, multiplies the image embedding by `sqrt(hidden_dim)`, preserves camera masks, and subsequently concatenates image, language and projected state embeddings (lines 760–852).

The selected candidate point is **after connector output and before the existing image-token scaling**. The downstream code uses a token sequence; exact batch/token/channel numbers and dtype must be observed from the live checkpoint. Do not assume a spatial feature grid or reshape tokens into a CNN grid without proving that mapping.

The same `embed_prefix` is called in training (lines 907–925) and action inference (lines 948–964), making this a shared candidate insertion point. The source alone does not prove runtime output parity or absence of integration errors.

### Freezing and autograd

`train_expert_only=True` freezes all parameters of `self.vlm`; `train()` keeps that VLM in evaluation mode (lines 1264–1312). A new adapter placed beneath that subtree could be frozen by the same rule. Our candidate registers adapter weights at `policy.model.m3_visual_adapter`, outside the frozen VLM, while inserting its output after the connector through one module-local forward hook.

Frozen weights do not mean all intermediate computations can be detached. A trainable adapter before frozen text-transformer layers still needs gradients through those layers. No `no_grad()` or `detach()` should sever this training path. This may increase activation memory and compute; a small parameter count is not sufficient to predict total training cost.

### Flow matching and N10

The implementation accepts explicit `noise` and `time` (lines 485–509 and 897–905). It computes:

`x_t = t * noise + (1 - t) * actions`

`u_t = noise - actions`

`v_t = action_out_proj(suffix_out)`

and elementwise `MSE(u_t, v_t)` (lines 907–932).

The public training forward returns the reduced loss and statistics, not signed `v_t`. The proposed consistency extension must access the raw prediction; equal or similar MSE values do not establish equal predicted actions.

The policy crops real action dimensions and accounts for `action_is_pad` when reducing its training loss (lines 509–540). A consistency loss must preserve that valid-timestep/real-action-dimension treatment.

### Defaults are not experiment history

The class defaults include weight decay `1e-10`, warmup `1000`, and decay `30000` (lines 116–125). They are not evidence that those values were used in N06's custom loop. Read the saved experiment handoff/manifests and actual training code. Likewise, the preprocessing comment mentions 224 while the configured resize can differ; the live saved configuration is authoritative for the runtime probe.

## 3. Proposed M3-RVA candidate

For visual tokens `z` shaped `[B, N, D]`:

`RVA(z) = z + alpha * Up(GELU(Down(LayerNorm(z))))`

- Bottleneck rank: candidate `r=64`, not a tuned or optimal value.
- Shared adapter across cameras/tokens.
- LayerNorm, down/up linear projections with biases, GELU, and a learned scalar scale.
- No dropout, spatial convolution, attention change, extra sensors, or extra language inputs in this candidate.
- Initialize up-projection weights/bias to zero and scale to nonzero `0.1`.
- Added parameter count: `2*D*r + 3*D + r + 1`.
- Obtain `D`, token counts, and actual parameter overhead at runtime.

The initial output is identical to the input. The first backward pass should update the up-projection; earlier adapter layers may have zero gradients until that projection becomes nonzero. Initializing both the projection and scale at zero would create a dead residual branch. Runtime checks explicitly distinguish those cases.

This is an adaptation of established residual-adapter ideas. A task-specific implementation/evaluation may be a contribution; do not claim invention of adapters or guaranteed robustness.

## 4. Scope of delivered Notebook 09 Stage A

1. Verify environment, source pin and checkpoint metadata.
2. Select M2-LE seed 42 as a local **engineering fixture only**.
3. Hash original checkpoint files.
4. Inspect real connector shapes and original fixed-input outputs using synthetic data.
5. Install candidate; verify original trainability flags and identity in training/inference.
6. Run two **adapter-only synthetic diagnostic updates** through the full network.
7. Measure gradient flow and peak GPU allocated/reserved memory at batch 1.
8. Save a small diagnostic adapter sidecar; reconstruct a new full policy instance with the hook reinstalled.
9. Compare fixed-noise outputs and verify original files/source remain unchanged.
10. Write a summary distinguishing engineering PASS from scientific NOT_EVALUATED.

The notebook does not import datasets or the simulator. It does not alter the existing model registry, original checkpoint weights, or upstream source. Diagnostic bundles are visibly marked as unsuitable for scientific evaluation.

A sidecar is sufficient here only because inherited weights are not updated. Final M3 training updates inherited expert/state components too; those weights and the processors must be saved in a complete final bundle. A normal SmolVLA loader does not automatically reconstruct custom hook behavior.

## 5. Remaining gates before full training

- Inspect and reuse the actual N06 training/augmentation path.
- Verify one real dataset batch, camera range, normalization, language mapping and action masks.
- Verify common pinned base initialization. Do not start final M3 from a 25k M2-LE checkpoint and claim an equal 25k training budget.
- Measure the intended effective batch-size memory and compute; do not silently reduce the batch or change training scope.
- Validate a complete trained-policy checkpoint and processors in a fresh kernel/process using an explicit M3 loader.
- Record adapter hyperparameters, tuning budget, seeds, primary contrasts and extension test protocol before new scientific runs.

The notebook uses the old M2-LE checkpoint only to test that adding an initially identity adapter preserves its interface. These synthetic updates are discarded as research evidence.

## 6. Two scientific experiments, with necessary controls

### N09: architectural contribution

Compare original SmolVLA under M2-LE augmentation against adapter-equipped SmolVLA under the same augmentation, same initialization, data split and training regimen. Record both success and cost. Existing outcomes are retrospective extension comparisons, not newly untouched evidence.

### N10: consistency-objective contribution

For paired views A and B with the same physical action target:

`L = 0.5 * (L_FM_A + L_FM_B) + lambda * L_consistency`

Use shared flow time and noise. Compute consistency on raw flow velocities with valid-action masks. Compare against an identical two-view supervised control with `lambda=0`; a one-view control does not isolate the new term. A zero coefficient must preserve the two-view code path and sampling budget.

This introduces a control condition within the second scientific question; it is not a third research hypothesis. Evaluate whether the complete combination helps and whether the consistency term adds value independently of added views/compute.

For a visual mechanism study, identical language across the paired views is a clean first design. Adding simultaneous paraphrases is a separate choice and must not be silently mixed into a claim about visual consistency alone.

### New holdout

E6 blur was a candidate in discussion, not a finalized protocol. Specify the actual corruption, severity, image resolution/domain, valid tasks, reset-state identities, model RNG seeds, inference settings, comparisons and uncertainty procedure before using it. Existing original conditions cannot be relabelled as previously unseen tests. Environment reset matching alone does not establish paired policy-noise draws; record/reset policy sampling RNG separately where needed.

## 7. What to return after running

Upload the executed Notebook 09 or its printed `m3_preflight_summary.json` plus the gradient/memory CSV. Numeric token dimensions, real checkpoint shapes, gradient checks, memory and reload errors—not guessed defaults—will determine whether the candidate progresses to real-data qualification.

## Reference boundaries

Source-derived facts above refer to the supplied local source snapshot and its displayed line numbers.
General framework references:

- PyTorch autograd mechanics: https://docs.pytorch.org/docs/stable/notes/autograd.html
- PyTorch module hooks and state dictionaries: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html

The delivered local checks ran with PyTorch 2.10.0+cpu and a stand-in network. They do not substitute for the user's PyTorch/LeRobot/RTX 5090 run.
