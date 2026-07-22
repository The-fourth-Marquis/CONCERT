# AnchorFlow Three-Hour Submission Design

Date: 2026-07-22  
Status: approved for design execution  
Time budget: three hours  

## Objective

Submit a scientifically defensible title and abstract immediately, then use the
remaining time for one frozen, leakage-safe confirmation experiment. The paper
is about few-shot, reference-conditioned gene-expression distribution
generation for observed spatial perturbations. It is not about individual
causal counterfactuals, strict unseen perturbations, or perturbation spatial
distribution completion.

## Provisional Title

Reference-Anchored Residual Flows for Few-Shot Spatial Perturbation Expression
Generation

## Provisional Abstract

Spatial perturbation transcriptomics profiles cellular responses in native
tissue contexts, but observations of a target perturbation are often sparse,
making response generation challenging. Existing end-to-end predictors must
simultaneously infer the target-state location and model context-dependent
distributional variation, an ill-conditioned objective in few-shot settings.
We formulate observed-perturbation prediction as reference-conditioned
gene-expression distribution generation rather than individual causal
counterfactual prediction. We introduce a reference-anchored conditional
residual flow that first constructs a context-aware expression anchor from the
reported target-reference pool and then learns a low-dimensional conditional
velocity field to refine the anchor toward the target distribution. This
decomposition separates statistically stable target localization from flexible
nonlinear distribution refinement, while retaining the anchor-only model as an
exact zero-horizon fallback. Training and model construction use only
target-reference observations, with evaluation expressions excluded from
representation learning, coupling, and inference. We evaluate the method under
random-stratified and spatial-block reference/evaluation splits using
E-distance as the primary endpoint and compare against anchor-only,
fixed-mixture, linear-residual, spatial-smoothing, and end-to-end flow
baselines. Preliminary experiments indicate that anchor initialization resolves
the target-location failure of direct conditional flows and that learned
residual dynamics can further reduce held-out distribution discrepancy beyond
anchor-only prediction.

## Method Story

The method decomposes a difficult few-shot learning problem into two parts:

```text
target-reference context anchor
+ learned conditional residual velocity
-> predicted target expression distribution
```

The anchor supplies low-variance target-state localization. The trainable flow
does not relearn the full perturbation response; it learns only the nonlinear
residual required to refine the anchor distribution. Horizon zero exactly
recovers the anchor, so the primary ablation and fallback share the same model
contract.

The allowed claim is:

```text
observed-perturbation, reference-conditioned expression distribution generation
```

The disallowed claims are per-source-spot causal counterfactual correctness,
strict unseen/OOD perturbation prediction, and source-geometry preservation.

## Frozen Confirmation Experiment

The experiment uses two AnchorFlow-untouched Mainline A tasks with distinct
target pools and relocation conditions:

1. `GSM5808055_border_surrogate_KP_to_Tgfbr2`
   - manifest index: 13
   - condition: same-slide border
   - target pool: GSM5808055 / Tgfbr2
2. `GSM5808055_cross_slide_surrogate_KP_to_Jak2_to_GSM5808054`
   - manifest index: 16
   - condition: cross-slide
   - target pool: GSM5808054 / Jak2

Each task is evaluated under `random_stratified` and `spatial_block`, producing
four task/split views. These are diagnostic views over two target pools, not
four independent biological replicates.

Frozen settings:

| Setting | Value |
| --- | ---: |
| reference fraction | 0.70 |
| seed | 20260711 |
| latent dimension | 8 |
| hidden dimension | 32 |
| epochs | 60 |
| integration steps | 16 |
| integration horizon | 0.25 |
| anchor temperature | 0.25 |
| source skip weight | 0.20 |
| distribution weight | 0.25 |
| identity weight | 0.05 |
| flow compute device | physical GPU 2 |
| process-visible CUDA device | `CUDA_VISIBLE_DEVICES=2` |
| PyTorch device inside process | `cuda:0` |
| precision | FP32, no mixed precision |

The horizon is fixed before reading these results. There is no grid search,
per-task tuning, alternate seed, or rescue run in the three-hour experiment.

The four views run sequentially on physical GPU 2 so that the experiment can
share the machine without launching competing jobs. NumPy preprocessing,
low-rank basis construction, context anchors, and Sinkhorn coupling remain on
CPU; the velocity network training and Euler integration run on CUDA. CPU and
CUDA RNGs are seeded from the frozen seed. The run records the physical GPU,
visible device mapping, GPU name, PyTorch version, CUDA version, and precision.

Every run reports three aligned predictions:

- fixed 80/20 baseline;
- initial reference-conditioned anchor;
- anchor-initialized conditional residual flow.

E-distance is the sole promotion endpoint. Delta PCC, boundary leakage, and
source individuality remain descriptive diagnostics and cannot veto or rescue
the E-distance decision.

## Success and Stop Rules

The confirmation is positive only if all conditions hold:

```text
mean E-distance(flow) < mean E-distance(initial anchor)
mean E-distance(flow) < mean E-distance(fixed 80/20)
mean flow difference is negative for random-stratified views
mean flow difference is negative for spatial-block views
flow wins against both comparators in at least 3 of 4 views
all evaluation-expression mutation guards pass
all predictions and metrics are finite
```

If any condition fails, the frozen decision is
`insufficient_confirmation_for_performance_claim`. The submitted abstract stays
method-forward and retains only the existing preliminary-result sentence. No
architecture, horizon, loss, task, split, or seed is changed after observing
the result.

## Implementation Boundary

Create a new isolated experiment directory:

```text
repro_runs/2026-07-22-anchorflow-submission-smoke/
```

The implementation is a thin wrapper around the existing
`run_anchor_flow_smoke.run_anchor_flow_smoke` function. It must not modify the
frozen Mainline A evidence or the existing v1.5 result directories.

Minimum contract checks before the real runs:

- the two frozen manifest indices resolve to the expected task IDs;
- the four task/split views are unique;
- all settings equal the frozen table above;
- CUDA is available and the process sees exactly the intended physical GPU 2;
- the output directory does not already contain result artifacts;
- summary aggregation uses only E-distance for the decision;
- the existing evaluation-expression mutation guard remains enabled.

Artifacts:

- `config.json` with the frozen protocol;
- one `summary.json` and `prediction.npz` per task/split view;
- `metrics.csv` with fixed, anchor, and flow rows;
- `aggregate_summary.json` with exact decision booleans;
- `README.md` with the claim boundary, results, and stop decision.

## Three-Hour Order of Work

1. Use the provisional title and abstract for registration immediately.
2. Add the isolated wrapper and minimal contract tests.
3. Run the four frozen views sequentially on physical GPU 2 without parallel
   result-dependent changes.
4. Verify artifacts and aggregate the E-distance decision.
5. Update the abstract only if the frozen confirmation succeeds; otherwise keep
   the provisional abstract unchanged.

