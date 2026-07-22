# AnchorFlow Three-Hour Submission Design

Date: 2026-07-22  
Status: approved for design execution  
Time budget: three hours  

## Objective

Submit a scientifically defensible title and abstract immediately, then use the
remaining time for one frozen, leakage-safe confirmation experiment. The paper
formalizes few-shot spatial perturbation relocation for an observed target
perturbation, establishes an access-matched evaluation framework that exposes
target-state localization as a key bottleneck, and introduces a
reference-anchored residual flow for gene-expression distribution generation
across spatial contexts. It is not about individual causal counterfactuals,
strict unseen perturbations, physical movement, or perturbation spatial
distribution completion.

## Provisional Title

Few-Shot Spatial Perturbation Relocation via Reference-Anchored Residual Flows

## Provisional Abstract

Spatial perturbation experiments often observe a perturbation in only a sparse
and spatially biased subset of tissue, leaving its response distribution unknown
in other regions, niches, or slides. We formalize this setting as observed
perturbation relocation: given a small target-reference pool and a requested
spatial context, the goal is to generate the corresponding gene-expression
distribution without claiming individual causal counterfactuals. We first
establish an access-matched evaluation framework and show that simple spatial
and target-reference baselines can outperform available end-to-end
counterfactual predictors, revealing target-state localization as a major
bottleneck in few-shot prediction. Motivated by this finding, we introduce a
reference-anchored conditional residual flow. The method first constructs a
context-aware expression anchor from the observed target-reference pool and
then learns a low-dimensional conditional velocity field that refines the
anchor toward the held-out target distribution. This decomposition separates
statistically stable localization from flexible nonlinear distribution
refinement and retains the anchor-only solution as an exact zero-horizon
fallback. Training uses source/control context and target-reference
observations, while target-evaluation expressions are excluded from
representation learning, coupling, selection, and inference. We evaluate the
method under same-slide, random-stratified, spatial-block, and cross-slide
relocation using E-distance as the primary endpoint and compare against strong
access-matched spatial baselines, anchor-only prediction, fixed mixtures, and
direct conditional flows. Preliminary results indicate that anchor
initialization resolves the localization failure of direct conditional flows
and that learned residual dynamics can improve held-out distribution matching
beyond anchor-only prediction.

## TLDR

We formalize few-shot spatial perturbation relocation, reveal target-state
localization as a key bottleneck through access-matched baselines, and introduce
reference-anchored residual flows for gene-expression distribution generation.

The TLDR is 234 characters including spaces and punctuation, below the
250-character submission limit.

## Contribution Structure

The paper has three linked contributions:

1. **Task:** formalize observed spatial perturbation relocation under sparse,
   explicitly reported target-reference access.
2. **Evaluation:** establish access-matched baselines and leakage-safe
   reference/evaluation splits that distinguish target-state localization from
   unsupported individual causal counterfactual claims.
3. **Method:** introduce a reference-anchored conditional residual flow that
   separates statistically stable localization from nonlinear distribution
   refinement.

## Relocation Task and Application Support

Observed perturbation relocation takes a small observed target-perturbation
reference pool and a requested source spot set or spatial context, then
generates the target-perturbation gene-expression distribution for that
requested context.

The task supports practical settings in which perturbation coverage is sparse
or spatially biased, a target perturbation is missing from a region or niche,
or an observed target state must be transferred across regions or slides. The
outputs can support response completion and prioritization of regions for
follow-up experiments. Relocation means computational transfer of an observed
perturbation state across spatial contexts; it does not mean physical movement
or identification of per-spot causal potential outcomes.

## Baseline Audit as Method Motivation

The baseline contribution is an evaluation and diagnosis framework, not a new
prediction algorithm. Frozen evidence motivates the proposed decomposition:

- target-state holdout: the filtered CONCERT retrain has E-distance 332.12,
  while the strongest simple baseline has E-distance 43.42;
- spatial-block holdout: the filtered CONCERT retrain has E-distance 111.68,
  while the strongest simple baseline has E-distance 51.56;
- available CONCERT outputs cover 11 of 37 frozen Mainline A tasks and beat the
  strong access-matched winner on 0 of those 11 tasks.

These protocol-specific results must not be summarized as a universal CONCERT
E-distance of approximately 300. They support the narrower conclusion that
target-state localization and reference access explain substantial apparent
counterfactual-prediction signal. The diagnostic chain is:

    strong access-matched baseline audit
    -> target-state localization bottleneck
    -> reference anchor for stable localization
    -> conditional residual flow for nonlinear refinement

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
few-shot observed spatial perturbation relocation with explicitly reported
target-reference access
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

1. Use the revised relocation title, abstract, and TLDR for registration
   immediately.
2. Add the isolated wrapper and minimal contract tests.
3. Run the four frozen views sequentially on physical GPU 2 without parallel
   result-dependent changes.
4. Verify artifacts and aggregate the E-distance decision.
5. Update the abstract only if the frozen confirmation succeeds; otherwise keep
   the provisional abstract unchanged.

