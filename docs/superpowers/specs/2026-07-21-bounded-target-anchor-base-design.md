# Bounded Query-Specific Target-Anchor Base Design

Date: 2026-07-21

## Decision

Advance route one as a new structural branch. The immediate research question is
whether a query-specific target anchor can repair target distribution and
boundary location while a provable pairwise-distortion bound preserves source
geometry. Low-rank residuals, kernel-versus-Bayesian head selection, and joint
"complementarity" are excluded from this stage.

## Claim Boundary

The experiment remains Mainline A:

```text
observed perturbation spatial relocation with reported target-reference access
```

It does not claim strict unseen/OOD perturbation prediction, spot-level causal
counterfactuals, optimal transport, or calibrated generalization beyond the
reported target-reference protocol.

## Evidence Motivating the Branch

The support-aware low-rank operator showed that replacing the shared global
shift with six reference-only mean, centroid, anchor, and blend estimators did
not produce one safe pseudo-validation candidate. The selected fallback retained
near-perfect individuality but failed E-distance, Delta PCC, and boundary.

The earlier anchor-guided factorized decoder demonstrated the opposite side of
the trade-off. On the first random-stratified task it achieved E-distance
31.8327 and Delta PCC 0.9905, but its normalized pairwise geometry error was
0.7338. In the 12-unit representative pilot only 4 units found a safe nonzero
strength; mean E-distance difference versus fixed 80/20 was +17.3936 and mean
boundary relative difference was +103.62%.

These results motivate a narrower hypothesis: anchor relocation is useful only
when it is query-support-aware and its induced geometry change is bounded by the
prediction contract rather than encouraged by a soft loss or screened only by
source-individuality correlation.

## New Experiment Boundary

All new code, tests, results, and reports will live under:

```text
repro_runs/2026-07-21-bounded-target-anchor-base/
```

The 2026-06-22 frozen evidence, NicheFlow-v2 loaders and metrics, the 2026-07-20
anchor experiments, and the 2026-07-21 low-rank experiment are read-only inputs.
No prior result artifact will be modified.

## Structural Baseline and Context

Use the current Mainline A task adapter and its context-only features:

```text
normalized spatial position
+ niche score
+ distance from the binary niche boundary
```

Context normalization is fit from eligible controls and target-reference rows.
Target-evaluation expression is excluded from normalization, matching, support
estimation, calibration, and prediction.

The shared fallback is:

```text
s_i = x_i + Delta_0(R)
```

`Delta_0(R)` is the existing matched-residual shift. It is not tuned in this
branch. Because every query receives the same shift, `s_i` exactly preserves
source pairwise differences.

## Query-Specific Target Anchor

For standardized query context `c_i` and target-reference contexts `c_j^R`, use
the frozen RBF kernel with bandwidth 0.75:

```text
w_ij = RBF(c_i, c_j^R) / sum_k RBF(c_i, c_k^R)
a_i = sum_j w_ij * y_j^R
```

The anchor displacement relative to the source-preserving fallback is:

```text
d_i = a_i - s_i
```

This is a mapping/anchor decoder, not transport or optimal transport: it has no
coupling objective, pushforward loss, or OT optimization.

## Support-Aware Deformation

Reuse the support calculation already tested by the low-rank operator:

```text
n_eff_i = 1 / sum_j(w_ij^2)
distance_i = min_j ||c_i - c_j^R||^2

rho_i = n_eff_i / (n_eff_i + 4)
        * exp(-distance_i / (2 * 0.75^2))
```

The only reference-calibrated model degree of freedom is:

```text
beta_max in {0.00, 0.25, 0.50, 0.75, 1.00}
```

The unprojected query-specific displacement is:

```text
v_i = beta_max * rho_i * d_i
```

When target-reference support is weak or distant, `rho_i` approaches zero and
the prediction structurally falls back to `s_i`. No group-specific
gene-dimensional lookup table is estimated. `rho_i` is deterministic and has no
per-query fitted parameter. It is described as kernel effective support, not an
independent sample count or a calibrated probability.

## Provable Hard Pairwise-Distortion Bound

Set the fixed distortion budget before evaluation:

```text
epsilon = 0.05
```

This value is pre-registered rather than selected from target evaluation. It is
roughly one order of magnitude above the largest pairwise distortion observed
for the safe low-rank operator (0.00569), leaving room for meaningful anchor
movement, while remaining far below the 0.7338 geometry error of the earlier
first-task anchor result. The experiment tests whether useful relocation exists
inside this safety envelope; failure does not authorize increasing `epsilon`
after inspecting outer metrics.

For all unordered query pairs, compute:

```text
source_scale = mean_{i<j} ||x_i - x_j||
deformation_bound = mean_{i<j} ||v_i - v_j|| / source_scale
```

If `source_scale` is effectively zero, use zero deformation. Otherwise define:

```text
gamma = min(1, epsilon / max(deformation_bound, tiny))
u_i = gamma * v_i
y_hat_i = s_i + u_i
```

By the reverse triangle inequality:

```text
abs(||(s_i + u_i) - (s_j + u_j)|| - ||s_i - s_j||)
<= ||u_i - u_j||
```

and `s_i - s_j = x_i - x_j`. Therefore the normalized mean pairwise-distance
distortion is guaranteed not to exceed `epsilon`, up to floating-point
tolerance. This is an inference-time invariant, not a training penalty.

The result metadata must expose `deformation_bound_before_projection`, `gamma`,
`pairwise_distortion_after_projection`, `epsilon`, support range, and whether the
hard bound is active.

## Reference-Deletion Stability Gate

Pairwise distortion cannot detect a large erroneous displacement shared by all
queries. The branch therefore adds an independent reference-sensitivity gate;
this gate does not modify the prediction and introduces no fitted parameter.

For each target-reference row `j`, remove it and recompute the complete
prediction path, including context normalization, `Delta_0`, anchor weights,
support, and hard projection:

```text
y_hat(R minus j)
```

Using the matched-reference residual scale from the full fit as the denominator,
define:

```text
reference_scale = max(reference_residual_scale, tiny)

deletion_instability
= max_j mean_i ||y_hat_i(R minus j) - y_hat_i(R)||
  / reference_scale
```

Compute the same quantity for the frozen fixed 80/20 baseline on the identical
reference subsets. A promotable candidate must satisfy:

```text
deletion_instability(candidate)
<= deletion_instability(fixed 80/20) + numerical_tolerance
```

Also report the maximum reference weight and the full per-deletion instability
vector. These are diagnostics rather than additional tunable gates. A method
cannot compensate for failed deletion stability with better E-distance or
pairwise distortion.

## Required Ablations

Each outer unit produces three predetermined outputs:

1. `shared_fallback`: `s_i`, equivalent to `beta_max=0`;
2. `support_anchor_unprojected`: `s_i + v_i`, diagnostic only;
3. `support_anchor_hard_projected`: `s_i + gamma*v_i`, the only promotable method.

The unprojected output is not a fallback and cannot be promoted even if its
outer metrics are favorable. It identifies how much location gain and geometry
damage the hard projection trades off.

No low-rank residual is added. Complementarity is a later ablation only if the
hard-projected anchor base first passes its own mechanism gates.

## Reference-Only Calibration

For each outer task/split unit:

1. create the reported outer target-reference/evaluation split;
2. subset the payload to outer reference rows only;
3. split that pool into pseudo-fit and pseudo-validation;
4. estimate anchors and support using pseudo-fit only;
5. evaluate the five fixed `beta_max` candidates on pseudo-validation;
6. compute candidate and fixed-80/20 reference-deletion instability using
   pseudo-fit rows only;
7. apply all mechanism gates before ranking candidates;
8. select the safe candidate with minimum pseudo-validation E-distance, then
   higher Delta PCC, lower boundary leakage, and smaller `beta_max` as ties;
9. if no candidate passes, record `no_safe_candidate` and fall back to
   `beta_max=0`;
10. refit the selected configuration on the full outer reference pool;
11. score each predetermined outer output once.

Use a pseudo split matching the outer split type: random-stratified for a random
outer unit and spatial-block for a spatial outer unit. If the outer-reference
pool cannot support the matching pseudo split, record `insufficient_calibration`
and use the zero-deformation fallback; do not silently replace it with a random
pseudo split.

## Pseudo-Validation and Outer Mechanism Gates

The hard-projected candidate must satisfy all of the following relative to the
uncalibrated fixed 80/20 baseline:

- held-out E-distance no worse;
- Delta PCC difference at least -0.02;
- boundary relative degradation at most 5%;
- source-individuality difference at least -0.02;
- normalized pairwise distortion at most 0.05 plus numerical tolerance;
- reference-deletion instability no worse than fixed 80/20;
- finite prediction, support, projection, and metric values;
- evaluation-expression mutation guard;
- `uses_evaluation_target=False`.

A favorable E-distance cannot override a failed distortion, Delta, boundary,
individuality, deletion-stability, or leakage gate.

## First Experimental Stage

Before any representative pilot, run the first exact Mainline A task with:

```text
reference fraction = 0.70
seed = 20260711
outer splits = random-stratified and spatial-block
```

The branch advances to the six-task representative pilot only if the
hard-projected output:

- finds a safe nonzero `beta_max` through reference-only calibration in both
  outer split types;
- passes every outer mechanism gate in both split types;
- passes the full mutation contract in both split types.

If either split falls back, fails a gate, or lacks a valid matching pseudo split,
the decision is `do_not_advance_to_representative_pilot`.

## Mutation and Contract Tests

Tests must prove:

1. anchor weights are finite, nonnegative, and row-normalized;
2. duplicate identical reference support cannot reduce `rho_i`;
3. a distant query receives lower support than a matched query;
4. `beta_max=0` exactly equals the shared fallback;
5. the hard projection never increases the unprojected deformation magnitude;
6. the reported pairwise distortion is at most 0.05 plus tolerance;
7. zero source pairwise scale produces zero deformation safely;
8. low support approaches the shared fallback;
9. deleting one reference recomputes the entire fit and produces finite
   instability diagnostics;
10. a synthetic single-reference-dominance case fails the deletion-stability
    gate even when pairwise distortion is zero;
11. pseudo-fit and pseudo-validation are disjoint and contained in outer
    reference;
12. changing only outer target-evaluation expression leaves candidate selection,
    selected `beta_max`, anchors, support, `gamma`, deletion-instability values,
    all three outputs, and all fit metadata unchanged;
13. all outputs and diagnostics are finite.

## Decision-Oriented Artifacts

The stage must produce:

```text
results/first_task_two_split_smoke/metrics.csv
results/first_task_two_split_smoke/summary.json
results/first_task_two_split_smoke/predictions.npz
reports/first_task_two_split_smoke_zh.md
```

The summary records pseudo candidate rows, gate results, fallback reason,
projection activity, maximum reference weight, per-deletion stability values,
outer comparisons, mutation checks, and one explicit decision:
`advance_to_representative_pilot` or
`do_not_advance_to_representative_pilot`.

## Relationship to the Future Complementary Model

If and only if this branch passes both first-task outer splits, the next ablation
may add the frozen support-aware capped low-rank residual:

```text
bounded anchor base
+ bounded low-rank context residual
```

That later experiment must compare the combined model directly with the bounded
anchor base and retain the residual only if it supplies stable incremental value
without increasing distortion or fallback frequency. The current design does
not assume that complementarity exists.
