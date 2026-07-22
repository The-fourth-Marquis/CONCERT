# Hierarchical Niche-Conditioned Low-Rank Affine Distribution Operator Design

Date: 2026-07-21

## Objective

Test one new structural hypothesis: a partially pooled, niche-conditioned
low-rank affine operator can reconstruct target distribution location, spread,
and covariance while retaining source-specific variation outside the learned
subspace.

The method is named a distribution `mapping` or `operator`. It is not called
transport or optimal transport because the proposed implementation has no OT
coupling or transport objective.

## Claim Boundary

The experiment remains Mainline A:

```text
observed perturbation spatial relocation
with explicitly reported target-reference access
```

It does not claim strict unseen/OOD perturbation prediction, causal
counterfactual correspondence, physical cell movement, or prediction without
target-reference access.

## Evidence Motivating the Design

The next method must not repeat any of the following failed structures:

- direct or anchor-initialized conditional flow;
- per-query target-anchor interpolation;
- one shared additive target shift;
- shared shift plus a small additive low-rank correction;
- unrestricted gene-dimensional niche lookup.

The reference-only distortion frontier showed that larger anchor movement can
improve E-distance while boundary behavior remains wrong and individuality
declines. The missing capacity is therefore not another mean-shift estimator.
The next falsifiable question is whether a shared low-dimensional affine
transformation of the source distribution can recover spread and covariance in
addition to location.

## Experiment Boundary

All new implementation and results live under:

```text
repro_runs/2026-07-21-hierarchical-niche-affine-operator/
```

The frozen baseline evidence and all earlier NicheFlow, low-rank, flow, and
bounded-anchor experiments are read-only dependencies.

## Fixed Niche Partition

The existing boundary metric threshold `niche >= 0.5` is unusable as the
operator hierarchy on the first task: all 31 query rows and all 15 outer target
references fall into the same group.

The operator therefore freezes a finer, expression-independent partition:

```text
interface_like: niche score < 0.90
core_like:      niche score >= 0.90
```

This threshold is fixed before model evaluation and is not selected using
target expression or any performance metric. In the first task, it gives:

| Split | pseudo-fit interface/core | pseudo-validation interface/core |
| --- | ---: | ---: |
| random-stratified | 5 / 5 | 1 / 4 |
| spatial-block | 4 / 6 | 2 / 3 |

The labels are a low-capacity conditional partition, not a claim that two
biological niches are universally sufficient. Missing or poorly supported
groups must fall back toward the global operator.

## Shared Low-Rank Basis

Only pseudo-fit target references may enter fitting. Let eligible source
controls be `S` and target pseudo-fit references be `R_fit`.

First estimate the existing reference-only matched residual shift:

```text
Delta_0 = mean_j(target_reference_j - matched_control_anchor_j)
```

Construct one task-level orthonormal expression basis `U`, with fixed maximum
rank 4:

1. the first available direction is the normalized global shift `Delta_0`;
2. remaining directions come from the centered source-control and target-fit
   variation after removing the first direction;
3. source and target variation blocks receive equal total Frobenius weight so
   the much larger source-control pool cannot dominate solely by row count;
4. the realized rank is reduced when fewer finite nonzero directions exist.

No niche receives a gene-dimensional basis or gene-dimensional free shift.

## Partially Pooled Niche Moments

For niche `g`, estimate the source full-gene mean `mu_s,g` from eligible source
controls and source latent covariance `C_s,g` in the shared basis. Estimate
target latent mean correction `b_raw,g` and covariance `C_t,g` from target-fit
references in the same niche.

`mu_s,g` is a deterministic centering statistic from the larger observed
source-control population; it is not a fitted target-side gene-dimensional
lookup. If a source niche has fewer than `rank + 2` controls, both its source
mean and covariance fall back to the global source moments.

The target support coefficient is fixed as:

```text
rho_g = n_target,g / (n_target,g + 4)
```

The niche mean correction is low-dimensional:

```text
b_g = rho_g * U^T(
        mean(target_fit_g) - (mu_s,g + Delta_0)
      )
```

Target covariance is partially pooled toward the global target covariance:

```text
C_t,pool,g = rho_g * C_t,g + (1 - rho_g) * C_t,global
```

Covariance matrices use a trace-relative eigenvalue floor, not an absolute
gene-expression-scale constant. If a niche has fewer than three target-fit
references, its covariance falls back to the global target covariance before
pooling. A group with no target-fit reference uses `b_g = 0` and `A_g = I`.

## Affine Distribution Operator

For source latent covariance `C_s,g` and pooled target latent covariance
`C_t,pool,g`, define a regularized CORAL-style affine map:

```text
A_raw,g = sqrt(C_t,pool,g) * invsqrt(C_s,g)
A_pool,g = I + rho_g * (A_raw,g - I)
```

Singular values of `A_pool,g` are clipped to the fixed interval `[0.5, 2.0]`.
This prevents collapse and explosive variance without forcing identity.

For query row `i` in niche `g`:

```text
z_i = U^T(x_i - mu_s,g)
r_i = (x_i - mu_s,g) - U * z_i

y_hat_i(alpha)
  = mu_s,g
  + Delta_0
  + alpha * U * b_g
  + U * [I + alpha * (A_safe,g - I)] * z_i
  + r_i
```

The orthogonal source residual `r_i` is carried through unchanged. At
`alpha = 0`, the prediction is exactly `x_i + Delta_0`. Nonzero alpha changes
only rank-4 niche mean and covariance components.

## Within-Niche Distortion Contract

A global pairwise-distortion constraint was already shown to suppress necessary
between-niche relocation. This design instead protects within-niche geometry.

For each niche with at least two query rows, project `A_pool,g` toward identity:

```text
A_safe,g = I + gamma_g * (A_pool,g - I)
```

Choose the largest deterministic `gamma_g` in `[0, 1]` such that the measured
mean pairwise-distance distortion among query rows in niche `g` is at most
20%. Translation by `Delta_0 + U*b_g` does not affect within-niche pairwise
differences. If a niche has fewer than two query rows, use `A_safe,g = I`
because its distortion cannot be certified.

The 20% bound is frozen before the real run. It is a structural safety limit,
not a candidate selected using outer evaluation.

## Reference-Only Calibration

For each outer split, divide the reported target-reference pool into the same
type of pseudo-fit and pseudo-validation split. Fit all moments and basis terms
using pseudo-fit only.

The frozen strength grid is:

```text
alpha in {0, 0.25, 0.50, 0.75, 1.00}
```

Each candidate is evaluated on pseudo-validation against fixed 80/20. A
candidate is safe only when all gates pass:

```text
held-out E-distance <= fixed 80/20
Delta PCC difference >= -0.02
boundary relative difference <= 5%
individuality difference >= -0.02
within-niche distortion <= 0.20
deletion instability <= fixed 80/20
all metrics finite
```

Among safe nonzero candidates, select the candidate with minimum E-distance,
then higher Delta PCC, lower boundary leakage, and finally lower alpha. If no
safe nonzero candidate exists, record `no_safe_candidate` and use alpha zero.
Outer evaluation is scored once after refitting the selected configuration on
the full outer reference pool.

## First Real-Task Decision

Run task `GSM5808054_patchclose_tumor_Jak2`, seed `20260711`, reference fraction
0.70, under both:

- random-stratified;
- spatial-block.

The method advances to the representative 6-task pilot only if both units:

1. select a safe nonzero alpha using reference-only pseudo-validation;
2. pass the same E-distance, Delta, boundary, individuality, distortion,
   deletion-stability, and mutation gates on their frozen outer evaluation;
3. expose finite basis, moment, shrinkage, and affine-map metadata.

Otherwise the decision is:

```text
do_not_advance_to_representative_pilot
```

No 6-task, 37-task, multi-seed, or broader hyperparameter sweep is run after a
failed first-task decision.

## Required Mutation and Stability Guards

1. Replacing only outer target-evaluation expression with extreme values must
   leave pseudo selection, basis, moments, alpha, predictions, fallback,
   deletion stability, and exposed metadata unchanged.
2. Replacing only pseudo-validation expression must not change fitted basis or
   candidate predictions before scoring.
3. Delete each target-fit reference, refit every component, and report the
   maximum normalized prediction change.
4. Candidate deletion instability must not exceed fixed 80/20.

## Operator Contract Tests

The implementation must prove:

1. realized rank never exceeds 4 and `U^T U = I` numerically;
2. niche parameters are rank-dimensional, never gene-dimensional lookup
   vectors;
3. `rho_g` is finite, bounded in `[0, 1]`, and increases with target support;
4. a missing target niche uses zero niche mean correction and identity affine
   map;
5. fewer than three target rows trigger global covariance fallback;
6. alpha zero is exactly the shared-shift prediction;
7. the orthogonal source residual is exactly preserved;
8. within-niche pairwise distortion is at most 20%;
9. singular values remain in `[0.5, 2.0]` before distortion projection;
10. a synthetic known mean-and-covariance change is recovered better than a
    shared shift;
11. expression scaling preserves dimensionless latent behavior and scales
    predictions consistently;
12. all outputs, covariances, eigenvalues, and diagnostics are finite;
13. all mutation and deletion guards pass.

## Artifacts

The experiment stores:

- tests and source under the new experiment directory;
- `results/first_task_two_split/summary.json`;
- `results/first_task_two_split/metrics.csv`;
- `results/first_task_two_split/predictions.npz`;
- `reports/first_task_two_split_zh.md`;
- a short README with the exact command and final `advance` or `do_not_advance`
  decision.

No git commit is created unless the user explicitly authorizes one.
