# Support-Aware Low-Rank Reference Operator Design

Date: 2026-07-21

## Objective

Build a small-sample reference-conditioned perturbation mapper that tests one
structural hypothesis: a shared low-rank context shift with support-aware
shrinkage can improve target localization without requiring unrestricted
query-specific corrections.

The experiment remains Mainline A: observed perturbation spatial relocation
with reported target-reference access. It does not claim strict unseen/OOD
perturbation prediction or spot-level causal counterfactuals.

## Experiment Boundary

All new code and outputs live under:

```text
repro_runs/2026-07-21-support-aware-low-rank-operator/
```

The frozen 2026-06-22 evidence, NicheFlow-v2 evaluator, and NicheFlow-v1.5
experiments are read-only inputs. No existing experiment output is modified.

## Unified Prediction Contract

For source/query expression `x_i`, context `c_i`, and target-reference set
`R`, predict:

```text
y_hat_i = x_i + Delta_0(R) + rho_i * Clip_kappa(U * h(c_i, R))
```

- `Delta_0(R)` is a shared task-level shift estimated from reference residuals.
- `U` is a rank-limited gene-response basis shared across contexts in a task.
- `h(c_i, R)` is a low-dimensional context coefficient.
- `rho_i` is a support/uncertainty shrinkage coefficient in `[0, 1]`.
- `Clip_kappa` caps the local correction norm.

At `rho_i = 0`, prediction falls back to source plus the shared shift. The
shared base exactly preserves source pairwise differences; only the bounded
low-rank correction can distort them.

## Reference Residual Construction

Target-reference rows are matched to eligible controls using context-only RBF
weights. Reference residual examples are:

```text
d_j = y_reference_j - control_anchor(c_reference_j)
```

No target-evaluation expression enters context normalization, matching, basis
construction, fitting, uncertainty estimation, or prediction.

The global shift is `mean_j(d_j)`. The centered residual matrix supplies a
rank-limited SVD gene basis. Rank is fixed before evaluation and cannot exceed
the number of nonzero centered reference directions.

## Two Comparable Heads

### Kernel head

RBF weights between query and reference contexts predict the latent residual
coefficient. Effective sample size and nearest-reference distance define
support shrinkage:

```text
n_eff_i = 1 / sum_j(w_ij^2)
rho_i = n_eff_i / (n_eff_i + tau)
        * exp(-nearest_distance_i^2 / (2 * bandwidth^2))
```

### Bayesian head

Independent Bayesian ridge heads predict each latent coefficient from the same
standardized context features. Posterior predictive uncertainty multiplies the
same support shrinkage. This is not a gene-wise Bayesian model: uncertainty is
estimated only in the low-rank coefficient space.

## Inductive Biases

1. Global perturbation main effect plus a smaller context interaction.
2. Low-rank gene response rather than per-context gene-dimensional lookup.
3. Partial pooling toward a shared shift under weak support.
4. Spatial/context locality through an RBF similarity kernel.
5. Support monotonicity and automatic low-support fallback.
6. Source-preserving shared base path.
7. Explicit local-correction norm cap and measurable pairwise distortion.
8. Low-capacity heads suitable for the available independent sample count.

## Required Contracts

1. Rank never exceeds the configured limit.
2. Shrinkage remains finite and in `[0, 1]`.
3. Duplicating identical supporting references cannot reduce kernel support
   shrinkage.
4. A far query receives less shrinkage than a supported query.
5. Zero centered reference residual produces shared-shift fallback.
6. Local correction norm never exceeds the configured cap.
7. Changing target-evaluation expression does not change prediction or exposed
   fit metadata.
8. All predictions and diagnostics are finite.

## First Real-Task Smoke

Use the first exact Mainline A task with reference fraction 0.70,
random-stratified split, and seed 20260711. Compare both heads against the
uncalibrated legacy fixed 80/20 baseline. Report held-out E-distance, Delta PCC,
boundary leakage, source individuality, pairwise distortion, shrinkage range,
rank, and mutation-guard status.

This smoke is diagnostic. It cannot promote the method to a representative
pilot. Advancement requires a later 6-task random/spatial-block evaluation.

## Decision Rule

Advance from the first task only if at least one low-rank head:

- passes the evaluation-expression mutation guard;
- has held-out E-distance no worse than fixed 80/20;
- has Delta PCC difference at least -0.02;
- has source individuality difference at least -0.02;
- respects the configured local-correction cap.

Boundary leakage is reported as a spatial diagnostic in this first slice, not
used to override the primary distribution/effect/distortion decision.

