# Cross-Pool Shared FiLM Residual Operator Design

Date: 2026-07-21

## Finality Contract

This is the final algorithm-development branch for the current dataset. The
method, data boundary, hyperparameters, calibration rule, final metrics, and
termination rule are frozen in this specification before final holdout target
expression is read.

If the final gate fails, the project stops algorithm exploration and pivots to
a benchmark/empirical-study paper. No architecture, loss, threshold, rank,
hidden width, or calibration grid may be changed and rerun on the final
holdout.

## Objective and Claim Boundary

Test whether parameter sharing across independent target pools can learn a
continuous context-dependent perturbation residual that per-task models could
not identify from roughly ten pseudo-fit references.

The claim boundary remains:

```text
observed perturbation spatial relocation
with explicitly reported target-reference access
```

The method does not claim strict unseen-perturbation prediction, spot-level
causal counterfactuals, physical movement, or prediction without target
references. It is a low-rank conditional mapping/operator, not optimal
transport.

## Why Cross-Pool Sharing Is the Last New Hypothesis

All completed methods fit their conditional structure independently inside one
task. Shared shifts, anchors, conditional flows, additive low-rank corrections,
and niche affine moments reached the same trade-off: distribution distance may
improve, but spatial-block transfer and boundary behavior remain unstable.

The 37 manifest tasks do not provide 37 independent target distributions. They
reuse only six raw target expression pools. Random task-level train/test splits
would leak identical target pools across training and evaluation. Route B
therefore shares parameters across target pools and evaluates only on target
samples absent from training.

## Pool-Level Data Partition

### Training and development pools

```text
GSM5808054 / Jak2   / tumor   n_target=21
GSM5808054 / Tgfbr2 / tumor   n_target=75
GSM5808055 / Ifngr2 / tumor   n_target=41
GSM5808055 / Tgfbr2 / tumor   n_target=62
```

These four pools provide 199 target rows before the deterministic internal
training/validation split. Each raw pool is loaded once even when several
manifest tasks reference it.

Training source controls come only from GSM5808054 and GSM5808055 same-slide KP
tumor pools. Training must not load source or target expression from
GSM5808056 or GSM5808057.

### Sealed final target pools

```text
GSM5808056 / Ifngr2 / tumor   n_target=64
GSM5808057 / Tgfbr2 / tumor   n_target=87
```

Manifest metadata and row counts may be used to construct the protocol, but
their target expression remains unread until the model, configuration, code,
and development artifacts are frozen.

The final runner requires an explicit final-evaluation flag and verifies a
saved hash of the frozen model/configuration before loading holdout expression.

## Training Example Construction

For each training target pool `k`:

1. construct eligible same-slide KP source controls;
2. normalize source and target positions separately to `[0, 1]` per axis;
3. create context features:

```text
c = [normalized_x, normalized_y, niche_score, abs(niche_score - 0.5)]
```

4. match every target row to a context-only RBF-weighted source-control anchor;
5. calculate the pool shared shift:

```text
Delta_k = mean_j(target_j - source_anchor_j)
```

6. calculate centered target residuals:

```text
residual_j = target_j - source_anchor_j - Delta_k
```

The perturbation one-hot order is frozen as:

```text
[Ifngr2, Jak2, Tgfbr2]
```

No slide identifier is provided to the model.

## Shared Expression Basis

Build one orthonormal rank-8 gene basis `U` from the centered residual rows of
all training pools. Each pool's centered residual block is divided by its
Frobenius norm before concatenation so a pool with more target cells cannot
dominate solely by row count.

Use a deterministic dual-Gram eigensolver for wide matrices. Reduce the
realized rank only when fewer than eight finite nonzero directions exist.

All target residuals and source-state features are projected into this shared
basis. Basis-orthogonal query expression is never modified by the neural
network.

## Shared FiLM Residual Network

For source/query expression `x`, source-pool mean `mu_s`, context `c`, and
perturbation one-hot `p`:

```text
z_state = U^T(x - mu_s)
input   = concat(z_state, c, p)

h       = tanh(W1 * input + b1)
delta   = W_delta * h + b_delta
scale   = 1 + 0.5 * tanh(W_scale * h + b_scale)

local_latent = delta + (scale - 1) * z_state
```

Frozen capacity:

```text
rank = 8
hidden width = 16
hidden layers = 1
activation = tanh
scale range = [0.5, 1.5]
```

The same weights are used for every pool, task, perturbation, context, and
slide. The network contains no task-specific or slide-specific lookup table.

## Task-Reference Prediction Contract

For a new task, compute a shared full-gene target shift from reported
target-reference rows only:

```text
Delta_task = mean_j(target_reference_j - source_anchor_j)
```

Let `sigma_task` be the RMS norm of centered reference residuals in the shared
latent basis and:

```text
rho_task = n_reference / (n_reference + 4)
local_safe = rho_task * Clip_sigma_task(local_latent)
```

Prediction at calibration strength `alpha` is:

```text
y_hat_i(alpha)
  = x_i
  + Delta_task
  + alpha * U * local_safe_i
```

At alpha zero the model is exactly a shared translation and preserves all
source pairwise differences. Every learned context correction remains in the
rank-8 basis. The full-gene task shift is an estimated shared statistic, not a
learned per-task parameter vector.

## Training Loss and Frozen Optimization

All losses operate in standardized rank-8 latent coordinates.

```text
L = L_huber
  + 0.25 * L_pool_mean
  + 0.10 * L_pool_covariance
  + 0.25 * L_context_bin_mean
```

- `L_huber`: per-reference local latent residual prediction;
- `L_pool_mean`: predicted versus observed centered residual mean per pool;
- `L_pool_covariance`: predicted versus observed covariance per pool;
- `L_context_bin_mean`: mean residual agreement in fixed niche bins
  `[0,0.5)`, `[0.5,0.9)`, and `[0.9,1.01]` when both sides contain at least two
  rows.

Optimization is frozen:

```text
framework = PyTorch CPU deterministic mode
seed = 20260721
optimizer = Adam
learning rate = 0.01
weight decay = 0.0001
maximum epochs = 400
early-stopping patience = 40
batching = full batch
```

Each training pool is deterministically split 80/20, stratified by the three
niche bins. Basis, shifts, standardization, and training loss use the 80% rows.
The combined validation loss across all four pools selects the epoch. After
selection, refit all preprocessing and the network on all 199 training rows for
exactly the selected number of epochs.

No loss-weight, rank, hidden-width, learning-rate, cap, context-feature, or
architecture sweep is permitted.

## Development Gates Before Unsealing Holdout

Before final evaluation, the implementation must pass:

1. synthetic cross-pool mean/context/covariance recovery better than shared
   shift;
2. identical raw target pools are deduplicated;
3. holdout samples GSM5808056/8057 cannot be loaded by the development runner;
4. training is deterministic, including selected epoch and state-dict hash;
5. expression scaling preserves standardized latent behavior;
6. perturbation/context input changes the prediction on a synthetic example;
7. alpha zero is exact shared translation;
8. scale is always in `[0.5, 1.5]`;
9. basis-orthogonal source residual is unchanged;
10. all outputs and losses are finite;
11. target-evaluation-expression mutation cannot change fit, alpha selection,
    prediction, or exposed metadata.

Failure of a structural/development gate is fixed using synthetic or training
data only. Holdout expression remains sealed.

## Reference-Only Final Calibration

For each final pool and outer split, divide the reported 70% target-reference
pool into the same-type pseudo-fit/pseudo-validation split. Compute
`Delta_task`, `sigma_task`, and predictions from pseudo-fit only.

Frozen alpha grid:

```text
alpha in {0, 0.5, 1.0}
```

A nonzero alpha is safe only when pseudo-validation passes:

```text
held-out E-distance <= fixed 80/20
Delta PCC difference >= -0.02
boundary relative difference <= 5%
individuality difference >= -0.02
deletion instability <= fixed 80/20
all metrics finite
```

Select minimum E-distance among safe nonzero candidates, then higher Delta,
lower boundary leakage, and lower alpha. If no safe nonzero alpha exists,
record `no_safe_candidate` and use alpha zero. Refit only the shared task shift
and residual scale on the full outer reference pool; neural weights and basis
remain frozen.

## Final Holdout Task Views

Evaluate all non-cross-slide manifest task views targeting the sealed pools:

```text
GSM5808056_border_surrogate_KP_to_Ifngr2
GSM5808056_cross_niche_core_to_border_surrogate_KP_to_Ifngr2
GSM5808056_niche_core_surrogate_KP_to_Ifngr2
GSM5808056_patch_surrogate_KP_to_Ifngr2
GSM5808056_periphery_surrogate_KP_to_Ifngr2

GSM5808057_border_surrogate_KP_to_Tgfbr2
GSM5808057_cross_niche_core_to_border_surrogate_KP_to_Tgfbr2
GSM5808057_niche_core_surrogate_KP_to_Tgfbr2
GSM5808057_patch_surrogate_KP_to_Tgfbr2
GSM5808057_periphery_surrogate_KP_to_Tgfbr2
```

Run random-stratified and spatial-block with outer reference fraction 0.70 and
seed 20260711. These produce 20 task-view rows but only four independent
pool/split units:

```text
8056-Ifngr2 / random
8056-Ifngr2 / spatial
8057-Tgfbr2 / random
8057-Tgfbr2 / spatial
```

Promotion statistics are computed at the four pool/split-unit level by first
averaging the five task views within each unit.

## Required Final Ablations

Report without using them for post-hoc selection:

1. `shared_shift_only`: alpha zero;
2. `frozen_film_alpha_one`: alpha one without pseudo-validation selection;
3. `reference_selected_film`: frozen alpha selection rule.

The third is the primary method. Fixed 80/20 remains the comparison baseline.

## One-Time Success and Termination Rule

The primary method succeeds only if all conditions hold across the four
pool/split units:

```text
overall mean E-distance difference < 0
spatial-block mean E-distance difference < 0
each held-out pool mean E-distance difference < 0
mean Delta PCC difference >= -0.02
mean boundary relative difference <= 5%
mean individuality difference >= -0.02
all mutation guards pass
```

The result must also report candidate selection frequency and pool-level
confidence intervals descriptively, without treating 20 task views as 20
independent biological replicates.

If any required condition fails:

```text
stop_all_algorithm_development
pivot_to_benchmark_or_empirical_study
```

No final-holdout-driven revision or second final run is permitted. A rerun is
allowed only to verify byte-level determinism of the identical frozen code and
configuration.

## Artifacts

All new files live under:

```text
repro_runs/2026-07-21-cross-pool-shared-film/
```

Required artifacts:

- training-pool manifest and holdout-lock manifest;
- preprocessing/basis metadata;
- deterministic model state and hash;
- internal training/validation loss history;
- synthetic and structural contract tests;
- reference-only calibration rows;
- final task-view and pool-level metrics;
- mutation evidence;
- final Chinese decision report;
- exact final configuration and commands.

No git commit is created unless explicitly authorized by the user.
