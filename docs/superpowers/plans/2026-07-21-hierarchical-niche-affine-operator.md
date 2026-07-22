# Hierarchical Niche Affine Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a reference-only, partially pooled rank-4 niche-conditioned affine distribution operator on the first Mainline A task under random-stratified and spatial-block splits.

**Architecture:** A pure NumPy operator builds one shared basis, estimates low-dimensional niche mean/covariance moments, preserves orthogonal source residuals, and projects each niche affine map to a 20% within-niche distortion bound. A task adapter, reference-only calibrator, and two-split runner reuse frozen NicheFlow-v2 data and metrics without modifying prior experiments.

**Tech Stack:** Python 3.10, NumPy, existing NicheFlow-v2 task/split/baseline/metric utilities, executable assertion-based tests, JSON/CSV/NPZ/Markdown artifacts.

---

No git commit is created because repository instructions require explicit user authorization before committing.

### Task 1: Pure affine operator contracts

**Files:**
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/src/niche_affine/__init__.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/src/niche_affine/operator.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_operator.py`

- [ ] **Step 1: Write failing operator tests**

Create deterministic synthetic source/query/reference arrays with six genes and two niche groups. Require this API:

```python
result = fit_predict_niche_affine(
    control_expr=control_expr,
    control_context=control_context,
    control_niche=control_niche,
    query_expr=query_expr,
    query_context=query_context,
    query_niche=query_niche,
    reference_expr=reference_expr,
    reference_context=reference_context,
    reference_niche=reference_niche,
    alpha=1.0,
    max_rank=4,
    support_prior=4.0,
    niche_threshold=0.90,
    distortion_epsilon=0.20,
)
```

Assert:

```python
assert result.basis.shape[1] <= 4
assert np.allclose(result.basis.T @ result.basis, np.eye(result.basis.shape[1]), atol=1e-6)
assert np.all((0 <= result.group_support) & (result.group_support <= 1))
assert np.max(result.group_distortion) <= 0.20 + 1e-6
assert np.allclose(result.shared_fallback, query_expr + result.global_shift)
```

Add separate tests for alpha-zero exact fallback, missing-target-niche identity fallback, orthogonal-residual preservation, singular-value clipping, expression scaling, and improvement over shared shift on a known synthetic mean/covariance change.

- [ ] **Step 2: Run the test and verify RED**

Run:

```text
rtk uv run python repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_operator.py
```

Expected: fail with `ModuleNotFoundError: niche_affine`.

- [ ] **Step 3: Implement the minimal operator**

Implement `NicheAffineResult` and `fit_predict_niche_affine`. The result exposes prediction, fallback, global shift, basis, orthogonal residuals, query groups, group mean corrections, raw/clipped/safe affine matrices, support, counts, distortion, gamma, reference weights, and finite metadata.

Use:

```text
rho_g = n_target,g / (n_target,g + support_prior)
A_raw,g = sqrt(C_t,pool,g) * invsqrt(C_s,g)
A_pool,g = I + rho_g * (A_raw,g - I)
A_safe,g = I + gamma_g * (A_clipped,g - I)
```

Build the first basis direction from the matched-residual global shift and the remaining directions from equal-Frobenius-weight source/target centered variation. Use symmetric eigendecomposition with a trace-relative floor. Use deterministic bisection for `gamma_g`. Source groups with fewer than `rank + 2` controls use global source moments; target groups with fewer than three rows use global target covariance; missing target groups use zero correction and identity.

- [ ] **Step 4: Run the operator test and verify GREEN**

Run the focused test until it prints `ok` with no warnings.

### Task 2: Mainline A adapter and metric contracts

**Files:**
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/src/niche_affine/task_adapter.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/src/niche_affine/evaluation.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_task_adapter.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_evaluation.py`

- [ ] **Step 1: Write failing adapter and evaluation tests**

Use a synthetic `TaskPayload` and `TargetSplit`. Require `predict_task(payload, split, alpha=...)` to use only `split.reference_indices`. Mutate only evaluation expression and assert every prediction/fit array and metadata item is unchanged.

Require evaluation helpers:

```python
metrics = metric_bundle(prediction, payload, split)
distortion = max_within_niche_distortion(prediction, payload.query_expr, payload.query_niche)
selection = select_alpha(candidate_rows, fixed_metrics)
```

Test that `select_alpha` selects the minimum-E safe nonzero candidate, applies all seven gates, and returns alpha zero with `no_safe_candidate` when any required gate is absent.

- [ ] **Step 2: Run both tests and verify RED**

Expected: imports fail because adapter/evaluation modules do not exist.

- [ ] **Step 3: Implement adapter and evaluation modules**

The adapter constructs cross-slide-aware position/niche/boundary context exactly from control, query, and reference rows. The evaluation module uses existing NicheFlow-v2 distribution, perturbation, boundary, and individuality metrics and implements the frozen gate thresholds:

```text
E <= fixed E
Delta difference >= -0.02
Boundary relative difference <= 0.05
Individuality difference >= -0.02
Within-niche distortion <= 0.20
Deletion instability <= fixed deletion instability
All finite
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Both files must print `ok`.

### Task 3: Reference-only calibration and deletion stability

**Files:**
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/src/niche_affine/calibration.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_calibration.py`

- [ ] **Step 1: Write a failing calibration test**

Require the frozen grid:

```python
ALPHA_GRID == (0.0, 0.25, 0.50, 0.75, 1.0)
```

Inject outer and pseudo splits. Assert calibration reports pseudo candidate rows, fixed metrics, safe/non-safe status, selected alpha, full refit, candidate/fixed leave-one-reference-out vectors, group support, and no outer-evaluation access. Mutate outer evaluation expression and require exact equality of selection, predictions, moments, basis, affine matrices, deletion evidence, and metadata.

- [ ] **Step 2: Run the test and verify RED**

Expected: `calibrate_task` import failure.

- [ ] **Step 3: Implement calibration**

Fit each alpha only on pseudo-fit target references, score against pseudo-validation, compute deletion instability by deleting and fully refitting one pseudo-fit reference at a time, select with `select_alpha`, then refit on the full outer reference pool. Normalize deletion change by the fitted reference-residual RMS. Keep alpha zero as the only conservative fallback.

- [ ] **Step 4: Run the calibration and prior tests**

Expected: all focused tests print `ok`.

### Task 4: Two-split runner and mutation contract

**Files:**
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/run_first_task_two_split.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/test_runner.py`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/run_all.py`

- [ ] **Step 1: Write a failing injected runner test**

Require `run_first_task_two_split(output_dir=..., payload=..., outer_splits=..., pseudo_splits=...)`. Assert JSON, CSV, and NPZ artifacts contain both split types, pseudo-selection evidence, outer metrics, group/basis arrays, deletion vectors, mutation guard, and a conservative aggregate decision. Require advancement only when both units selected nonzero alpha and passed all outer gates.

- [ ] **Step 2: Run the runner test and verify RED**

Expected: runner module is missing.

- [ ] **Step 3: Implement the runner**

Use task index 0, reference fraction 0.70, seed 20260711, rank 4, bandwidth 0.75, support prior 4, niche threshold 0.90, distortion limit 0.20. For each split, run calibration twice after adding 1000 to outer evaluation expression and fail unless every scientific output remains identical. Score outer evaluation once after selection. Write sorted JSON, stable CSV column order, and compressed arrays.

- [ ] **Step 4: Run all tests**

Run:

```text
rtk uv run python repro_runs/2026-07-21-hierarchical-niche-affine-operator/tests/run_all.py
```

Expected: all test files pass.

### Task 5: Real run, report, and deterministic verification

**Files:**
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/results/first_task_two_split/summary.json`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/results/first_task_two_split/metrics.csv`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/results/first_task_two_split/predictions.npz`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/reports/first_task_two_split_zh.md`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/README.md`
- Create: `repro_runs/2026-07-21-hierarchical-niche-affine-operator/verify_artifacts.py`

- [ ] **Step 1: Run the real two-split experiment**

Run:

```text
rtk uv run python repro_runs/2026-07-21-hierarchical-niche-affine-operator/run_first_task_two_split.py
```

- [ ] **Step 2: Write the decision report from artifacts**

Report pseudo-validation and outer metrics separately, group counts/support, covariance fallbacks, singular values, within-niche distortion, deletion stability, mutation guard, and the exact `advance` or `do_not_advance` decision. Do not reinterpret a numerically stable run as scientific success.

- [ ] **Step 3: Verify artifacts and reproducibility**

The verifier must cross-check report numbers against JSON/CSV, all claim boundaries, all gate keys, mutation invariance, and absence of unrendered LaTeX. Hash JSON/CSV/NPZ, rerun the experiment, and require identical hashes. Run all tests fresh after the rerun.
