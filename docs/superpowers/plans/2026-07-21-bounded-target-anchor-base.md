# Bounded Query-Specific Target-Anchor Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement and evaluate a support-aware query-specific target-anchor base with a provable 5% pairwise-distortion bound and reference-deletion stability gate on the first Mainline A task under random-stratified and spatial-block splits.

**Architecture:** A pure NumPy operator builds a matched-residual shared fallback, context-RBF target anchor, deterministic support shrinkage, and a globally projected query-specific deformation. Separate task-level calibration splits only the outer reference pool, selects one frozen `beta_max`, computes leave-one-reference-out stability against fixed 80/20, and exposes a full outer-evaluation mutation contract. The runner scores the three predeclared ablations and decides whether both split types justify a representative pilot.

**Tech Stack:** Python 3.10, NumPy, pandas, existing scikit-learn/NicheFlow-v2 data and metric utilities, executable assertion tests.

---

All new experiment files live in `repro_runs/2026-07-21-bounded-target-anchor-base/`. No commit is created because repository instructions require explicit authorization before committing.

### Task 1: Implement the pure bounded-anchor operator

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/__init__.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/operator.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_operator.py`

- [x] Write a failing test importing `fit_predict_bounded_anchor` and asserting finite row-normalized weights, finite support in `[0,1]`, exact `beta_max=0` fallback, and lower support for a distant query.
- [x] Run `rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_operator.py`; expect import failure because the package is absent.
- [x] Implement input validation, context standardization, RBF weights, matched-residual `Delta_0`, target anchors, and support:

```python
support = effective_support / (effective_support + support_prior)
support *= np.exp(-0.5 * nearest_squared / bandwidth**2)
raw_deformation = beta_max * support[:, None] * (anchor - shared_fallback)
```

- [x] Add a failing test with nonconstant query deformation asserting projection scale `gamma <= 1`, actual normalized pairwise distortion `<= epsilon + 1e-6`, and projected deformation no larger than raw deformation.
- [x] Implement the provable projection:

```python
bound = mean_pairwise_norm(raw_deformation) / source_pairwise_scale
gamma = min(1.0, epsilon / max(bound, tiny))
prediction = shared_fallback + gamma * raw_deformation
```

- [x] Add and pass a zero-source-scale test requiring zero deformation and finite metadata.
- [x] Re-run the operator test; expect `ok`.

### Task 2: Add reference-deletion stability and mechanism gates

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/stability.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/evaluation.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_stability.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_evaluation.py`

- [x] Write a failing test for:

```python
deletion_instability(full, leave_one_out, reference_scale)
```

requiring the maximum normalized mean query change and the full per-deletion vector.
- [x] Add a synthetic shared-wrong-shift example whose pairwise distortion is zero but whose deletion instability exceeds a stable baseline.
- [x] Run the test; expect import failure.
- [x] Implement finite shape validation and deletion-instability calculation.
- [x] Write failing selection tests that require all E-distance, Delta, boundary, individuality, 5% distortion, finite-value, and fixed-80/20 deletion-stability gates before selecting minimum E-distance; require `no_safe_candidate` fallback to `beta_max=0`.
- [x] Implement `select_beta` and `decide_two_split_stage` with the frozen thresholds and nonzero-safe-candidate requirement in both split types.
- [x] Run both test files; expect `ok`.

### Task 3: Add Mainline A task adaptation and reference-only calibration

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/task_adapter.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/src/bounded_anchor/calibration.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_calibration.py`

- [x] Write a failing synthetic `TaskPayload` test for context construction and `predict_task` using target-reference indices only.
- [x] Implement the current cross-slide-aware context normalization and operator adapter without importing old experiment outputs.
- [x] Write a failing calibration test with injected outer and pseudo splits requiring pseudo-fit/pseudo-validation containment, the frozen beta grid `(0, .25, .5, .75, 1)`, matching pseudo split type, candidate rows, fixed-80/20 deletion stability, conservative fallback, and full-reference refit.
- [x] Implement a full-reference-only prediction helper for each leave-one-out subset and compute candidate/baseline stability by refitting normalization, shift, anchors, support, and projection after every deletion.
- [x] Add a failing mutation test that changes only outer evaluation expression by an extreme value and requires identical selection, selected beta, anchors, weights, support, gamma, shared/unprojected/projected outputs, deletion vectors, and metadata.
- [x] Implement calibration metadata and make the mutation test pass without reading outer evaluation expression.
- [x] Run the calibration test; expect `ok`.

### Task 4: Build the two-split first-task runner

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/run_first_task_two_split.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/test_runner.py`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/tests/run_all.py`

- [x] Write a failing injected synthetic runner test requiring random-stratified and spatial-block units, matching pseudo split types, all three ablations, fixed 80/20 comparisons, mutation evidence, deletion vectors, a decision, `metrics.csv`, `summary.json`, and `predictions.npz`.
- [x] Run the test; expect import failure because the runner is absent.
- [x] Implement the runner with frozen defaults:

```text
reference_fraction = 0.70
seed = 20260711
bandwidth = 0.75
support_prior = 4.0
epsilon = 0.05
beta_grid = 0, 0.25, 0.50, 0.75, 1.00
outer splits = random_stratified, spatial_block
pseudo split type = matching outer type
```

- [x] Require `insufficient_calibration` plus zero fallback if a matching pseudo split is unavailable; never substitute a random pseudo split.
- [x] Persist arrays for shared fallback, anchor, support, raw deformation, unprojected prediction, projected prediction, selected beta, gamma, and per-deletion instability.
- [x] Run the synthetic runner test and then `tests/run_all.py`; expect all files to pass.

### Task 5: Run the real two-split experiment and report the decision

**Files:**
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/results/first_task_two_split_smoke/metrics.csv`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/results/first_task_two_split_smoke/summary.json`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/results/first_task_two_split_smoke/predictions.npz`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/reports/first_task_two_split_smoke_zh.md`
- Create: `repro_runs/2026-07-21-bounded-target-anchor-base/README.md`

- [x] Run `rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/run_first_task_two_split.py` on the first real task.
- [x] Inspect pseudo rows before outer metrics and verify both selection paths use only their outer reference pools.
- [x] Write the Chinese report directly from the artifacts, including selected beta/status, projection activity, raw-versus-projected location/geometry trade-off, deletion stability versus fixed 80/20, all outer gates, claim boundary, and the explicit advance/stop decision.
- [x] State the next structural hypothesis without running a 6-task pilot unless both split units pass every required gate with safe nonzero beta.

### Task 6: Completion audit

**Files:**
- Modify: `docs/superpowers/plans/2026-07-21-bounded-target-anchor-base.md`

- [x] Run `rtk uv run python repro_runs/2026-07-21-bounded-target-anchor-base/tests/run_all.py`; require zero failures.
- [x] Hash `summary.json`, `metrics.csv`, and `predictions.npz`, rerun the real experiment, and require identical hashes.
- [x] Verify every report number, fallback claim, mutation claim, gate, and file path against JSON/CSV/NPZ.
- [x] Scan the report for unsupported OT/causal/OOD terminology and unrendered LaTeX markers.
- [x] Confirm every required plan item and design contract has direct test or artifact evidence.
- [x] Run `rtk git diff --check -- repro_runs/2026-07-21-bounded-target-anchor-base docs/superpowers/specs/2026-07-21-bounded-target-anchor-base-design.md docs/superpowers/plans/2026-07-21-bounded-target-anchor-base.md`.
