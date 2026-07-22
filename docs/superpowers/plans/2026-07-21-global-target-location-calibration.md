# Global Target-Location Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace only the shared global target-location estimator in the support-aware low-rank operator, select it with an outer-reference-only pseudo-fit/pseudo-validation split, and evaluate the frozen operator once on the outer held-out rows.

**Architecture:** The existing matched-residual shift continues to center the reference residuals and therefore leaves the rank-4 basis, coefficient head, support shrinkage, and correction cap unchanged. A selected `global_shift_override` changes only `base_prediction = query_expr + shift`. Candidate shifts are estimated from pseudo-fit rows, screened on pseudo-validation against fixed 80/20, refit on the full outer reference pool, and then applied to both frozen heads; outer evaluation expression is never read during selection or prediction.

**Tech Stack:** Python 3.10, NumPy, existing scikit-learn operator heads, NicheFlow-v2 task/split/metric adapters, executable assertion tests.

---

No git commit is created because repository instructions require explicit user authorization before committing.

### Task 1: Add a base-only global-shift override

**Files:**
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_operator.py`
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/operator.py`

- [x] Add one test that calls each head with two finite gene-length overrides and asserts that `base_prediction` and `prediction` move by exactly the override difference while `local_correction`, `shrinkage`, rank, residual scale, correction cap, and maximum correction norm remain unchanged.
- [x] Run `rtk uv run python repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_operator.py`; expect `TypeError` because `global_shift_override` is not accepted.
- [x] Add optional `global_shift_override` to both heads, validate shape `(n_genes,)` or `(1, n_genes)` and finiteness, retain the matched-residual mean for centering/basis/correction construction, and use the override only in `base_prediction`.
- [x] Re-run the operator test; expect `ok`.

### Task 2: Generate the frozen global-location candidate family

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_global_location.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/global_location.py`

- [x] Add a synthetic test for the ordered candidate names `matched_residual`, `centroid`, `query_anchor`, `matched_query_blend_025`, `matched_query_blend_050`, and `matched_query_blend_075`; assert correct gene shape, finite values, exact centroid formula, and exact fixed-alpha blends.
- [x] Run the test; expect import failure because `global_location.py` is absent.
- [x] Implement candidate generation using the operator's existing context standardization and RBF matching. Define `matched_residual` as the current reference-minus-control-anchor mean, `centroid` as target-reference mean minus eligible-control mean, and `query_anchor` as mean reference-conditioned query anchor minus query expression.
- [x] Re-run the candidate test; expect `ok`.

### Task 3: Implement reference-only selection and conservative fallback

**Files:**
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_global_location.py`
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/global_location.py`

- [x] Add a test in which multiple candidates are supplied, one fails the Delta gate, one fails the boundary gate, and the minimum-E-distance safe candidate is selected.
- [x] Add a test in which no candidate passes all gates and assert `selection_status == "no_safe_candidate"` with fallback to `matched_residual`.
- [x] Run the test and confirm both selection tests fail because the selector is absent.
- [x] Implement frozen gates: held-out E-distance no worse than fixed 80/20, Delta PCC difference at least `-0.02`, boundary relative degradation at most `0.05`, source-individuality difference at least `-0.02`, and finite metrics. Select the safe candidate by E-distance, then higher Delta PCC, then lower boundary leakage, then frozen candidate order.
- [x] Re-run the test; expect `ok`.

### Task 4: Calibrate using only the outer reference pool

**Files:**
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_global_location.py`
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/global_location.py`
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/src/reference_operator/task_adapter.py`

- [x] Add a synthetic task test that supplies a manual pseudo-fit/pseudo-validation split inside the outer reference subset, calibrates a candidate, mutates only outer evaluation expression, recalibrates, and asserts identical selected name, refit shift, candidate rows, selection metadata, predictions, base predictions, local corrections, shrinkage, and operator metadata.
- [x] Run the test and confirm failure because task-level calibration is absent.
- [x] Add `global_shift_override` passthrough to `predict_task` and implement task-level calibration: subset to outer reference, form or accept a pseudo split, estimate candidates on pseudo-fit, score shared-base predictions on pseudo-validation, select without outer evaluation access, regenerate the selected candidate on the full outer reference, and return auditable calibration metadata.
- [x] Re-run task and global-location tests; expect `ok`.

### Task 5: Run the calibrated second-round smoke

**Files:**
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/tests/test_global_location_runner.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/run_global_location_calibrated_smoke.py`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/results/global_location_calibrated_smoke/summary.json`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/results/global_location_calibrated_smoke/predictions.npz`

- [x] Add an injected synthetic runner test asserting frozen settings, reference-only calibration metadata, both fixed heads, complete mutation evidence, fixed 80/20 comparison, decision gates, and output files.
- [x] Run the test; expect import failure because the runner is absent.
- [x] Implement the runner with rank `4`, bandwidth `0.75`, support prior `4.0`, correction cap `0.5`, pseudo-reference fraction `0.70`, pseudo seed `outer seed + 1000`, and random-stratified pseudo split. The runner must calibrate once, refit on full outer reference, run both heads with the same selected shift, and rerun the full path after an extreme outer-evaluation mutation.
- [x] Run the synthetic runner test; expect `ok`.
- [x] Run `rtk uv run python repro_runs/2026-07-21-support-aware-low-rank-operator/run_global_location_calibrated_smoke.py`; expect the two output artifacts.

### Task 6: Report the decision and verify all contracts

**Files:**
- Modify: `repro_runs/2026-07-21-support-aware-low-rank-operator/README.md`
- Create: `repro_runs/2026-07-21-support-aware-low-rank-operator/reports/global_location_calibrated_smoke_zh.md`

- [x] Write the Chinese report directly from `summary.json`, including pseudo-selection status and rows, selected/refit estimator, frozen invariants, outer metrics versus fixed 80/20, mutation results, claim boundary, and an explicit `advance_to_representative_pilot` or `do_not_advance` decision.
- [x] Run `rtk uv run python repro_runs/2026-07-21-support-aware-low-rank-operator/tests/run_all.py`; expect all experiment test files to pass.
- [x] Re-run the real smoke and compare the complete JSON and NPZ arrays for deterministic equality.
- [x] Read back the report and verify every reported number and claim against `summary.json`.
- [x] Run `rtk git diff --check -- repro_runs/2026-07-21-support-aware-low-rank-operator docs/superpowers/plans/2026-07-21-global-target-location-calibration.md`.
