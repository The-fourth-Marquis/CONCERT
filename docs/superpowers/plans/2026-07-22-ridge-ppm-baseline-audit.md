# Ridge PPM Strong-Baseline Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a leakage-safe B4/B5/B6 conditional point-process ridge audit on the frozen Cd52 K=4 episode and conditionally run an `mgcv` B7 diagnostic.

**Architecture:** A provenance adapter imports the already verified first-smoke loader without modifying its worktree. Focused numerical modules build frozen coordinate/context/support features and optimize the exact grouped conditional-softmax likelihood with SciPy; a single-use runner performs pseudo-fit/pseudo-validation selection, target evaluation, optional `mgcv` diagnostic, and atomic artifact publication.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy L-BFGS-B, existing AnnData loader, R `mgcv 1.8-39`, unittest, JSON/CSV/NPZ.

---

## File map

- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/README.md`: frozen scope, command, and final decision.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/provenance.py`: upstream dependency hashing and exact frozen input adapter.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/features.py`: B3/B4/B5/B6 feature groups and stable pseudo split.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/objective.py`: exact conditional-softmax objective, gradient, prediction, and ridge fit.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/selection.py`: lambda selection and full-training refit.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/run_audit.py`: fail-closed one-run orchestration and atomic artifacts.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/run_b7_gam.R`: post-trigger `mgcv::bam` spatial correction over the frozen B6 linear predictor.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/verify_audit.py`: independent artifact and decision checks.
- `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_*.py`: minimal scientific contracts.

Git commits are intentionally omitted because project instructions prohibit commits unless explicitly requested.

### Task 1: Provenance and frozen-input adapter

**Files:**
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/__init__.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/provenance.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_provenance.py`

- [ ] **Step 1: Write the failing provenance test**

```python
def test_upstream_contract_matches_frozen_cd52_smoke():
    bundle = load_frozen_bundle()
    assert bundle.selected_episode.episode_id == "ot1_day4_rep2__Cd52"
    assert int(bundle.selected_split.support_counts.sum()) == 4
    assert int(bundle.selected_split.heldout_counts.sum()) == 40
    assert len(bundle.training_records) == 12
    assert "Cd52" not in {episode.target_id for episode, _ in bundle.training_records}
    assert bundle.value_roles == ("development",)
```

- [ ] **Step 2: Run RED**

Run:

```bash
rtk /root/code/nicheformer/.venv/bin/python -m unittest discover -s repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests -p "test_provenance.py" -v
```

Expected: FAIL because `ridge_ppm.provenance` does not exist.

- [ ] **Step 3: Implement the exact adapter**

`load_frozen_bundle()` must import `/tmp/CONCERT-spacseq-elrr/repro_runs/2026-07-22-spatial-distribution-completion/run_first_smoke.py`, call its `_read_sources`, `_inventory`, `_load_context_and_episodes`, and `_split_episode` functions, assert `EXPECTED_EPISODE`, and return a dataclass containing selected/training episodes, exact splits, raw `n_genes`, and provenance. Hash the upstream runner, six `spatial_completion` modules, ELRR assignment/library modules, source manifest, source Table S1, and original first-smoke summary.

- [ ] **Step 4: Run GREEN**

Run the focused unittest command and require PASS with 4 support, 40 held-out, 12 training episodes, development-only roles.

### Task 2: Frozen features and exact ridge objective

**Files:**
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/features.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/objective.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_objective.py`

- [ ] **Step 1: Write RED for likelihood, gradient, nesting, and split**

```python
def test_conditional_objective_matches_hand_calculation():
    X = np.array([[0.0], [1.0], [2.0]])
    counts = np.array([0, 1, 1])
    value, grad = conditional_objective_and_gradient(
        np.array([np.log(2.0)]), [(X, counts)], ridge_lambda=0.0
    )
    probability = np.array([1.0, 2.0, 4.0]) / 7.0
    expected = -(np.log(probability[1]) + np.log(probability[2])) / 2.0
    assert np.isclose(value, expected)
    assert np.allclose(grad, approx_fprime(np.array([np.log(2.0)]), lambda b: conditional_objective_and_gradient(b, [(X, counts)], 0.0)[0], 1e-6), atol=1e-5)


def test_b6_zero_support_coefficients_exactly_recover_b4():
    b4 = predict_probability(base, beta_base)
    b6 = predict_probability(np.column_stack([base, support]), np.r_[beta_base, np.zeros(support.shape[1])])
    assert np.array_equal(b4, b6)


def test_stable_pseudo_split_is_eight_four_and_excludes_target():
    fit_ids, validation_ids = stable_pseudo_split(training_ids)
    assert len(fit_ids) == 8 and len(validation_ids) == 4
    assert set(fit_ids).isdisjoint(validation_ids)
    assert "Cd52" not in set(fit_ids) | set(validation_ids)
```

- [ ] **Step 2: Run RED**

Run:

```bash
rtk /root/code/nicheformer/.venv/bin/python -m unittest discover -s repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests -p "test_objective.py" -v
```

Expected: FAIL because feature/objective functions are missing.

- [ ] **Step 3: Implement minimal feature and numerical modules**

`build_feature_bundle()` must produce shared standardized coordinate polynomial features, existing standardized guide-blind context, and per-episode standardized support features containing `d_min/D`, four log mean-RBF fields, Euclidean context distance, and cosine context similarity. `feature_indices()` must freeze B3/B4/B5/B6 column groups. `conditional_objective_and_gradient()` must use stable log-sum-exp and event-weighted averaging; `fit_ridge()` must call SciPy `minimize(method="L-BFGS-B", jac=True)` and reject non-convergence or non-finite output.

- [ ] **Step 4: Run GREEN and finite-difference verification**

Run the focused tests and require exact B4 nesting plus gradient agreement within `1e-5`.

### Task 3: Pseudo-validation selection and fail-closed runner

**Files:**
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/src/ridge_ppm/selection.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/run_audit.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_selection.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_runner_contract.py`

- [ ] **Step 1: Write RED for lambda choice and decision rules**

```python
def test_lambda_tie_prefers_stronger_regularization():
    rows = [{"lambda": value, "validation_nll": 2.0} for value in (0.1, 1.0, 10.0)]
    assert choose_lambda(rows) == 10.0


def test_decision_requires_positive_support_gain():
    assert decide({"B4": 12.50, "B6": 12.51}) == "do_not_scale_no_support_gain"
    assert decide({"B4": 12.50, "B6": 12.49}) == "support_gain_positive"


def test_b7_trigger_is_frozen():
    assert should_run_b7(12.619395462034924)
    assert not should_run_b7(12.619395462034926)
```

- [ ] **Step 2: Run RED**

Run the two focused test files and expect missing selection/runner behavior.

- [ ] **Step 3: Implement selection and one-run orchestration**

For each B3–B6, fit the nine frozen lambdas on eight pseudo-fit episodes, score four pseudo-validation episodes, choose minimum validation NLL with larger-lambda tie-break, and refit all 12 training episodes. Evaluate Cd52 once, run held-out-count mutation through all predictors, compare frozen neural/RBF constants, and stage `metrics.csv`, `selection_metrics.csv`, `probability_fields.npz`, `coefficients.npz`, and `summary.json` before atomic rename.

- [ ] **Step 4: Run all synthetic tests**

Run:

```bash
rtk /root/code/nicheformer/.venv/bin/python -m unittest discover -s repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests -v
```

Require all tests PASS before touching real held-out metrics.

### Task 4: Conditional B7 open-source GAM diagnostic

**Files:**
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/run_b7_gam.R`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests/test_b7_contract.py`

- [ ] **Step 1: Write RED static contract**

```python
def test_b7_script_uses_frozen_mgcv_contract():
    text = B7_SCRIPT.read_text()
    for token in ("mgcv::bam", "te(x, y, k = 16)", "method = \"fREML\"", "discrete = TRUE", "offset"):
        assert token in text
```

- [ ] **Step 2: Run RED and implement the R script**

The R script must read only runner-exported training rows, fit episode-specific intercepts plus `te(x, y, k=16)` with the frozen B6 linear predictor as an offset, predict Cd52 using its B6 offset, normalize the target field, and write a compact JSON/CSV result marked `post_trigger_diagnostic`. It must never read guide files directly.

- [ ] **Step 3: Run GREEN and package check**

Run:

```bash
rtk Rscript -e 'stopifnot(requireNamespace("mgcv", quietly=TRUE)); cat(as.character(packageVersion("mgcv")), "\n")'
rtk /root/code/nicheformer/.venv/bin/python -m unittest discover -s repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests -p "test_b7_contract.py" -v
```

Require `mgcv 1.8.39` and PASS. Run B7 only when the frozen B6 threshold triggers.

### Task 5: Real audit, verifier, and decision report

**Files:**
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/verify_audit.py`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/README.md`
- Create: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/reports/ridge_ppm_audit_zh.md`
- Create atomically: `repro_runs/2026-07-22-ridge-ppm-baseline-audit/results/ridge_ppm_audit/*`

- [ ] **Step 1: Confirm no prior result and run exactly once**

```bash
rtk find repro_runs/2026-07-22-ridge-ppm-baseline-audit/results -maxdepth 2 -type f
rtk /root/code/nicheformer/.venv/bin/python repro_runs/2026-07-22-ridge-ppm-baseline-audit/run_audit.py
```

Expected: one compact JSON line naming Cd52, B4/B5/B6 NLL, support gain, B7 status, decision, and result directory.

- [ ] **Step 2: Implement and run independent verification**

Verifier must reload NPZ/CSV/JSON, recompute field normalization and conditional NLL from frozen held-out counts, verify selected-target exclusion, lambda choice, mutation guard, B7 trigger/status, dependency hashes, decision code, and zero external/final/reserve reads.

```bash
rtk /root/code/nicheformer/.venv/bin/python repro_runs/2026-07-22-ridge-ppm-baseline-audit/verify_audit.py
```

Expected: `status=verified` with exact artifact count and decision.

- [ ] **Step 3: Write decision-oriented Chinese report**

Report the raw B3–B6 and frozen neural/RBF NLLs, `support gain = B4 - B6`, chosen lambda per method, pseudo-validation results, whether B7 triggered, access/mutation guards, exact decision, claim boundary, and one allowed next structural hypothesis. If B6 does not beat B4, stop expansion.

- [ ] **Step 4: Fresh verification before completion**

```bash
rtk /root/code/nicheformer/.venv/bin/python -m unittest discover -s repro_runs/2026-07-22-ridge-ppm-baseline-audit/tests -v
rtk /root/code/nicheformer/.venv/bin/python -m compileall -q repro_runs/2026-07-22-ridge-ppm-baseline-audit
rtk /root/code/nicheformer/.venv/bin/python repro_runs/2026-07-22-ridge-ppm-baseline-audit/verify_audit.py
rtk git status --short repro_runs/2026-07-22-ridge-ppm-baseline-audit docs/superpowers/plans/2026-07-22-ridge-ppm-baseline-audit.md docs/superpowers/specs/2026-07-22-ridge-ppm-baseline-audit-design.md
```

Require all tests PASS, compilation success, verifier `status=verified`, and only scoped new files.

## Self-review

- Spec coverage: frozen episode/split, exact likelihood, B3–B6 nesting, support summaries, 8/4 selection, nine lambdas, SciPy reuse, conditional mgcv B7, mutation/access guards, artifacts, report, and stop rule are assigned to concrete tasks.
- Scope: one development Cd52 audit only; no extra seed, support draw, episode, external/final/reserve access, LGCP, or neural redesign.
- Type consistency: feature matrices are location-by-feature; counts and probabilities are location-aligned; method names remain `B3`, `B4`, `B5`, `B6`; B7 remains diagnostic.
