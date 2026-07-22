# Cross-Pool Shared FiLM Operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Implement, freeze, and evaluate the final cross-pool shared FiLM residual operator, then make the precommitted success-or-stop decision without holdout-driven revision.

**Architecture:** A guarded pool loader deduplicates four training target pools and refuses sealed samples. Shared preprocessing builds context-matched residual examples, a pool-balanced rank-8 basis, and standardized latent tensors; one deterministic CPU PyTorch FiLM network is trained across pools. A task adapter estimates only the reference-only full-gene shift and scale, a frozen alpha selector controls the bounded shared residual, and a hash-locked final runner unseals GSM5808056/8057 exactly once.

**Tech Stack:** Python 3.10, NumPy, pandas, h5py, PyTorch CPU, existing NicheFlow-v2 task/split/baseline/metric utilities, assertion-based executable tests, JSON/CSV/NPZ/PT/Markdown artifacts.

---

No git commit is created because repository instructions require explicit user authorization. Work stays in the current checkout because this experiment depends on uncommitted prior modules; all new writes stay under `repro_runs/2026-07-21-cross-pool-shared-film/` plus this plan.

### File map

- `src/shared_film/config.py`: the only frozen constants, pool identities, task identities, and final success thresholds.
- `src/shared_film/data.py`: manifest-only protocol construction, guarded raw-pool loading, deduplication, and deterministic internal splits.
- `src/shared_film/preprocessing.py`: context features/RBF anchors, shared shifts, centered residuals, pool-balanced basis, and latent standardization.
- `src/shared_film/model.py`: one-hidden-layer shared FiLM network and bounded scale contract.
- `src/shared_film/training.py`: deterministic loss, early stopping, full-data refit, serialization, and state/config hashes.
- `src/shared_film/prediction.py`: frozen-model task prediction with reference-only shift/scale and alpha contract.
- `src/shared_film/calibration.py`: pseudo-fit/pseudo-validation alpha selection and deletion stability.
- `src/shared_film/evaluation.py`: metric/gate aggregation at task-view and independent pool/split levels.
- `src/shared_film/locking.py`: development/final access guards, freeze manifest, and exact hash verification.
- `run_development.py`: synthetic gates, four-pool training, internal validation, freezing; never loads 8056/8057.
- `run_final.py`: explicit one-time final evaluation and deterministic replay mode.
- `verify_artifacts.py`: cross-artifact, claim-boundary, hash, metric, and report verification.

### Task 1: Frozen protocol and holdout access lock

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/__init__.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/config.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/locking.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_protocol_lock.py`

- [x] **Step 1: Write the failing protocol-lock test**

Require exact frozen constants and a loader callback that is rejected before it can be called:

```python
assert TRAIN_POOL_KEYS == (
    ("GSM5808054", "Jak2", "tumor"),
    ("GSM5808054", "Tgfbr2", "tumor"),
    ("GSM5808055", "Ifngr2", "tumor"),
    ("GSM5808055", "Tgfbr2", "tumor"),
)
assert SEALED_POOL_KEYS == (
    ("GSM5808056", "Ifngr2", "tumor"),
    ("GSM5808057", "Tgfbr2", "tumor"),
)
with pytest.raises(HoldoutAccessError):
    guard_pool_access(("GSM5808056", "Ifngr2", "tumor"), mode="development")
```

Also require rank 8, width 16, seed 20260721, alpha grid `(0.0, 0.5, 1.0)`, optimizer/loss constants, exact ten final task IDs, and final success thresholds.

- [x] **Step 2: Run and verify RED**

Run `rtk uv run python repro_runs/2026-07-21-cross-pool-shared-film/tests/test_protocol_lock.py`.

Expected: `ModuleNotFoundError: shared_film`.

- [x] **Step 3: Implement the minimal frozen protocol and lock**

Use immutable tuples/dataclasses. `guard_pool_access` must reject sealed samples in development mode. `verify_final_access` must require `final_evaluation=True`, all development gates true, and exact saved SHA-256 hashes for configuration, preprocessing, model state, source files, and development summary before returning.

- [x] **Step 4: Run and verify GREEN**

Expected: test prints `ok`; no data file is opened.

### Task 2: Deduplicated pool loading and deterministic split

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/data.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_data.py`

- [x] **Step 1: Write failing injected-loader tests**

Construct repeated manifest rows backed by an injected `raw_loader(path)` counter. Require `build_pool_specs` to return each `(sample, perturbation, tissue)` once, `load_training_pools` to call the raw loader once per sample file rather than once per task, and exact target counts `{21, 75, 41, 62}`. Require source controls to be same-slide `KP/tumor`, shared genes aligned once, context position normalized separately per source/target pool, and no path containing `GSM5808056` or `GSM5808057` passed to the loader.

Require `stratified_internal_split(niche, seed=20260721)` to be deterministic, disjoint, exhaustive, and approximately 80/20 within bins `[0,.5)`, `[.5,.9)`, `[.9,1.01]`.

- [x] **Step 2: Run and verify RED**

Expected: import failure for `shared_film.data`.

- [x] **Step 3: Implement minimal loader and split**

Reuse only v2 string decoding, gene alignment semantics, and niche-score definition. Do not call `load_task_payload` for training because it reloads the same raw target pool per task. Keep `PoolPayload` explicit: pool key, perturbation index, genes, source/target expression, positions, niche, and provenance.

- [x] **Step 4: Run and verify GREEN**

Expected: injected tests pass and counter proves two raw sample files, four target masks, zero sealed reads.

### Task 3: Pool-balanced preprocessing and rank-8 basis

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/preprocessing.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_preprocessing.py`

- [x] **Step 1: Write failing preprocessing tests**

Create four synthetic pools with unequal row counts and known context-matched residuals. Require:

```python
examples = build_training_examples(pools, train_indices)
state = fit_preprocessor(examples, max_rank=8)
assert state.basis.shape[1] <= 8
assert np.allclose(state.basis.T @ state.basis, np.eye(state.realized_rank), atol=1e-8)
assert state.solver == "dual_gram_eigh"
assert np.allclose(per_pool_centered_means, 0.0, atol=1e-8)
```

Verify context is `[x_norm, y_norm, niche, abs(niche-.5)]`, RBF anchors use context only, each residual block is divided by its Frobenius norm before concatenation, a duplicated large pool does not change another pool's relative block weight, latent standardization is fit on training rows only, transforms are finite/invertible, and multiplying expression by a constant yields identical standardized latent coordinates after refitting.

- [x] **Step 2: Run and verify RED**

Expected: import failure.

- [x] **Step 3: Implement preprocessing**

Compute one shared full-gene shift per pool, center residuals per pool, use deterministic dual-Gram eigendecomposition for wide matrices, canonicalize eigenvector signs, discard nonfinite/zero directions only, project source state and residual targets into the shared basis, and serialize every fitted statistic needed for inference.

- [x] **Step 4: Run and verify GREEN**

Expected: all basis, balance, matching, scale, and finite assertions pass.

### Task 4: Shared FiLM model, losses, and deterministic training

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/model.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/training.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_model.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_training.py`

- [x] **Step 1: Write failing model tests**

Require input width `rank + 4 + 3`, one tanh hidden layer of width 16, no embedding/lookup parameters, latent output `delta + (scale - 1) * z_state`, and `0.5 <= scale <= 1.5`. On synthetic inputs, changing context or perturbation one-hot must change output after assigning deterministic nonzero weights. All outputs must be finite.

- [x] **Step 2: Run model test and verify RED, then implement minimal model and verify GREEN**

Implement only `SharedFiLMResidual(torch.nn.Module)` and typed output; run the focused test until it passes.

- [x] **Step 3: Write failing loss/training tests**

Require the exact loss:

```text
Huber + 0.25 * pool_mean + 0.10 * pool_covariance + 0.25 * context_bin_mean
```

Test that covariance uses centered `(n-1)` estimates, niche-bin mean terms require at least two rows on both sides, validation selects the earliest minimum epoch under patience 40, final refit uses all rows for exactly the selected epoch count, and two CPU runs with seed 20260721 return identical selected epoch, loss history, predictions, and state-dict SHA-256.

- [x] **Step 4: Run training test and verify RED**

Expected: missing `train_shared_film`.

- [x] **Step 5: Implement deterministic training and verify GREEN**

Set deterministic PyTorch algorithms and single-thread CPU execution, initialize from the frozen seed, use full-batch Adam (`lr=.01`, `weight_decay=1e-4`), maximum 400 epochs, patience 40, save best epoch state, then reinitialize and refit all 199 rows for that exact epoch count. Reject any override of frozen capacity, optimizer, features, weights, or seed.

### Task 5: Frozen task prediction and structural contracts

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/prediction.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_prediction.py`

- [x] **Step 1: Write failing prediction tests**

With a frozen synthetic preprocessor/network, require target references to be context-matched to eligible controls, then:

```text
Delta_task = mean(reference - source_anchor)
rho_task = n_reference / (n_reference + 4)
local_safe = rho_task * Clip_sigma_task(local_latent)
prediction = query + Delta_task + alpha * U * local_safe
```

Require alpha zero to equal `query + Delta_task` bit-for-bit, scale bounds, basis-orthogonal query differences unchanged, alpha limited to the frozen grid, all network weights shared, and no task/slide lookup. Define `Clip_sigma_task` as radial clipping of each latent correction to `sigma_task`, where `sigma_task` is the RMS norm of centered reference residuals in latent space.

- [x] **Step 2: Run and verify RED**

Expected: import failure.

- [x] **Step 3: Implement prediction path**

Expose prediction, shared fallback, task shift, sigma/rho, unclipped/clipped local latent, network delta/scale, reference anchor weights, and finite metadata. Accept only reference indices; never accept evaluation expression.

- [x] **Step 4: Run and verify GREEN**

Expected: all translation, clipping, orthogonal, and shared-parameter contracts pass.

### Task 6: Reference-only calibration, deletion guard, and mutation invariance

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/calibration.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/src/shared_film/evaluation.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_calibration.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_evaluation.py`

- [x] **Step 1: Write failing calibration/evaluation tests**

Require alpha candidates `{0,.5,1}` to fit `Delta_task` and `sigma_task` on pseudo-fit only and score pseudo-validation only. A nonzero alpha is safe only when E-distance is no worse than fixed 80/20, Delta PCC difference is at least `-.02`, boundary relative difference at most `.05`, individuality difference at least `-.02`, deletion instability no worse than fixed 80/20, and all values finite. Select by E-distance, then higher Delta, lower boundary, lower alpha; otherwise return alpha zero with `no_safe_candidate`.

Delete each pseudo-fit reference, fully refit only task shift/scale, and compare against fixed 80/20 deletion instability. Mutate outer evaluation expression by `+1000`; require exact equality of alpha selection, fit arrays, prediction, deletion vectors, and exposed metadata. Confirm metrics alone change when evaluation expression is scored.

- [x] **Step 2: Run and verify RED**

Expected: imports fail.

- [x] **Step 3: Implement calibration and evaluation**

Reuse v2 metric implementations and the prior experiment's boundary/individuality definitions, but remove the old affine distortion gate because FiLM corrections are bounded by the frozen task-scale contract. Keep fitting and scoring APIs separate so the mutation guard can prove absence of evaluation access.

- [x] **Step 4: Run and verify GREEN**

Expected: selector, fallback, deletion, and mutation tests pass.

### Task 7: Development runner, synthetic recovery gates, and freeze manifest

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/run_development.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_development_runner.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/run_all.py`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/training_pool_manifest.json`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/preprocessor.npz`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/preprocessing_metadata.json`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/model_state.pt`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/loss_history.csv`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/development_summary.json`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/development/freeze_manifest.json`

- [x] **Step 1: Write failing injected development-runner test**

Require the runner to generate all artifacts from injected pools, run all eleven development gates, refuse a sealed pool before loader invocation, store source/config/artifact hashes, and write the freeze manifest only when every gate is true. Synthetic recovery must compare against shared shift on mean/context/covariance error and require strict improvement in all three.

- [x] **Step 2: Run and verify RED**

Expected: runner import failure.

- [x] **Step 3: Implement runner and test aggregator**

The runner first executes synthetic gates, then loads only 8054/8055, checks exact row counts and deduplication, trains/refits, reruns training for determinism, executes scale/context/translation/bounds/orthogonal/finite/mutation contracts, writes stable artifacts, hashes them, and finally writes `freeze_manifest.json`. No command-line option may weaken a gate or change a frozen hyperparameter.

- [x] **Step 4: Run all synthetic/injected tests**

Run `rtk uv run python repro_runs/2026-07-21-cross-pool-shared-film/tests/run_all.py`; expected all files pass.

- [x] **Step 5: Run real development training and freeze**

Run `rtk uv run python repro_runs/2026-07-21-cross-pool-shared-film/run_development.py`. Inspect only development artifacts. If a structural gate fails, fix code against synthetic/training evidence and rerun development; never inspect holdout expression.

### Task 8: Hash-locked final runner and one-time evaluation

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/run_final.py`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/tests/test_final_runner.py`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/final/reference_calibration.csv`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/final/task_view_metrics.csv`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/final/pool_split_metrics.csv`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/final/mutation_evidence.json`
- Create after run: `repro_runs/2026-07-21-cross-pool-shared-film/results/final/final_summary.json`

- [x] **Step 1: Write failing injected final-runner test**

Require refusal without `final_evaluation=True`, refusal when any frozen hash differs, refusal when a final-output sentinel already records a completed run, and loader invocation only after all checks. With synthetic payloads, require ten task views times random/spatial, three ablations, reference-only alpha evidence, mutation guards, aggregation of five views into four independent pool/split units, and the exact seven success gates.

- [x] **Step 2: Run and verify RED**

Expected: final runner import failure.

- [x] **Step 3: Implement the final runner**

Load the frozen state read-only after hash verification. For each of the ten non-cross-slide task views, make outer 70/30 random-stratified and spatial-block splits at seed 20260711, calibrate inside the 70% references, refit shift/scale on all outer references, evaluate fixed 80/20 plus `shared_shift_only`, `frozen_film_alpha_one`, and `reference_selected_film`, and run evaluation-expression mutation guards. Average task views within each pool/split before promotion statistics; compute descriptive pool-level bootstrap intervals with pool/split as the resampling unit and never present 20 views as independent replicates.

- [x] **Step 4: Run final-runner injected test and full pre-unseal suite**

Expected: all tests pass while the real sealed loader has still never been invoked.

- [x] **Step 5: Execute the one-time real holdout evaluation**

Run exactly:

```text
rtk uv run python repro_runs/2026-07-21-cross-pool-shared-film/run_final.py --final-evaluation
```

Do not edit model, configuration, thresholds, architecture, selection, or evaluation code after seeing results. A second run is permitted only in deterministic replay mode against the same frozen hashes and must not overwrite the first result.

### Task 9: Final report, deterministic replay, and termination audit

**Files:**
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/reports/final_decision_zh.md`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/README.md`
- Create: `repro_runs/2026-07-21-cross-pool-shared-film/verify_artifacts.py`

- [x] **Step 1: Write the report directly from frozen artifacts**

State the Mainline A access boundary, pool-level independence, selection/fallback frequency, all four pool/split rows, both held-out pool means, overall/spatial means, Delta/boundary/individuality differences, mutation status, ablations, descriptive intervals, and exact final decision. If any gate fails, the report must say `stop_all_algorithm_development` and `pivot_to_benchmark_or_empirical_study`; it must not propose another model tweak.

- [x] **Step 2: Implement and run artifact verification**

Cross-check JSON/CSV/report values, all ten task IDs and 20 task-view rows, four independent units, exact pool counts, selected alphas, success logic, hashes, source/config immutability, mutation guards, claim boundaries, renderer-safe formulas, and absence of unrendered LaTeX markers.

- [x] **Step 3: Deterministic replay**

Run the final runner in replay mode to a separate temporary directory and require identical stable CSV/JSON hashes and tensor predictions; account for container metadata if PyTorch archive bytes are not stable by comparing canonical state tensors and the canonical state hash.

- [x] **Step 4: Fresh full verification**

Run all route-B tests, `verify_artifacts.py`, scoped git status, and a requirement-by-requirement audit against the design specification. Only then mark the goal complete.

