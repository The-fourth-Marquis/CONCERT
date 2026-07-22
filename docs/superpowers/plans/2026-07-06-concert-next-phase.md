# CONCERT Next Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current CONCERT reproduction into a defensible paper-comparison package by adding perturbation-aware metrics, stricter CONCERT-vs-baseline comparisons, and clearer figure-level evidence.

**Architecture:** Keep all new work under `repro_runs/2026-06-22-full-repro/` so it remains separate from upstream source code. Extend existing benchmark scripts rather than rewriting them, and write every new table/figure into a named output directory with a README.

**Tech Stack:** Python 3.10 venv at `/root/code/CONCERT/.venv`, pandas, numpy, scipy/sklearn, matplotlib, anndata/h5ad, W&B optional logging.

---

### Task 1: Add Perturbation-Aware Metrics

**Files:**
- Modify: `repro_runs/2026-06-22-full-repro/stage2_strong_baselines.py`
- Modify: `repro_runs/2026-06-22-full-repro/test_stage2_strong_baselines.py`
- Output: `repro_runs/2026-06-22-full-repro/stage2_strong_baselines/metrics.csv`

- [ ] Add metrics for perturbation effect vectors: `delta_mae`, `delta_pcc`, `delta_spearman`, `delta_r2`.
- [ ] Define delta as `mean(predicted target) - mean(source/control reference)` versus `mean(observed target) - mean(source/control reference)`.
- [ ] Add DE-focused metrics on top 20 and top 50 observed-effect genes: `de_top20_mse`, `de_top20_direction_acc`, `de_top50_mse`, `de_top50_direction_acc`.
- [ ] Add tests with tiny matrices where effect direction and DE gene sets are known.
- [ ] Run:

```bash
rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/test_stage2_strong_baselines.py
rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/stage2_strong_baselines.py --seed 20260706 --wandb --wandb-project concert-stage2 --wandb-run stage2_metric_extension --wandb-mode online
```

Expected: tests pass, `metrics.csv` gains the new columns, and README ranks methods by both E-distance and effect metrics.

### Task 2: Run Strict CONCERT Holdout Comparisons Where Defined

**Files:**
- Modify: `repro_runs/2026-06-22-full-repro/prepare_stage1_concert_holdouts.py`
- Modify: `repro_runs/2026-06-22-full-repro/run_stage1_concert_holdouts.sh`
- Output: `repro_runs/2026-06-22-full-repro/stage1_diagnostic_benchmark/`

- [ ] Keep existing strict target-state and spatial-block filtered retrains.
- [ ] Add a clear manifest explaining which leave-slide and leave-perturbation CONCERT tasks are well-defined and which are only baseline proxies.
- [ ] For well-defined leave-slide cases with target spots in multiple batches, run CONCERT retraining with held-out batch removed.
- [ ] Score each retrained output with the same E-distance, MAE, PCC, R2, and new delta/DE metrics from Task 1.
- [ ] Run the queue with nohup and W&B:

```bash
rtk nohup /root/code/CONCERT/repro_runs/2026-06-22-full-repro/run_stage1_concert_holdouts.sh > /root/code/CONCERT/repro_runs/2026-06-22-full-repro/logs/strict_holdouts_next.log 2>&1 &
```

Expected: strict CONCERT rows are present where the model design supports them; impossible cases are documented instead of silently omitted.

### Task 3: Strengthen Fig. 5 Transplant Reproduction

**Files:**
- Modify: `repro_runs/2026-06-22-full-repro/fig5_transplant/fig5_transplant.py`
- Output: `repro_runs/2026-06-22-full-repro/fig5_transplant/`

- [ ] Keep the deterministic surrogate spot selections, because exact author spot lists are not shipped.
- [ ] Add DE top-gene and pathway/module score tables for observed Jak2 core, predicted core-context transplant, predicted surface-context transplant, and normal reference.
- [ ] Add spatial marker maps for Plac8 and any additional marker genes used in the paper figure.
- [ ] Add a README section separating exact reproduction evidence from surrogate-analysis evidence.
- [ ] Run:

```bash
rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/fig5_transplant/test_fig5_transplant.py
rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/fig5_transplant/fig5_transplant.py
```

Expected: Fig. 5 outputs include marker boxplots, spatial context plots, gene-wise correlations, and DE/module-level summary tables.

### Task 4: Produce Final Paper-Comparison Report

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/final_report/`
- Create: `repro_runs/2026-06-22-full-repro/final_report/CONCERT_reproduction_summary.md`

- [ ] Summarize paper-reported E-distance values and local comparable values.
- [ ] Separate evidence into exact, partial, surrogate, and not reproducible from public assets.
- [ ] Include Stage 1 leakage diagnostics, Stage 2 strong-baseline results, Fig. 5 transplant reproduction, DSS/STROKE case-study status, and missing-data limitations.
- [ ] Include a table of strongest conclusions and caveats.
- [ ] Include links to all CSVs, logs, W&B runs, and figures.

Expected: one report can be used as the current reproduction deliverable without requiring the reader to inspect the whole run directory.

### Self-Review

- Stage 1 and Stage 2 are already completed at baseline/diagnostic level.
- The main remaining scientific gap is not more E-distance runs, but perturbation-effect and DE/pathway metrics.
- Strict CONCERT holdout retraining should only be claimed where the current categorical perturbation design makes the task well-defined.
- The final report must distinguish public-repo reproducibility limits from model performance.
