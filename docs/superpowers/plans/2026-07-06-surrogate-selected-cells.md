# Surrogate Selected Cells Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate executable rule-based surrogate selected-cell files for CONCERT paper subtask benchmarks and visualize why the selections match the intended biological contexts.

**Architecture:** Add one standalone generator under `repro_runs/2026-06-22-full-repro/` that reads Perturb-map `.h5` files, computes local tissue/perturbation neighborhood scores, writes 1-based selected-cell files, writes a manifest, and exports figures. Add one smoke test module for deterministic scoring and manifest behavior.

**Tech Stack:** Python 3.10 venv at `/root/code/CONCERT/.venv`, h5py, numpy, pandas, matplotlib.

---

### Task 1: Scoring and File Helpers

**Files:**
- Create: `repro_runs/2026-06-22-full-repro/test_generate_surrogate_selected_cells.py`
- Create: `repro_runs/2026-06-22-full-repro/generate_surrogate_selected_cells.py`

- [ ] Write tests for local label fractions excluding self, deterministic top-k selection, 1-based selected-cell file writing, and task manifest rows.
- [ ] Run: `rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/test_generate_surrogate_selected_cells.py`
- [ ] Expected red: import fails because `generate_surrogate_selected_cells.py` does not exist yet.
- [ ] Implement the helper functions and rerun the same test.
- [ ] Expected green: tests print `ok`.

### Task 2: Generate Surrogate Tasks

**Files:**
- Modify: `repro_runs/2026-06-22-full-repro/generate_surrogate_selected_cells.py`
- Output: `repro_runs/2026-06-22-full-repro/surrogate_select_cells/`

- [ ] Generate `patch`, `border`, `niche_core`, `periphery`, `cross_niche_core_to_border`, and `cross_slide` surrogate files where the sample has enough candidate and target spots.
- [ ] Each `.txt` contains 1-based indices compatible with `src/run_concert_map.py`.
- [ ] Each manifest row records `is_exact_author_file=false`, `selection_rule`, `n_spots`, tissue/perturbation context, and summary neighborhood scores.
- [ ] Run: `rtk /root/code/CONCERT/.venv/bin/python repro_runs/2026-06-22-full-repro/generate_surrogate_selected_cells.py`
- [ ] Expected: manifest and selected-cell files are written.

### Task 3: Visual Evidence

**Files:**
- Modify: `repro_runs/2026-06-22-full-repro/generate_surrogate_selected_cells.py`
- Output: `repro_runs/2026-06-22-full-repro/surrogate_select_cells/fig_surrogate_selection_overview.png`

- [ ] Plot background spots by tissue, observed target perturbation spots, and selected surrogate spots.
- [ ] Include one overview panel per generated task family with selection rule and spot count in the title.
- [ ] Save `.png` and `.pdf`.
- [ ] Verify files exist and manifest has nonzero task rows.

### Self-Review

- The plan does not reconstruct author Shiny/lasso selections.
- All generated files must be labeled surrogate.
- The output must be executable by CONCERT eval because files use 1-based row indices.
