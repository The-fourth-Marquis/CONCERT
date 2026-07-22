# AGENTS.md

Project-level instructions for agents working in this CONCERT repository.

## Shell Command Rule

Always prefix shell commands with `rtk`. This repository is used in an environment where raw shell output can be expensive and many commands are filtered through RTK.

Examples:

```bash
rtk rg "pattern" .
rtk sed -n '1,160p' path/to/file.py
rtk uv run python path/to/test.py
```

## Namespace / Bwrap Constraint

This workspace may fail with errors like:

```text
bwrap: No permissions to create a new namespace
```

This is an environment sandbox limitation, not a project-code failure. To reduce token waste and repeated failed attempts:

1. Prefer approved `rtk` commands for reads and verification: `rtk rg`, `rtk sed`, `rtk find`, `rtk uv run python ...`.
2. For Markdown, CSV, and report edits, prefer a small `rtk uv run python -c ...` script that performs an exact write or marker-based replacement.
3. For code edits, try `apply_patch` once when appropriate. If it fails with the namespace error, immediately switch to a precise `rtk uv run python -c ...` or approved `perl -0pi -e` edit. Do not repeatedly retry the same failing tool path.
4. Avoid very long inline shell one-liners for verification. Put nontrivial checks in a small `test_*.py` or `verify_*.py` file and run it with `rtk uv run python`.
5. If a task needs network access, dependency installation, GUI access, or system configuration changes, request escalation instead of probing repeatedly inside the sandbox.


## Documentation Fast Path

Use this fast path when the primary requested output is a Markdown report, summary, explanation, project-progress document, decision record, or README update based on existing evidence. It does not apply when the task also changes code behavior, creates a reusable report generator, or modifies an experiment pipeline.

For documentation-only tasks:

- Write the document directly. Do not require brainstorming approval, a design specification, an implementation plan, execution-mode confirmation, or a worktree.
- Make reasonable assumptions about the audience, structure, title, and output path from repository context. Ask the user only when a missing choice would materially change the result.
- Do not use TDD or deliberately create a failing/red test for static prose. A red test is appropriate only when the user requests it, the document is machine-consumed, or the task creates a reusable report-generation/validation program.
- After writing, verify factual claims, key numeric values, file links, claim boundaries, and Markdown readback. Use a small verification script only when it materially reduces transcription risk.
- Do not create separate design or plan documents for the report unless the user explicitly requests them.
- Do not create a git commit unless the user explicitly requests a commit or the documentation is already part of an authorized implementation workflow.
- Prefer one-pass delivery: inspect compact evidence, write the document, verify it, and return the file link. Offer revision after delivery instead of asking for repeated pre-writing confirmations.
- Keep evidence reads compact: compute aggregates from CSVs and avoid copying entire prior reports into the conversation.

If a request mixes documentation with a code or experiment change, apply the normal development workflow to the code/experiment portion and this fast path only to the documentation portion.


## Common Token Traps In This Project

The CONCERT benchmark workspace has several recurring token-expensive failure modes. Agents should treat these as project-specific hazards.

### 1. Repeating failing namespace operations

Symptom:

```text
bwrap: No permissions to create a new namespace
```

Response:

- Do not retry the same command path more than once.
- If `apply_patch` fails with this error, switch to a precise `rtk uv run python -c ...`, approved `perl -0pi -e`, or request escalation.
- If `rtk uv run python -c ...` also fails with this error, request escalation rather than trying many variants.

### 2. Shell command substitution inside inline Python strings

Markdown backticks inside `python -c "..."` can be interpreted by the shell before Python sees them. This can execute accidental commands, destroy intended file content, and produce huge help output.

Response:

- Avoid literal backticks in shell-quoted inline Python.
- If a generated file needs Markdown fences or inline code marks, build them inside Python with `chr(96)`.
- Prefer a short script file for large document generation.
- Always read back generated Markdown after writing.

### 3. Huge CSV or dataframe dumps

Many result tables in `repro_runs/2026-06-22-full-repro/` are wide and can flood the transcript.

Response:

- Print shapes and selected columns instead of full rows.
- Use `.head().to_string(index=False)` only with a small, explicit column subset.
- For summaries, compute aggregates directly and print compact values.
- Prefer writing summary CSVs or Markdown reports, then reference their paths.

### 4. Noisy recursive file discovery

The repo contains many baseline and output files. Broad `find` or `ls -R` calls produce too much output.

Response:

- Use `rtk rg --files` or `rtk find <dir> -maxdepth N -type f`.
- Scope searches to `repro_runs/2026-06-22-full-repro/`, a named active `repro_runs/<run>/` directory, `src/`, or a named baseline directory.
- Avoid compound `find` predicates through `rtk find`; use simple scoped commands.

### 5. Long inline verification commands

Long `python -c` checks are hard to quote, easy to break, and often print noisy warnings.

Response:

- Put nontrivial verification into `test_*.py` or `verify_*.py`.
- Keep inline checks to short existence, keyword, shape, or count assertions.
- Avoid shell pipes and command substitutions in verification unless necessary.

### 6. Re-running full experiments unnecessarily

Several benchmark runners can be expensive or produce large outputs. Many user requests only need existing CSVs and READMEs.

Response:

- Before running an experiment, check whether the expected output directory already exists.
- Prefer reading `README.md`, `metadata.json`, `method_ranking.csv`, `compare_to_a2_best.csv`, and `task_type_summary.csv`.
- Re-run full scripts only when outputs are missing, stale, or explicitly requested.

### 7. Confusing Mainline A with strict OOD

This is a scientific-token trap: explanations can balloon if the protocol boundary is not stated early.

Response:

- State early that Mainline A is observed perturbation spatial relocation with reported target-pool access.
- Keep strict unseen/OOD as Stage6 appendix or stress test unless the user explicitly asks for that protocol.
- Do not describe NicheFlow-v1 as predicting never-seen perturbations.

### 8. Dirty worktree noise

This repo often has unrelated dirty or untracked files.

Response:

- Use scoped `git status --short <path>` for the files touched in the current task.
- Do not summarize unrelated dirty files unless relevant.
- Never revert unrelated user changes.

## Scientific Exploration Fast Path

Use this fast path when a new model or structural hypothesis has not yet shown
a positive signal on the first real development task. Its purpose is to
falsify weak ideas cheaply before building publication-grade infrastructure.

### 1. Use minimal RED -> GREEN contract tests

Before the first real smoke, test only contracts that could invalidate the
scientific conclusion:

- split and reference/evaluation isolation;
- evaluation-expression mutation/leakage guards;
- exact frozen-baseline reproduction;
- the core formula on a hand-calculated example;
- output shape, finiteness, normalization, and structural invariants;
- metric direction, mechanism-gate thresholds, and deterministic seed behavior.

For these contracts, observe a meaningful RED caused by the missing behavior,
then implement the smallest GREEN. Do not create exhaustive tests for every
CLI option, malformed input, report, or artifact format before model viability
is known.

### 2. Run the smallest real smoke immediately

After the minimal contracts pass, run one predeclared representative
development unit with one seed and one split. Freeze the comparator, primary
metric, mechanism gates, and `advance` / `redesign` / `do_not_scale` rule before
reading the result.

The first smoke is diagnostic, not calibrated evidence and not a publication
result. Keep it in a separate, date-specific experiment directory.

### 3. Stop expansion on a negative smoke

If the candidate fails a primary mechanism gate, loses to the frozen baseline,
or fails to beat its relevant nonlearned/no-query null, record `do_not_scale`.
Do not rescue it by adding seeds, spatial splits, tasks, epochs, hyperparameter
sweeps, or new data engineering. A follow-up run requires a new structural
hypothesis that directly addresses the observed failure.

### 4. Harden only after scientific signal

Use this order of escalation:

```text
minimal contract tests
-> one real first-task smoke
-> representative random and spatial-block units
-> more tasks and seeds
-> reusable pipeline, exhaustive tests, provenance, and release artifacts
-> external/final evaluation
```

Before the candidate passes the first smoke, do not require a complete
production design, a large implementation plan, exhaustive malformed-artifact
coverage, multi-round code review, or polished reporting for disposable
experiment code. A short frozen protocol in the runner/README plus the minimal
contract tests is sufficient. Normal development workflow still applies when
the task changes reusable core code or a shared experiment pipeline.

### 5. Front-load only the minimum validity and access guards

Before the first smoke, implement only the safeguards needed to keep the result
scientifically interpretable and prevent irreversible access:

- use an explicit development-only allowlist and deny external/final/reserve
  value access by default;
- keep target-evaluation expression out of fit, prediction, and selection, and
  run one mutation guard;
- verify split integrity and exact frozen-baseline reproduction;
- verify guide/perturbation assignment truth only for the labels actually used
  by the smoke;
- write outputs to an isolated experiment directory and never mutate frozen
  evidence.

These are minimum guards, not a requirement to build the complete access
matrix, source-checksum universe, artifact verifier, transactional seal, or
publication bundle before the model has a positive signal. If the smoke uses an
already frozen benchmark, assignment and provenance work for unrelated future
datasets should remain deferred.

### 6. Move formal compliance after positive signal

After a candidate passes the first smoke, and before representative-task or
multi-seed scaling, add the data manifest, provenance, schema validation,
calibrated selection artifacts, reproducibility checks, and stronger code
review that the promoted experiment actually needs.

Build the full role-by-source access matrix, source checksums, transactional
unseal, atomic evidence publication, external/final runners, and final artifact
verifier only after development evidence justifies requesting external access.
Before that point, external/final/reserve need a hard deny-by-default boundary,
not a fully implemented evaluation workflow.

Assignment correctness and leakage isolation cannot be postponed for data used
in the smoke, because errors there make the apparent model signal meaningless.
The surrounding publication-grade compliance machinery can and should be
postponed until scientific viability is demonstrated.

## Scientific Experiment Discipline

Use these rules for perturbation-model experiments, method promotion, and
scientific reporting.

### 1. Separate diagnostic sweeps from valid model selection

A hyperparameter sweep evaluated on held-out target rows is diagnostic only. It
may identify a promising region, but it is not a calibrated result and cannot
promote a method.

Before final evaluation:

1. Split the reported target-reference pool into pseudo-fit and
   pseudo-validation subsets.
2. Freeze the candidate grid, selection rule, and fallback behavior.
3. Select the configuration using pseudo-validation only.
4. Evaluate the real target-evaluation rows once after selection.

Keep uncalibrated smoke results and reference-calibrated results in separate
artifacts.

### 2. Use conservative calibration fallbacks

If no pseudo-validation candidate passes all configured mechanism gates, record
`no_safe_candidate` and fall back to the least deforming prediction, normally
zero local correction or the shared-shift base.

Never select the largest correction merely because no candidate passed. This
can silently destroy individuality while making calibration look successful.

### 3. Require mechanism gates before scaling

Do not scale because one task has a strong E-distance result. A candidate must
first pass the representative-task mechanism gates:

- held-out E-distance;
- perturbation Delta PCC;
- boundary leakage;
- source individuality;
- evaluation-expression mutation guard.

After the first-task smoke, run representative tasks under both
random-stratified and spatial-block splits. Only then consider more seeds or the
full task set.

### 4. Prefer structural invariants over loss reweighting

When a property such as source pairwise geometry is required at inference,
encode it in the prediction contract where possible. Training-set identity loss
does not guarantee geometry preservation on query spots.

For a shared additive shift:

```text
(x_i + shift) - (x_j + shift) = x_i - x_j
```

Any context-dependent residual must have an explicit hard bound and a contract
test measuring pairwise distortion.

### 5. Guard scientific method terminology

Use method names only when the implementation supports the claim:

- `transport` requires an explicit coupling, pushforward objective, or
  distributional transport loss;
- `optimal transport` requires an actual OT objective, not merely a
  distribution-moving interpretation;
- `geometry-preserving` requires an exact invariant or a stated distortion
  bound;
- `hierarchical` requires implemented parameter sharing or partial pooling,
  not independent per-group parameters;
- `calibrated` requires selection without target-evaluation access.

Otherwise use narrower terms such as `mapping`, `decoder`, or
`geometry-regularized`.

### 6. Avoid per-group high-dimensional lookup tables

Do not estimate one unconstrained gene-dimensional shift per niche or spatial
block when reference counts are small. Prefer:

```text
global shift
+ partially pooled low-rank context shift
+ bounded local residual
```

Groups must share a low-rank basis or conditional predictor. Low-support and
unseen groups must shrink toward the global component.

### 7. Treat pseudo-validation transfer as evidence, not a guarantee

A leakage-safe pseudo-validation split can still select configurations that do
not generalize to the final spatial block or slide. Always report:

- how many task/split units found a safe nonzero candidate;
- fallback frequency;
- pseudo-validation versus evaluation gate agreement;
- results by random, spatial-block, and cross-slide condition.

### 8. Run mutation guards for every new prediction path

For each new task wrapper, replace only the held-out target-evaluation
expression with an extreme value and rerun prediction. Both the primary output
and exposed initialization/fallback predictions must remain unchanged.

### 9. Keep experiment artifacts decision-oriented

For each method stage, store:

- raw `summary.json` or `metrics.csv`;
- leakage and calibration metadata;
- comparison with the frozen baseline;
- a short report stating `advance`, `redesign`, or `do_not_scale`;
- the exact next structural hypothesis.

Do not describe a numerically stable training run as a scientifically
successful model.

### 10. Prefer renderer-safe formulas in Markdown reports

If a report renderer does not reliably support LaTeX delimiters such as `\[`
or `\(`, use fenced ASCII formulas:

```text
Delta(c_i) = Delta_0 + rho_i * U * h(c_i)
lambda_i = lambda_max * rho_i * q_i
```

After writing, read back the affected section and scan for unrendered LaTeX
markers.

## Editing Safety

- Do not revert unrelated dirty files.
- Treat frozen evidence directories such as `repro_runs/2026-06-22-full-repro/` as read-only unless the user explicitly requests an update there.
- Put new benchmark experiments and reports in a named, date-specific `repro_runs/YYYY-MM-DD-<topic>/` directory unless the task explicitly targets core CONCERT source.
- For Mainline A work, keep the claim boundary explicit: observed perturbation spatial relocation with reported target-pool access, not strict unseen/OOD perturbation prediction.

## Current Mainline A Report Artifacts

Frozen baseline evidence:

- `repro_runs/2026-06-22-full-repro/mainline_a_report/NicheFlow_Mainline_A_summary.md`
- `repro_runs/2026-06-22-full-repro/mainline_a_report/NicheFlow_当前方法中文说明.md`
- `repro_runs/2026-06-22-full-repro/mainline_a_report/result_index.csv`

Active NicheFlow-v1.5 method development:

- `repro_runs/2026-07-20-nicheflow-v15/README.md`
- `repro_runs/2026-07-20-nicheflow-v15/reports/2026-07-20_work_archive_and_hrcgt_plan_review_zh.md`
- `repro_runs/2026-07-20-nicheflow-v15/reports/anchor_guided_calibrated_pilot.md`
- `repro_runs/2026-07-20-nicheflow-v15/results/anchor_guided_calibrated_pilot_final/metrics.csv`
- `repro_runs/2026-07-20-nicheflow-v15/results/anchor_guided_calibrated_pilot_final/summary.json`
