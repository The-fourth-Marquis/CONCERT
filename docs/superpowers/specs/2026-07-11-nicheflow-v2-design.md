# NicheFlow-v2 Evaluation and Method Upgrade Design

Date: 2026-07-11

## 1. Purpose

NicheFlow-v1 established a strong result for Mainline A transductive perturbation relocation, but its 37/37 E-distance wins are mainly driven by observed-target anchoring and target-pool calibration. NicheFlow-v2 must determine whether performance remains when evaluation targets are held out and whether the model predicts perturbation effects, differential-expression direction, and spatial-context behavior rather than only matching the global target distribution.

This design freezes the evaluation contract before adding model complexity. The implementation order is:

1. split the observed target pool into reference and evaluation subsets;
2. add held-out distribution, perturbation-effect, DE-direction, and spatial-context metrics;
3. re-evaluate existing methods without changing their predictions;
4. implement an explicit residual model with perturbation-invariant matching and reliability gating;
5. consider graph and latent-flow modules only if the lightweight model passes the new evidence gates.

## 2. Scope and Claim Boundary

The primary scope remains Mainline A:

```text
observed perturbation response in a reference pool
    + user-selected source spots and context
    -> perturbation-state prediction at the source spots
```

This supports claims about observed perturbation spatial relocation. It does not support claims about never-seen perturbations, strict causal effects, or de novo OOD prediction. Strict unseen perturbation evaluation remains a separate stress-test lane.

NicheFlow-v1 remains frozen as an anchored transductive baseline. NicheFlow-v2 is not allowed to overwrite its scripts, metrics, reports, or output directories.

## 3. Workspace Layout

All new work lives under one new experiment root:

```text
repro_runs/2026-07-11-nicheflow-v2/
  README.md
  configs/
  src/
  tests/
  results/
    evaluation_contract/
    existing_method_reassessment/
    residual_gating/
  reports/
```

The frozen evidence source is:

```text
repro_runs/2026-06-22-full-repro/
```

New code may import stable helpers or read artifacts from the frozen root, but it must not write there. Paths must be centralized in one configuration module rather than duplicated across runners.

## 4. Evaluation Contract

### 4.1 Target-pool split

For every task with observed target perturbation spots, split the target pool into disjoint subsets:

```text
T_reference intersect T_evaluation = empty
T_reference union T_evaluation = T_observed
```

`T_reference` may be used for:

- target-pool anchors;
- residual construction;
- matching or coupling;
- optional calibration;
- model fitting and validation performed inside the reference side.

`T_evaluation` may be used only for final metrics. It must not influence anchor construction, matching, residuals, calibration, hyperparameter selection, early stopping, or gating thresholds.

### 4.2 Split policy

The default split is deterministic and stratified where labels permit:

- default reference fraction: 0.70;
- default evaluation fraction: 0.30;
- minimum reference size: 5 spots;
- minimum evaluation size: 5 spots;
- fixed master seed: 20260711;
- repeated split seeds: 5 for the first full assessment;
- stratification fields, in priority order: tissue/region, niche bin, spatial block;
- if a target pool is too small for a valid disjoint split, mark the task `insufficient_target_pool` rather than silently reverting to full-pool evaluation.

Spatial blocking must be used as a sensitivity analysis because a random split can leave near-duplicate spatial neighbors on both sides. The first release reports both random-stratified and spatial-block splits when feasible.

### 4.3 Access ladder

Every evaluated method receives an explicit access level:

```text
L0: no target reference pool
L1: uncalibrated target anchor
L2: calibrated target anchor
L3: target anchor plus residual branch
L4: gated anchor/residual model
```

Results from different access levels must not be merged into one unlabeled ranking. Every metric row records access level, reference fraction, split seed, split type, calibration status, and whether target-pool information was used.

### 4.4 Full-pool compatibility

The existing full-pool E-distance remains a compatibility metric and transductive upper-reference result. It is not the primary v2 selection metric. The primary v2 result is held-out evaluation under a disjoint target split.

## 5. Metrics

No single composite score is used in the first release. Metrics remain separate so one strong distribution score cannot conceal incorrect perturbation direction or spatial behavior.

### 5.1 Distribution metrics

- `heldout_edistance`: primary paper-aligned v2 metric, lower is better;
- `fullpool_edistance`: compatibility metric, lower is better;
- `niche_stratified_edistance`: held-out E-distance within task-relevant niche strata;
- reference-fraction sensitivity at 0.25, 0.50, and 0.70 where pool size permits.

### 5.2 Perturbation-effect metrics

For each source group or matched niche stratum:

```text
predicted_delta = mean(predicted target) - mean(matched source/control)
true_delta = mean(held-out observed target) - mean(matched source/control)
```

Report:

- `delta_mae`, lower is better;
- `delta_rmse`, lower is better;
- `delta_pcc`, higher is better;
- `delta_spearman`, higher is better;
- `delta_r2`, higher is better when numerically defined.

These are group- or niche-level effects for unpaired data. They must not be described as spot-level counterfactual accuracy.

### 5.3 Differential-expression metrics

DE genes are selected using reference-side data only. Evaluation uses held-out target data. Report top-20 and top-50 versions of:

- delta correlation;
- MSE;
- direction accuracy;
- opposite-direction fraction.

If too few valid genes exist, the metric is missing with an explicit reason. The implementation must not replace undefined correlations with zero.

### 5.4 Spatial and context diagnostics

Retain and version the existing diagnostics:

- boundary leakage, lower is better;
- source individuality, higher is better;
- distance-response decay, higher is better under the current signed definition;
- niche-wise delta correlation;
- cross-niche performance drop;
- spatial-map correlation for selected marker/DE genes when a registered map is meaningful.

These are diagnostics, not causal proof. Their definitions and directionality must be stored in metric metadata.

## 6. Existing-Method Reassessment

Before implementing NicheFlow-v2 modules, run the new evaluator on reconstructable predictions from:

- A2.1 strong transductive baselines;
- SpatialProp-style spot-GNN;
- NicheFlow-lite;
- A2.3 anchored NicheFlow;
- NicheFlow-v1;
- available CONCERT outputs, with coverage reported explicitly.

The reassessment must distinguish methods whose predictions can be reconstructed under each reference split from methods with only fixed full-pool output. A fixed output that used forbidden evaluation targets is reported as `not_eligible_for_heldout_primary`, not silently scored as a held-out method.

The reassessment produces a coverage table showing, for every method and task, whether held-out reconstruction is exact, approximate, unavailable, or ineligible.

## 7. First NicheFlow-v2 Module

### 7.1 Explicit residual contract

The code must distinguish a pure perturbation residual from a full predicted expression:

```text
delta_context(i) = sum_j pi(j | i) * (x_target(j) - x_control_anchor(j))
residual_prediction(i) = x_source(i) + delta_context(i)
```

Functions returning `delta_context` must never return `source + delta`. Names and output schemas must encode the distinction.

The control-anchor candidate pool is the eligible control/KP population from the source sample after protocol filtering, not only the user-selected query spots. The selected source spots are prediction queries. Restricting anchors to the query set is permitted only as an explicitly named ablation.

### 7.2 Perturbation-invariant matching

The matching representation is built from reference-side, baseline-like context features that should be less sensitive to the perturbation response:

- normalized spatial coordinates within a slide;
- tissue/region label;
- local tumor/normal or niche composition;
- boundary score;
- control-derived expression embedding when available;
- slide/batch indicators for cross-slide tasks.

Response-aware target expression may define the residual target but may not define the matching representation without an explicit ablation label.

The first implementation uses transparent soft matching. Optimal transport, graph encoders, and adversarial perturbation removal are deferred.

### 7.3 Reliability gating

Replace the global 0.80/0.20 mixture with an interpretable gate:

```text
P_anchor(i) = target-reference anchor prediction
P_residual(i) = source(i) + delta_context(i)
g_i in [0, 1]
P_v2(i) = (1 - g_i) * P_anchor(i) + g_i * P_residual(i)
```

The first gate is deterministic and auditable. Candidate features are matching entropy, nearest-match distance, reference density, boundary proximity, cross-slide status, and disagreement between anchor and residual predictions. Gate parameters are selected without evaluation-pool access.

Required ablations are:

- anchor only;
- residual prediction only;
- frozen 0.80/0.20 ensemble reproduced with correct terminology;
- reliability-gated ensemble;
- each of the above with and without target calibration where applicable.

Calibration remains a labeled transductive post-processing ablation. It is not part of the default core mechanism unless held-out evidence shows a benefit without degrading delta, DE, and context metrics.

## 8. Deferred Modules

The following are deliberately deferred until the lightweight model passes the evidence gates:

- leakage-controlled spatial Graph Transformer;
- learned perturbation embeddings or biological semantics;
- conditional latent residual flow;
- Best-of-K biological reranking;
- generative uncertainty calibration;
- strict unseen perturbation mainline claims.

Deferral prevents model complexity from obscuring whether better matching and residual semantics already solve the observed problem.

## 9. Success Gates

The lightweight NicheFlow-v2 module advances to graph/flow development only if all of the following hold on the frozen task set where evaluation is feasible. Paired comparisons use task-split-seed units and a task-clustered bootstrap with 10,000 resamples. Percentage-point margins are absolute, while percentage margins are relative to the comparison baseline:

1. held-out E-distance improves over the corresponding A2.1 access-matched baseline on more than 50% of eligible task-split units, the paired mean difference is below zero, and the 95% bootstrap confidence interval upper bound is below zero;
2. mean `delta_pcc` and `delta_spearman` are no worse by more than 0.02, while mean `delta_mae` and `delta_rmse` are no worse by more than 5%;
3. DE direction accuracy is no worse by more than 2 percentage points and opposite-direction fraction increases by no more than 2 percentage points;
4. boundary leakage is no worse by more than 5%, and source individuality is no worse by more than 0.02, relative to reconstructable NicheFlow-v1 results under the same split;
5. the gain remains at reduced reference fractions and is not confined to the 0.70 reference setting;
6. calibration-free performance has a negative paired mean held-out E-distance difference and wins on at least 50% of eligible task-split units, demonstrating that the result is not solely moment alignment;
7. split-seed variability and task coverage are reported.

For gate 5, reduced-access support requires a negative paired mean held-out E-distance difference at both 0.50 reference fraction and every feasible 0.25 reference-fraction task subset. If fewer than 15 tasks are feasible at 0.25, the 0.25 result is reported as exploratory and does not block progression; the 0.50 gate remains mandatory.

The report must include failures as well as wins. A method is not promoted based only on full-pool E-distance.

## 10. Outputs

Each experiment writes:

- `task_manifest.csv` with split membership and provenance;
- `metrics.csv` in long format;
- `method_ranking.csv` limited to methods eligible for that protocol;
- `coverage.csv`;
- `metadata.json` containing metric definitions, seeds, access levels, and source artifact hashes or paths;
- `README.md` with claim boundaries and compact summaries;
- figures only after tabular outputs pass validation.

The top-level report compares full-pool and held-out conclusions and states whether v1's ranking survives the new protocol.

## 11. Error Handling and Reproducibility

- Validate disjoint target splits before any prediction.
- Fail closed if an evaluation index appears in a reference input.
- Record insufficient-pool tasks instead of changing protocol silently.
- Use deterministic seeds and stable task ordering.
- Preserve NaN for undefined metrics with a reason column.
- Avoid absolute paths in new manifests when a repository-relative path is sufficient.
- Do not write into the frozen 2026-06-22 experiment root.

## 12. Verification Strategy

Tests must cover:

1. reference and evaluation indices are disjoint and deterministic;
2. evaluation expression changes cannot alter anchors, residuals, calibration, gates, or selected hyperparameters;
3. pure residual and full residual prediction have distinct shapes/contracts and satisfy `P_residual = source + delta`;
4. synthetic predictions with correct perturbation direction outperform reversed predictions on delta and DE metrics;
5. global distribution matching can score well on E-distance while failing direction/context tests, demonstrating why the new metrics are necessary;
6. access-level metadata is complete for every metric row;
7. frozen-root write protection is enforced by output-path validation;
8. smoke tests reproduce eligible v1/full-pool compatibility values within documented tolerance;
9. full runs produce all required CSV, JSON, README, and coverage artifacts.

## 13. Delivery Sequence

Implementation proceeds in this exact order:

1. create the new experiment skeleton and central configuration;
2. implement and test target-pool splitting plus leakage guards;
3. implement held-out/full-pool E-distance and access metadata;
4. implement delta and DE metrics;
5. port/version spatial diagnostics;
6. reassess existing methods and publish the coverage audit;
7. implement explicit residual matching;
8. implement reliability gating and calibration ablations;
9. run the frozen benchmark, verify artifacts, and write the v2 report;
10. decide whether graph or latent-flow development is justified.

