# NicheFlow Mainline A Research Summary

Generated: 2026-07-09

## Executive Summary

Mainline A should be framed as **observed perturbation spatial relocation**: given a perturbation state that has been observed somewhere in the spatial dataset, predict how that state should be relocated to a specified source spot set or spatial context. Under this protocol, access to the observed target perturbation pool is allowed for matching, anchoring, residual transfer, and calibration if it is explicitly reported.

The current evidence supports NicheFlow-v1 as a strong transductive relocation method. It beats the strongest A2.1 transductive baseline on 37/37 frozen tasks, but the A2.4 audit shows that most of the gain comes from target-pool anchoring and target-pool calibration. The residual branch remains useful as an interpretable mechanism layer, but it is not yet the dominant performance source. Therefore, the paper-facing claim should be precise: NicheFlow-v1 improves spatial relocation under observed target-state access; it should not be described as strict unseen or de novo OOD perturbation prediction.

## Scientific Question

Current spatial perturbation technologies often observe perturbation effects at sampled tissue locations but do not let us arbitrarily place a perturbation into every desired spatial niche, boundary, or slide. The computational question for Mainline A is:

> Given observed target perturbation spots and a source spatial context, can a model relocate the target perturbation state to the requested spatial spots while preserving spatial context and spot individuality?

This is different from strict unseen perturbation prediction. Mainline A asks whether observed perturbation information can be transferred across locations or contexts. Stage6 strict-unseen experiments remain useful as stress tests, but they should not be mixed into the primary Mainline A claim.

## Protocol

- Frozen task set: 37 Mainline A tasks.
- Shared gene space: 1053 genes.
- Primary metric: E-distance, lower is better.
- Main protocol: `transductive_perturbation_relocation`.
- Target-pool access: allowed for Mainline A if reported.
- Strict unseen target-pool access: forbidden only in the separate strict-unseen/OOD lane.

The benchmark includes one exact author-selected patch task and surrogate tasks covering patch, border, periphery, niche-core, cross-niche core-to-border, and cross-slide relocation settings.

## Compared Methods

| Family | Artifact | Role |
| --- | --- | --- |
| A2.1 strong transductive baselines | `mainline_a_transductive_baselines/` | Primary non-oracle baseline set; includes spatial kNN, kernel smoothing, source identity, linear covariate, and graph diffusion controls. |
| A2.2 SpatialProp-style spot-GNN | `mainline_a_spatialprop_style_spot_gnn/` | Spot-level GNN adaptation using upstream SpatialProp components. |
| Available CONCERT outputs | `concert_available_comparison/` | Limited comparison using only locally available CONCERT prediction files. |
| A2.3 anchored NicheFlow | `nicheflow_anchored_baseline/` | Strong anchored residual baseline with target-pool moment calibration. |
| A2.4 audit | `nicheflow_anchored_audit/` | Gain-source decomposition for anchor, residual branch, and calibration. |
| NicheFlow-v1 | `nicheflow_v1/` | Promoted Mainline A method after audit, with explicit anchor/residual/calibration components. |
| Mechanism metrics | `mainline_a_mechanism_metrics/` | Diagnostics beyond E-distance: boundary leakage, source individuality, and distance-response decay. |

## Main Results

| Model family | Tasks | Mean E-distance | Beats A2.1 best |
| --- | ---: | ---: | ---: |
| A2.1 strong transductive best | 37 | 48.2582 | reference |
| SpatialProp-style spot-GNN | 37 | 256.3295 | 0/37 |
| A2.3 anchored NicheFlow | 37 | 24.9837 | 37/37 |
| NicheFlow-v1 | 37 | 25.1836 | 37/37 |

The A2.1 results show that simple observed-target-pool baselines are already strong: spatial kNN and kernel smoothing dominate the winner counts. This makes the target-state relocation problem nontrivial and establishes the proper comparison level for NicheFlow-v1.

SpatialProp-style spot-GNN did not beat the A2.1 best method on any of the 37 frozen tasks. This does not rule out SpatialProp as a method family, but it indicates that directly adapting the available spot-GNN implementation is not the strongest route for Mainline A under the current data and protocol.

NicheFlow-v1 beats A2.1 on every task type. Its mean E-distance improvements are largest for periphery, border, cross-niche, niche-core, and cross-slide settings, and more modest for patch surrogate tasks.

| Task type | Tasks | NicheFlow-v1 mean E-distance | A2.1 mean E-distance | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| exact patch | 1 | 18.3163 | 46.6524 | -28.3361 |
| periphery | 6 | 24.9218 | 53.0710 | -28.1492 |
| border | 6 | 26.0744 | 52.5523 | -26.4779 |
| cross-niche core-to-border | 6 | 28.4788 | 52.0639 | -23.5851 |
| niche core | 6 | 28.4249 | 50.9792 | -22.5543 |
| cross-slide | 6 | 17.4227 | 39.7238 | -22.3012 |
| patch surrogate | 6 | 26.9237 | 41.4265 | -14.5029 |

## Gain-Source Audit

A2.4 decomposes the A2.3 gain into five families:

| Audit family | Mean E-distance | Wins |
| --- | ---: | ---: |
| anchor + residual + calibration | 24.9837 | 36 |
| anchor only + calibration | 34.6885 | 0 |
| anchor + residual, no calibration | 41.9256 | 0 |
| anchor only | 52.4261 | 0 |
| residual only | 125.2747 | 1 |

The dominant gain source is:

| Dominant source | Tasks |
| --- | ---: |
| anchor | 27 |
| calibration | 10 |

Interpretation: NicheFlow-v1's current performance is mainly anchored by observed target-pool information. The residual branch improves the calibrated anchor on average and gives the model an interpretable spatial-transfer component, but the current version should not claim that residual transfer alone explains the result.

## Mechanism Diagnostics

Mechanism metrics are diagnostics, not causal proof. They ask whether lower E-distance is accompanied by spatially interpretable behavior.

| Model family | Mean E-distance | Boundary leakage lower is better | Source individuality higher is better | Distance-response decay higher is better |
| --- | ---: | ---: | ---: | ---: |
| A2.3 anchored NicheFlow | 24.9837 | 0.5863 | 0.5542 | -0.0479 |
| NicheFlow-v1 | 25.1836 | 0.5704 | 0.5941 | -0.0458 |
| A2.1 strong transductive best | 48.2582 | 0.7055 | 0.2233 | -0.1184 |
| SpatialProp-style spot-GNN | 256.3295 | 3.3612 | 0.0656 | -0.0266 |

NicheFlow-v1 improves boundary leakage and source individuality relative to the A2.1 winner set. This supports the idea that the model is not only reducing global distribution distance but also preserving more spot-level structure. However, distance-response decay remains weak and negative, so the current evidence does not support a strong claim about spatial dose-like response gradients.

## CONCERT Comparison Status

The current CONCERT comparison is limited by available local outputs:

- Available CONCERT coverage: 11/37 tasks.
- Cross-slide CONCERT coverage: 0/6 tasks.
- Available CONCERT beats A2.1 best: 0/11 covered tasks.

This is enough to include CONCERT as a limited available-output comparison, but not enough for a definitive full-method comparison across Mainline A. A first report can fairly state that available CONCERT outputs do not beat A2.1 on covered tasks, while reserving full retraining as a later paper-grade comparison if CONCERT is positioned as a central baseline.

## Caveats

- Many tasks are deterministic surrogate tasks rather than experimentally selected holdouts.
- Mainline A permits observed target-pool access, so its results are not strict unseen/OOD perturbation prediction.
- NicheFlow-v1's strongest gains are anchor/calibration driven; the residual branch is not yet the main performance driver.
- SpatialProp-style spot-GNN is an adaptation of available SpatialProp components, not a full claim about every possible SpatialProp retraining or redesign.
- CONCERT comparison is incomplete because only 11/37 matching local outputs are currently available.
- Mechanism metrics are proxy diagnostics and should be reported as such.

## Paper-Facing Figures

Recommended figures for the next research draft:

1. Mainline A protocol schematic: observed target perturbation pool, source spots, relocation prediction, and evaluation target.
2. E-distance bar/box plot across A2.1, SpatialProp-style, A2.3, and NicheFlow-v1.
3. Task-type stratified delta versus A2.1 best.
4. A2.4 audit plot separating anchor, residual, calibration, and combined model.
5. Mechanism metric panel: boundary leakage, source individuality, distance-response decay.
6. Coverage panel for available CONCERT outputs, clearly marking missing cross-slide coverage.

## Recommended Next Steps

1. Write `concert_retraining_decision.md` to decide whether full CONCERT retraining is required before the first NicheFlow report.
2. Add an access-level ladder experiment: no target pool, uncalibrated target anchor, calibrated target anchor, anchor plus residual.
3. Strengthen the residual branch as a mechanism component, especially for distance-response decay.
4. Keep strict unseen/OOD prediction as a separate appendix or stress-test lane rather than the Mainline A headline.

