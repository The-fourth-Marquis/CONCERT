# NicheFlow-v2 Experiments

This root contains the leakage-controlled evaluation and approved lightweight NicheFlow-v2 upgrade.

Scientific scope: Mainline A observed-perturbation spatial relocation with reported target-reference access. This is not a strict unseen/OOD perturbation claim. The 2026-06-22 experiment tree is a frozen read-only evidence source.

## Main artifacts

- Final report: reports/NicheFlow_v2_report.md
- Existing-method reassessment: results/existing_method_reassessment/
- Residual/gating evidence and G1-G7: results/residual_gating/
- Interim reassessment report: reports/existing_method_reassessment.md
- Frozen configuration: configs/default.json

## Reproduction commands

    rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_reassessment.py --config repro_runs/2026-07-11-nicheflow-v2/configs/default.json --exclude-expensive
    rtk uv run python repro_runs/2026-07-11-nicheflow-v2/run_residual_gating.py --config repro_runs/2026-07-11-nicheflow-v2/configs/default.json
    rtk uv run python repro_runs/2026-07-11-nicheflow-v2/tests/verify_full_outputs.py --mode complete

Implementation follows docs/superpowers/specs/2026-07-11-nicheflow-v2-design.md and docs/superpowers/plans/2026-07-11-nicheflow-v2.md.

Current promotion decision: do_not_advance_to_graph_flow.

SpatialProp-style reference-only retraining is complete for all 1,110 task/split units and is merged into the held-out ranking as an approximate A2.2 comparator.
