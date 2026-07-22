# Surrogate Selected-Cell Files

These files are deterministic, rule-based surrogate selected-cell files.
They are executable by `src/run_concert_map.py` because each `.txt` file
contains 1-based spot indices. They are not the original author Shiny/lasso
selected-cell files.

## Generated File Counts

| subtask_type | files |
| --- | ---: |
| border | 6 |
| cross_niche_core_to_border | 6 |
| cross_slide | 6 |
| niche_core | 6 |
| patch | 6 |
| periphery | 6 |

## Biological Definition

- `patch`: KP tumor spots nearest to observed target perturbation spots.
- `border`: KP tumor spots whose local tumor-neighbor fraction is closest to 0.5.
- `niche_core`: KP tumor spots with the highest local tumor-neighbor fraction.
- `periphery`: KP tumor spots with the highest local normal-neighbor fraction.
- `cross_niche_core_to_border`: KP tumor core source spots for a border-context target benchmark.
- `cross_slide`: KP tumor core source spots in a sample where the target perturbation is absent, paired with a sample where that target exists.

## Outputs

- `surrogate_selected_cells_manifest.csv`: task metadata and neighborhood summaries.
- `fig_surrogate_selection_overview.png/.pdf`: spatial overview of representative selections.
- `fig_surrogate_neighborhood_scores.png/.pdf`: selected-neighborhood score distributions.
