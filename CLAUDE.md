# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CONCERT is a spatial-aware GP-VAE model for counterfactual perturbation prediction in spatial transcriptomics. It predicts spatially-informed perturbation responses and enables counterfactual "what-if" gene expression predictions by switching cell attributes.

## Repo structure

- `src/` — Core CONCERT for Perturb-map data (multi-kernel, perturbation-specific spatial kernels)
- `src/CONCERT-3D/` — 2D/3D variant for brain stroke data
- `src/CONCERT-long/` — Longitudinal variant for gut inflammation data
- `src/single_kernel/` — Single shared kernel ablation variant
- `notebooks/` — Tutorial/analysis Jupyter notebooks
- `select_cells/` — R Shiny app and pre-saved spot selection files for counterfactual prediction
- `datasets/` — README with data download links (H5 files not stored in repo)
- `Outputs/` — Example output files

## Common commands

All commands run from the repo root. Data goes in `datasets/`.

**Perturb-map — train:**
```bash
python src/run_concert_map.py \
  --config src/config.yaml \
  --sample GSM5808054 \
  --project_index map \
  --stage train \
  --data_file datasets/GSM5808054_data.h5 \
  --model_file model_map_GSM5808054.pt \
  --multi_kernel_mode True \
  --wandb --wandb_project concert-map --wandb_run train
```

**Perturb-map — counterfactual eval:**
```bash
python src/run_concert_map.py \
  --config src/config.yaml \
  --sample GSM5808054 \
  --project_index map \
  --stage eval \
  --data_file datasets/GSM5808054_data.h5 \
  --model_file model_map_GSM5808054.pt \
  --multi_kernel_mode True \
  --pert_cells select_cells/pert_cells_GSM5808054_patchclose_tumor_Jak2.txt \
  --target_cell_tissue tumor \
  --target_cell_perturbation Jak2
```

Other entry points follow the same pattern (config-first, CLI overrides):
- `src/run_concert_map_impute.py` — Perturb-map with imputation + CP
- `src/CONCERT-3D/run_concert_2D_stroke.py` / `run_concert_3D_stroke.py` — Stroke data
- `src/CONCERT-long/run_concert_gut.py` — Gut inflammation
- `src/single_kernel/run_concert_sk.py` — Single kernel ablation

## Architecture

### GP-VAE backbone (`concert_map.py` → `CONCERT` class)

The latent space has two blocks:
- **GP block** (`GP_dim`, default 2): Spatially structured latents via Sparse Variational GP (SVGP). Each GP latent dimension has its own variational posterior with inferred mean/variance at every spatial location.
- **Gaussian block** (`Normal_dim`, default 8): Non-spatial (diagonal Gaussian) latent effects.

### LORD encoder (`lord_batch.py` → `LordEncoder`)

Latent Orthogonal Representation Disentanglement — produces three embeddings per cell:
- **Basal latent**: Per-sample learned embedding (noise-regularized).
- **Tissue latent**: Categorical embedding for tissue type.
- **Perturbation latent**: Categorical embedding for perturbation identity.

These are concatenated into `total_latent` and fed into a `DenseEncoder` that produces posterior mean/variance for the combined GP+Gaussian latents.

### SVGP module (`SVGP_Batch.py` → `SVGP`)

Sparse variational GP using inducing points (grid or KMeans). The multi-kernel mode uses `BatchedCauchyKernel_CONCERT_flex` — a batched Cauchy kernel with per-perturbation learnable length scales. A learnable mask cutoff (`mask_cutoff`) controls spatial propagation scope per cell via a sigmoid gate on the kernel matrix.

### Decoder (`VAE_utils.py`)

Dense layers → `MeanAct` (clamped exp for NB mean) + NB dispersion (shared or batch-specific). Loss is negative binomial reconstruction loss.

### Dynamic beta scheduling (`I_PID.py` → `PIDControl`)

PID controller automatically adjusts the KL divergence weight (beta) during training to keep KL near a target value, preventing posterior collapse.

### Data flow

1. Load H5 file (`X`, `pos`, `perturbation`, optionally `tissue`)
2. Build tissue/perturbation integer codes; create one-hot batch vectors
3. MinMax-scale positions to `[0, loc_range]`, append one-hot batch columns → `pos_batched`
4. Build inducing points (grid or KMeans, with batch one-hot blocks)
5. Normalize counts via scanpy (size factors + log1p + scale)
6. Train CONCERT with early stopping on validation ELBO
7. For counterfactuals: overwrite selected cells' attributes, re-encode through LORD, decode with GP+Gaussian latents

### Variant differences

| Variant | Kernel | Batch encoding | Key difference |
|---------|--------|---------------|----------------|
| `src/` (Perturb-map) | Multi-kernel Cauchy with cutoff gate | One-hot per perturbation | Original full model |
| `CONCERT-3D/` | 2D or 3D Cauchy with batch scales | One-hot per batch | Separate encoder (no LORD), stroke-specific preprocessing |
| `CONCERT-long/` | Single Cauchy kernel | Ordinal (time) | Time as continuous attribute, no multi-kernel |
| `single_kernel/` | Single Cauchy kernel | None | Ablation: shared kernel across all perturbations |

## Key configuration patterns

- YAML/JSON config files set defaults; CLI flags override only explicitly passed arguments
- `--config` points to the YAML file; other `--flags` take precedence
- `--batch_size "auto"` auto-selects 128/256/512 based on dataset size
- `--multi_kernel_mode True` enables perturbation-specific kernel length scales
- W&B logging is opt-in via `--wandb`
- `--stage` is either `train` or `eval`

## Data format

H5 files with datasets: `X` (N×G counts), `pos` (2×N), `perturbation` (N strings), optionally `tissue` (N strings). The stroke variant uses `Pos`, `Batch`, `Celltype_coarse`, and `Barcode`.
