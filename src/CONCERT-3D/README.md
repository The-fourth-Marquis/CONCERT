CONCERT with a single 2D/3D kernel for stroke data.
## Randomly perturbing spots or patches on healthy brain with 2D space.  
1. Train model
```python
python src/CONCERT-3D/run_concert_2D_stroke.py \
  --config src/CONCERT-3D/config_2D.yaml \
  --index random \
  --stage train \
  --data_file ../../datasets/Mouse_brain_stroke_all_data.h5 \
  --pert_cells select_cells/pert_cells_stroke_sham1_ICA.txt \
  --model_file model_stroke2d.pt \
  --lr 0.0001 \
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --pert_cell_number 20 \
  --wandb \
  --wandb_project concert-stroke-2D \
  --wandb_run train
```

2. Adjust sampling strategies as random, also change the number of perturbed spots
```python
python src/CONCERT-3D/run_concert_2D_stroke.py \
  --config src/CONCERT-3D/config_2D.yaml \
  --index random \
  --stage eval \
  --data_file ../../datasets/Mouse_brain_stroke_all_data.h5 \
  --pert_cells select_cells/pert_cells_stroke_sham1_ICA.txt \
  --model_file ../../model_stroke2d.pt \
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --pert_cell_number 20 \
  --wandb \
  --wandb_project concert-stroke-2D \
  --wandb_run eval
```

3. Adjust sampling strategies as patch, also change the number of perturbed spots
```python
python src/CONCERT-3D/run_concert_2D_stroke.py \
  --config src/CONCERT-3D/config_2D.yaml \
  --index patch \
  --stage eval \
  --data_file ../../datasets/Mouse_brain_stroke_all_data.h5 \
  --pert_cells ../../select_cells/pert_cells_stroke_sham1_ICA.txt \
  --model_file model_stroke2d.pt \
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --pert_cell_number 20 \
  --wandb \
  --wandb_project concert-stroke-2D \
  --wandb_run eval
</pre> 

## Predicting perturbations on healthy brain with 3D space.  
1. Train model
<pre> 
 python src/CONCERT-3D/run_concert_3D_stroke.py  \
  --config src/CONCERT-3D/config_3D.yaml \
  --stage train \
  --data_file ../../datasets/Mouse_brain_stroke_all_data.h5 \
  --model_file model_stroke_3D_stable.pt  \
  --wandb_project concert-stroke-3D \
  --wandb_run train
``` 

2. Do perturbation prediction on the specified spots. Here we predict stroke on healthy (sham) brains. The arguments `--pert_batch` and  `--target_cell_perturbation` indicate the slide to be perturbed and the target perturbation state (ICA: ischemia central area)  
```python
python src/CONCERT-3D/run_concert_3D_stroke.py \
  --config src/CONCERT-3D/config_3D.yaml \
  --stage eval \
  --project_index stroke_3D_eval \
  --data_file ../../datasets/Mouse_brain_stroke_all_data.h5 \
  --pert_cells ../../select_cells/pert_cells_stroke_sham1_ICA.txt \
  --model_file model_stroke_3D_stable.pt \
  --pert_batch Sham1 \
  --target_cell_perturbation ICA \
  --wandb_project concert-stroke-3D \
  --wandb_run eval
```
