## Scripts to run CONCERT on Perturb-Map data  
run_concert_map.py - do counterfactual prediction on specified spots  
run_concert_map_impute.py - do imputation and imputation + counterfactual prediction on specified unseen spots

### Example for imputation:
<pre> python -u run_concert_map_impute.py \
--data_file datasets/GSM5808054_data.h5 \
--sample GSM5808054 \
--data_index impute_spots \
--pert_cells select_cells/GSM5808054_impute_spots.txt \
--model_file model.pt \
--target_cell_tissue tumor \
--target_cell_perturbation Jak2
<pre>
