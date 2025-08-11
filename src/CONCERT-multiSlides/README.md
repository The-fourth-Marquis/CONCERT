CONCERT applied on gut DSS data with a single kernel for multi-slides.

1. Train model.
<pre> python run_concert_gut.py  
  --stage train  
  --data_file datasets/processed_gut_data.h5  
  --inducing_point_steps 6  
  --model_file model_gut.pt
</pre> 

2. Do perturbation prediction on the specified time points. Here we predict inflammed spots after 18 and 61 days of recovery. The arguments `--pert_cells` and `--target_cell_day` indicate the day of data to perturb and the target day of transcriptomic state.
<pre> python run_concert_gut.py  
  --stage infer  
  --data_file datasets/processed_gut_data.h5  
  --inducing_point_steps 6  
  --model_file model_gut.pt  
  --pert_cells D12  
  --target_cell_day 30.

  python run_concert_3D_stroke.py  
  --stage infer  
  --data_file datasets/processed_gut_data.h5  
  --inducing_point_steps 6  
  --model_file model_gut.pt  
  --pert_cells D12 
  --target_cell_day 73.
</pre> 
