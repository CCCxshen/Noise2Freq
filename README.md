<div style="width: 100%;
            text-align:center;" > 
    <div style="width: 100%; height: 100px;"></div>
    <h1 style = "font-size: 80px;"> Noise2Freq </h1>
    <span>
        <b>Noise2Freq: A Frrequency-Aware Self-Supervised Framework with Dual-Decoder Structure for LDCT Denoising<b>		
    </span>
</div>
<div style="page-break-after: always;"></div>



<div style="page-break-after: always;"></div>

# Project Description

Only the inference code for the method is currently available.



# Data preprocessing instructions

Using Mayo-2016 dataset with data range clipped to [-1024, 1000] and test data IDs: L333, L506.

The data is stored according to slices in the format np.ndarray two-dimensional array of size (512*512).

Naming Convention：$[ID]\_[slice\_index]\_input/target.npy$

Stored as follows：

```bash
/mayo-2016/
	L333_1_input.npy
	L333_1_target.npy
	...
	L506_[final_Slice_index]_input.npy
	L506_[final_Slice_index]_target.npy
	
```



# Inference

1. Change the `data_dir` in $inference.py$ on line 35 to the absolute path where you store the mayo-2016 dataset.
2. Modify the `reference_version` variable in $inference.py$ on line 29.
   1. `reference_version == mayo_ldct`：Reproducing the results of training with `ldct only`.
   2. `reference_version == mayo_ldct`：Reproducing the results of training with `ndct only`.
3. Run the $inference.py$ file and the results will be stored in the `inference` folder.



# Pre-training Weight

Pre-training weights are stored in the `result/*/checkpoints/` folder.



