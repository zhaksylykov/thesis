# NJ-ODE as Generative Framework

This repository contains the code used for my master's thesis.

## Thesis Context

The goal is to adapt the NJ-ODEs framework ([floriankrach.github.io/njode](https://floriankrach.github.io/njode)), which was originally developed for prediction, into a generative model.

- **Academic supervision:** ETH Zurich
- **Industry supervision:** Zürcher Kantonalbank (ZKB)
- **Original codebase:** [FlorianKrach/PD-NJODE](https://github.com/FlorianKrach/PD-NJODE)
- *Note: Some files irrelevant to this thesis have been removed for clarity and focus.*

---

## What’s Different in This Repository. 

- **Added configs:** 3 configs files added for 1d OU, 3d OU, and 1d GMB process 
- **Z process datasets generation** Generation of datasets from (`$Z_t = (X_{\tau(t)} - X_t)(X_{\tau(t)} - X_t)^T$`)
- **New loss functions** Introduced the "easy_vol" loss
- **`generate.py`**: The main script for producing new synthetic time series using trained NJ-ODE models
- **Jupyter notebooks**: Includes notebooks with all experimental test runs.
- **Thesis PDF**: The PDF version of the master’s thesis is included in the repository.

---  
## Requirements. 
Same as in the original codebase, see [FlorianKrach/PD-NJODE/README](https://github.com/FlorianKrach/PD-NJODE?tab=readme-ov-file) 

---   
## Usage, License & Citation.  
Same as in the original codebase, see [FlorianKrach/PD-NJODE/README](https://github.com/FlorianKrach/PD-NJODE?tab=readme-ov-file) 

---   
## Acknowledgements and References 
Same as in the original codebase, see [FlorianKrach/PD-NJODE/README](https://github.com/FlorianKrach/PD-NJODE?tab=readme-ov-file)  

## Instructions for generating synthetic data as in the thesis 

Experiments were conducted using 3 processes, specifically 1-dimensional Geometric Brownian motion, 1-dimensional Ornstein-Uhlenbeck, and 3-dimensional Ornstein-Uhlenbeck processes. Configs for them can be found in the following folder: `NJODE/configs/`.

### How to run synthetic data generation

Before generating synthetic data, you must first choose the appropriate config file for your experiment.  
To do this, update the import path inside `generate.py` to point to the correct configuration (e.g., `1dBS`, `1dOU`, or `3dOU`).

---

#### If no trained model exists

You need to follow two steps:

**Step 1: Create dataset and train NJ-ODE models**

```bash
python generate.py --task=train
```
This will: 

- Generates a synthetic dataset based on the selected process
- Trains NJ-ODE models for both drift (`mu`) and volatility (`vol`)
- Saves the trained model checkpoints inside `data/saved_models/

#### Step 2: Generate synthetic data using trained models

After training, you need to know the ID numbers of the saved models for the `X` and `Z` processes.  
These IDs correspond to folder names inside the `data/saved_models/` directory.

Then run:

```bash
python generate.py --task=generate --n_paths=1000 --mu_model_ckpt=1 --vol_model_ckpt=2 --output_name=synthetic_data.npy
```

Replace `1` and `2` with the actual checkpoint folder names for `mu` and `vol` models. 

---

### If trained models already exist

If the NJ-ODE models have already been trained and saved, you can directly generate synthetic data using:

```bash
python generate.py --task=generate --n_paths=1000 --mu_model_ckpt=1 --vol_model_ckpt=2 --output_name=synthetic_data.npy
```

Make sure to:
- Set the correct config import inside `generate.py`
- Use the correct IDs for the model checkpoints found in `data/saved_models/`

You can change the number of synthetic paths by adjusting the `--n_paths` argument. The name of the saved output file can be changed by adjusting the `--output_name=`. 

### Note 
For smaller experiments, I found it convenient to use the notebook [explainability_njodes.ipynb](https://gist.github.com/FlorianKrach/7a610cd88d9739b2f8bbda8455a558b4).  
Experiments run using this notebook are  inside the `Method I` and `Method II` folders in this repository.



