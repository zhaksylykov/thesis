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
- **Z process datasets generation** Generation of datasets from (`Z_t = (X_{$\tau(t)$} - X_t)(X_{$\tau(t)$} - X_t)^T`)
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

Experiments were conducted using 3 processes, speciffically 1-dimensional Geometric Brownian motion, 1-dimensional Ornstein-Uhlenbeck, 3-dimensional Ornstein-Uhlenbeck process. Configs for them can be found in the following folder NJODE/configs/.  

To generate synthetic data, first you have to create dataset from known process then train NJ-ODE model. Then you can 








