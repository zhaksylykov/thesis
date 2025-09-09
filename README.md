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

- **Added configs:** 2 configs files added for 1d OU, and 1d GMB process 
- **Z process datasets** Generation of datasets from (`Z_t = (X_t - X_{τ(t)})(X_t - X_{τ(t)})^T`)
- **New loss functions** Introduced the "vola" loss function 
- **`train_gen.py`** A new file was created for the train function, and minor updates were made to other files.
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

---
## Instructions for generating synthetic data as in the thesis 

Experiments were conducted using 2 processes, specifically 1-dimensional Geometric Brownian motion and 1-dimensional Ornstein-Uhlenbeck processes. Configs for them can be found in the following folder: `NJODE/configs/`.

---
## How to Run

The refactored script (`generate.py`) automates the entire pipeline. In a single run, it will:
1.  Load the dataset configuration.
2.  Train the NJODE model for the drift (`mu`).
3.  Train the NJODE model for the volatility (`sigma`).
4.  Immediately use these newly trained models to generate the synthetic data paths.

---

### Usage Examples

Here are the primary ways to use the script.

#### Scenario 1: Standard Generation 

This is the most common use case. It trains the models and generates a new set of paths from t = 0.

```bash
python generate.py
```
This command will use the default settings:

* Generate **10,000** paths (`--n_paths=10000`).
* Start generation from scratch (`--start_index=1`).
* Save the output to the `../data/generated_data/` directory.

#### Scenario 2: Conditional Generation
This feature enables the generation of paths from a chosen point in the historical data, with full control over the number of paths and the save location.

```bash
python generate.py --start_index 50 --n_paths 500 --output_dir ./results/ 
```
This command will use the following settings:

* `--n_paths`: (Integer) Specifies the total number of synthetic paths to generate.
* `--start_index`: (Integer) Sets the time step index from which to begin generation. A value of `1` starts the generation from t = 0. 
* `--output_dir`: (String) Defines the directory path where the resulting `.npy` file containing the generated paths will be saved. 


### Note 
For smaller experiments, I found it convenient to use the notebook [explainability_njodes.ipynb](https://gist.github.com/FlorianKrach/7a610cd88d9739b2f8bbda8455a558b4).  
Experiments run using this notebook are  inside the `Method I` and `Method II` folders in this repository.



