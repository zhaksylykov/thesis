# NJ-ODE as Generative Framework

This repository contains the code used for my master's thesis.

## Thesis Context

The goal is to adapt the NJ-ODEs framework ([floriankrach.github.io/njode](https://floriankrach.github.io/njode)), which was originally developed for prediction, into a generative model.

- **Industry supervision:** Zürcher Kantonalbank (ZKB)
- **Academic supervision:** ETH Zurich
- **Original codebase:** [FlorianKrach/PD-NJODE](https://github.com/FlorianKrach/PD-NJODE)
- *Note: Some files irrelevant to this thesis have been removed for clarity and focus.*

---

## What’s Different in This Repository. 

- **Added configs:** 3 configs files added for 1d OU, 3d OU, and 1d GMB process 
- **Z process datasets** (`Z_t = (X_{\tau(t)} - X_t)(X_{\tau(t)} - X_t)^T`)
- **New loss functions** It is named "easy_vol"
- **`generate.py`**: The main script for producing new synthetic time series using trained NJ-ODE models

---


