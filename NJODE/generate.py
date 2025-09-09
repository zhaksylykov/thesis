"""
Main script for training NJODE models and generating synthetic data.

This script provides a comprehensive workflow for:
1.  Training the drift (mu) and volatility (sigma) models based on the
    Neural Jump Ordinary Differential Equation (NJODE) framework.
2.  Generating synthetic time-series paths using the trained models.

Author: Azamat Zhaksylykov
"""

# =====================================================================================
# Imports
# =====================================================================================
import argparse
import os
from typing import Tuple, Dict, Any

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

import configs.config_generate1dBS as config
from data_utils import (create_dataset, create_dataset_Z_minus_n_plus_n_X,
                        IrregularDataset, makedirs )
from models import NJODE
from train_gen import train

# =====================================================================================
# Constants
# =====================================================================================
RANDOM_SEED = 1
TEST_SPLIT_RATIO = 0.2


# =====================================================================================
# Generation Functions
# =====================================================================================

def generate_next_value(X_t: torch.Tensor, mu_t: torch.Tensor,
                        sigma_t: torch.Tensor, delta_t: float) -> torch.Tensor:
    """
    Generates the next value in a time series using the Euler-Maruyama scheme.

    Args:
        X_t (torch.Tensor): Current value tensor of shape (batch_size, dim).
        mu_t (torch.Tensor): Drift coefficient tensor of shape (batch_size, dim).
        sigma_t (torch.Tensor): Diffusion coefficient tensor of shape (batch_size, dim, dim).
        delta_t (float): Time step increment.

    Returns:
        torch.Tensor: The next value in the series, with shape (batch_size, dim).
    """

    delta_Wt = torch.randn_like(X_t) * np.sqrt(delta_t)
    delta_Wt_sigma_t = torch.bmm(delta_Wt.unsqueeze(1), sigma_t.transpose(1, 2)).squeeze(1)
    X_t_next = X_t + mu_t * delta_t + delta_Wt_sigma_t

    return X_t_next




def generate_synthetic_paths(
    n_paths: int, model_mu: NJODE, model_vol: NJODE, output_dir: str,
    start_index: int, historical_data: np.ndarray
) -> None:
    """
    Generates synthetic time series paths using trained drift and volatility models.

    The generation can start from scratch (t=0) or be conditioned on a provided
    historical segment up to a specified `start_index`.

    Args:
        n_paths (int): The number of synthetic paths to generate.
        model_mu (NJODE): The trained drift model.
        model_vol (NJODE): The trained volatility model.
        output_dir (str): Directory to save the generated paths.
        start_index (int): The time step index to start generating new points from.
                           If set to 1, generation starts from t=0.
        historical_data (np.ndarray): A numpy array of shape (n_paths, dim, n_steps)
                                      containing the historical path segments.
    """
    print(f"Generating {n_paths} paths starting from step {start_index}...")

    # --- 1. Configure Generation Parameters ---
    batch_size = n_paths
    mu_conf = config.mu_dataset_dict 
    dim = mu_conf["dimension"]
    n_steps = mu_conf["nb_steps"]
    delta_t = mu_conf['maturity'] / n_steps

    # --- 2. Initialize State and History ---
    # `start_index` <= 1 means starting from scratch at t=0.
    # `start_index` > 1 means continuing from a historical path.
    if start_index > 1:
        start_X = torch.tensor(historical_data[:batch_size, :, 0], dtype=torch.float)
        X_t = torch.tensor(historical_data[:batch_size, :, start_index - 1], dtype=torch.float)
    else:
        start_X = torch.ones((batch_size, dim), dtype=torch.float)
        X_t = start_X

    history_steps = max(0, start_index - 1)
    
    times = np.arange(delta_t, (history_steps + 1) * delta_t, delta_t)
    time_ptr = np.arange(0, (history_steps + 1) * batch_size, batch_size)
    obs_idx = torch.arange(batch_size).repeat(history_steps)
    n_obs_ot = torch.full((batch_size,), float(history_steps), dtype=torch.float)

    if history_steps > 0:
        X_hist_seq = torch.tensor(historical_data[:batch_size, :, 1:start_index], dtype=torch.float)
        X = X_hist_seq.permute(2, 0, 1).reshape(history_steps * batch_size, dim)
    else:
        X = torch.tensor([], dtype=torch.float)

    start_ZX = torch.cat((torch.zeros_like(start_X), torch.zeros_like(start_X), start_X), dim=1)

    for i in tqdm.tqdm(range(start_index, n_steps + 1), desc="Generating Path"):
        current_time = delta_t * i
        
        mu_pred = model_mu.get_pred(times, time_ptr, X, obs_idx, delta_t, current_time, start_X, n_obs_ot)["pred"][-1].detach()
        
        vol_time = delta_t * (i - 1) + (delta_t * 3) 
        ZX = torch.cat((torch.zeros_like(X), torch.zeros_like(X), X), dim=1)
        sigma_pred = model_vol.get_pred(times, time_ptr, ZX, obs_idx, delta_t, vol_time, start_ZX, n_obs_ot)["pred"][-1].detach()

        mu_hat_t = (mu_pred - X_t) / delta_t
        sigma_hat_t = sigma_pred.view(batch_size, dim, dim)

        X_next = generate_next_value(X_t, mu_hat_t, sigma_hat_t, delta_t).detach()

        X_t = X_next
        X = torch.cat((X, X_next), dim=0)
        times = np.append(times, current_time)
        time_ptr = np.append(time_ptr, i * batch_size)
        obs_idx = torch.cat((obs_idx, torch.arange(batch_size)))
        n_obs_ot += 1.0

    X_generated_seq = X.view(n_steps, batch_size, dim).permute(1, 2, 0).detach()

    X_final = torch.cat((start_X.unsqueeze(2), X_generated_seq), dim=2)

    makedirs(output_dir)
    output_file = os.path.join(output_dir, f'generated_paths_start_{start_index}.npy')
    with open(output_file, 'wb') as f:
        np.save(f, X_final.numpy())
    print(f"Generated data successfully saved to: {output_file}")


# =====================================================================================
# Helper Functions for Main Execution
# =====================================================================================

def _setup_arg_parser() -> argparse.ArgumentParser:
    """Configures and returns the argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Train NJODE models and generate synthetic time-series data."
    )
    parser.add_argument(
        "--n_paths",
        type=int,
        default=10000,
        help="Number of synthetic paths to generate."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Time step index to start generation from. "
             "Set to 1 to generate from scratch (t=0). "
             "Set > 1 to condition on a historical path up to this index."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/generated_data/",
        help="Directory to save the generated data."
    )
    return parser


def _train_drift_model(
    app_config: Any, seed: int
) -> Tuple[NJODE, Dict[str, Any]]:
    """Loads data and trains the drift (mu) model."""
    dataset = create_dataset(
        app_config.mu_dataset_dict["model_name"],
        app_config.mu_dataset_dict,
        seed=0
    )
    dataset_metadata = dataset[-2]

    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]),
        test_size=TEST_SPLIT_RATIO,
        random_state=seed
    )

    data_train = IrregularDataset(dataset, idx=train_idx)
    data_val = IrregularDataset(dataset, idx=val_idx)

    app_config.mu_train_config['data_train'] = data_train
    app_config.mu_train_config['data_val'] = data_val
    app_config.mu_train_config['dataset_metadata'] = dataset_metadata

    model, _, _, _, _, _, _ = train(**app_config.mu_train_config)
    return model, dataset


def _train_volatility_model(
    app_config: Any, seed: int
) -> NJODE:
    """Loads data and trains the volatility (sigma) model."""
    dataset_components = create_dataset_Z_minus_n_plus_n_X(
        app_config.vol_dataset_dict["model_name"],
        app_config.vol_dataset_dict,
        seed=0,
        divide_by_t=True
    )
    (Z_minus_flat, Z_plus_flat, x_proc, obs_dates, nb_obs, metadata, noise) = dataset_components

    Z_X = np.concatenate((Z_plus_flat, Z_minus_flat, x_proc), axis=1)
    dataset_vol = (Z_X, obs_dates, nb_obs, metadata, noise)

    train_idx, val_idx = train_test_split(
        np.arange(metadata["nb_paths"]),
        test_size=TEST_SPLIT_RATIO,
        random_state=seed
    )

    data_vol_train = IrregularDataset(dataset_vol, idx=train_idx)
    data_vol_val = IrregularDataset(dataset_vol, idx=val_idx)
    if metadata['model_name'] == "BlackScholes":
        metadata['model_name'] = "BlackScholes_Z_CV"
    else: 
        metadata['model_name'] = "OrnsteinUhlenbeck_Z_CV"

    app_config.vol_train_config['data_train'] = data_vol_train
    app_config.vol_train_config['data_val'] = data_vol_val
    app_config.vol_train_config['dataset_metadata'] = metadata

    model_vol, _, _, _, _, _, _ = train(**app_config.vol_train_config)
    return model_vol


# =====================================================================================
# Main Execution Block
# =====================================================================================

def main():
    """Main execution function to run the complete training and generation pipeline."""
    args = _setup_arg_parser().parse_args()

    print("--- Starting GenNJODE Workflow ---")

    # --- Step 1: Train Drift (mu) Model ---
    print("\n[Phase 1/3] Training Drift Model (mu)...")
    drift_model, historical_dataset = _train_drift_model(
        config, seed=RANDOM_SEED
    )

    # --- Step 2: Train Volatility (sigma) Model ---
    print("\n[Phase 2/3] Training Volatility Model (sigma)...")
    volatility_model = _train_volatility_model(
        config, seed=RANDOM_SEED
    )

    # --- Step 3: Generate Synthetic Data ---
    print("\n[Phase 3/3] Generating Synthetic Paths...")
    generate_synthetic_paths(
        n_paths=args.n_paths,
        model_mu=drift_model,
        model_vol=volatility_model,
        output_dir=args.output_dir,
        start_index=args.start_index,
        historical_data=historical_dataset[0]  
    )

    print("\n--- Workflow Completed Successfully ---")

if __name__ == '__main__':
    main()