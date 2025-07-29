"""
Author: Azamat Zhaksylykov
congifs for 1-d Ornstein-Uhlenbeck process 
"""

import numpy as np
from configs.config_utils import get_parameter_array, data_path, training_data_path


# Paths
data_path = data_path
training_data_path = training_data_path

# Dataset parameters for Z_t 
vol_dataset_dict = {
    'model_name': 'OrnsteinUhlenbeckZ',
    'volatility': 0.3,
    'mean': 1.5,
    "speed" : 0.3,
    'nb_paths': 10000,
    'nb_steps': 100,
    'S0': 2,
    'maturity': 1.,
    'dimension': 1,
    'obs_perc': 1.0,
    'scheme': 'euler',
    'return_vol': False,
    'v0': 1,
    'hurst':0.75,
    'FBMmethod':"daviesharte"
}


# Dataset parameters for X_t
mu_dataset_dict = {
    'model_name': 'OrnsteinUhlenbeck',
    'volatility': 0.3,
    'mean': 1.5,
    "speed" : 0.3,
    'nb_paths': 10000,
    'nb_steps': 100,
    'S0': 2,
    'maturity': 1.,
    'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler',
    'return_vol': False,
    'v0': 1,
    'hurst':0.75,
    'FBMmethod':"daviesharte"
}

 
# Train parameters for Z_t
vol_param_dict = {
    'which_loss': "easy_vol",
    'dataset': 'OrnsteinUhlenbeckZ',
    'use_cond_exp': False,
    'eval_use_true_paths': True,
    'plot': True,
    'paths_to_plot': (0,),
}


# Train parameters for X_t
mu_param_dict = {
    'which_loss': "easy",
    'dataset': 'OrnsteinUhlenbeck',
    'plot': True,
    'paths_to_plot': (0, ),
}


# Model parameters for mu_model
mu_params_dict = {
    'input_size': 1,
    'hidden_size': 10,
    'output_size': 1,
    'ode_nn': ((50, "tanh"), (50, "tanh")),
    'readout_nn': ((50, "tanh"), (50, "tanh")),
    'enc_nn': ((50, "tanh"), (50, "tanh")),
    'use_rnn': False,
    'options': {'which_loss': 'easy'},
    "input_coords": np.arange(1),
    "output_coords": np.arange(1),
    "signature_coords": np.arange(1)
}

# Model parameters for vol_model
vol_params_dict = {
    'input_size': 1,
    'hidden_size': 10,
    'output_size': 1,
    'ode_nn': ((50, "tanh"), (50, "tanh")),
    'readout_nn': ((50, "tanh"), (50, "tanh")),
    'enc_nn': ((50, "tanh"), (50, "tanh")),
    'use_rnn': False,
    'options': {'which_loss': 'easy_vol'},
    "input_coords": np.arange(1),
    "output_coords": np.arange(1),
    "signature_coords": np.arange(1)
}
