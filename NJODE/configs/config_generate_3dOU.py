"""
Author: Azamat Zhaksylykov
configs for 3D OU process 
"""

 
import numpy as np

from configs.config_utils import get_parameter_array, data_path, training_data_path

 

# Paths
data_path = data_path
training_data_path = training_data_path

 
# Dataset parameters for Z 
vol_dataset_dict = {
    'volatility': np.array([[0.2, 0.1, 0.1], [0.1, 0.25, 0.1], [0.1, 0.1, 0.3]]).tolist(),
    'mean': np.array([1.2, 1.0, 1.5]).tolist(),
    'speed': np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]).tolist(),
    'nb_paths': 10000,
    'nb_steps': 100,
    'S0': np.array([1.0, 1.5, 2.0]).tolist(),
    'maturity': 1.0,
    'dimension': 3,
    'obs_perc': 1.0,
    'scheme': 'euler',
    'model_name': 'OrnsteinUhlenbeckMulti_Z',
    'dt': 0.01
}

 

# Dataset parameters for X
mu_dataset_dict = {
    'volatility': np.array([[0.2, 0.1, 0.1], [0.1, 0.25, 0.1], [0.1, 0.1, 0.3]]).tolist(),
    'mean': np.array([1.2, 1.0, 1.5]).tolist(),
    'speed': np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]).tolist(),
    'nb_paths': 10000,
    'nb_steps': 100,
    'S0': np.array([1.0, 1.5, 2.0]).tolist(),
    'maturity': 1.0,
    'dimension': 3,
    'obs_perc': 1.0,
    'scheme': 'euler',
    'model_name': 'OrnsteinUhlenbeckMultiDimensional',
    'dt': 0.01
}

 

 

# Train parameters for Z
vol_param_dict = {
    'which_loss': ["easy_vol"],
    'dataset': ['OrnsteinUhlenbeckMulti_Z'],
    'use_cond_exp': [False],
    'eval_use_true_paths': [True],
    'plot': [True],
    'paths_to_plot': [(0,)],
}

vol_params_list = get_parameter_array(
    param_dict=vol_param_dict)

 

# Train parameters for X
mu_param_dict = {
    'which_loss': ["easy"],
    'dataset': ['OrnsteinUhlenbeckMultiDimensional'],
    'plot': [True],
    'paths_to_plot': [(0, )],
}

mu_params_list= get_parameter_array(
    param_dict=mu_param_dict)

 

 

# Model parameters for mu_model
mu_params_dict = {
    'input_size': 3,
    'hidden_size': 10,
    'output_size': 3,
    'ode_nn': ((50, "tanh"), (50, "tanh")),
    'readout_nn': ((50, "tanh"), (50, "tanh")),
    'enc_nn': ((50, "tanh"), (50, "tanh")),
    'use_rnn': False,
    'options': {'which_loss': 'easy'},
    "input_coords": np.arange(3),
    "output_coords": np.arange(3),
    "signature_coords": np.arange(3)
}

 

# Model parameters for vol_model
vol_params_dict = {
    'input_size': 9,
    'hidden_size': 10,
    'output_size': 9,
    'ode_nn': ((50, "tanh"), (50, "tanh")),
    'readout_nn': ((50, "tanh"), (50, "tanh")),
    'enc_nn': ((50, "tanh"), (50, "tanh")),
    'use_rnn': False,
    'options': {'which_loss': 'easy_vol'},
    "input_coords": np.arange(9),
    "output_coords": np.arange(9),
    "signature_coords": np.arange(9)
}
