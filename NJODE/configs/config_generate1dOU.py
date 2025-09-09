"""
Author: Azamat Zhaksylykov
congifs for 1-d Ornstein-Uhlenbeck process 
"""

# Dataset parameters for the OU process X_t
mu_dataset_dict = {
    'model_name': "OrnsteinUhlenbeck",
    'nb_paths': 10000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1, 'mean': 4, 'volatility': 0.3, 'speed': 2, 'S0': 1.,
}

# Dataset parameters for the Z-process 
vol_dataset_dict = {
    'model_name': "OrnsteinUhlenbeck",
    'nb_paths': 10000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'mean': 4, 'volatility': 0.3, 'speed': 2, 'S0': 1.,
}

# --------------------------------------------------------------------------
# Configuration for the Drift Estimator Model
# --------------------------------------------------------------------------
mu_train_config = {
    'seed': 2,
    'data_train': None,
    'data_val': None,
    'dataset_metadata': None,
    'epochs': 11,
    'batch_size': 200,
    'learning_rate': 0.01,
    'hidden_size': 100,
    'bias': True,
    'dropout_rate': 0.1,
    'ode_nn': ((50, 'relu'), (50, 'relu')),
    'readout_nn': ((50, 'relu'), (50, 'relu')),
    'enc_nn': ((50, 'relu'), (50, 'relu')),
    'dataset_id': None,
    'which_loss': 'easy',
    'use_y_for_ode': True,
    'use_rnn': False,
    'input_sig': False,
    'level': 2,
    'masked': False,
    'evaluate': True,
    'compute_variance': False,
    'residual_enc_dec': True,
    'ode_input_scaling_func': "identity", 
}

# --------------------------------------------------------------------------
# Configuration for the Volatility Estimator Model
# --------------------------------------------------------------------------
vol_train_config = {
    'seed': 2,
    'data_train': None,
    'data_val': None,
    'dataset_metadata': None,
    'epochs': 40,
    'input_coords': [0, 2],
    'output_coords': [1],
    'batch_size': 200,
    'learning_rate': 0.001,
    'hidden_size': 100,
    'bias': True,
    'dropout_rate': 0.1,
    'ode_nn': ((50, 'relu'),),
    'readout_nn': ((50, 'relu'),),
    'enc_nn': ((50, 'relu'),),
    'dataset_id': None,
    'which_loss': 'vola',
    'use_y_for_ode': False,
    'use_rnn': False,
    'input_sig': False,
    'level': 2,
    'masked': False,
    'evaluate': False,
    'use_cond_exp': True,
    'compute_variance': False,
    'residual_enc_dec': True,
    'ode_input_scaling_func': "identity",
    'input_var_t_helper': True,
}