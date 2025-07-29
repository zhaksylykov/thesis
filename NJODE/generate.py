"""

author:  Azamat Zhaksylykov

 

code for generating synthetic data

"""

 

import numpy as np

import torch

import gc

from absl import app, flags

from configs import config_generate1dOU as config_generate

import train

import data_utils

from models import get_ckpt_model, NJODE

 

FLAGS = flags.FLAGS

 

flags.DEFINE_string("task", None, "Task to perform: 'train' or 'generate'")

flags.DEFINE_integer("n_paths", None, "Number of paths to generate'")

#flags.DEFINE_string("mu_dataset_param", None, "Dataset param for mu_model")

#flags.DEFINE_string("vol_dataset_param", None, "Dataset param for vol_model")

flags.DEFINE_integer("mu_model_ckpt", None, "Checkpoint path number for mu_model")

flags.DEFINE_integer("vol_model_ckpt", None, "Checkpoint path number for vol_model")

 

def generate_next_value( X_t, mu_t, sigma_t, delta_t):

    """

    Generate the next value in the time series using the Euler-Maruyama scheme.

 

    :param X_t: current value tensor of shape (batch_size, d)

    :param mu_t: drift coefficient tensor of shape (batch_size, d)

    :param sigma_t: diffusion coefficient tensor of shape ( d, d)

    :param delta_t: time difference float

    :return: next value tensor of shape (batch_size, d)

    """

    delta_Wt = torch.randn_like(X_t) * np.sqrt(delta_t)

    delta_Wt_sigma_t = torch.bmm(delta_Wt.unsqueeze(1), sigma_t).squeeze(1)

    X_t_next = X_t + mu_t * delta_t + delta_Wt_sigma_t

 

    return X_t_next

 

def generate_1k_paths(n=500):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 

    # Load trained models

    model_mu = NJODE(**config_generate.mu_params_dict).to(device)

    optimizer = torch.optim.Adam(model_mu.parameters())

    get_ckpt_model(f"../data/saved_models/id-{FLAGS.mu_model_ckpt}/best_checkpoint/", model_mu, optimizer, device)

    #get_ckpt_model(f"../data/saved_models/id-{80}/best_checkpoint/", model_mu, optimizer, device)

    model_mu.eval()

 

    model_vol = NJODE(**config_generate.vol_params_dict).to(device)

    optimizer_vol = torch.optim.Adam(model_vol.parameters())

    get_ckpt_model(f"../data/saved_models/id-{FLAGS.vol_model_ckpt}/best_checkpoint/", model_vol, optimizer_vol, device)

    #get_ckpt_model(f"../data/saved_models/id-{81}/best_checkpoint/", model_vol, optimizer_vol, device)

    model_vol.eval()

 

    # Generate synthetic data in parallel

    batch_size=n

    dim=config_generate.mu_dataset_dict["dimension"]

    times=np.array([])

    time_ptr = np.array([0])

    obs_idx = torch.tensor([],dtype=torch.long)

    start_Z = torch.tensor(np.zeros((1,)), dtype=torch.float).unsqueeze(0)

    start_Z = start_Z.repeat(batch_size, dim)

    Z = torch.tensor([], dtype=torch.float)

    delta_t=0.01

    X= torch.tensor([], dtype=torch.float)

    start_X = torch.tensor([config_generate.mu_dataset_dict["S0"]], dtype=torch.float).unsqueeze(0)

    start_X = start_X.repeat(batch_size, dim)

    n_obs_ot=torch.tensor([0]*batch_size, dtype=torch.float)

 

    for i in range(1, 101):

        T=delta_t*i

        if i==100:

            print(i)

        mu_pred=model_mu.get_pred(times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot)["pred"][-1]

 

        if i==1:

            X_t=start_X

        elif i==2:

            X_t = X_next

            X_t_minus = start_X

            # Compute the difference

            diff = X_t - X_t_minus

 

            # Compute the outer product for each element in the batch

            # diff.unsqueeze(2) has shape (batch_size, dim, 1)

            # diff.unsqueeze(1) has shape (batch_size, 1, dim)

            Z_tensor = diff.unsqueeze(2) @ diff.unsqueeze(1)

            Z=torch.cat((Z,Z_tensor.view(batch_size,-1)))

 

        else:

            X_t = X_next

            X_t_minus = X[batch_size*(i-3):batch_size*(i-2), :]

            # Compute the difference

            diff = X_t - X_t_minus

 

            # Compute the outer product for each element in the batch

            # diff.unsqueeze(2) has shape (batch_size, dim, 1)

            # diff.unsqueeze(1) has shape (batch_size, 1, dim)

            Z_tensor = diff.unsqueeze(2) @ diff.unsqueeze(1)

            Z=torch.cat((Z,Z_tensor.view(batch_size,-1)))

 

 

 

        sigma_pred=model_vol.get_pred(times, time_ptr, Z, obs_idx, delta_t, T, start_Z, n_obs_ot )["pred"][-1]

        mu_hat_t = (mu_pred - X_t)/delta_t

        sigma_hat_t = sigma_pred.view(batch_size, dim, dim)/np.sqrt(delta_t)

 

 

        X_next = generate_next_value(X_t, mu_hat_t, sigma_hat_t, delta_t)

 

        X= torch.cat((X, X_next), dim=0)

        times=np.append(times, delta_t*i)

        time_ptr = np.append(time_ptr, i*batch_size)

        obs_idx=torch.cat((obs_idx, torch.arange(batch_size)))

        n_obs_ot+=torch.tensor([1]*batch_size, dtype=torch.float)

 

    return X.view(100, batch_size, dim).permute(1, 2, 0).detach()

 

 

def generate():

    paths = [generate_1k_paths() for i in range(FLAGS.n_paths)]

    combined = torch.cat(paths, dim=0)

    X_final=torch.cat((torch.ones(FLAGS.n_paths*1000,1,1, dtype=torch.flaot), combined), dim =2)

 

 

    output_dir = "../data/generated_data/"

    data_utils.makedirs(output_dir)

    with open('{}data_ou3.npy'.format(output_dir), 'wb') as f:

        np.save(f, X_final.detach().numpy())

 

def main(argv):

    del argv

    if FLAGS.task == "train":

        # Train models

        _, vol_dataset_id= data_utils.create_dataset(config_generate.vol_dataset_dict["model_name"],config_generate.vol_dataset_dict )

        train.train(**config_generate.vol_param_dict, dataset_id=vol_dataset_id)

 

        _, mu_dataset_id = data_utils.create_dataset(config_generate.mu_dataset_dict["model_name"], config_generate.mu_dataset_dict)

        train.train(**config_generate.mu_param_dict, dataset_id=mu_dataset_id)

 

    elif FLAGS.task == "generate":

        # Generate paths

        generate()

if __name__ == '__main__':

    flags.mark_flag_as_required("task")

    app.run(main)
