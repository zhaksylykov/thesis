"""
author:  Azamat Zhaksylykov
code for generating synthetic data
"""

 
import numpy as np
import torch
import gc
from absl import app, flags
from configs import config_generate1dOU as config_generate
#from configs import config_generate3dOU as config_generate
#from configs import config_generate1dBS as config_generate
import train
import data_utils
from models import get_ckpt_model, NJODE

 

FLAGS = flags.FLAGS
flags.DEFINE_string("task", None, "Task to perform: 'train' or 'generate'")
flags.DEFINE_integer("n_paths", None, "Number of paths to generate")
flags.DEFINE_integer("mu_model_ckpt", None, "Checkpoint path number for mu_model")
flags.DEFINE_integer("vol_model_ckpt", None, "Checkpoint path number for vol_model")
flags.DEFINE_string("output_name", "synthetic_data.npy", "Name of the output .npy file")

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    delta_t = 0.01
    num_steps = 100
    batch_size = FLAGS.n_paths
    dim = config_generate.mu_dataset_dict["dimension"]
 
    # Load trained models
    model_mu = NJODE(**config_generate.mu_params_dict).to(device)
    optimizer = torch.optim.Adam(model_mu.parameters())
    get_ckpt_model(f"../data/saved_models/id-{FLAGS.mu_model_ckpt}/best_checkpoint/", model_mu, optimizer, device)
    model_mu.eval()

    model_vol = NJODE(**config_generate.vol_params_dict).to(device)
    optimizer_vol = torch.optim.Adam(model_vol.parameters())
    get_ckpt_model(f"../data/saved_models/id-{FLAGS.vol_model_ckpt}/best_checkpoint/", model_vol, optimizer_vol, device)
    model_vol.eval()

    # Generate synthetic data in parallel
    times=np.array([])
    time_ptr = np.array([0])
    obs_idx = torch.tensor([],dtype=torch.long) 
    start_Z = torch.tensor(np.zeros((1,)), dtype=torch.float).unsqueeze(0)
    start_Z = start_Z.repeat(batch_size, dim)
    Z = torch.tensor([], dtype=torch.float)
    X= torch.tensor([], dtype=torch.float)
    start_X = torch.tensor([config_generate.mu_dataset_dict["S0"]], dtype=torch.float).unsqueeze(0)
    start_X = start_X.repeat(batch_size, dim)
    n_obs_ot=torch.tensor([0]*batch_size, dtype=torch.float)

    for i in range(1, num_steps+1):
        T=delta_t*i
        mu_pred=model_mu.get_pred(times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot)["pred"][-1].detach()

        if i==1:
            X_t=start_X
        elif i==2:
            X_t = X[batch_size*(i-2):batch_size*(i-1), :]
            X_t_minus = start_X
            diff = X_t - X_t_minus
            Z_tensor = diff.unsqueeze(2) @ diff.unsqueeze(1)
            Z=torch.cat((Z,Z_tensor.view(batch_size,-1))).detach()
        else:
            X_t = X[batch_size*(i-2):batch_size*(i-1), :]
            X_t_minus = X[batch_size*(i-3):batch_size*(i-2), :]
            diff = X_t - X_t_minus
            Z_tensor = diff.unsqueeze(2) @ diff.unsqueeze(1)
            Z=torch.cat((Z,Z_tensor.view(batch_size,-1))).detach()



        sigma_pred=model_vol.get_pred(times, time_ptr, Z, obs_idx, delta_t, T, start_Z, n_obs_ot )["pred"][-1].detach()
        
        mu_hat_t = (mu_pred - X_t)/delta_t
        sigma_hat_t = sigma_pred.view(batch_size, dim, dim)/np.sqrt(delta_t)

        X_next = generate_next_value(X_t, mu_hat_t, sigma_hat_t, delta_t).detach()

        X= torch.cat((X, X_next), dim=0)
        times=np.append(times, delta_t*i)
        time_ptr = np.append(time_ptr, i*batch_size)
        obs_idx=torch.cat((obs_idx, torch.arange(batch_size)))
        n_obs_ot+=torch.tensor([1]*batch_size, dtype=torch.float)

    paths=X.view(100, batch_size, dim).permute(1, 2, 0).detach()
    initial_points=start_X.unsqueeze(0).permute(1,0,2)
    full_paths=torch.cat((initial_points,paths),dim=2)

    output_dir = "../data/generated_data/"
    data_utils.makedirs(output_dir)
    with open('{}FLAGS.output_name'.format(output_dir), 'wb') as f:
        np.save(f, full_paths.detach().numpy()) 


def generate_next_value( X_t, mu_t, sigma_t, delta_t):
    """
    Generate the next value in the time series using the Euler-Maruyama scheme.
    """
    delta_Wt = torch.randn_like(X_t) * np.sqrt(delta_t)
    delta_Wt_sigma_t = torch.bmm(delta_Wt.unsqueeze(1), sigma_t).squeeze(1)
    X_t_next = X_t + mu_t * delta_t + delta_Wt_sigma_t

    return X_t_next

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
