from typing import List 
import numpy as np
import json, os, time
from torch.utils.data import DataLoader
import torch
import tqdm 
import pandas as pd
import warnings

from loss_functions import LOSS_FUN_DICT
from data_utils import CustomCollateFnGen
from models import NJODE 
import synthetic_datasets
_STOCK_MODELS = synthetic_datasets.DATASETS



METR_COLUMNS: List[str] = [
    'epoch', 'train_time', 'val_time', 'train_loss', 'val_loss',
    'optimal_val_loss', 'test_loss', 'optimal_test_loss', 'evaluation_mean_diff']
default_nn = ((50, 'tanh'), (50, 'tanh'))

def train(
    model=None, seed=None,
    data_train=None, data_val=None, data_test=None,
    dataset_metadata=None, testset_metadata=None,
    epochs=100, batch_size=100, save_every=1,
    learning_rate=0.001,
    hidden_size=10, bias=True, dropout_rate=0.1,
    ode_nn=default_nn, readout_nn=default_nn,
    enc_nn=default_nn, use_rnn=False,
    solver="euler",
    **options
):


    """
    training function for NJODE model (models.NJODE),
    the model is automatically saved in the model-save-path with the given
    model id, also all evaluations of the model are saved there

    :param model: None or a NJODE model instance
    :param data_train: see data_utils
    :param data_val: see data_utils
    :param data_test: see data_utils
    :param dataset_metadata: see data_utils
    :param testset_metadata: see data_utils
    :param epochs: int, number of epochs to train, each epoch is one cycle
            through all (random) batches of the training data
    :param batch_size: int
    :param save_every: int, defined number of epochs after each of which the
            model is saved and plotted if wanted. whenever the model has a new
            best eval-loss it is also saved, independent of this number (but not
            plotted)
    :param learning_rate: float
    :param hidden_size: see models.NJODE
    :param bias: see models.NJODE
    :param dropout_rate: float
    :param ode_nn: see models.NJODE
    :param readout_nn: see models.NJODE
    :param enc_nn: see models.NJODE
    :param use_rnn: see models.NJODE
    :param solver: see models.NJODE
    :param saved_models_path: str, where to save the models
    :param options: kwargs, used keywords:
        'test_data_dict'    None, str or dict, if no None, this data_dict is
                        used to define the dataset for plot_only and
                        evaluation (if evaluate=True)
        'func_appl_X'   list of functions (as str, see data_utils)
                        to apply to X
        'masked'        bool, whether the data is masked (i.e. has
                        incomplete observations)
        'which_loss'    default: 'standard', see models.LOSS_FUN_DICT for
                        choices. suggested: 'easy' or 'very_easy'
        'residual_enc_dec'  bool, whether resNNs are used for encoder and
                        readout NN, used by models.NJODE. the provided value
                        is overwritten by 'residual_enc' & 'residual_dec' if
                        they are provided. default: False
        'residual_enc'  bool, whether resNNs are used for encoder NN,
                        used by models.NJODE.
                        default: True if use_rnn=False, else: False (this is
                        for backward compatibility)
        'residual_dec'  bool, whether resNNs are used for readout NN,
                        used by models.NJODE. default: True
        'use_y_for_ode' bool, whether to use y (after jump) or x_impute for
                        the ODE as input, only in masked case, default: True
        'use_current_y_for_ode' bool, whether to use the current y as input
                        to the ode. this should make the training more
                        stable in the case of long windows without
                        observations (cf. literature about stable output
                        feedback). default: False
        'coord_wise_tau'    bool, whether to use a coordinate wise tau
        'input_sig'     bool, whether to use the signature as input
        'level'         int, level of the signature that is used
        'input_current_t'   bool, whether to additionally input current time
                        to the ODE function f, default: False
        'enc_input_t'   bool, whether to use the time as input for the
                        encoder network. default: False
        'evaluate'      bool, whether to evaluate the model in the test set
                        (i.e. not only compute the val_loss, but also
                        compute the mean difference between the true and the
                        predicted paths comparing at each time point)
        'use_observation_as_input'  bool, whether to use the observations as
                        input to the model or whether to only use them for
                        the loss function (this can be used to make the
                        model learn to predict well far into the future).
                        can be a float in (0,1) to use an observation with
                        this probability as input. can also be a string
                        defining a function (when evaluated) that takes the
                        current epoch as input and returns a bool whether to
                        use the current observation as input (this can be a
                        random function, i.e. the output can depend on
                        sampling a random variable). default: true
        'val_use_observation_as_input'  bool, None, float or str, same as
                        'use_observation_as_input', but for the validation
                        set. default: None, i.e. same as for training set
        'ode_input_scaling_func'    None or str in {'id', 'tanh'}, the
                        function used to scale inputs to the neuralODE.
                        default: tanh
        'use_cond_exp'  bool, whether to use the conditional expectation
                        as reference for model evaluation, default: True
        'which_val_loss'   str, see models.LOSS_FUN_DICT for choices, which
                        loss to use for evaluation, default: 'easy'
        'input_coords'  list of int or None, which coordinates to use as
                        input. overwrites the setting from dataset_metadata.
                        if None, then all coordinates are used.
        'output_coords' list of int or None, which coordinates to use as
                        output. overwrites the setting from
                        dataset_metadata. if None, then all coordinates are
                        used.
        'signature_coords'  list of int or None, which coordinates to use as
                        signature coordinates. overwrites the setting from
                        dataset_metadata. if None, then all input
                        coordinates are used.
        'compute_variance'   None, bool or str, if None, then no variance
                        computation is done. if bool, then the (marginal)
                        variance is computed. if str "covariance", then the
                        covariance matrix is computed. ATTENTION: the model
                        output corresponds to the square root of the
                        variance (or the Cholesky decomposition of the
                        covariance matrix, respectivel), so if W is the
                        model's output corresponding to the variance, then
                        the models variance estimate is V=W^T*W or W^2,
                        depending whether the covariance or marginal
                        variance is estimated.
                        default: None
        'var_weight'    float, weight of the variance loss term in the loss
                        function, default: 1
        'input_var_t_helper'    bool, whether to use 1/sqrt(Delta_t) as
                        additional input to the ODE function f. this should help
                        to better learn the variance of the process.
                        default: False
        'which_var_loss'   None or int, which loss to use for the variance loss
                        term. default: None, which leads to using default choice
                        of the main loss function (which aligns with structure
                        of main loss function as far as reasonable). see
                        models.LOSS_FUN_DICT for choices (currently in {1,2,3}).

    :return: model, df_metric, dl, dl_val, dl_test, stockmodel, stockmodel_test
    """
    use_cond_exp = True
    masked = False
    if 'masked' in options:
        masked = options['masked']
    device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # get input and output coordinates of the dataset
    input_coords = None
    output_coords = None
    signature_coords = None
    if "input_coords" in options:
        input_coords = options["input_coords"]
    elif "input_coords" in dataset_metadata:
        input_coords = dataset_metadata["input_coords"]
    if "output_coords" in options:
        output_coords = options["output_coords"]
    elif "output_coords" in dataset_metadata:
        output_coords = dataset_metadata["output_coords"]
    if "signature_coords" in options:
        signature_coords = options["signature_coords"]
    elif "signature_coords" in dataset_metadata:
        signature_coords = dataset_metadata["signature_coords"]
    if input_coords is None:
        input_size = dataset_metadata['dimension']
        input_coords = np.arange(input_size)
    else:
        input_size = len(input_coords)
    if output_coords is None:
        output_size = dataset_metadata['dimension']
        output_coords = np.arange(output_size)
    else:
        output_size = len(output_coords)
    if signature_coords is None:
        signature_coords = input_coords

    initial_print = '\ninput_coords: {}\noutput_coords: {}'.format(
        input_coords, output_coords)
    initial_print += '\ninput_size: {}\noutput_size: {}'.format(
        input_size, output_size)
    initial_print += '\nsignature_coords: {}'.format(signature_coords)
    dimension = dataset_metadata['dimension']
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']
    original_output_dim = output_size
    original_input_dim = input_size

    # get functions to apply to the paths in X
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        initial_print += '\napply functions to X'
        functions = options['func_appl_X']
        collate_fn, mult = CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
        input_coords = np.concatenate(
            [np.array(input_coords)+dimension*i for i in range(mult)])
        output_coords = np.concatenate(
            [np.array(output_coords)+dimension*i for i in range(mult)])
        initial_print += '\nnew input_coords: {}'.format(input_coords)
        initial_print += '\nnew output_coords: {}'.format(output_coords)
    else:
        functions = None
        collate_fn, mult = CustomCollateFnGen(None)
        mult = 1

    # get variance or covariance coordinates if wanted
    compute_variance = None
    var_size = 0
    if 'compute_variance' in options:
        if functions is not None:
            warnings.warn(
                "function application to X and concurrent variance/covariance "
                "computation might lead to problems! Use carefully!",
                UserWarning)
        compute_variance = options['compute_variance']
        if compute_variance == 'covariance':
            var_size = output_size**2
            initial_print += '\ncompute covariance of size {}'.format(var_size)
        elif compute_variance not in [None, False]:
            compute_variance = 'variance'
            var_size = output_size
            initial_print += '\ncompute (marginal) variance of size {}'.format(
                var_size)
        else:
            compute_variance = None
            initial_print += '\nno variance computation'
            var_size = 0
        # the models variance output is the Cholesky decomposition of the
        #   covariance matrix or the square root of the marginal variance.
        # for Y being the entire model output, the variance output is
        #   W=Y[:,-var_size:]
        output_size += var_size

    # get data-loader for training
    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=0)
    dl_val = DataLoader(  # class to iterate over validation data
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=0)
    stockmodel = _STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    if data_test is not None:
        dl_test = DataLoader(  # class to iterate over test data
            dataset=data_test, collate_fn=collate_fn,
            shuffle=False, batch_size=len(data_test),
            num_workers=0)
        stockmodel_test = _STOCK_MODELS[
            testset_metadata['model_name']](**testset_metadata)
    else:
        dl_test = dl_val
        stockmodel_test = stockmodel
        testset_metadata = dataset_metadata

    # validation loss function
    which_val_loss = 'easy'
    if 'which_loss' in options:
        which_val_loss = options['which_loss']
    if 'which_val_loss' in options:
        which_val_loss = options['which_val_loss']
    assert which_val_loss in LOSS_FUN_DICT

    if compute_variance is not None:
        warnings.warn(
            "optimal loss might be wrong, since the conditional "
            "variance is also learned, which is not accounted for in "
            "computation of the optimal loss",
            UserWarning)
    store_cond_exp = True
    if dl_val != dl_test:
        store_cond_exp = False
    if functions is not None and len(functions) > 0:
        initial_print += '\nWARNING: optimal loss computation for ' \
                          'power=2 not implemented for this model'
        corrected_string = "(corrected: only original X used) "
    else:
        corrected_string = ""
    opt_val_loss = compute_optimal_val_loss(
        dl_val, stockmodel, delta_t, T, mult=mult,
        store_cond_exp=store_cond_exp, return_var=False,
        which_loss=which_val_loss)
    initial_print += '\noptimal {}val-loss (achieved by true cond exp): ' \
                  '{:.5f}'.format(corrected_string, opt_val_loss)

    # get params_dict
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver,
        'learning_rate': learning_rate, 'test_size': 0.2, 'seed': seed,
        'optimal_val_loss': opt_val_loss, 'options': options}

    # add additional values to params_dict (not to be shown in the description)
    params_dict['input_coords'] = input_coords
    params_dict['output_coords'] = output_coords
    params_dict['signature_coords'] = signature_coords
    params_dict['compute_variance'] = compute_variance
    params_dict['var_size'] = var_size

    # get the model & optimizer
    if model is None:
        model = NJODE(**params_dict)  # get NJODE model class from
        initial_print += '\ninitiate new model ...'
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.0005)
    best_val_loss = np.inf
    metr_columns = METR_COLUMNS

    # ---------------- TRAINING ----------------
    initial_print += '\n\nmodel overview:'
    print(initial_print)
    print(model, '\n')

    # compute number of parameters
    nr_params = 0
    for name, param in model.named_parameters():
        nr_params += param.nelement()  # count number of parameters
    print('# parameters={}\n'.format(nr_params))

    metric_app = []
    while model.epoch <= epochs:
        t = time.time()
        model.train()  # set model in train mode (e.g. BatchNorm)
        for i, b in tqdm.tqdm(enumerate(dl)):  # iterate over the dataloader
            optimizer.zero_grad()  # reset the gradient
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"]
            if M is not None:
                M = M.to(device)
            start_M = b["start_M"]
            if start_M is not None:
                start_M = start_M.to(device)
            start_X = b["start_X"].to(device)
            obs_idx = b["obs_idx"]
            n_obs_ot = b["n_obs_ot"].to(device)

            hT, loss = model(
                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                return_path=False, get_loss=True, M=M, start_M=start_M,)
            loss.backward()  # compute gradient of each weight regarding loss function
            optimizer.step()  # update weights with ADAM optimizer
        train_time = time.time() - t

        # -------- evaluation --------
        print("evaluating ...")
        t = time.time()
        batch = None
        with torch.no_grad():  # no gradient needed for evaluation
            loss_val = 0
            num_obs = 0
            eval_msd = 0
            model.eval()  # set model in evaluation mode
            for i, b in enumerate(dl_val):
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"]
                if M is not None:
                    M = M.to(device)
                start_M = b["start_M"]
                if start_M is not None:
                    start_M = start_M.to(device)
                start_X = b["start_X"].to(device)
                obs_idx = b["obs_idx"]
                n_obs_ot = b["n_obs_ot"].to(device)
                true_paths = b["true_paths"]
                true_mask = b["true_mask"]

                hT, c_loss = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X,
                    n_obs_ot, return_path=False, get_loss=True, M=M,
                    start_M=start_M, which_loss=which_val_loss,)
                loss_val += c_loss.detach().numpy()
                num_obs += 1  # count number of observations

            # mean squared difference evaluation
            if 'evaluate' in options and options['evaluate']:
                eval_msd = evaluate_model(
                    model=model, dl_test=dl_test, device=device,
                    stockmodel_test=stockmodel_test,
                    testset_metadata=testset_metadata,
                    mult=mult, use_cond_exp=use_cond_exp,
                    eval_use_true_paths=False,)

            val_time = time.time() - t
            loss_val = loss_val / num_obs
            eval_msd = eval_msd / num_obs
            train_loss = loss.detach().numpy()
            print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                        "optimal-val-loss={:.5f}, val-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_val_loss, loss_val)
            print(print_str)

        curr_metric = [model.epoch, train_time, val_time, train_loss,
                               loss_val, opt_val_loss, None, None]
        if 'evaluate' in options and options['evaluate']:
            curr_metric.append(eval_msd)
            print("evaluation mean square difference (test set): {:.5f}".format(
                eval_msd))
        else:
            curr_metric.append(None)
        metric_app.append(curr_metric)

        if loss_val < best_val_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_val_loss, loss_val, model.epoch))
            best_val_loss = loss_val
        print("-"*100)

        model.epoch += 1

    df_metric = pd.DataFrame(data=metric_app, columns=metr_columns)

    return model, df_metric, dl, dl_val, dl_test, stockmodel, stockmodel_test


def compute_optimal_val_loss(
        dl_val, stockmodel, delta_t, T, mult=None,
        store_cond_exp=False, return_var=False, which_loss='easy'):
    """
    compute optimal evaluation loss (with the true cond. exp.) on the
    test-dataset
    :param dl_val: torch.DataLoader, used for the validation dataset
    :param stockmodel: stock_model.StockModel instance
    :param delta_t: float, the time_delta
    :param T: float, the terminal time
    :param mult: None or int, the factor by which the dimension is multiplied
    :param store_cond_exp: bool, whether to store the conditional expectation
    :return: float (optimal loss)
    """
    opt_loss = 0
    num_obs = 0
    for i, b in enumerate(dl_val):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].detach().numpy()
        start_X = b["start_X"].detach().numpy()
        obs_idx = b["obs_idx"].detach().numpy()
        n_obs_ot = b["n_obs_ot"].detach().numpy()
        M = b["M"]
        if M is not None:
            M = M.detach().numpy()
        num_obs += 1
        opt_loss += stockmodel.get_optimal_loss(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot, M=M,
            mult=mult, store_and_use_stored=store_cond_exp,
            return_var=return_var, which_loss=which_loss)
    return opt_loss / num_obs


def evaluate_model(
        model, dl_test, device, stockmodel_test, testset_metadata,
        mult, use_cond_exp, eval_use_true_paths):
    """
    evaluate the model on the test set

    Args:
        model:
        dl_test:
        device:
        stockmodel_test:
        testset_metadata:
        mult:
        use_cond_exp:
        eval_use_true_paths:

    Returns: evaluation metric

    """
    eval_msd = 0.
    for i, b in enumerate(dl_test):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].to(device)
        M = b["M"]
        if M is not None:
            M = M.to(device)
        start_M = b["start_M"]
        if start_M is not None:
            start_M = start_M.to(device)
        start_X = b["start_X"].to(device)
        obs_idx = b["obs_idx"]
        n_obs_ot = b["n_obs_ot"].to(device)
        true_paths = b["true_paths"]
        true_mask = b["true_mask"]

        if use_cond_exp and not eval_use_true_paths:
            true_paths = None
            true_mask = None
        _eval_msd = model.evaluate(
            times=times, time_ptr=time_ptr, X=X,
            obs_idx=obs_idx,
            delta_t=testset_metadata["dt"],
            T=testset_metadata["maturity"],
            start_X=start_X, n_obs_ot=n_obs_ot,
            stockmodel=stockmodel_test, return_paths=False, M=M,
            start_M=start_M, true_paths=true_paths,
            true_mask=true_mask, mult=mult,
            use_stored_cond_exp=True, )
        eval_msd += _eval_msd

    return eval_msd