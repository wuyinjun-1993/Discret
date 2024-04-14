import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from MIRACLE import MIRACLE
import torch
import impyute as impy

def impute_train_eval(X_MISSING, miracle=None, reg_lambda=0.1, reg_beta=0.1, ckpt_file="tmp.ckpt", window=10, max_steps=2, batch_size=128):
    missing_idxs = np.where(np.any(np.isnan(X_MISSING), axis=0))[0]

    X = X_MISSING.copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    imputed_data_x = X
    
    # Initialize MIRACLE
    
    miracle = MIRACLE(
        num_inputs=X_MISSING.shape[1],
        reg_lambda=reg_lambda,
        reg_beta=reg_beta,
        n_hidden=32,
        ckpt_file=ckpt_file,
        missing_list=missing_idxs,
        reg_m=0.1,
        lr=0.0001,
        batch_size = batch_size,
        window=window,
        max_steps=max_steps,
    )

    # Train MIRACLE
    miracle_imputed_data_x = miracle.fit(
        X_MISSING,
        X_seed=imputed_data_x,
    )
    
    
    return torch.from_numpy(miracle_imputed_data_x), miracle


def impute_train_eval_all(X_MISSING_train, X_MISSING_val, X_MISSING_test):
    all_X_missing = torch.cat([X_MISSING_train, X_MISSING_val, X_MISSING_test], dim=0)    
    # all_X_missing = impy.fast_knn(all_X_missing.numpy())
    all_X_missing, _ = impute_train_eval(all_X_missing.numpy())
    X_MISSING_train, X_MISSING_val, X_MISSING_test = torch.split(all_X_missing, [X_MISSING_train.shape[0], X_MISSING_val.shape[0], X_MISSING_test.shape[0]], dim=0)
    return X_MISSING_train.type(torch.float), X_MISSING_val.type(torch.float), X_MISSING_test.type(torch.float)
