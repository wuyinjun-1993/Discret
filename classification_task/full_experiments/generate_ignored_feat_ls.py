import pandas as pd
import os, sys
from parse_args import *


import random
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pickle

import torch
import yaml

from create_language import *
from trainer import *
# import rl_enc_dec.synthetic_lang as synthetic_lang
# from rl_enc_dec.train_rl_synthesizer_2 import Trainer6_2,Trainer7_2,obtain_feat_range_mappings,obtain_numeric_categorical_value_count
from baselines.baseline_main import construct_feat_label_mat
from baselines.dt import *
from baselines.gb import *
from baselines.rf import *
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import logging

# from rl_enc_dec.ehr_lang import *


from datetime import datetime
from datasets.EHR_datasets import EHRDataset, read_data, obtain_feat_range_mappings, obtain_numeric_categorical_value_count

from full_experiments.pretrain_main import obtain_embeddings_over_data, construct_model


from utils_mortality.metrics import metrics_maps
from trainer import Trainer_all


other_attrs_to_remove = {"EMPI"}

def get_all_attributes_with_std_suffix(attr_ls):
    removed_attr_ls = []
    
    for attr in attr_ls:
        if attr.endswith("std"):
            removed_attr_ls.append(attr)
    
    return removed_attr_ls


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    rl_config, model_config = load_configs(args)
    train_data, valid_data, test_data, _ = read_data(args.data_folder, dataset_name=args.dataset_name)

    removed_cols_ls = get_all_attributes_with_std_suffix(train_data.columns)

    removed_cols_ls.extend(other_attrs_to_remove)

    root_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(root_dir,"removed_cols_" + args.dataset_name + ".txt"), "w") as f:
        for col in removed_cols_ls:
            f.write(col+"\n")

        f.close() 
        
        
    with open(os.path.join(root_dir,"all_cols_" + args.dataset_name + ".txt"), "w") as f:
        for col in list(train_data.columns):
            f.write(col+"\n")

        f.close() 