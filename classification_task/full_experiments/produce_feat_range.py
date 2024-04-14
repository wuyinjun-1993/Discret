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
import rl_enc_dec.synthetic_lang as synthetic_lang
# from rl_enc_dec.train_rl_synthesizer_2 import Trainer6_2,Trainer7_2,obtain_feat_range_mappings,obtain_numeric_categorical_value_count
from baselines.baseline_main import construct_feat_label_mat
from baselines.dt import *
from baselines.gb import *
from baselines.rf import *
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import logging

from rl_enc_dec.ehr_lang import *


from datetime import datetime
from datasets.EHR_datasets import EHRDataset, read_data, one, two, three, four, seven

import math

import pickle



additional_range_count = 10

def generate_age_ranges(age_count=10):
    range_ls = []
    age_gap = 100/age_count
    for i in range(age_count):
        range_ls.append((i*age_gap, (i+1)*age_gap))

    return range_ls

def create_ranges_by_interval_count_no_open_bound(lower_val, upper_val, additional_range_count):
    if np.isnan(upper_val):
        upper_val = lower_val + np.abs(lower_val)*3
    if np.isnan(lower_val):
        lower_val = upper_val - np.abs(upper_val)*3


    sub_interval_size = (upper_val - lower_val)/additional_range_count


    range_ls = []
    # if is_lb:
    #     range_ls.append((-np.inf, lower_val))
    prev_val = lower_val
    for k in range(additional_range_count-1):
        curr_val = prev_val + sub_interval_size
        range_ls.append((prev_val, curr_val))
        prev_val = curr_val

    range_ls.append((curr_val, upper_val))

    return range_ls

def create_ranges_by_interval_data_points_no_open_bound(lower_val, upper_val, additional_range_count):
    if np.isnan(upper_val):
        upper_val = lower_val + np.abs(lower_val)*3
    if np.isnan(lower_val):
        lower_val = upper_val - np.abs(upper_val)*3


    sub_interval_size = (upper_val - lower_val)/additional_range_count


    data_points_ls = []
    # if is_lb:
    #     range_ls.append((-np.inf, lower_val))
    prev_val = lower_val
    for k in range(additional_range_count-1):
        curr_val = prev_val + sub_interval_size
        data_points_ls.append(prev_val)
        prev_val = curr_val

    data_points_ls.append(upper_val)

    return data_points_ls

def create_ranges_by_interval_count(lower_val, upper_val, is_lb=True):
    assert lower_val <= upper_val
    if upper_val == lower_val:
        if is_lb:
            return [(-np.inf, upper_val)]
        else:
            return [(upper_val, np.inf)]

    sub_interval_size = (upper_val - lower_val)/additional_range_count


    range_ls = []
    if is_lb:
        range_ls.append((-np.inf, lower_val))
    prev_val = lower_val
    for k in range(additional_range_count-1):
        curr_val = prev_val + sub_interval_size
        range_ls.append((prev_val, curr_val))
        prev_val = curr_val

    range_ls.append((curr_val, upper_val))

    if not is_lb:
        range_ls.append((upper_val, np.inf))

    return range_ls

def create_all_ranges2(feat_min, feat_max, bound_gap_count=5):
    range_ls = []
    if np.isnan(feat_min) or np.isinf(feat_min):
        feat_min = -np.inf
    if np.isnan(feat_max) or np.isinf(feat_max):
        feat_max = np.inf

    if (not np.isinf(feat_max)) and (not np.isinf(feat_min)):
        gap = feat_max - feat_min
        lb = feat_min - bound_gap_count*gap
        ub = feat_max + bound_gap_count*gap
    else:
        if np.isinf(feat_max):
            lb = feat_min - np.abs(feat_min)*2
        else:
            ub = feat_max + np.abs(feat_max)*2

    if (not np.isnan(feat_min)) and (not np.isinf(feat_min)):
        range_ls.extend(create_ranges_by_interval_count(lb, feat_min, is_lb=True))

    # range_ls.append((feat_min, feat_max))
    range_ls.extend(create_ranges_by_interval_count_no_open_bound(feat_min, feat_max, 5))
    if (not np.isnan(feat_max)) and (not np.isinf(feat_max)):
        range_ls.extend(create_ranges_by_interval_count(feat_max, ub, is_lb=False))
    return range_ls

def replace_values_in_intervals_with_normal_range_vals(all_range_ls, min_val, max_val):
    all_range_arr = np.array(all_range_ls)
    all_distance = np.abs(all_range_arr - min_val)
    arr_idx1 = all_distance.argmin()

    all_distance = np.abs(all_range_arr - max_val)
    arr_idx2 = all_distance.argmin()

    if arr_idx1 != arr_idx2:
        all_range_ls[arr_idx1] = min_val
        all_range_ls[arr_idx2] = max_val
    else:
        all_range_ls[arr_idx1] = min_val
        all_range_ls.insert(arr_idx1 + 1, max_val)
        if arr_idx1 + 1 >= len(all_range_ls)/2:
            del all_range_ls[0]
        # if arr_idx1 + 1 == len(all_range_ls) - 1:
        #     del all_range_ls[0]
        else:
            del all_range_ls[-1]
    return all_range_ls

def create_all_ranges3(feat_min, feat_max, col_min, col_max, bound_gap_count=20):
    if np.isinf(feat_min) or np.isnan(feat_min):
        normalized_feat_min = 0
    else:
        normalized_feat_min = (feat_min - col_min)/(col_max - col_min)
    
    if np.isinf(feat_max) or np.isnan(feat_max):
        normalized_feat_max = 1
    else:
        normalized_feat_max = (feat_max - col_min)/(col_max - col_min)

    all_range_ls = []
    all_range_ls.extend(create_ranges_by_interval_data_points_no_open_bound(0, 1, bound_gap_count))

    all_range_ls = replace_values_in_intervals_with_normal_range_vals(all_range_ls, normalized_feat_min, normalized_feat_max)
    # all_range_ls = replace_values_in_intervals_with_normal_range_vals(all_range_ls, normalized_feat_max)

    print(all_range_ls)

  
    min_idx = all_range_ls.index(normalized_feat_min)
    max_idx = all_range_ls.index(normalized_feat_max)

    return all_range_ls, min_idx, max_idx




def create_all_ranges(feat_min, feat_max, col_min, col_max):
    range_ls = []
    if (not np.isnan(feat_min)) and (not np.isnan(feat_max)):
        if col_min <= feat_min:
            range_ls.extend(create_ranges_by_interval_count(col_min, feat_min, is_lb=True))
        range_ls.append((feat_min, feat_max))
        if col_max >= feat_max:
            range_ls.extend(create_ranges_by_interval_count(feat_max, col_max, is_lb=False))
    else:
        if np.isnan(feat_min):
            range_ls.append((-np.inf, feat_max))
            if col_max >= feat_max:
                range_ls.extend(create_ranges_by_interval_count(feat_max, col_max, is_lb=False))
            else:
                range_ls.append((feat_max, np.inf))
        else:
            if np.isnan(feat_max):
                if col_min <= feat_min:
                    range_ls.extend(create_ranges_by_interval_count(col_min, feat_min, is_lb=True))
                else:
                    range_ls.append((-np.inf, feat_min))

                range_ls.append((feat_min, np.inf))

    return range_ls

def read_feat_range_from_files(feat_range_file_name, data, col_ranges=None):
    feat_ranges = pd.read_csv(feat_range_file_name)

    feat_ls = list(feat_ranges["feat_name"])

    col_ls = list(data.columns)

    if col_ranges is None:
        new_col_ranges = dict()

    all_feat_ranges=dict()

    feat_ranges_mappings = dict()

    for col in col_ls:
        if col_ranges is None:
            col_min = data[col].min()
            col_max = data[col].max()
            new_col_ranges[col] = (col_min, col_max)
        else:
            col_min, col_max = all_feat_ranges[col]
        if col.startswith("CARBOXYHEMOGLOBIN"):
            print()

        if col.endswith("std"):
            curr_ranges = create_ranges_by_interval_count(col_min, col_max, is_lb=True)
            curr_ranges.append((col_max, np.inf))
            feat_ranges_mappings[col] = curr_ranges
            print(col, curr_ranges)
        else:
            if ".." in col:
                col_name_ls = col.split("..")
                col_suffix = ".." + col_name_ls[-1]
                col_prefix = col.split(col_suffix)[0]
                print(col_prefix, col)
                for feat in feat_ls:
                    feat_min = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "min"])[0])
                    feat_max = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "max"])[0])

                    if col.lower().startswith(feat.lower()) and feat.lower().startswith(col_prefix.lower()):
                        if (not math.isnan(feat_max)) and (not math.isnan(feat_min)):
                            overlap = (min(feat_max, col_max) - max(feat_min, col_min))/(max(feat_max, col_max) - min(feat_min, col_min))
                            assert overlap > 0
                        curr_ranges = create_all_ranges2(feat_min, feat_max)
                        # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
                        print(col, curr_ranges)
                        feat_ranges_mappings[col] = curr_ranges

    feat_ranges_mappings["PAT_AGE"] = generate_age_ranges()
    
    feat_ranges_mappings["SEX_C.y"] = [(1,1), (2,2)]
    
    print("number of attributes::", len(feat_ranges_mappings))

    return feat_ranges, feat_ranges_mappings



def read_feat_range_from_files2(feat_range_file_name, num_attr_ls, data, discrete_count_num=20, col_ranges=None):
    feat_ranges = pd.read_csv(feat_range_file_name)

    feat_ls = list(feat_ranges["feat_name"])

    # col_ls = list(data.columns)

    if col_ranges is None:
        new_col_ranges = dict()

    all_feat_ranges=dict()

    feat_ranges_mappings = dict()

    for col in num_attr_ls:
        if col_ranges is None:
            col_min = data[col].min()
            col_max = data[col].max()
            new_col_ranges[col] = (col_min, col_max)
        else:
            col_min, col_max = all_feat_ranges[col]

        if col.endswith("std") or col.endswith("count"):
            # curr_ranges = create_ranges_by_interval_count(col_min, col_max, is_lb=True)
            # curr_ranges.append((col_max, np.inf))
            curr_ranges = create_all_ranges3(-np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
            feat_ranges_mappings[col] = curr_ranges
            print(col, curr_ranges)
        else:
            if col.startswith("PCO2"):
                print()
            if ".." in col:
                col_name_ls = col.split("..")
                col_suffix = ".." + col_name_ls[-1]
                col_prefix = col.split(col_suffix)[0]
                print(col_prefix, col)
            
                for feat in feat_ls:
                    feat_min = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "min"])[0])
                    feat_max = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "max"])[0])

                    if col.lower().startswith(feat.lower()) and feat.lower().startswith(col_prefix.lower()):
                        if (not math.isnan(feat_max)) and (not math.isnan(feat_min)):
                            overlap = (min(feat_max, col_max) - max(feat_min, col_min))/(max(feat_max, col_max) - min(feat_min, col_min))
                            assert overlap > 0 or (col_max < feat_max and col_min > feat_min)
                        curr_ranges = create_all_ranges3(feat_min, feat_max, col_min, col_max, bound_gap_count=discrete_count_num)
                        # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
                        print(col, curr_ranges)
                        feat_ranges_mappings[col] = curr_ranges

            else:
                col_prefix = col
                curr_ranges = create_all_ranges3(col_min, col_max, col_min, col_max, bound_gap_count=discrete_count_num)
                # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
                print(col, curr_ranges)
                feat_ranges_mappings[col] = curr_ranges

    # col_min = data["SEX_C.y"].min()
    # col_max = data["SEX_C.y"].max()

    feat_ranges_mappings["PAT_AGE"] = create_all_ranges3(np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
    
    # feat_ranges_mappings["SEX_C.y"] = [0.5]*len(curr_ranges)
    
    
    print("expected number of attributes::", len(num_attr_ls))
    
    print("number of attributes::", len(feat_ranges_mappings))
    
    remaining_attrs = set(num_attr_ls)
    
    remaining_attrs = remaining_attrs.difference(set(feat_ranges_mappings.keys()))
    
    print("remaining attributes::", remaining_attrs)

    return feat_ranges, feat_ranges_mappings

def read_feat_range_from_files_for_seven(feat_range_file_name, num_attr_ls, data, discrete_count_num=20, col_ranges=None):
    feat_ranges = pd.read_csv(feat_range_file_name)

    feat_ls = list(feat_ranges["feat_name"])

    # col_ls = list(data.columns)

    if col_ranges is None:
        new_col_ranges = dict()

    all_feat_ranges=dict()

    feat_ranges_mappings = dict()
    feat_range_minmax_ids = dict()

    for col in num_attr_ls:
        if col_ranges is None:
            col_min = data[col].min()
            col_max = data[col].max()
            new_col_ranges[col] = (col_min, col_max)
        else:
            col_min, col_max = all_feat_ranges[col]

        if col.endswith("std") or col.endswith("diag"):
            # curr_ranges = create_ranges_by_interval_count(col_min, col_max, is_lb=True)
            # curr_ranges.append((col_max, np.inf))
            curr_ranges, min_idx, max_idx = create_all_ranges3(-np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
            feat_ranges_mappings[col] = curr_ranges
            feat_range_minmax_ids[col] = (min_idx, max_idx)
            print(col, curr_ranges)
        else:
            # if col.startswith("PCO2"):
            #     print()
            if col.endswith("_min") or col.endswith("_max") or col.endswith("_avg"):
                sub_col_ls = col.split("_")
                col_prefix = "_".join(sub_col_ls[:-1])
                if col_prefix in feat_ls:
                    feat_min = float(list(feat_ranges.loc[feat_ranges["feat_name"] == col_prefix, "min"])[0])
                    feat_max = float(list(feat_ranges.loc[feat_ranges["feat_name"] == col_prefix, "max"])[0])
                    
                    curr_ranges, min_idx, max_idx = create_all_ranges3(feat_min, feat_max, col_min, col_max, bound_gap_count=discrete_count_num)
                else:
                    curr_ranges, min_idx, max_idx = create_all_ranges3(-np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
            else:
                curr_ranges, min_idx, max_idx = create_all_ranges3(-np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
            # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
            print(col, curr_ranges)
            feat_ranges_mappings[col] = curr_ranges
            feat_range_minmax_ids[col] = (min_idx, max_idx)
            # if ".." in col:
            #     col_name_ls = col.split("..")
            #     col_suffix = ".." + col_name_ls[-1]
            #     col_prefix = col.split(col_suffix)[0]
            #     print(col_prefix, col)
            
            #     for feat in feat_ls:
            #         feat_min = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "min"])[0])
            #         feat_max = float(list(feat_ranges.loc[feat_ranges["feat_name"] == feat, "max"])[0])

            #         if col.lower().startswith(feat.lower()) and feat.lower().startswith(col_prefix.lower()):
            #             if (not math.isnan(feat_max)) and (not math.isnan(feat_min)):
            #                 overlap = (min(feat_max, col_max) - max(feat_min, col_min))/(max(feat_max, col_max) - min(feat_min, col_min))
            #                 assert overlap > 0 or (col_max < feat_max and col_min > feat_min)
            #             curr_ranges = create_all_ranges3(feat_min, feat_max, col_min, col_max)
            #             # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
            #             print(col, curr_ranges)
            #             feat_ranges_mappings[col] = curr_ranges

            # else:
            #     col_prefix = col
            #     curr_ranges = create_all_ranges3(col_min, col_max, col_min, col_max)
            #     # curr_ranges = create_all_ranges(feat_min, feat_max, col_min, col_max)
            #     print(col, curr_ranges)
            #     feat_ranges_mappings[col] = curr_ranges

    # col_min = data["SEX_C.y"].min()
    # col_max = data["SEX_C.y"].max()

    feat_ranges_mappings["age"], min_idx, max_idx = create_all_ranges3(-np.inf, np.inf, col_min, col_max, bound_gap_count=discrete_count_num)
    feat_range_minmax_ids["age"] = (min_idx, max_idx)
    
    # feat_ranges_mappings["SEX_C.y"] = [0.5]*len(curr_ranges)
    
    
    print("expected number of attributes::", len(num_attr_ls))
    
    print("number of attributes::", len(feat_ranges_mappings))
    
    remaining_attrs = set(num_attr_ls)
    
    remaining_attrs = remaining_attrs.difference(set(feat_ranges_mappings.keys()))
    
    print("remaining attributes::", remaining_attrs)

    return feat_ranges, feat_ranges_mappings, feat_range_minmax_ids



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
    if args.group_aware:
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(root_dir, "feature_group_names"), "rb") as f:
            feat_group_names = pickle.load(f)
    else:
        feat_group_names = None

    removed_feat_ls = []
    if args.removed_feats_file_name is not None:
        with open(args.removed_feats_file_name) as f:
            for line in f:
                removed_feat_ls.append(line.strip())

    # col_str = None
    # for col in train_data.columns:
    #     if col_str is None:
    #         col_str = col
    #     else:
    #         col_str +=",  " + col
    # print(col_str)
    program_max_len = args.program_max_len
    print("program max len::", program_max_len)
    patient_max_appts = 1
    # train_dataset = EHRDataset(data= train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    # train_valid_dataset = EHRDataset(data= pd.concat([train_data, valid_data]), drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    # valid_dataset = EHRDataset(data = valid_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    # test_dataset = EHRDataset(data = test_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    train_dataset, train_valid_dataset, valid_dataset, test_dataset, feat_range_mappings = create_train_val_test_datasets(train_data, valid_data, test_data, patient_max_appts, synthetic_lang_mappings, args)
    # feat_ranges, feat_bound_mappings = read_feat_range_from_files(os.path.join(args.data_folder, feat_range_file_name_mappings[args.dataset_name]), train_data)
    feat_range_minmax_ids = None
    if args.dataset_name == seven:
        feat_ranges, feat_bound_point_mappings, feat_range_minmax_ids = read_feat_range_from_files_for_seven(os.path.join(args.data_folder, feat_range_file_name_mappings[args.dataset_name]), list(set(train_dataset.num_cols)), train_data, discrete_count_num=rl_config["discretize_feat_value_count"])
    else:
        feat_ranges, feat_bound_point_mappings = read_feat_range_from_files2(os.path.join(args.data_folder, feat_range_file_name_mappings[args.dataset_name]), list(set(train_dataset.num_cols)), train_data, discrete_count_num=rl_config["discretize_feat_value_count"])

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # with open(os.path.join(curr_dir, "feat_range_mappings"), "wb") as f:
    #     pickle.dump(feat_bound_mappings, f)
    with open(os.path.join(curr_dir, "feat_bound_point_mappings_" + args.dataset_name), "wb") as f:
        pickle.dump(feat_bound_point_mappings, f)
        
    with open(os.path.join(curr_dir, "feat_range_mappings_" + args.dataset_name), "wb") as f:
        pickle.dump(train_dataset.feat_range_mappings, f)
    
    if feat_range_minmax_ids is not None:
        with open(os.path.join(curr_dir, "feat_range_minmax_ids_" + args.dataset_name), "wb") as f:
            pickle.dump(feat_range_minmax_ids, f)
        # f.write(feat_ranges_mappings)
