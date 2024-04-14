import os, sys
import pandas as pd
import torch

def read_normal_range_files(data_folder,dataset_name):
    normal_range_file = os.path.join(data_folder, "feat_range_" + str(dataset_name) + ".txt")
    df = pd.read_csv(normal_range_file)
    return df

def check_feat_in_list(feat, feat_ls):
    for curr_feat in feat_ls:
        if feat.startswith(curr_feat) and (feat.endswith("_min") or feat.endswith("_max") or feat.endswith("_avg")):
            return True
    return False

def generate_abnormal_value_indicator_mat(df, train_dataset, feature_range_mappings, feat_bound_point_ls):
    origin_feat_mat = train_dataset.origin_features
    all_feat = train_dataset.num_cols + train_dataset.cat_cols
    all_abnormal_indicators = []
    all_activated_indicators = []
    
    normal_range_feat_ls = df["feat_name"].values.tolist()
    for idx in range(len(all_feat)):
        feat = all_feat[idx]
        # if feat not in df["feat_name"].values.tolist():
        if check_feat_in_list(feat, normal_range_feat_ls) == False:
            all_abnormal_indicators.append(torch.zeros(origin_feat_mat.shape[0], dtype=torch.bool))
            all_activated_indicators.append(torch.zeros((origin_feat_mat.shape[0], len(feat_bound_point_ls[list(feat_bound_point_ls.keys())[0]])), dtype=torch.bool))
            continue
        
        feat_prefix = feat.split("_")[0]
        
        min_val = df[df["feat_name"] == feat_prefix]["min"].values[0]
        max_val = df[df["feat_name"] == feat_prefix]["max"].values[0]
        curr_feat_val = torch.from_numpy(origin_feat_mat[:, idx].reshape(-1).astype(float))
        
        feat_range = feature_range_mappings[feat]
        curr_feat_bound_point = torch.tensor(feat_bound_point_ls[feat])
        rescale_feat_bound_point = curr_feat_bound_point*(feat_range[1] - feat_range[0]) + feat_range[0]
        
        lower_value_boolean = curr_feat_val.view(-1,1) >= rescale_feat_bound_point.view(1,-1)
        upper_value_boolean = curr_feat_val.view(-1,1) <= rescale_feat_bound_point.view(1,-1)
        
        max_val_upper_value_boolean = max_val <= rescale_feat_bound_point.view(1,-1)
        min_val_lower_value_boolean = min_val >= rescale_feat_bound_point.view(1,-1)
        
        upper_activated_ranges = torch.logical_and(lower_value_boolean, max_val_upper_value_boolean)
        lower_activated_ranges = torch.logical_and(upper_value_boolean, min_val_lower_value_boolean)
        
        
        greater_than_indicator = (curr_feat_val > max_val)
        
        less_than_indicator = (curr_feat_val < min_val)
        
        upper_activated_ranges = torch.logical_and(upper_activated_ranges, greater_than_indicator.view(-1,1))
        lower_activated_ranges = torch.logical_and(lower_activated_ranges, less_than_indicator.view(-1,1))
        
        curr_abnormal_indicator = torch.logical_or(greater_than_indicator, less_than_indicator)
        curr_activated_ranges = torch.logical_or(upper_activated_ranges, lower_activated_ranges)
        
        
        all_abnormal_indicators.append(curr_abnormal_indicator.view(-1))
        all_activated_indicators.append(curr_activated_ranges)
    
    all_abnormal_feature_indicators = torch.stack(all_abnormal_indicators, dim=1)
    
    train_dataset.set_abnormal_feature_vals(all_abnormal_feature_indicators, torch.stack(all_activated_indicators,dim=1), all_activated_indicators[0].shape[-1])
    # all_activated_indicator_tensor = torch.stack(all_activated_indicators, dim=1)
    return all_abnormal_feature_indicators, all_activated_indicators


