import numpy as np
from scipy.stats import spearmanr

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets.EHR_datasets import read_data
from rl_enc_dec.ehr_lang import *
import pickle

import openai




def compute_correlations(tensor):
    num_features = tensor.shape[1]
    correlations = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(i, num_features):
            rho, _ = spearmanr(tensor[:, i], tensor[:, j])
            correlations[i, j] = rho
            correlations[j, i] = rho
    return correlations

def split_features_by_correlation(tensor, threshold):
    correlations = compute_correlations(tensor)
    num_features = tensor.shape[1]
    groups = []
    visited = set()
    for i in range(num_features):
        if i not in visited:
            group = {i}
            visited.add(i)
            for j in range(i+1, num_features):
                if j not in visited and np.abs(correlations[i,j]) >= threshold:
                    group.add(j)
                    visited.add(j)
            groups.append(group)

    transformed_groups = []
    for group in groups:
        transformed_groups.append(list(group))
    return transformed_groups

def order_column_names_by_clinic_use(feature_id_groups, columns):
    for feature_id_group in feature_id_groups:
        feature_name_ls = []
        for feature_id in feature_id_group:
            feature_name_ls.append(columns[feature_id])

        print()

def extract_feature_origin_names(columns):
    feature_name_set = dict()
    for column in columns:
        if ".." in column:

            cln_names = column.split("..")

            stats = cln_names[-1]

            feature_name = column.split(".." + stats)[0]

            if ".." in feature_name:
                feature_name = feature_name.split("..")[-1]
        elif "." in column:
            feature_name = column.split(".")[0]
        else:
            feature_name = column

        feature_name_set[feature_name] = column

    return feature_name_set
        

def get_feature_name_groups(columns, feature_id_groups):
    feature_name_groups = []
    for feature_id_group in feature_id_groups:
        cln_name_ls =[]
        for feature_id in feature_id_group:
            cln_name = columns[feature_id]
            cln_name_ls.append(cln_name)
        feature_name_groups.append(cln_name_ls)

    return feature_name_groups
    


if __name__ == "__main__":
    data_folder = "/data6/wuyinjun/cancer_data/"
    
    train_data, valid_data, test_data, _ = read_data(data_folder)

    selected_feats = train_data.columns
    
    selected_feats = [feat for feat in selected_feats if feat not in DROP_FEATS]
    
    feature_name_set = extract_feature_origin_names(selected_feats)

    selected_train_data = np.array(train_data[selected_feats])


    feature_id_groups = split_features_by_correlation(selected_train_data, 0.7)
    
    print(feature_id_groups)

    feature_name_groups = get_feature_name_groups(selected_feats, feature_id_groups)

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    with open(os.path.join(root_dir, "feature_group_names"), "wb") as f:
        pickle.dump(feature_name_groups, f)

    with open(os.path.join(root_dir, "feature_id_groups"), "wb") as f:
        pickle.dump(feature_id_groups, f)


    # order_column_names_by_clinic_use(feature_id_groups, selected_feats)