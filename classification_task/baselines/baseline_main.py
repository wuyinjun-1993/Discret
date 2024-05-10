import os
import logging
import sys
import numpy as np
import pandas as pd
import random
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from torch.utils.data import Dataset
# from rl_enc_dec.train_rl_synthesizer import EHRDataset, split_train_valid_set
from rl_enc_dec.ehr_lang import *
from datetime import datetime
from baselines.gb import *
from baselines.rf import *
from baselines.dt import *
from sklearn.metrics import recall_score, f1_score, roc_auc_score

class EHRDataset(Dataset):
    def __init__(self, data, drop_cols,patient_max_appts, balance, other_data=None):
        self.data = data
        self.patient_max_appts = patient_max_appts
        self.drop_cols = drop_cols
        self.patient_ids = self.data['PAT_ID'].unique().tolist()
        self.other_data = other_data
        if balance:
            most = self.data['label'].value_counts().max()
            for label in self.data['label'].unique():
                match  = self.data.loc[self.data['label'] == label]['PAT_ID'].to_list()
                samples = [random.choice(match) for _ in range(most-len(match))]
                self.patient_ids.extend(samples)
        random.shuffle(self.patient_ids)

    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[idx]]
        if self.other_data is None:
            all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[idx]]
        else:
            all_other_pats = self.other_data
        m = [appts['label'].max()]
        y = torch.FloatTensor(m)
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.FloatTensor([0]*len(X[0]))]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y

    @staticmethod
    def collate_fn(data):
        all_other_pats_ls = [data[idx][0][0] for idx in range(len(data))]
        all_x_pd_ls = [data[idx][0][1] for idx in range(len(data))]
        all_x_ls = [data[idx][0][2] for idx in range(len(data))]
        y_ls = [data[idx][1] for idx in range(len(data))]
        
        return (all_other_pats_ls, all_x_pd_ls, all_x_ls), y_ls
        
def split_train_valid_set(train_data, valid_ratio=0.2):
    total_count = len(train_data)
    random_train_ids = torch.randperm(total_count)
    valid_ids = random_train_ids[0:int(total_count*valid_ratio)]
    train_ids = random_train_ids[int(total_count*valid_ratio):]
    
    valid_data = train_data.iloc[valid_ids]
    train_data = train_data.iloc[train_ids]
    
    return train_data, valid_data

def construct_feat_label_mat(dataset, var_clns=None):
    if var_clns is None:
        var_clns = []
        var_clns.extend(list(dataset.cat_cols))
        var_clns.extend(list(dataset.num_cols))
    
    # full_data = pd.concat([dataset.data.loc[dataset.data["PAT_ID"] == dataset.patient_ids[i]] for i in range(len(dataset.patient_ids))])
    # full_data = pd.concat([dataset.data[dataset.data.index == dataset.patient_ids[i]] for i in range(len(dataset.patient_ids))])
    # var_clns.remove("PAT_ID")
    # var_clns.remove("label")
    # feat_mat = np.array(full_data[var_clns])
    feat_mat = dataset.transformed_features.numpy()
    # label_mat = np.array(list(full_data["label"]))
    label_mat = dataset.labels.numpy()
    
    return feat_mat, label_mat.reshape(-1,1).astype(int)


def remove_empty_classes(train_labels, valid_labels, test_labels):
    all_labels = np.concatenate([train_labels, valid_labels, test_labels])
    selected_label_ids = (np.sum(all_labels, axis=0) > 0)
    train_labels = train_labels[:, selected_label_ids]
    valid_labels = valid_labels[:, selected_label_ids]
    test_labels = test_labels[:, selected_label_ids]
    return train_labels, valid_labels, test_labels


if __name__=="__main__":
    seed = 0
    work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    train_data_path = os.path.join(work_dir,"simple_df")# "synthetic_dataset.pd")#"simple_df"
    test_data_path = os.path.join(work_dir,"simple_df")# "synthetic_test_dataset.pd")#"simple_df"
    precomputed_path = os.path.join(work_dir,"ehr_precomputed.npy")# "synthetic_precomputed.npy")#"ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 0.00001
    batch_size = 16
    gamma = 0.999
    epsilon = 0.9
    epsilon_falloff = 0.9
    target_update = 20
    epochs = 100
    program_max_len = 2
    patient_max_appts = 1
    provenance = "difftopkproofs"
    latent_size = 40
    is_log = True

    if is_log:
        if len(sys.argv) >= 2 and sys.argv[1] == "new":
            os.makedirs(os.path.join(work_dir,'logs_new/'), exist_ok=True)
            log_path = os.path.join(work_dir,'logs_new/')+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        else:
            os.makedirs(os.path.join(work_dir,'logs_old/'), exist_ok=True)
            log_path = os.path.join(work_dir,'logs_old/')+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        logging.basicConfig(filename=log_path,
                filemode='a',
                format='%(message)s',
                level=logging.DEBUG)
        logging.info("EHR Explanation Synthesis\n Seed: {}, train_path: {}, test_path: {}, precomputed_path: {}, mem_cap: {}, learning_rate: {}, batch_\
        size: {}, gamma: {}, epsilon: {}, epsilon_falloff: {}, target_update: {}, epochs: {}, prog_max_len: {}, pat_max_appt: {}, latent_size: {}".format(
            seed, train_data_path,test_data_path,precomputed_path,replay_memory_capacity,learning_rate,batch_size,gamma,epsilon,epsilon_falloff,target_update,epochs,
            program_max_len,patient_max_appts,latent_size))

    np.random.seed(seed)
    random.seed(seed)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    train_data, valid_data = split_train_valid_set(train_data, valid_ratio=0.2)
    
    
    train_dataset = EHRDataset(data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    valid_dataset = EHRDataset(data=valid_data, other_data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    test_dataset = EHRDataset(data = test_data, other_data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)

    train_feat, train_labels = construct_feat_label_mat(train_dataset)
    valid_feat, valid_labels = construct_feat_label_mat(valid_dataset)
    test_feat, test_labels = construct_feat_label_mat(test_dataset)

    gb_model = gb_model_train(feat_mats=train_feat, label_mats=train_labels)

    gb_pred_labels, pred_prob_labels = gb_model_pred(test_feat, gb_model)


    # gb_model = rf_model_train(feat_mats=train_feat, label_mats=train_labels)

    # gb_pred_labels, pred_prob_labels = rf_model_pred(test_feat, gb_model)

    gb_model = dt_model_train(feat_mats=train_feat, label_mats=train_labels)

    gb_pred_labels, pred_prob_labels = dt_model_pred(test_feat, gb_model)

    auc_score = roc_auc_score(test_labels, pred_prob_labels[:,1])

    auc_score2 = roc_auc_score(test_labels, (pred_prob_labels[:,1]>0.5).astype(float))

    pred_prob_labels_int = (pred_prob_labels[:,1]>0.5).astype(int)

    accuracy = np.mean(test_labels.reshape(-1) == pred_prob_labels_int.reshape(-1))

    print("auc score::", auc_score)

    print("auc score 2::", auc_score2)

    print("accuracy::", accuracy)