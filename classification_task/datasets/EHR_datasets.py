import torch
from torch.utils.data import Dataset
import random
import os
import pandas as pd
import numpy as np
# from rl_enc_dec.ehr_lang import *
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# import full_experiments.synthetic_lang_one as synthetic_lang_one
# import full_experiments.synthetic_lang_two as synthetic_lang_two
# import full_experiments.synthetic_lang_three as synthetic_lang_three
# import full_experiments.synthetic_lang_four as synthetic_lang_four
# import full_experiments.synthetic_lang_five as synthetic_lang_five
# import full_experiments.synthetic_lang_six as synthetic_lang_six
# import full_experiments.synthetic_lang_seven as synthetic_lang_seven
# import pickle
# from baselines.rf import rf_model_pred
# from baselines.baseline_main import construct_feat_label_mat

# one="one"
# two="two"
# three="three"
# four="four"
# five="five"
# six="six"
# seven = "seven"

# feat_range_file_name_mappings = {one: "feat_range_one.csv", two: "feat_range_two.csv", three: "feat_range_three.csv", four:"feat_range_four.csv", seven:"feat_range_seven.csv"}

# dataset_name_mappings = {one:"enc_data_static_12month", two:"static_encounters_variables.csv", three:"all_encounters_variables.csv", four:"final_thoracic_patients.csv", five:"final_patients_treatments_ALL.csv", six:"Patients_including_no_treatment.csv", seven:"flatiron/processed/full_df.csv"}#, seven:"flatiron/augmented_featuriized_data_first_line.csv"}
# synthetic_lang_mappings = {one: synthetic_lang_one, two: synthetic_lang_two, three: synthetic_lang_three, four: synthetic_lang_four, five: synthetic_lang_five, six: synthetic_lang_six, seven: synthetic_lang_seven}
# pred_mortality_feat="predicted_mortality"


def obtain_numeric_categorical_value_count(dataset):
    # lang = Language(data=train_data, precomputed=col_thres_mappings, lang=synthetic_lang)
    col_ls = list(dataset.data.columns)
    numeric_Count = len(col_ls) - 2
    category_count = ()

    return numeric_Count, category_count

def obtain_feat_range_mappings(train_dataset):
    # cln_names = list(train_dataset.data.columns)
    cln_names = list(train_dataset.num_cols)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

def drop_unnecessary_columns(cancer_df, missing_ratio_thres=0.5):
    if "EnhancedCohort" in cancer_df.columns:
        cancer_df = cancer_df.drop(columns = ["EnhancedCohort"])
    
    for col in cancer_df.columns:
        # missing_ratio = np.sum(cancer_df[col].isna().to_numpy())*1.0/len(cancer_df)
        # if (missing_ratio > missing_ratio_thres):
        #     cancer_df = cancer_df.drop(columns = [col])
        #     continue
        if "date" in col.lower():
            cancer_df = cancer_df.drop(columns = [col])
            continue
        # unique_vals = len(cancer_df[col].unique())
        # print("unique values::", col, unique_vals)
    # 
    return cancer_df
        
    


def read_data(data_folder, dataset_name, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2, select_feat_ratio=-1):
    #  change here if you want to import your own data
    data_path = os.path.join(data_folder, dataset_name, "cardio_train2.csv")
    label_column = "cardio"
    id_column = "id"
    input_df = pd.read_csv(data_path, sep=";")
    # cancer_df = cancer_df.drop("")
    # filtered_clns = [key for key in cancer_df.columns if key.endswith("mean") or key.endswith("std") or key.endswith("count") or key.endswith("last") or key.endswith("first")]
    print(input_df.columns)
        
    column_ls = list(input_df.columns)
    if select_feat_ratio > 0:
        random.shuffle(column_ls)
        column_ls = list(input_df.columns)
        print("selected column count::", len(column_ls))

    print("positive sample count::", np.sum(input_df[label_column] == 1))
    print("negative sample count::", np.sum(input_df[label_column] == 0))
    random_sample_ids = list(range(len(input_df)))
    random.shuffle(random_sample_ids)
    random_sample_ids = np.array(random_sample_ids)
    train_count = int(len(input_df)*train_ratio)
    valid_count = int(len(input_df)*valid_ratio)
    test_count = int(len(input_df)*test_ratio)
    train_ids = random_sample_ids[0:train_count]
    valid_ids = random_sample_ids[train_count: train_count + valid_count]
    test_ids = random_sample_ids[train_count + valid_count:train_count + valid_count+test_count]
    train_df = input_df.iloc[train_ids]
    valid_df = input_df.iloc[valid_ids]
    test_df = input_df.iloc[test_ids]
    
    return train_df, valid_df, test_df, label_column, id_column

class EHRDataset(Dataset):
    def __init__(self, data, drop_cols, balance, lang, cat_unique_count_mappings=None, cat_unique_vals_id_mappings=None, cat_id_unique_vals_mappings=None, other_data=None, pid_attr="id", label_attr="label"):
        self.data = data
        self.drop_cols = drop_cols
        self.data.index = list(range(len(self.data)))
        self.pid_attr = pid_attr
        # self.patient_ids = self.data[self.pid_attr].unique().tolist()
        self.patient_ids = list(range(len(self.data[self.pid_attr])))
        self.labels = torch.from_numpy(self.data[label_attr].to_numpy()).type(torch.float)
        self.label_attr = label_attr
        self.lang = lang
        self.other_data = other_data
        self.cat_cols = list(set(self.lang.CAT_FEATS))
        self.cat_cols = [col for col in self.cat_cols if col in self.data.columns]
        self.cat_cols.sort()
        self.lang.CAT_FEATS = self.cat_cols
        self.num_cols = [col for col in data.columns if col not in self.cat_cols and not col == self.pid_attr and not col in drop_cols and not col == self.label_attr]
        self.num_cols.sort()
        if cat_unique_count_mappings is not None and cat_unique_vals_id_mappings is not None and cat_id_unique_vals_mappings is not None:
            self.cat_unique_count_mappings = cat_unique_count_mappings
            self.cat_unique_vals_id_mappings = cat_unique_vals_id_mappings
            self.cat_id_unique_vals_mappings = cat_id_unique_vals_mappings
        else:
            self.cat_unique_count_mappings = {}
            self.cat_unique_vals_id_mappings = {}
            self.cat_id_unique_vals_mappings = {}
            self.create_cat_feat_mappings()
        self.cat_unique_val_count_ls = [self.cat_unique_count_mappings[col] for col in self.cat_cols]
        
        self.cat_sum_count = sum([len(self.cat_unique_vals_id_mappings[feat]) for feat in self.cat_unique_vals_id_mappings])
        self.init_onehot_mats()
        if balance:

            most = self.data[label_attr].value_counts().max()
            for label in list(self.data[label_attr].value_counts().index):
                match = torch.nonzero(self.labels.view(-1) == label).view(-1).tolist()
                samples = [random.choice(match) for _ in range(most-len(match))]
                self.patient_ids.extend(samples)

                
        # random.shuffle(self.patient_ids)

    def transform_imputed_data(self):
        start_cln_idx = len(self.num_cols)
        self.imputed_features[:,0:start_cln_idx] = self.transformed_features[:, 0:start_cln_idx]

        imputed_cat_feat_ls = []
        for idx in range(len(self.cat_cols)):
            curr_cat_unique_count = self.cat_unique_count_mappings[self.cat_cols[idx]]
            imputed_cat_feat_ls.append(torch.argmax(self.transformed_features[:,start_cln_idx:start_cln_idx + curr_cat_unique_count], dim=1))
            start_cln_idx = start_cln_idx + curr_cat_unique_count

        self.imputed_features[:,len(self.num_cols):] = torch.stack(imputed_cat_feat_ls, dim=1)

    def create_imputed_data(self):
        self.imputed_features = torch.clone(self.features)

        if torch.sum(torch.isnan(self.imputed_features)) > 0:
            self.transform_imputed_data()

    def init_data(self):
        # feat_onehot_mat_mappings = [item[0][6] for item in data][0]
        X_num_tar = torch.from_numpy(self.r_data[self.num_cols].to_numpy()).type(torch.float)
        origin_X_num_tar = self.data[self.num_cols].to_numpy()
        
        if len(self.cat_cols) > 0:
            curr_cat_data = self.r_data[self.cat_cols].to_numpy()
            curr_cat_data[curr_cat_data != curr_cat_data] = -1
            X_cat_tar = torch.from_numpy(curr_cat_data.astype(float)).type(torch.float)
            origin_X_cat_tar = self.data[self.cat_cols].to_numpy()
            
            X_cat_onehot_ls = []
            for idx in range(len(self.cat_cols)):
                cat_feat = X_cat_tar[:, idx].type(torch.long)
                curr_feat_mappings = self.feat_onehot_mat_mappings[self.cat_cols[idx]]
                no_nan_cat_feat = cat_feat[cat_feat >= 0]
                all_onehot_encodings = torch.zeros((len(cat_feat), self.cat_unique_count_mappings[self.cat_cols[idx]]))
                all_onehot_encodings[:] = np.nan
                all_onehot_encodings[cat_feat >= 0] = curr_feat_mappings[no_nan_cat_feat]
                X_cat_onehot_ls.append(all_onehot_encodings)
            
            
            # X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat_tar[:, idx].type(torch.long)] for idx in range(len(self.cat_cols))]
            
            X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1)        
            self.transformed_features = torch.cat([X_num_tar, X_cat_onehot], dim=-1)
            X_cat_tar[X_cat_tar <0] = np.nan
            self.features = torch.cat([X_num_tar, X_cat_tar], dim=-1)
            self.origin_features = np.concatenate([origin_X_num_tar, origin_X_cat_tar], axis=-1)
        else:
            self.transformed_features = X_num_tar
            self.features = X_num_tar
            self.origin_features = origin_X_num_tar
            
        # if self.treatment_attr is not None:
        #     self.treatment_array = torch.from_numpy(self.r_data[self.treatment_attr].to_numpy()).type(torch.float)
        # if self.treatment_graph is not None:
        #     self.init_treatment_graph()

        
        # self.outcome_array = torch.from_numpy(self.r_data[self.outcome_attr].to_numpy()).type(torch.float)
        # if self.count_outcome_attr is not None:
        #     self.count_outcome_array = torch.from_numpy(self.r_data[self.count_outcome_attr].to_numpy()).type(torch.float)
        # else:
        #     self.count_outcome_array = None
        
        # if self.dose_attr is not None:
        #     self.dose_array = torch.from_numpy(self.r_data[self.dose_attr].to_numpy()).type(torch.float)
        # else:
        #     self.dose_array = None
    def init_onehot_mats(self):
        self.feat_onehot_mat_mappings = dict()
        for cat_feat in self.cat_unique_count_mappings:            
            self.feat_onehot_mat_mappings[cat_feat] = torch.eye(self.cat_unique_count_mappings[cat_feat])

    def create_cat_feat_mappings(self):
        for cat_feat in self.cat_cols:
            unique_vals = list(self.data[cat_feat].unique())
            
            self.cat_unique_vals_id_mappings[cat_feat] = dict()
            self.cat_id_unique_vals_mappings[cat_feat] = dict()
            unique_vals = [val for val in unique_vals if not (type(val) is not str and np.isnan(val))]
            self.cat_unique_count_mappings[cat_feat] = len(unique_vals)
            unique_vals.sort()
            for val_idx, unique_val in enumerate(unique_vals):
                self.cat_id_unique_vals_mappings[cat_feat][val_idx] = unique_val
                self.cat_unique_vals_id_mappings[cat_feat][unique_val] = val_idx

    def rescale_data(self, feat_range_mappings):
        self.r_data = self.data.copy()
        self.feat_range_mappings = feat_range_mappings
        for feat in list(self.r_data.columns):
            if feat == self.label_attr:
                continue
            # if not feat == 'PAT_ID' and (not feat in self.cat_cols):
            if feat in self.num_cols:
                lb, ub = feat_range_mappings[feat][0], feat_range_mappings[feat][1]
                if lb < ub:
                    self.r_data[feat] = (self.data[feat]-lb)/(ub-lb)
                else:
                    self.r_data[feat] = 0
            else:
                if feat in self.cat_cols:
                    self.r_data[feat] = self.data[feat]
                    for unique_val in self.cat_unique_vals_id_mappings[feat]:
                        self.r_data.loc[self.r_data[feat] == unique_val, feat] = self.cat_unique_vals_id_mappings[feat][unique_val]
                    print(list(self.r_data[feat].unique()))
        self.init_data()

    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        
        idx = self.patient_ids[idx]

        appts2 = self.features[idx]
        X = self.transformed_features[idx]
        X_num = self.imputed_features[idx, 0:len(self.num_cols)]
        X_cat = self.imputed_features[idx, len(self.num_cols):].long()
        
        if self.other_data is None:  
            all_other_pats2 = torch.ones(len(self.features)).bool()  #torch.cat([self.features[0:idx], self.features[idx+1:]], dim=0)
            all_other_pats2[idx] = False
        else:
            all_other_pats2 = torch.ones(len(self.other_data)).bool()
        
        y = self.labels[idx]
        appts = self.r_data[self.r_data.index == idx]
        # appts = self.r_data.loc[self.r_data['PAT_ID'] == self.patient_ids[idx]]
        if self.other_data is None:
            all_other_pats = self.r_data[self.r_data.index != idx]
        else:
            all_other_pats = self.other_data
            
        abnormal_feature_indicator = self.abnormal_feature_indicators[idx]
        activated_indicator = self.activated_indicators[idx]
        
        # # full_pats = self.r_data#.loc[self.r_data['PAT_ID'] != self.patient_ids[idx]]
        
            
        
        # y = torch.tensor(m, dtype=torch.float)
        # X_pd = appts.drop(self.drop_cols, axis=1)
        
        # # X_num = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.num_cols].to_numpy(dtype=np.float64)][0]
        # # X_cat = [torch.tensor(i, dtype=torch.float) for i in X_pd[self.cat_cols].to_numpy(dtype=np.float64)][0].type(torch.long)
        
        # X_num = torch.from_numpy(X_pd[self.num_cols].to_numpy()).type(torch.float)
        # X_cat = torch.from_numpy(X_pd[self.cat_cols].to_numpy()).type(torch.long)
        
        # if len(self.cat_cols) > 0:
        #     X_cat_onehot_ls = [self.feat_onehot_mat_mappings[self.cat_cols[idx]][X_cat.view(-1)[idx]] for idx in range(len(self.cat_cols))]
            
        #     X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1).unsqueeze(0)
            
        #     X = [torch.cat([X_num, X_cat_onehot], dim=-1)]
        # else:
        #     X = X_num
        #zero pad
        # X.extend([torch.tensor([0]*len(X[0]), dtype=torch.float) ]*(len(X)-self.patient_max_appts))
        return (all_other_pats2, appts2, X, idx, appts, all_other_pats, (X_num, X_cat), (abnormal_feature_indicator, activated_indicator)), y
    
    @staticmethod
    def collate_fn(data):
        all_other_pats_ls2 = [item[0][0] for item in data]
        X_pd_ls2 = [item[0][1] for item in data]
        X_ls = [item[0][2].view(1,-1) for item in data]
        X_sample_ids = [item[0][3] for item in data]
        X_pd_ls = [item[0][4] for item in data]
        all_other_pats_ls = [item[0][5] for item in data]
        X_num_tensor = torch.stack([item[0][6][0] for item in data])
        X_cat_tensor = torch.stack([item[0][6][1] for item in data])
        abnormal_feature_indicator = torch.stack([item[0][7][0] for item in data])
        activated_indicator = torch.stack([item[0][7][1] for item in data])
        # patient_id_ls = [item[0][5] for item in data]
        # full_data = [item[0][6] for item in data][0]
        # num_cols = [item[0][6] for item in data][0]
        # cat_cols = [item[0][7] for item in data][0]
        # feat_onehot_mat_mappings = [item[0][8] for item in data][0]
        
        # X_pd_array = pd.concat(X_pd_ls)
        
        # X_num_tar = torch.from_numpy(X_pd_array[num_cols].to_numpy()).type(torch.float)
        # X_cat_tar = torch.from_numpy(X_pd_array[cat_cols].to_numpy()).type(torch.float)
        # if len(cat_cols) > 0:
        
        #     X_cat_onehot_ls = [feat_onehot_mat_mappings[cat_cols[idx]][X_cat_tar[:, idx].type(torch.long)] for idx in range(len(cat_cols))]
            
        #     X_cat_onehot = torch.cat(X_cat_onehot_ls, dim=-1)
            
        #     X = torch.cat([X_num_tar, X_cat_onehot], dim=-1)
        # else:
        #     X = X_num_tar
        
        X_tensor = torch.cat(X_ls)
        # assert torch.norm(X_tensor - X) <= 0
        y_ls = [item[1].view(1,-1) for item in data]
        y_tensor = torch.cat(y_ls)
        X_sample_ids_tensor = torch.tensor(X_sample_ids)
        return (all_other_pats_ls, all_other_pats_ls2, X_pd_ls2, X_tensor, X_sample_ids_tensor, X_pd_ls, (X_num_tensor, X_cat_tensor), (abnormal_feature_indicator, activated_indicator)), y_tensor

    def set_abnormal_feature_vals(self, abnormal_feature_indicators, activated_indicators, val_count):
        if abnormal_feature_indicators is None:
            abnormal_feature_indicators = torch.zeros((self.features.shape[0], self.origin_features.shape[1]), dtype=torch.bool)
        if activated_indicators is None:
            activated_indicators = torch.zeros((self.features.shape[0], self.origin_features.shape[1], val_count), dtype=torch.bool)
        
        
        self.abnormal_feature_indicators = abnormal_feature_indicators
        self.activated_indicators = activated_indicators

def reset_data_index(train_data):
    train_data = train_data.reset_index()
    train_data = train_data.rename(columns = {"index": "PAT_ID"})
    return train_data






def transform_labels2(train_data, column_ls):
    for cln in column_ls:
        train_data[cln] = train_data[cln].apply(lambda x: int(x))

    return train_data[column_ls].apply(list, axis=1)

    
def create_train_val_test_datasets(train_data, valid_data, test_data, rule_lang, args):
    pid_attr=args.id_cln
    label_attr = args.label_cln
    DROP_FEATS = rule_lang.DROP_FEATS
    
    all_data = pd.concat([train_data, valid_data, test_data])
    # if args.dataset_name == four:
    #     all_dataset = EHRDataset(data= all_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False, lang = lang)
    #     train_dataset = EHRDataset(data= train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False, lang = lang, cat_unique_count_mappings=all_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=all_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=all_dataset.cat_id_unique_vals_mappings)
    # else:
    all_dataset = EHRDataset(data= all_data, drop_cols=DROP_FEATS, balance=True, lang = rule_lang, pid_attr=pid_attr, label_attr=label_attr)
    train_dataset = EHRDataset(data= train_data, drop_cols=DROP_FEATS, balance=True, lang = rule_lang, cat_unique_count_mappings=all_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=all_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=all_dataset.cat_id_unique_vals_mappings, pid_attr=pid_attr, label_attr=label_attr)
    
    feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    
    train_dataset.rescale_data(feat_range_mappings) 

    
    train_valid_dataset = EHRDataset(data= pd.concat([train_data, valid_data]), drop_cols=DROP_FEATS, balance=True, lang = rule_lang, cat_unique_count_mappings=all_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=all_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=all_dataset.cat_id_unique_vals_mappings,pid_attr=pid_attr, label_attr=label_attr)
    valid_dataset = EHRDataset(data = valid_data, drop_cols=DROP_FEATS, balance=False, lang = rule_lang, cat_unique_count_mappings=all_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=all_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=all_dataset.cat_id_unique_vals_mappings, other_data=train_dataset.r_data,pid_attr=pid_attr, label_attr=label_attr)
    test_dataset = EHRDataset(data = test_data, drop_cols=DROP_FEATS, balance=False, lang = rule_lang, cat_unique_count_mappings=all_dataset.cat_unique_count_mappings, cat_unique_vals_id_mappings=all_dataset.cat_unique_vals_id_mappings, cat_id_unique_vals_mappings=all_dataset.cat_id_unique_vals_mappings, other_data=train_dataset.r_data,pid_attr=pid_attr, label_attr=label_attr)
    
    valid_dataset.rescale_data(feat_range_mappings) 
    test_dataset.rescale_data(feat_range_mappings)
        
    return train_dataset, train_valid_dataset, valid_dataset, test_dataset, feat_range_mappings