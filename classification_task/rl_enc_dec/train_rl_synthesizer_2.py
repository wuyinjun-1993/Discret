import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rl_synthesizer import *
from rl_synthesizer_2 import *
from rl_synthesizer_4 import *
from rl_synthesizer_5 import *
from rl_synthesizer_6 import *
from rl_synthesizer_7 import *

from rl_synthesizer_3 import Transition

import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import synthetic_lang
import logging
from datetime import datetime
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import operator

from sklearn.metrics import brier_score_loss
from scipy import stats

import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.EHR_datasets import *
from utils_mortality.metrics import metrics_maps

# class EHRDataset(Dataset):
#     def __init__(self, data, drop_cols,patient_max_appts, balance):
#         self.data = data
#         self.patient_max_appts = patient_max_appts
#         self.drop_cols = drop_cols
#         self.patient_ids = self.data['PAT_ID'].unique().tolist()
#         if balance:
#             most = self.data['label'].value_counts().max()
#             for label in self.data['label'].unique():
#                 match  = self.data.loc[self.data['label'] == label]['PAT_ID'].to_list()
#                 samples = [random.choice(match) for _ in range(most-len(match))]
#                 self.patient_ids.extend(samples)
                
#         # random.shuffle(self.patient_ids)

#     def rescale_data(self, feat_range_mappings):
#         self.r_data = self.data.copy()
#         self.feat_range_mappings = feat_range_mappings
#         for feat in feat_range_mappings:
#             if not feat == 'PAT_ID':
#                 lb, ub = feat_range_mappings[feat][0], feat_range_mappings[feat][1]
#                 self.r_data[feat] = (self.data[feat]-lb)/(ub-lb)

#     def __len__(self):
#         return len(self.patient_ids)
#     def __getitem__(self, idx):
#         appts = self.r_data.loc[self.r_data['PAT_ID'] == self.patient_ids[idx]]
#         all_other_pats = self.r_data.loc[self.r_data['PAT_ID'] != self.patient_ids[idx]]
#         m = [appts['label'].max()]
#         y = torch.tensor(m, dtype=torch.float)
#         X_pd = appts.drop(self.drop_cols, axis=1)
#         X = [torch.tensor(i, dtype=torch.float) for i in X_pd.to_numpy(dtype=np.float64)]
#         #zero pad
#         X.extend([torch.tensor([0]*len(X[0]), dtype=torch.float) ]*(len(X)-self.patient_max_appts))
#         return (all_other_pats, appts, X), y
    
#     @staticmethod
#     def collate_fn(data):
#         all_other_pats_ls = [item[0][0] for item in data]
#         X_pd_ls = [item[0][1] for item in data]
#         X_ls = [item[0][2][0].view(1,-1) for item in data]
#         X_tensor = torch.cat(X_ls)
#         y_ls = [item[1].view(1,-1) for item in data]
#         y_tensor = torch.cat(y_ls)
#         return (all_other_pats_ls, X_pd_ls, X_tensor), y_tensor

class EHRDeathsOnlyDataset(Dataset):
    def __init__(self, data, drop_cols,patient_max_appts):
        self.data = data
        self.drop_cols = drop_cols
        self.patient_max_appts = patient_max_appts
        self.patient_ids = sorted(self.data['PAT_ID'].unique())
        self.deaths_only = []
        for pat_id in sorted(self.data['PAT_ID'].unique()):
            appts = self.data.loc[self.data['PAT_ID'] == pat_id]
            y = torch.tensor([appts['label'].max()])
            if y[0] == 1:
                self.deaths_only.append(pat_id)
    def __len__(self):
        return len(self.deaths_only)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.deaths_only[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.deaths_only[idx]]
        y = torch.tensor([appts['label'].max()])
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.tensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.tensor([0]*len(X[0]))]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y

def obtain_numeric_categorical_value_count(dataset):
    col_ls = list(dataset.data.columns)
    numeric_Count = len(col_ls) - 2
    category_count = ()

    return numeric_Count, category_count


class Trainer:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p):
        self.dqn = DQN(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        all_rand_ids = torch.randperm(len(self.train_dataset))
        for episode_i, sample_idx in iterator:
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, program=program, epsilon=self.epsilon)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
                #derive reward
                reward = db_cons if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
                #record transition in buffer
                if done:
                    next_program = None
                transition = Transition(X,X_pd, program, atom, next_program, reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model()
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            success_rate = (success / (episode_i + 1)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        for episode_i, val in iterator:
            (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, program=program, epsilon=0)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                #check constraints
                x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
                db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                #derive reward
                done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
                if done:
                    next_program = None
                #update next step
                if done: #stopping condition
                    if self.is_log:
                        msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                        self.logger.log(level=logging.DEBUG, msg=msg)
                    if y == y_pred: success += 1
                    else: failure += 1
                    y_true_ls.append(y.item())
                    y_pred_ls.append(y_pred)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats

            y_true_array = np.array(y_true_ls, dtype=float)
            y_pred_array = np.array(y_pred_ls, dtype=float)
            y_pred_array[y_pred_array < 0] = 0.5
            if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
            #     recall = 0
            #     f1 = 0
                auc_score= 0
            else:
                auc_score = roc_auc_score(y_true_array, y_pred_array)


            # Print information
            success_rate = (success / (episode_i + 1)) * 100.00
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        self.test_epoch(0)
        for i in range(1, self.epochs + 1):
            self.train_epoch(i)
            self.test_epoch(i)


class Trainer3:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed):
        self.dqn = DQN4(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        if self.is_log:
            self.logger = logging.getLogger()
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls

    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        all_rand_ids = torch.randperm(len(self.train_dataset))
        for episode_i, sample_idx in iterator:
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            program_atom_ls = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=self.epsilon)

                atom_ls = self.identify_op(X_pd, atom)

                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)

                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                #record transition in buffer
                if done:
                    next_program = None
                transition = Transition(X,X_pd, program, atom, next_program, reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model()
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            success_rate = (success / (episode_i + 1)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        for episode_i, val in iterator:
            (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            program_atom_ls = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                atom_ls = self.identify_op(X_pd, atom)

                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                #derive reward
                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                if done:
                    next_program = None
                #update next step
                if done: #stopping condition
                    if self.is_log:
                        msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                        self.logger.log(level=logging.DEBUG, msg=msg)
                    if y == y_pred: success += 1
                    else: failure += 1
                    y_true_ls.append(y.item())
                    y_pred_ls.append(y_pred)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats

            y_true_array = np.array(y_true_ls, dtype=float)
            y_pred_array = np.array(y_pred_ls, dtype=float)
            y_pred_array[y_pred_array < 0] = 0.5
            if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
            #     recall = 0
            #     f1 = 0
                auc_score= 0
            else:
                auc_score = roc_auc_score(y_true_array, y_pred_array)


            # Print information
            success_rate = (success / (episode_i + 1)) * 100.00
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        self.test_epoch(0)
        for i in range(1, self.epochs + 1):
            self.train_epoch(i)
            self.test_epoch(i)


# class Trainer3_2:
#     def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed):
#         self.dqn = DQN4(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed)
#         self.epsilon = epsilon
#         self.batch_size = batch_size
#         self.epsilon_falloff = epsilon_falloff
#         self.epochs = epochs
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
#         self.lang = lang
#         self.target_update = target_update
#         self.program_max_len = program_max_len
#         self.is_log = is_log
#         if self.is_log:
#             self.logger = logging.getLogger()

#     def identify_op_ls(self, batch_size:int, atom:dict):

#         atom_ls = []        

#         # atom1 = [dict()]*batch_size
#         atom1 = []
#         for _ in range(batch_size):
#             atom1.append(dict())
#         for k in atom:
#             # if k not in self.lang.syntax["num_feat"]:
#             if type(k) is not tuple:
#                 if type(atom[k]) is not dict:
#                     for atom_id in range(batch_size):
#                         atom1[atom_id][k] = atom[k]
#                 else:
#                     # atom1[k] = [None]*batch_size
#                     for selected_item in atom[k]:
#                         sample_ids = atom[k][selected_item]
#                         for sample_id in sample_ids:
#                             atom1[sample_id.item()][k] = selected_item
#             else:
                
#                 # atom1[k] = [None]*batch_size
#                 # atom1[k + "_prob"] = [None]*batch_size

#                 for selected_item in atom[k][2]:
#                     sample_ids = atom[k][2][selected_item]
#                     for sample_id_id in range(len(sample_ids)):
#                         atom1[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][0][selected_item][sample_id_id]
#                         atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
#                         # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


#                 # atom1[k] = atom[k][0][0]
#                 # atom1[k + "_prob"] = atom[k][1][0]
#                 # atom1[k + "_sample_ids"] = atom[k][2][0]
#         for sample_id in range(len(atom1)):
#             atom1[sample_id]["num_op"] = operator.__ge__   


#         atom2 = []
#         for _ in range(batch_size):
#             atom2.append(dict())
#         for k in atom:
#             # if k not in self.lang.syntax["num_feat"]:
#             if type(k) is not tuple:
#                 if type(atom[k]) is not dict:
#                     for atom_id in range(batch_size):
#                         atom2[atom_id][k] = atom[k]
#                 else:
#                     for selected_item in atom[k]:
#                         sample_ids = atom[k][selected_item]
#                         for sample_id in sample_ids:
#                             atom2[sample_id.item()][k] = selected_item
#             else:
                
#                 for selected_item in atom[k][2]:
#                     sample_ids = atom[k][2][selected_item]
#                     for sample_id_id in range(len(sample_ids)):
#                         atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
#                         atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
#                         # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


#                 # atom1[k] = atom[k][0][0]
#                 # atom1[k + "_prob"] = atom[k][1][0]
#                 # atom1[k + "_sample_ids"] = atom[k][2][0]
#         for sample_id in range(len(atom2)):
#             atom2[sample_id]["num_op"] = operator.__le__  


#         # atom2 = dict()
#         # for k in atom:
#         #     if k not in self.lang.syntax["num_feat"]:
#         #         atom2[k] = atom[k]
#         #     else:
#         #         atom2[k] = atom[k][0][1]
#         #         atom2[k + "_prob"] = atom[k][1][1]
#         # atom2["num_op"] = operator.__le__
#         atom_ls.append(atom1)
#         atom_ls.append(atom2)
            
#         return atom_ls
#     def identify_op(self, X:pd, atom:dict):

#         atom_ls = []
        

#         atom1 = dict()
#         for k in atom:
#             if k not in self.lang.syntax["num_feat"]:
#                 atom1[k] = atom[k]
#             else:
#                 atom1[k] = atom[k][0][0]
#                 atom1[k + "_prob"] = atom[k][1][0]

#         atom1["num_op"] = operator.__ge__

#         atom2 = dict()
#         for k in atom:
#             if k not in self.lang.syntax["num_feat"]:
#                 atom2[k] = atom[k]
#             else:
#                 atom2[k] = atom[k][0][1]
#                 atom2[k + "_prob"] = atom[k][1][1]
#         atom2["num_op"] = operator.__le__
#         atom_ls.append(atom1)
#         atom_ls.append(atom2)
            
#         return atom_ls

#     def get_test_decision_from_db(self, data: pd.DataFrame):
#         if data.shape[0] == 0:
#             return -1
#         return data['label'].value_counts().idxmax()

#     def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
#         if len(data) == 0:
#             return 0
#         same = data.loc[data['label'] == y]["PAT_ID"].nunique()
#         total = data['PAT_ID'].nunique()
#         return same / total

#     def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
#         return lang.evaluate_atom_on_sample(atom, X)

#     def check_program_constraint(self, prog: list) -> bool:
#         return len(prog) < self.program_max_len
    
#     def check_db_constrants_ls(self, data_ls,  y_ls):
#         # if len(data) == 0:
#         #     return 0
#         rwd_ls = []
#         for idx in range(len(data_ls)):
#             data = data_ls[idx]
#             y = int(y_ls[idx].item())
#             same = data.loc[data['label'] == y]["PAT_ID"].nunique()
#             total = data['PAT_ID'].nunique()
#             # if total == 0:
#             #     rwd_ls.append(0) 
#             # else:
#             rwd_ls.append(same / total) 
#         return np.array(rwd_ls)

#     def train_epoch(self, train_loader, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
#         # iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
#         # all_rand_ids = torch.randperm(len(self.train_dataset))
#         # for episode_i, sample_idx in iterator:
#         iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
#         for episode_i, val in iterator:
#             (all_other_pats_ls, X_pd_ls, X), y = val
#             # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
#             # (all_other_pats, X_pd, X), y = val
#             program = []
#             program_str = []
#             program_atom_ls = []
#             while True: # episode
#                 atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=self.epsilon)

#                 # atom_ls = self.identify_op_ls(len(X_pd_ls), atom)
#                 atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
#                 reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

#                 next_program = program.copy()
#                 next_program_str = program_str.copy()
#                 for new_atom_ls in atom_ls_ls:

#                     curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

#                     next_program.append(torch.stack(curr_vec_ls))

#                     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

#                     for vec_idx in range(len(curr_vec_ls)):
#                         vec = curr_vec_ls[vec_idx]
#                         atom_str = curr_atom_str_ls[vec_idx]
                        
#                         next_program_str[vec_idx].append(atom_str)
#                         program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
#                         reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])



#                 # next_program = program.copy()
#                 # next_program_str = program_str.copy()
#                 # for new_atom in atom_ls:
#                 #     next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
#                 #     # atom["num_op"] = atom_op
                    
                    
#                 #     next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
#                     # program_atom_ls.append(new_atom)

#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats)
#                 # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
#                 #check constraints
#                 # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 db_cons = self.check_db_constrants_ls(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
#                 #derive reward
#                 reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
#                 done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
#                 #record transition in buffer
#                 if done:
#                     next_program = None
#                 transition = Transition(X,X_pd_ls, program, atom, next_program, reward)
#                 self.dqn.observe_transition(transition)
#                 #update model
#                 loss = self.dqn.optimize_model_ls()
#                 sum_loss += loss
#                 #update next step
#                 if done: #stopping condition
#                     if reward > 0.5: success += 1
#                     else: failure += 1
#                     break
#                 else:
#                     program = next_program
#                     program_str = next_program_str
#                     all_other_pats = next_all_other_pats

#             # Update the target net
#             if episode_i % self.target_update == 0:
#                 self.dqn.update_target()
#             # Print information
#             success_rate = (success / (episode_i + 1)) * 100.0
#             avg_loss = sum_loss/(episode_i+1)
#             desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
#             iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
#         self.epsilon *= self.epsilon_falloff

#     def test_epoch(self, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
#         y_true_ls=[]
#         y_pred_ls=[]
#         for episode_i, val in iterator:
#             (all_other_pats, X_pd, X), y = val
#             program = []
#             program_str = []
#             program_atom_ls = []
#             while True: # episode
#                 atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
#                 atom_ls = self.identify_op(X_pd, atom)

#                 next_program = program.copy()
#                 next_program_str = program_str.copy()
#                 for new_atom in atom_ls:
#                     next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
#                     # atom["num_op"] = atom_op
                    
                    
#                     next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
#                     program_atom_ls.append(new_atom)
#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
#                 # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
#                 #check constraints
#                 # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
#                 db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
#                 #derive reward
#                 done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
#                 if done:
#                     next_program = None
#                 #update next step
#                 if done: #stopping condition
#                     if self.is_log:
#                         msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
#                         self.logger.log(level=logging.DEBUG, msg=msg)
#                     if y == y_pred: success += 1
#                     else: failure += 1
#                     y_true_ls.append(y.item())
#                     y_pred_ls.append(y_pred)
#                     break
#                 else:
#                     program = next_program
#                     program_str = next_program_str
#                     all_other_pats = next_all_other_pats

#             y_true_array = np.array(y_true_ls, dtype=float)
#             y_pred_array = np.array(y_pred_ls, dtype=float)
#             y_pred_array[y_pred_array < 0] = 0.5
#             if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#             #     recall = 0
#             #     f1 = 0
#                 auc_score= 0
#             else:
#                 auc_score = roc_auc_score(y_true_array, y_pred_array)


#             # Print information
#             success_rate = (success / (episode_i + 1)) * 100.00
#             avg_loss = sum_loss/(episode_i+1)
#             desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
#             iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
#     def run(self):
#         # self.test_epoch(0)
#         train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
#         for i in range(1, self.epochs + 1):
#             self.train_epoch(train_loader, i)
#             self.test_epoch(i)



class Trainer2:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, seed):
        self.dqn = DQN4(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, seed=seed)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.batch_size = batch_size
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        all_rand_ids = torch.randperm(len(self.train_dataset))
        # all_rand_ids = torch.tensor(list(range((len(self.train_dataset)))))
        for episode_i, sample_idx in iterator:
            # (all_other_pats, X_pd, X), y = val
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = []
            program_atom_ls = []
            prev_reward = 0
            while True: # episode
                atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=self.epsilon)
                atom_ls = self.identify_op(X_pd, atom)

                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                #x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                #record transition in buffer
                if done:
                    next_program = None
                transition = Transition(X, X_pd,program, atom, next_program, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model()
                sum_loss += loss
                # print(loss)
                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats
                    
                prev_reward = reward

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            success_rate = (success / (episode_i + 1)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)


                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
    def run(self):
        self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        for i in range(1, self.epochs + 1):
            self.train_epoch( i)
            self.test_epoch(i)

class Trainer2_2:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed, work_dir):
        self.dqn = DQN4(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed)
        self.epsilon = epsilon
        self.work_dir = work_dir
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        self.batch_size = batch_size
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()
    
    def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        for data in data_ls:
            if len(data) == 0:
                label_ls.append(-1)
                continue
            label = data['label'].value_counts().idxmax()
            label_ls.append(label)
        return label_ls

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_db_constrants_ls(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            data = data_ls[idx]
            y = int(y_ls[idx].item())
            same = data.loc[data['label'] == y]["PAT_ID"].nunique()
            total = data['PAT_ID'].nunique()
            if total == 0:
                rwd_ls.append(0) 
            else:
                rwd_ls.append(same / total) 
        return np.array(rwd_ls)

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    
    def identify_op_ls(self, batch_size:int, atom:dict):

        atom_ls = []        

        # atom1 = [dict()]*batch_size
        atom1 = []
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size

                for selected_item in atom[k][2]:
                    sample_ids = atom[k][2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        atom1[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][0][selected_item][sample_id_id]
                        atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   


        atom2 = []
        for _ in range(batch_size):
            atom2.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom2[atom_id][k] = atom[k]
                else:
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom2[sample_id.item()][k] = selected_item
            else:
                
                for selected_item in atom[k][2]:
                    sample_ids = atom[k][2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
                        atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom2)):
            atom2[sample_id]["num_op"] = operator.__le__  


        # atom2 = dict()
        # for k in atom:
        #     if k not in self.lang.syntax["num_feat"]:
        #         atom2[k] = atom[k]
        #     else:
        #         atom2[k] = atom[k][0][1]
        #         atom2[k + "_prob"] = atom[k][1][1]
        # atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    def train_epoch(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        # all_rand_ids = torch.randperm(len(self.train_dataset))
        # for episode_i, sample_idx in iterator:
        for episode_i, val in iterator:
            (all_other_pats_ls, X_pd_ls, X), y = val
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = [[] for _ in range(len(X_pd_ls))]
            program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            while True: # episode
                atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=self.epsilon)
                atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom_ls in atom_ls_ls:

                    curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                    next_program.append(torch.stack(curr_vec_ls))

                    curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    for vec_idx in range(len(curr_vec_ls)):
                        vec = curr_vec_ls[vec_idx]
                        atom_str = curr_atom_str_ls[vec_idx]
                        
                        next_program_str[vec_idx].append(atom_str)
                        program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                        reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # atom["num_op"] = atom_op
                    
                    
                    # next_program_str = next_program_str + []
                    
                    # program_atom_ls.append(new_atom_ls)
                #apply new atom
                next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats_ls)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                #record transition in buffer
                if done:
                    next_program = None
                transition = Transition(X, X_pd_ls,program, atom, next_program, reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model_ls()
                # print(loss)
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(reward > 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            total_count = ((episode_i + 1)*self.batch_size)
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    
    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 28:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)

                # if episode_i == self.batch_size:
                #     print(y_true_array.reshape(-1))
                #     print(y_pred_array.reshape(-1))

                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        return y_pred_array
    
    def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None):
        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():

            for episode_i, val in iterator:
                (all_other_pats_ls, X_pd_ls, X), y = val
                # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
                program = []
                program_str = [[] for _ in range(len(X_pd_ls))]
                program_atom_ls = [[] for _ in range(len(X_pd_ls))]
                while True: # episode
                    atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
                    atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                    reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom_ls in atom_ls_ls:

                        curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                        next_program.append(torch.stack(curr_vec_ls))

                        curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                        for vec_idx in range(len(curr_vec_ls)):
                            vec = curr_vec_ls[vec_idx]
                            atom_str = curr_atom_str_ls[vec_idx]
                            
                            next_program_str[vec_idx].append(atom_str)
                            program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                            reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                        # atom["num_op"] = atom_op
                        
                        
                        # next_program_str = next_program_str + []
                        
                        # program_atom_ls.append(new_atom_ls)
                    #apply new atom
                    next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats_ls)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                    # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
                    y_pred = self.get_test_decision_from_db_ls(next_all_other_pats_ls)

                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        # if self.is_log:
                        #     msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                        #     self.logger.log(level=logging.DEBUG, msg=msg)
                        # if y == y_pred: success += 1
                        # else: failure += 1
                        success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
                        failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
                        y_true_ls.extend(y.view(-1).tolist())
                        y_pred_ls.extend(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats_ls = next_all_other_pats_ls

            # for episode_i, val in iterator:
            #     (all_other_pats, X_pd, X), y = val
            #     program = []
            #     program_str = []
            #     program_atom_ls = []
            #     while True: # episode
            #         atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
            #         atom_ls = self.identify_op(X_pd, atom)
            #         next_program = program.copy()
            #         next_program_str = program_str.copy()
            #         for new_atom in atom_ls:
            #             next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
            #             # atom["num_op"] = atom_op
                        
                        
            #             next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
            #             program_atom_ls.append(new_atom)
            #         #apply new atom
            #         next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
            #         # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
            #         # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
            #         #check constraints
            #         x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
            #         prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
            #         y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
            #         db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
            #         #derive reward
            #         done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
            #         if done:
            #             next_program = None
            #         #update next step
            #         if done: #stopping condition
            #             if self.is_log:
            #                 msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
            #                 self.logger.log(level=logging.DEBUG, msg=msg)
            #             if y == y_pred: success += 1
            #             else: failure += 1
            #             y_true_ls.append(y.item())
            #             y_pred_ls.append(y_pred)
            #             break
            #         else:
            #             program = next_program
            #             program_str = next_program_str
            #             all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array.reshape(-1), y_pred_array.reshape(-1))


                # Print information
                success_rate = (success / len(y_pred_array)) * 100.00
                avg_loss = sum_loss/len(y_pred_array)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        
        if exp_y_pred_arr is not None:
            nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
            print(nonzero_ids[0])
          
        self.dqn.policy_net.train()
        self.dqn.target_net.train()

    def run(self):
        # exp_pred_array = self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        self.test_epoch_ls(test_loader, 0)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        for i in range(1, self.epochs + 1):
            self.train_epoch(i, train_loader)
            torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
            # self.test_epoch(i)
            self.test_epoch_ls(test_loader, i)
            torch.cuda.empty_cache() 

            # self.test_epoch_ls(test_loader, i)


def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.data.columns)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class Trainer4:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, seed):
        self.dqn = DQN5(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, seed=seed)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.batch_size = batch_size
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()

    def check_db_constrants(self, data_ls: list,  y: int) -> float:
        reward_ls = []
        for data in data_ls:
            if len(data) == 0:
                reward_ls.append(0)
                continue
            same = data.loc[data['label'] == y]["PAT_ID"].nunique()
            total = data['PAT_ID'].nunique()
            reward_ls.append(same / total)
        return reward_ls

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if self.dqn.policy_net.get_prefix(k) not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                if k.endswith("_lb"):
                    atom1[self.dqn.policy_net.get_prefix(k)] = atom[k][0]
                    atom1[self.dqn.policy_net.get_prefix(k) + "_prob"] = atom[k][1]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if self.dqn.policy_net.get_prefix(k) not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                if k.endswith("_ub"):
                    atom2[self.dqn.policy_net.get_prefix(k)] = atom[k][0]
                    atom2[self.dqn.policy_net.get_prefix(k) + "_prob"] = atom[k][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        all_rand_ids = torch.randperm(len(self.train_dataset))
        # all_rand_ids = torch.tensor(list(range((len(self.train_dataset)))))
        for episode_i, sample_idx in iterator:
            # (all_other_pats, X_pd, X), y = val
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = []
            program_atom_ls = []
            prev_reward = 0
            while True: # episode
                atom, origin_atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=self.epsilon)
                atom_ls = self.identify_op(X_pd, atom)

                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants([next_all_other_pats], y=int(y.clone().detach()[0]))[0] #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                #record transition in buffer
                if done:
                    next_program = None
                transition = Transition(X, X_pd,program, (atom, origin_atom), next_program, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model()
                sum_loss += loss
                # print(loss)
                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats
                    
                prev_reward = reward

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            success_rate = (success / (episode_i + 1)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom,_ = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants([next_all_other_pats], y=y_pred)[0]  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)


                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
    def run(self):
        self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        for i in range(1, self.epochs + 1):
            self.train_epoch( i)
            self.test_epoch(i)



class Trainer4_2:
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed, work_dir):
        self.dqn = DQN5(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed)
        self.epsilon = epsilon
        self.work_dir = work_dir
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        self.batch_size = batch_size
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()
    
    def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for data in data_ls:
            if len(data) == 0:
                label_ls.append(-1)
                prob_label_ls.append(-1)
                continue
            label = data['label'].value_counts().idxmax()
            prob_label = np.mean(data['label'])
            label_ls.append(label)
            prob_label_ls.append(prob_label)
        return label_ls, prob_label_ls

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_db_constrants_ls(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            data = data_ls[idx]
            y = int(y_ls[idx].item())
            same = data.loc[data['label'] == y]["PAT_ID"].nunique()
            total = data['PAT_ID'].nunique()
            if total == 0:
                rwd_ls.append(0) 
            else:
                rwd_ls.append(same / total) 
        return np.array(rwd_ls)

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    
    def identify_op_ls(self, batch_size:int, atom:dict):

        atom_ls = []        

        # atom1 = [dict()]*batch_size
        atom1 = []

        atom2 = []
        for _ in range(batch_size):
            atom2.append(dict())
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                        atom2[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
                            atom2[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size
                if k[0].endswith("_lb"):
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom1[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                else:
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom2[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        for sample_id in range(len(atom2)):
            atom2[sample_id]["num_op"] = operator.__le__   

        
        # for k in atom:
        #     # if k not in self.lang.syntax["num_feat"]:
        #     if type(k) is not tuple:
        #         if type(atom[k]) is not dict:
        #             for atom_id in range(batch_size):
        #                 atom2[atom_id][k] = atom[k]
        #         else:
        #             for selected_item in atom[k]:
        #                 sample_ids = atom[k][selected_item]
        #                 for sample_id in sample_ids:
        #                     atom2[sample_id.item()][k] = selected_item
        #     else:
                
        #         for selected_item in atom[k][2]:
        #             sample_ids = atom[k][2][selected_item]
        #             for sample_id_id in range(len(sample_ids)):
        #                 atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
        #                 atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
        #                 # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


        #         # atom1[k] = atom[k][0][0]
        #         # atom1[k + "_prob"] = atom[k][1][0]
        #         # atom1[k + "_sample_ids"] = atom[k][2][0]
        # for sample_id in range(len(atom2)):
        #     atom2[sample_id]["num_op"] = operator.__le__  


        # atom2 = dict()
        # for k in atom:
        #     if k not in self.lang.syntax["num_feat"]:
        #         atom2[k] = atom[k]
        #     else:
        #         atom2[k] = atom[k][0][1]
        #         atom2[k + "_prob"] = atom[k][1][1]
        # atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    # def train_epoch(self, epoch, train_loader):
    #     success, failure, sum_loss = 0, 0, 0.
    #     # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
    #     # for episode_i, val in iterator:
    #     iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
    #     # pos_count = np.sum(self.train_dataset.data["label"] == 1)
    #     # neg_count = np.sum(self.train_dataset.data["label"] == 0)
    #     # sample_weights = torch.ones(len(self.train_dataset.data))
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
    #     # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
    #     # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
    #     # episode_i = 0
    #     # for val in iterator:
    #     # all_rand_ids = torch.randperm(len(self.train_dataset))
    #     # for episode_i, sample_idx in iterator:
    #     for episode_i, val in iterator:
    #         (all_other_pats_ls, X_pd_ls, X), y = val
    #         # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
    #         program = []
    #         program_str = [[] for _ in range(len(X_pd_ls))]
    #         program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            
    #         while True: # episode
    #             atom,origin_atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=self.epsilon)
    #             atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
    #             reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

    #             next_program = program.copy()
    #             next_program_str = program_str.copy()
    #             for new_atom_ls in atom_ls_ls:

    #                 curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

    #                 next_program.append(torch.stack(curr_vec_ls))

    #                 curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

    #                 for vec_idx in range(len(curr_vec_ls)):
    #                     vec = curr_vec_ls[vec_idx]
    #                     atom_str = curr_atom_str_ls[vec_idx]
                        
    #                     next_program_str[vec_idx].append(atom_str)
    #                     program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                     reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                 # atom["num_op"] = atom_op
                    
                    
    #                 # next_program_str = next_program_str + []
                    
    #                 # program_atom_ls.append(new_atom_ls)
    #             #apply new atom
    #             next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats_ls)
    #             # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
    #             # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
    #             #check constraints
    #             # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
    #             prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
    #             db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
    #             #derive reward
    #             reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
    #             done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
    #             #record transition in buffer
    #             if done:
    #                 next_program = None
    #             transition = Transition(X, X_pd_ls,program, (atom, origin_atom), next_program, reward)
    #             self.dqn.observe_transition(transition)
    #             #update model
    #             loss = self.dqn.optimize_model_ls()
    #             # print(loss)
    #             sum_loss += loss
    #             #update next step
    #             if done: #stopping condition
    #                 # if reward > 0.5: success += 1
    #                 # else: failure += 1
    #                 success += np.sum(reward > 0.5)
    #                 break
    #             else:
    #                 program = next_program
    #                 program_str = next_program_str
    #                 all_other_pats_ls = next_all_other_pats_ls

    #         # Update the target net
    #         if episode_i % self.target_update == 0:
    #             self.dqn.update_target()
    #         # Print information
    #         total_count = ((episode_i + 1)*self.batch_size)
    #         success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
    #         avg_loss = sum_loss/(episode_i+1)
    #         desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
    #         iterator.set_description(desc)
    #     if self.is_log:
    #         self.logger.log(level=logging.DEBUG, msg = desc)
    #     self.epsilon *= self.epsilon_falloff

    def train_epoch(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        # all_rand_ids = torch.randperm(len(self.train_dataset))
        # for episode_i, sample_idx in iterator:
        
        col_list = list(self.train_dataset.data.columns)
        
        op_ls = list([operator.__le__, operator.__ge__])
        
        col_op_ls = []
        
        last_col = None

        last_op = None
        
        for col in col_list:
            if col == "PAT_ID" or col == "label":
                continue
            last_col = col
            for op in op_ls:
                col_op_ls.append((col, op))
                last_op = op
        
        
        for episode_i, val in iterator:
            (all_other_pats_ls, X_pd_ls, X), y = val
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = [[] for _ in range(len(X_pd_ls))]
            program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            X_pd_full = pd.concat(X_pd_ls)
            
            
            
            # col_comp_ls = zip()
            # while True: # episode
            # for 
            # for col_id in col_ids:
            prev_reward = np.zeros(len(X))
            random.shuffle(col_op_ls)
            last_col , last_op  = col_op_ls[-1]


            for arr_idx in range(len(col_op_ls)):
                (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=self.epsilon, col=col, op=op)
                
                next_program = program.copy()
                
                curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls, col, op)

                next_program.append(curr_vec_ls)

                curr_atom_str_ls = self.lang.atom_to_str_ls0(atom_ls, col, op, pred_v_key)
                # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                
                next_program_str = program_str.copy()
                # for new_atom_ls in atom_ls_ls:

                #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                #     next_program.append(torch.stack(curr_vec_ls))

                #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                for vec_idx in range(len(curr_vec_ls)):
                    # vec = curr_vec_ls[vec_idx]
                    atom_str = curr_atom_str_ls[vec_idx]
                    
                    next_program_str[vec_idx].append(atom_str)
                        # program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # atom["num_op"] = atom_op
                    
                    
                    # next_program_str = next_program_str + []
                    
                    # program_atom_ls.append(new_atom_ls)
                #apply new atom
                next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset0(atom_ls, all_other_pats_ls, col, op, pred_v_key)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                done = (col == last_col) and (op == last_op)
                #record transition in buffer
                if done:
                    next_state = None
                    next_program = None
                else:
                    next_state = (next_program, col_op_ls[arr_idx+1][0], col_op_ls[arr_idx+1][1])
                transition = Transition(X, X_pd_full,(program, col, op), atom_ls, next_state, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model_ls0()
                # print(loss)
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(reward > 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            total_count = ((episode_i + 1)*self.batch_size)
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    
    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 28:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)

                # if episode_i == self.batch_size:
                #     print(y_true_array.reshape(-1))
                #     print(y_pred_array.reshape(-1))

                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        return y_pred_array
    
    def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None):
        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        y_pred_prob_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            col_list = list(self.train_dataset.data.columns)
        
            op_ls = list([operator.__le__, operator.__ge__])
            
            col_op_ls = []
            
            last_col = None

            last_op = None
            
            for col in col_list:
                if col == "PAT_ID" or col == "label":
                    continue
                last_col = col
                for op in op_ls:
                    col_op_ls.append((col, op))
                    last_op = op
            for episode_i, val in iterator:
                (all_other_pats_ls, X_pd_ls, X), y = val
                # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
                program = []
                program_str = [[] for _ in range(len(X_pd_ls))]
                program_atom_ls = [[] for _ in range(len(X_pd_ls))]
                
                X_pd_full = pd.concat(X_pd_ls)
                
                for arr_idx in range(len(col_op_ls)):
                    (col, op) = col_op_ls[arr_idx]
                    # col_name = col_list[col_id]
                    atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=0, col=col, op=op)
                    
                    next_program = program.copy()
                    
                    curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls, col, op)

                    next_program.append(curr_vec_ls)

                    curr_atom_str_ls = self.lang.atom_to_str_ls0(atom_ls, col, op, pred_v_key)
                    # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                    # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    
                    next_program_str = program_str.copy()
                    # for new_atom_ls in atom_ls_ls:

                    #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                    #     next_program.append(torch.stack(curr_vec_ls))

                    #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    for vec_idx in range(len(curr_vec_ls)):
                        # vec = curr_vec_ls[vec_idx]
                        atom_str = curr_atom_str_ls[vec_idx]
                        
                        next_program_str[vec_idx].append(atom_str)
                # while True: # episode
                #     atom,_ = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
                #     atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                #     reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                #     next_program = program.copy()
                #     next_program_str = program_str.copy()
                #     for new_atom_ls in atom_ls_ls:

                #         curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                #         next_program.append(torch.stack(curr_vec_ls))

                #         curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                #         for vec_idx in range(len(curr_vec_ls)):
                #             vec = curr_vec_ls[vec_idx]
                #             atom_str = curr_atom_str_ls[vec_idx]
                            
                #             next_program_str[vec_idx].append(atom_str)
                #             program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                #             reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                        # atom["num_op"] = atom_op
                        
                        
                        # next_program_str = next_program_str + []
                        
                        # program_atom_ls.append(new_atom_ls)
                    #apply new atom
                    next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset0(atom_ls, all_other_pats_ls, col, op, pred_v_key)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                    # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
                    y_pred, y_pred_prob = self.get_test_decision_from_db_ls(next_all_other_pats_ls)

                    # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    done = (col == last_col) and (op == last_op)
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        # if self.is_log:
                        #     msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                        #     self.logger.log(level=logging.DEBUG, msg=msg)
                        # if y == y_pred: success += 1
                        # else: failure += 1
                        success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
                        failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
                        y_true_ls.extend(y.view(-1).tolist())
                        y_pred_ls.extend(y_pred)
                        y_pred_prob_ls.extend(y_pred_prob)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats_ls = next_all_other_pats_ls

            # for episode_i, val in iterator:
            #     (all_other_pats, X_pd, X), y = val
            #     program = []
            #     program_str = []
            #     program_atom_ls = []
            #     while True: # episode
            #         atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
            #         atom_ls = self.identify_op(X_pd, atom)
            #         next_program = program.copy()
            #         next_program_str = program_str.copy()
            #         for new_atom in atom_ls:
            #             next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
            #             # atom["num_op"] = atom_op
                        
                        
            #             next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
            #             program_atom_ls.append(new_atom)
            #         #apply new atom
            #         next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
            #         # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
            #         # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
            #         #check constraints
            #         x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
            #         prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
            #         y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
            #         db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
            #         #derive reward
            #         done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
            #         if done:
            #             next_program = None
            #         #update next step
            #         if done: #stopping condition
            #             if self.is_log:
            #                 msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
            #                 self.logger.log(level=logging.DEBUG, msg=msg)
            #             if y == y_pred: success += 1
            #             else: failure += 1
            #             y_true_ls.append(y.item())
            #             y_pred_ls.append(y_pred)
            #             break
            #         else:
            #             program = next_program
            #             program_str = next_program_str
            #             all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_prob_array = np.array(y_pred_prob_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                y_pred_prob_array[y_pred_prob_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                    auc_score_2 = 0
                else:
                    auc_score = roc_auc_score(y_true_array.reshape(-1), y_pred_array.reshape(-1))
                    auc_score_2 = roc_auc_score(y_true_array.reshape(-1), y_pred_prob_array.reshape(-1))

                # Print information
                success_rate = (success / len(y_pred_array)) * 100.00
                avg_loss = sum_loss/len(y_pred_array)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score: {auc_score}, auc score 2:{auc_score_2}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        
        if exp_y_pred_arr is not None:
            nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
            print(nonzero_ids[0])
          
        self.dqn.policy_net.train()
        self.dqn.target_net.train()

    def run(self):
        # exp_pred_array = self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        if self.valid_dataset is not None:
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        if self.valid_dataset is not None:
            self.test_epoch_ls(valid_loader, 0)    
        self.test_epoch_ls(test_loader, 0)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(1, self.epochs + 1):
                self.train_epoch(i, train_loader)
                # torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
                # self.test_epoch(i)
                if self.valid_dataset is not None:
                    self.test_epoch_ls(valid_loader, i)    
                self.test_epoch_ls(test_loader, i)
                torch.cuda.empty_cache() 

            # self.test_epoch_ls(test_loader, i)


class Trainer6_2:
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, tf_latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed, work_dir, numeric_count=None, category_count=None, train_feat_embeddings=None, valid_feat_embeddings=None, test_feat_embeddings=None, model="mlp", pretrained_model_path=None, topk_act=1):
        self.dqn = DQN6(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, tf_latent_size=tf_latent_size, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed, numeric_count=numeric_count, category_count=category_count, has_embeddings=(train_feat_embeddings is not None), model=model, pretrained_model_path=pretrained_model_path,topk_act=topk_act)
        self.epsilon = epsilon
        self.work_dir = work_dir
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        self.batch_size = batch_size
        self.train_feat_embeddings=train_feat_embeddings
        self.valid_feat_embeddings=valid_feat_embeddings
        self.test_feat_embeddings=test_feat_embeddings
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()
    
    def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for data in data_ls:
            if len(data) == 0:
                label_ls.append(-1)
                prob_label_ls.append(-1)
                continue
            label = data['label'].value_counts().idxmax()
            prob_label = np.mean(data['label'])
            label_ls.append(label)
            prob_label_ls.append(prob_label)
        return label_ls, prob_label_ls

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_db_constrants_ls(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            data = data_ls[idx]
            y = int(y_ls[idx].item())
            same = data.loc[data['label'] == y]["PAT_ID"].nunique()
            total = data['PAT_ID'].nunique()
            if total == 0:
                rwd_ls.append(0) 
            else:
                rwd_ls.append(same / total) 
        return np.array(rwd_ls)

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    
    def identify_op_ls(self, batch_size:int, atom:dict):

        atom_ls = []        

        # atom1 = [dict()]*batch_size
        atom1 = []

        atom2 = []
        for _ in range(batch_size):
            atom2.append(dict())
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                        atom2[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
                            atom2[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size
                if k[0].endswith("_lb"):
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom1[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                else:
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom2[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        for sample_id in range(len(atom2)):
            atom2[sample_id]["num_op"] = operator.__le__   

        
        # for k in atom:
        #     # if k not in self.lang.syntax["num_feat"]:
        #     if type(k) is not tuple:
        #         if type(atom[k]) is not dict:
        #             for atom_id in range(batch_size):
        #                 atom2[atom_id][k] = atom[k]
        #         else:
        #             for selected_item in atom[k]:
        #                 sample_ids = atom[k][selected_item]
        #                 for sample_id in sample_ids:
        #                     atom2[sample_id.item()][k] = selected_item
        #     else:
                
        #         for selected_item in atom[k][2]:
        #             sample_ids = atom[k][2][selected_item]
        #             for sample_id_id in range(len(sample_ids)):
        #                 atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
        #                 atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
        #                 # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


        #         # atom1[k] = atom[k][0][0]
        #         # atom1[k + "_prob"] = atom[k][1][0]
        #         # atom1[k + "_sample_ids"] = atom[k][2][0]
        # for sample_id in range(len(atom2)):
        #     atom2[sample_id]["num_op"] = operator.__le__  


        # atom2 = dict()
        # for k in atom:
        #     if k not in self.lang.syntax["num_feat"]:
        #         atom2[k] = atom[k]
        #     else:
        #         atom2[k] = atom[k][0][1]
        #         atom2[k + "_prob"] = atom[k][1][1]
        # atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    # def train_epoch(self, epoch, train_loader):
    #     success, failure, sum_loss = 0, 0, 0.
    #     # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
    #     # for episode_i, val in iterator:
    #     iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
    #     # pos_count = np.sum(self.train_dataset.data["label"] == 1)
    #     # neg_count = np.sum(self.train_dataset.data["label"] == 0)
    #     # sample_weights = torch.ones(len(self.train_dataset.data))
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
    #     # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
    #     # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
    #     # episode_i = 0
    #     # for val in iterator:
    #     # all_rand_ids = torch.randperm(len(self.train_dataset))
    #     # for episode_i, sample_idx in iterator:
    #     for episode_i, val in iterator:
    #         (all_other_pats_ls, X_pd_ls, X), y = val
    #         # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
    #         program = []
    #         program_str = [[] for _ in range(len(X_pd_ls))]
    #         program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            
    #         while True: # episode
    #             atom,origin_atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=self.epsilon)
    #             atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
    #             reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

    #             next_program = program.copy()
    #             next_program_str = program_str.copy()
    #             for new_atom_ls in atom_ls_ls:

    #                 curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

    #                 next_program.append(torch.stack(curr_vec_ls))

    #                 curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

    #                 for vec_idx in range(len(curr_vec_ls)):
    #                     vec = curr_vec_ls[vec_idx]
    #                     atom_str = curr_atom_str_ls[vec_idx]
                        
    #                     next_program_str[vec_idx].append(atom_str)
    #                     program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                     reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                 # atom["num_op"] = atom_op
                    
                    
    #                 # next_program_str = next_program_str + []
                    
    #                 # program_atom_ls.append(new_atom_ls)
    #             #apply new atom
    #             next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats_ls)
    #             # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
    #             # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
    #             #check constraints
    #             # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
    #             prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
    #             db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
    #             #derive reward
    #             reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
    #             done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
    #             #record transition in buffer
    #             if done:
    #                 next_program = None
    #             transition = Transition(X, X_pd_ls,program, (atom, origin_atom), next_program, reward)
    #             self.dqn.observe_transition(transition)
    #             #update model
    #             loss = self.dqn.optimize_model_ls()
    #             # print(loss)
    #             sum_loss += loss
    #             #update next step
    #             if done: #stopping condition
    #                 # if reward > 0.5: success += 1
    #                 # else: failure += 1
    #                 success += np.sum(reward > 0.5)
    #                 break
    #             else:
    #                 program = next_program
    #                 program_str = next_program_str
    #                 all_other_pats_ls = next_all_other_pats_ls

    #         # Update the target net
    #         if episode_i % self.target_update == 0:
    #             self.dqn.update_target()
    #         # Print information
    #         total_count = ((episode_i + 1)*self.batch_size)
    #         success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
    #         avg_loss = sum_loss/(episode_i+1)
    #         desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
    #         iterator.set_description(desc)
    #     if self.is_log:
    #         self.logger.log(level=logging.DEBUG, msg = desc)
    #     self.epsilon *= self.epsilon_falloff

    def train_epoch(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        # all_rand_ids = torch.randperm(len(self.train_dataset))
        # for episode_i, sample_idx in iterator:
        
        col_list = list(self.train_dataset.data.columns)
        
        op_ls = list([operator.__le__, operator.__ge__])
        
        col_op_ls = []
        
        last_col = None

        last_op = None
        
        for col in col_list:
            if col == "PAT_ID" or col == "label":
                continue
            last_col = col
            for op in op_ls:
                col_op_ls.append((col, op))
                last_op = op
        
        
        for episode_i, val in iterator:
            (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
            X_feat_embedding = None
            if self.train_feat_embeddings is not None:
                X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = [[] for _ in range(len(X_pd_ls))]
            program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            X_pd_full = pd.concat(X_pd_ls)
            
            
            
            # col_comp_ls = zip()
            # while True: # episode
            # for 
            # for col_id in col_ids:
            prev_reward = np.zeros(len(X))
            random.shuffle(col_op_ls)
            last_col , last_op  = col_op_ls[-1]


            # for arr_idx in range(len(col_op_ls)):
            for arr_idx in range(self.program_max_len):
                # (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                if X_feat_embedding is None:
                    atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=self.epsilon)
                else:
                    atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, epsilon=self.epsilon)
                
                next_program = program.copy()
                
                curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

                next_program.append(curr_vec_ls)

                curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings)#(atom_ls, pred_v_key)
                # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                
                next_program_str = program_str.copy()
                # for new_atom_ls in atom_ls_ls:

                #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                #     next_program.append(torch.stack(curr_vec_ls))

                #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                for vec_idx in range(len(curr_vec_ls)):
                    # vec = curr_vec_ls[vec_idx]
                    atom_str = curr_atom_str_ls[vec_idx]
                    
                    next_program_str[vec_idx].append(atom_str)
                        # program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # atom["num_op"] = atom_op
                    
                    
                    # next_program_str = next_program_str + []
                    
                    # program_atom_ls.append(new_atom_ls)
                #apply new atom
                next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)#(atom_ls, all_other_pats_ls, pred_v_key)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                # done = (col == last_col) and (op == last_op)
                done = (arr_idx == self.program_max_len-1)
                #record transition in buffer
                if done:
                    next_state = None
                    next_program = None
                else:
                    # next_state = (next_program, col_op_ls[arr_idx+1][0], col_op_ls[arr_idx+1][1])
                    next_state = next_program
                if X_feat_embedding is None:
                    transition = Transition(X, X_pd_full,program, atom_ls, next_state, reward - prev_reward)
                else:
                    transition = Transition((X, X_feat_embedding), X_pd_full,program, atom_ls, next_state, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model_ls0()
                # print(loss)
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(reward > 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            total_count = ((episode_i + 1)*self.batch_size)
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    
    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 28:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)

                # if episode_i == self.batch_size:
                #     print(y_true_array.reshape(-1))
                #     print(y_pred_array.reshape(-1))

                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        return y_pred_array
    
    def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None, feat_embedding = None):
        pd.options.mode.chained_assignment = None

        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        y_pred_prob_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            col_list = list(self.train_dataset.data.columns)
        
            op_ls = list([operator.__le__, operator.__ge__])
            
            col_op_ls = []
            
            last_col = None

            last_op = None
            
            for col in col_list:
                if col == "PAT_ID" or col == "label":
                    continue
                last_col = col
                for op in op_ls:
                    col_op_ls.append((col, op))
                    last_op = op
            for episode_i, val in iterator:
                (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
                
                X_feat_embeddings = None
                if feat_embedding is not None:
                    X_feat_embeddings = feat_embedding[X_sample_ids]
                
                # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
                program = []
                program_str = [[] for _ in range(len(X_pd_ls))]
                program_col_ls = [[] for _ in range(len(X_pd_ls))]
                
                X_pd_full = pd.concat(X_pd_ls)
                
                # for arr_idx in range(len(col_op_ls)):
                for arr_idx in range(self.program_max_len):
                    (col, op) = col_op_ls[arr_idx]
                    # col_name = col_list[col_id]
                    if X_feat_embeddings is None:
                        atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=0)
                    else:
                        atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, epsilon=0)
                    
                    next_program = program.copy()
                    
                    curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

                    next_program.append(curr_vec_ls)

                    curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings)
                    # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                    # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    
                    next_program_str = program_str.copy()
                    # for new_atom_ls in atom_ls_ls:

                    #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                    #     next_program.append(torch.stack(curr_vec_ls))

                    #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    for vec_idx in range(len(curr_vec_ls)):
                        # vec = curr_vec_ls[vec_idx]
                        atom_str = curr_atom_str_ls[vec_idx]
                        program_col_ls[vec_idx].append(atom_ls[col_key][vec_idx])
                        next_program_str[vec_idx].append(atom_str)
                # while True: # episode
                #     atom,_ = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
                #     atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                #     reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                #     next_program = program.copy()
                #     next_program_str = program_str.copy()
                #     for new_atom_ls in atom_ls_ls:

                #         curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                #         next_program.append(torch.stack(curr_vec_ls))

                #         curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                #         for vec_idx in range(len(curr_vec_ls)):
                #             vec = curr_vec_ls[vec_idx]
                #             atom_str = curr_atom_str_ls[vec_idx]
                            
                #             next_program_str[vec_idx].append(atom_str)
                #             program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                #             reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                        # atom["num_op"] = atom_op
                        
                        
                        # next_program_str = next_program_str + []
                        
                        # program_atom_ls.append(new_atom_ls)
                    #apply new atom
                    next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                    # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
                    y_pred, y_pred_prob = self.get_test_decision_from_db_ls(next_all_other_pats_ls)

                    # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    # done = (col == last_col) and (op == last_op)
                    done = (arr_idx == self.program_max_len - 1)
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            for pat_idx in range(len(y_pred)):
                                col_ls = list(set(program_col_ls[pat_idx]))
                                col_ls.append("PAT_ID")
                                x_pat_sub = X_pd_ls[pat_idx][col_ls]
                                x_pat_sub.reset_index(inplace=True)
                                for col in col_ls:
                                    if not col == "PAT_ID":
                                        x_pat_sub[col] = x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]

                                msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[pat_idx]), y_pred[pat_idx], y_pred_prob[pat_idx], str(x_pat_sub.to_dict()),str(next_program_str[pat_idx]))
                                self.logger.log(level=logging.DEBUG, msg=msg)
                        # if y == y_pred: success += 1
                        # else: failure += 1
                        success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
                        failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
                        y_true_ls.extend(y.view(-1).tolist())
                        y_pred_ls.extend(y_pred)
                        y_pred_prob_ls.extend(y_pred_prob)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats_ls = next_all_other_pats_ls

            # for episode_i, val in iterator:
            #     (all_other_pats, X_pd, X), y = val
            #     program = []
            #     program_str = []
            #     program_atom_ls = []
            #     while True: # episode
            #         atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
            #         atom_ls = self.identify_op(X_pd, atom)
            #         next_program = program.copy()
            #         next_program_str = program_str.copy()
            #         for new_atom in atom_ls:
            #             next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
            #             # atom["num_op"] = atom_op
                        
                        
            #             next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
            #             program_atom_ls.append(new_atom)
            #         #apply new atom
            #         next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
            #         # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
            #         # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
            #         #check constraints
            #         x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
            #         prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
            #         y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
            #         db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
            #         #derive reward
            #         done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
            #         if done:
            #             next_program = None
            #         #update next step
            #         if done: #stopping condition
            #             if self.is_log:
            #                 msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
            #                 self.logger.log(level=logging.DEBUG, msg=msg)
            #             if y == y_pred: success += 1
            #             else: failure += 1
            #             y_true_ls.append(y.item())
            #             y_pred_ls.append(y_pred)
            #             break
            #         else:
            #             program = next_program
            #             program_str = next_program_str
            #             all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_prob_array = np.array(y_pred_prob_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                y_pred_prob_array[y_pred_prob_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    # auc_score= 0
                    auc_score_2 = 0
                else:
                    # auc_score = roc_auc_score(y_true_array.reshape(-1), y_pred_array.reshape(-1))
                    auc_score_2 = roc_auc_score(y_true_array.reshape(-1), y_pred_prob_array.reshape(-1))
                success_rate = (success / len(y_pred_array)) * 100.00
                avg_loss = sum_loss/len(y_pred_array)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score:{auc_score_2}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        

        additional_score_str = ""
        full_y_pred_prob_array = np.stack([1 - y_pred_prob_array.reshape(-1), y_pred_prob_array.reshape(-1)], axis=1)
        for metric_name in metrics_maps:
            curr_score = metrics_maps[metric_name](y_true_array.reshape(-1),full_y_pred_prob_array)
            additional_score_str += metric_name + ": " + str(curr_score) + " "
        print(additional_score_str)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = additional_score_str)
        # Print information
        
        # if exp_y_pred_arr is not None:
        #     nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
        #     print(nonzero_ids[0])
          
        self.dqn.policy_net.train()
        self.dqn.target_net.train()

    def run(self):
        # exp_pred_array = self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        if self.valid_dataset is not None:
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        # if self.valid_dataset is not None:
        #     self.test_epoch_ls(valid_loader, 0)    
        self.test_epoch_ls(test_loader, 0, feat_embedding=self.test_feat_embeddings)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(1, self.epochs + 1):
                self.train_epoch(i, train_loader)
                torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
                torch.save(self.dqn.target_net.state_dict(), os.path.join(self.work_dir, "target_net_" + str(i)))
                torch.save(self.dqn.memory, os.path.join(self.work_dir, "memory_" + str(i)))
                # self.test_epoch(i)
                if self.valid_dataset is not None:
                    self.test_epoch_ls(valid_loader, i, feat_embedding=self.valid_feat_embeddings)    
                self.test_epoch_ls(test_loader, i, feat_embedding=self.test_feat_embeddings)
                torch.cuda.empty_cache() 

            # self.test_epoch_ls(test_loader, i)

class Trainer7_2:
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, tf_latent_size, is_log, dropout_p, feat_range_mappings, mem_sample_size, seed, work_dir, numeric_count=None, category_count=None, train_feat_embeddings=None, valid_feat_embeddings=None, test_feat_embeddings=None, model="mlp", pretrained_model_path=None, topk_act=1):
        self.topk_act =topk_act
        self.dqn = DQN7(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, tf_latent_size=tf_latent_size, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed, numeric_count=numeric_count, category_count=category_count, has_embeddings=(train_feat_embeddings is not None), model=model, pretrained_model_path=pretrained_model_path,topk_act=topk_act)
        self.epsilon = epsilon
        self.work_dir = work_dir
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        self.batch_size = batch_size
        self.train_feat_embeddings=train_feat_embeddings
        self.valid_feat_embeddings=valid_feat_embeddings
        self.test_feat_embeddings=test_feat_embeddings
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data['label'].value_counts().idxmax()
    
    def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for data in data_ls:
            if len(data) == 0:
                label_ls.append(-1)
                prob_label_ls.append(-1)
                continue
            label = data['label'].value_counts().idxmax()
            prob_label = np.mean(data['label'])
            label_ls.append(label)
            prob_label_ls.append(prob_label)
        return label_ls, prob_label_ls
    
    def get_test_decision_from_db_ls_multi(self, data_ls):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for sub_data_ls in data_ls:
            sub_label_ls = []
            sub_prob_label_ls = []
            for data in sub_data_ls:
                if len(data) == 0:
                    # sub_label_ls.append(-1)
                    # sub_prob_label_ls.append(-1)
                    continue
                label = data['label'].value_counts().idxmax()
                prob_label = np.mean(data['label'])
                sub_label_ls.append(label)
                sub_prob_label_ls.append(prob_label)
            if len(sub_label_ls) <= 0:    
                label_ls.append(-1)
                prob_label_ls.append(-1)
            else:
                
                prob_label = np.mean(np.array(sub_prob_label_ls))
                prob_label_ls.append(prob_label)
                
                if prob_label == 0.5:
                    label_ls.append(-1)
                elif prob_label > 0.5:
                    label_ls.append(1)
                else:
                    label_ls.append(0)
                
            
        return label_ls, prob_label_ls

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_db_constrants_ls(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            sub_data_ls = data_ls[idx]
            sub_rwd_ls = []
            for data in sub_data_ls:
                y = int(y_ls[idx].item())
                same = data.loc[data['label'] == y]["PAT_ID"].nunique()
                total = data['PAT_ID'].nunique()
                if total == 0:
                    sub_rwd_ls.append(0)
                else:
                    sub_rwd_ls.append(same / total)
            
            rwd_ls.append(sub_rwd_ls) 
            # if total == 0:
            #     rwd_ls.append(0)
            # else:
            #     rwd_ls.append(same / total) 
        return np.array(rwd_ls)

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    
    def identify_op_ls(self, batch_size:int, atom:dict):

        atom_ls = []        

        # atom1 = [dict()]*batch_size
        atom1 = []

        atom2 = []
        for _ in range(batch_size):
            atom2.append(dict())
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                        atom2[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
                            atom2[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size
                if k[0].endswith("_lb"):
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom1[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                else:
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom2[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        for sample_id in range(len(atom2)):
            atom2[sample_id]["num_op"] = operator.__le__   

        
        # for k in atom:
        #     # if k not in self.lang.syntax["num_feat"]:
        #     if type(k) is not tuple:
        #         if type(atom[k]) is not dict:
        #             for atom_id in range(batch_size):
        #                 atom2[atom_id][k] = atom[k]
        #         else:
        #             for selected_item in atom[k]:
        #                 sample_ids = atom[k][selected_item]
        #                 for sample_id in sample_ids:
        #                     atom2[sample_id.item()][k] = selected_item
        #     else:
                
        #         for selected_item in atom[k][2]:
        #             sample_ids = atom[k][2][selected_item]
        #             for sample_id_id in range(len(sample_ids)):
        #                 atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
        #                 atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
        #                 # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


        #         # atom1[k] = atom[k][0][0]
        #         # atom1[k + "_prob"] = atom[k][1][0]
        #         # atom1[k + "_sample_ids"] = atom[k][2][0]
        # for sample_id in range(len(atom2)):
        #     atom2[sample_id]["num_op"] = operator.__le__  


        # atom2 = dict()
        # for k in atom:
        #     if k not in self.lang.syntax["num_feat"]:
        #         atom2[k] = atom[k]
        #     else:
        #         atom2[k] = atom[k][0][1]
        #         atom2[k + "_prob"] = atom[k][1][1]
        # atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    # def train_epoch(self, epoch, train_loader):
    #     success, failure, sum_loss = 0, 0, 0.
    #     # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
    #     # for episode_i, val in iterator:
    #     iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
    #     # pos_count = np.sum(self.train_dataset.data["label"] == 1)
    #     # neg_count = np.sum(self.train_dataset.data["label"] == 0)
    #     # sample_weights = torch.ones(len(self.train_dataset.data))
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
    #     # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
    #     # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
    #     # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
    #     # episode_i = 0
    #     # for val in iterator:
    #     # all_rand_ids = torch.randperm(len(self.train_dataset))
    #     # for episode_i, sample_idx in iterator:
    #     for episode_i, val in iterator:
    #         (all_other_pats_ls, X_pd_ls, X), y = val
    #         # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
    #         program = []
    #         program_str = [[] for _ in range(len(X_pd_ls))]
    #         program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            
    #         while True: # episode
    #             atom,origin_atom = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=self.epsilon)
    #             atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
    #             reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

    #             next_program = program.copy()
    #             next_program_str = program_str.copy()
    #             for new_atom_ls in atom_ls_ls:

    #                 curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

    #                 next_program.append(torch.stack(curr_vec_ls))

    #                 curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

    #                 for vec_idx in range(len(curr_vec_ls)):
    #                     vec = curr_vec_ls[vec_idx]
    #                     atom_str = curr_atom_str_ls[vec_idx]
                        
    #                     next_program_str[vec_idx].append(atom_str)
    #                     program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                     reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
    #                 # atom["num_op"] = atom_op
                    
                    
    #                 # next_program_str = next_program_str + []
                    
    #                 # program_atom_ls.append(new_atom_ls)
    #             #apply new atom
    #             next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset(reorg_atom_ls_ls, all_other_pats_ls)
    #             # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
    #             # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
    #             #check constraints
    #             # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
    #             prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
    #             db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
    #             #derive reward
    #             reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
    #             done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
    #             #record transition in buffer
    #             if done:
    #                 next_program = None
    #             transition = Transition(X, X_pd_ls,program, (atom, origin_atom), next_program, reward)
    #             self.dqn.observe_transition(transition)
    #             #update model
    #             loss = self.dqn.optimize_model_ls()
    #             # print(loss)
    #             sum_loss += loss
    #             #update next step
    #             if done: #stopping condition
    #                 # if reward > 0.5: success += 1
    #                 # else: failure += 1
    #                 success += np.sum(reward > 0.5)
    #                 break
    #             else:
    #                 program = next_program
    #                 program_str = next_program_str
    #                 all_other_pats_ls = next_all_other_pats_ls

    #         # Update the target net
    #         if episode_i % self.target_update == 0:
    #             self.dqn.update_target()
    #         # Print information
    #         total_count = ((episode_i + 1)*self.batch_size)
    #         success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
    #         avg_loss = sum_loss/(episode_i+1)
    #         desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
    #         iterator.set_description(desc)
    #     if self.is_log:
    #         self.logger.log(level=logging.DEBUG, msg = desc)
    #     self.epsilon *= self.epsilon_falloff

    def train_epoch(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
        # pos_count = np.sum(self.train_dataset.data["label"] == 1)
        # neg_count = np.sum(self.train_dataset.data["label"] == 0)
        # sample_weights = torch.ones(len(self.train_dataset.data))
        # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
        # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
        # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
        # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
        # episode_i = 0
        # for val in iterator:
        # all_rand_ids = torch.randperm(len(self.train_dataset))
        # for episode_i, sample_idx in iterator:
        
        col_list = list(self.train_dataset.data.columns)
        
        op_ls = list([operator.__le__, operator.__ge__])
        
        col_op_ls = []
        
        last_col = None

        last_op = None
        
        for col in col_list:
            if col == "PAT_ID" or col == "label":
                continue
            last_col = col
            for op in op_ls:
                col_op_ls.append((col, op))
                last_op = op
        
        
        for episode_i, val in iterator:
            (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
            all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)
            X_feat_embedding = None
            if self.train_feat_embeddings is not None:
                X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            # program_str = [[] for _ in range(len(X_pd_ls))]
            # program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            
            X_pd_full = pd.concat(X_pd_ls)
            
            
            
            # col_comp_ls = zip()
            # while True: # episode
            # for 
            # for col_id in col_ids:
            prev_reward = np.zeros([len(X), self.topk_act])
            # random.shuffle(col_op_ls)
            # last_col , last_op  = col_op_ls[-1]


            # for arr_idx in range(len(col_op_ls)):
            for arr_idx in range(self.program_max_len):
                # (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                if X_feat_embedding is None:
                    atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=self.epsilon)
                else:
                    atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, epsilon=self.epsilon)
                
                curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings)
                
                next_program = program.copy()
                
                next_program_str = program_str.copy()
                
                curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

                if len(program) > 0:                        
                        

                    next_program, program_col_ls, next_program_str= self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls)



                else:

                    next_program.append(curr_vec_ls)

                    for vec_idx in range(len(curr_vec_ls)):
                        # vec = curr_vec_ls[vec_idx]
                        atom_str = curr_atom_str_ls[vec_idx]
                        for k in range(len(atom_ls[col_key][vec_idx])):
                            program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                            next_program_str[vec_idx][k].append(atom_str[k])
                # next_program.append(curr_vec_ls)

                # curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings)#(atom_ls, pred_v_key)
                # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                
                # next_program_str = program_str.copy()
                # for new_atom_ls in atom_ls_ls:

                #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                #     next_program.append(torch.stack(curr_vec_ls))

                #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                # for vec_idx in range(len(curr_vec_ls)):
                #     # vec = curr_vec_ls[vec_idx]
                #     atom_str = curr_atom_str_ls[vec_idx]
                    
                #     next_program_str[vec_idx].append(atom_str)
                        # program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                    # atom["num_op"] = atom_op
                    
                    
                    # next_program_str = next_program_str + []
                    
                    # program_atom_ls.append(new_atom_ls)
                #apply new atom
                # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)#(atom_ls, all_other_pats_ls, pred_v_key)
                next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                # done = (col == last_col) and (op == last_op)
                done = (arr_idx == self.program_max_len-1)
                #record transition in buffer
                if done:
                    next_state = None
                    next_program = None
                else:
                    # next_state = (next_program, col_op_ls[arr_idx+1][0], col_op_ls[arr_idx+1][1])
                    next_state = next_program
                if X_feat_embedding is None:
                    transition = Transition(X, X_pd_full,program, atom_ls, next_state, reward - prev_reward)
                else:
                    transition = Transition((X, X_feat_embedding), X_pd_full,program, atom_ls, next_state, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model_ls0()
                # print(loss)
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(np.max(reward, axis = -1) > 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            total_count = ((episode_i + 1)*self.batch_size)
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    
    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 28:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)

                # if episode_i == self.batch_size:
                #     print(y_true_array.reshape(-1))
                #     print(y_pred_array.reshape(-1))

                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        return y_pred_array
    
    def integrate_curr_program_with_prev_programs(self, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls):
        prev_prog_ids = atom_ls[prev_prog_key].cpu()
        curr_col_ids = atom_ls[col_key]
        program = []
        sample_ids = torch.arange(len(next_program[0]))
        # program length
        for pid in range(len(next_program)):
            program.append(torch.stack([next_program[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])],dim=1))
            
        program.append(curr_vec_ls)
        new_program_col_ls = []
        new_program_str = []
        for idx in range(len(program_col_ls)):
            curr_sample_new_program_col_ls = []
            curr_sample_new_program_str = []
            for k in range(self.topk_act):
                curr_new_program_col_ls = []
                curr_new_program_str = []
                # for pid in range(len(program_col_ls[idx])):
                
                #     curr_new_program_col_ls.append(program_col_ls[idx][prev_prog_ids[idx,k].item()][pid])
                #     # [k].append()
                #     curr_new_program_str.append(next_program_str[idx][prev_prog_ids[idx,k].item()][pid])
                curr_new_program_col_ls.extend(program_col_ls[idx][prev_prog_ids[idx,k].item()])
                curr_new_program_str.extend(next_program_str[idx][prev_prog_ids[idx,k].item()])
                
                
                curr_new_program_col_ls.append(curr_col_ids[idx][k])
                curr_new_program_str.append(curr_atom_str_ls[idx][k])
                curr_sample_new_program_col_ls.append(curr_new_program_col_ls)
                curr_sample_new_program_str.append(curr_new_program_str)
            new_program_col_ls.append(curr_sample_new_program_col_ls)
            new_program_str.append(curr_sample_new_program_str)
        return program, new_program_col_ls, new_program_str

    def copy_data_in_database(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(all_other_pats_ls[idx].copy())
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None, feat_embedding = None):
        pd.options.mode.chained_assignment = None

        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        y_pred_prob_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            col_list = list(self.train_dataset.data.columns)
        
            op_ls = list([operator.__le__, operator.__ge__])
            
            col_op_ls = []
            
            last_col = None

            last_op = None
            
            for col in col_list:
                if col == "PAT_ID" or col == "label":
                    continue
                last_col = col
                for op in op_ls:
                    col_op_ls.append((col, op))
                    last_op = op
            for episode_i, val in iterator:
                (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
                all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)
                
                X_feat_embeddings = None
                if feat_embedding is not None:
                    X_feat_embeddings = feat_embedding[X_sample_ids]
                
                # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
                program = []
                program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = pd.concat(X_pd_ls)
                
                # for arr_idx in range(len(col_op_ls)):
                for arr_idx in range(self.program_max_len):
                    (col, op) = col_op_ls[arr_idx]
                    # col_name = col_list[col_id]
                    if X_feat_embeddings is None:
                        atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, epsilon=0)
                    else:
                        atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, epsilon=0)
                    
                    curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings)

                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)
                    if len(program) > 0:                        
                        

                        next_program, program_col_ls, next_program_str= self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls)



                    else:

                        next_program.append(curr_vec_ls)

                        for vec_idx in range(len(curr_vec_ls)):
                            # vec = curr_vec_ls[vec_idx]
                            atom_str = curr_atom_str_ls[vec_idx]
                            for k in range(len(atom_ls[col_key][vec_idx])):
                                program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                                next_program_str[vec_idx][k].append(atom_str[k])
                            # next_program_str[vec_idx].append(atom_str)

                    
                    # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                    # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    
                    
                    # for new_atom_ls in atom_ls_ls:

                    #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                    #     next_program.append(torch.stack(curr_vec_ls))

                    #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    
                # while True: # episode
                #     atom,_ = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
                #     atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                #     reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                #     next_program = program.copy()
                #     next_program_str = program_str.copy()
                #     for new_atom_ls in atom_ls_ls:

                #         curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                #         next_program.append(torch.stack(curr_vec_ls))

                #         curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                #         for vec_idx in range(len(curr_vec_ls)):
                #             vec = curr_vec_ls[vec_idx]
                #             atom_str = curr_atom_str_ls[vec_idx]
                            
                #             next_program_str[vec_idx].append(atom_str)
                #             program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                #             reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                        # atom["num_op"] = atom_op
                        
                        
                        # next_program_str = next_program_str + []
                        
                        # program_atom_ls.append(new_atom_ls)
                    #apply new atom
                    # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                    next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                    # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
                    y_pred, y_pred_prob = self.get_test_decision_from_db_ls_multi(next_all_other_pats_ls)
                    # final_y_pred,_ = stats.mode(np.array(y_pred), axis = -1)
                    # final_y_pred_prob = np.mean(np.array(y_pred_prob), axis = -1)
                    
                    # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    # done = (col == last_col) and (op == last_op)
                    done = (arr_idx == self.program_max_len - 1)
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            for pat_idx in range(len(y_pred)):
                                col_ls = list(set(program_col_ls[pat_idx]))
                                col_ls.append("PAT_ID")
                                x_pat_sub = X_pd_ls[pat_idx][col_ls]
                                x_pat_sub.reset_index(inplace=True)
                                for col in col_ls:
                                    if not col == "PAT_ID":
                                        x_pat_sub[col] = x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]

                                msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[pat_idx]), y_pred[pat_idx], y_pred_prob[pat_idx], str(x_pat_sub.to_dict()),str(next_program_str[pat_idx]))
                                self.logger.log(level=logging.DEBUG, msg=msg)
                        # if y == y_pred: success += 1
                        # else: failure += 1
                        success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
                        failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
                        y_true_ls.extend(y.view(-1).tolist())
                        y_pred_ls.extend(y_pred)
                        y_pred_prob_ls.extend(y_pred_prob)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats_ls = next_all_other_pats_ls

            # for episode_i, val in iterator:
            #     (all_other_pats, X_pd, X), y = val
            #     program = []
            #     program_str = []
            #     program_atom_ls = []
            #     while True: # episode
            #         atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
            #         atom_ls = self.identify_op(X_pd, atom)
            #         next_program = program.copy()
            #         next_program_str = program_str.copy()
            #         for new_atom in atom_ls:
            #             next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
            #             # atom["num_op"] = atom_op
                        
                        
            #             next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
            #             program_atom_ls.append(new_atom)
            #         #apply new atom
            #         next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
            #         # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
            #         # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
            #         #check constraints
            #         x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
            #         prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
            #         y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
            #         db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
            #         #derive reward
            #         done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
            #         if done:
            #             next_program = None
            #         #update next step
            #         if done: #stopping condition
            #             if self.is_log:
            #                 msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
            #                 self.logger.log(level=logging.DEBUG, msg=msg)
            #             if y == y_pred: success += 1
            #             else: failure += 1
            #             y_true_ls.append(y.item())
            #             y_pred_ls.append(y_pred)
            #             break
            #         else:
            #             program = next_program
            #             program_str = next_program_str
            #             all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_prob_array = np.array(y_pred_prob_ls, dtype=float)
                # y_pred_prob_array = np.concatenate(y_pred_prob_ls, axis = 0)
                y_pred_array[y_pred_array < 0] = 0.5
                y_pred_prob_array[y_pred_prob_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    # auc_score= 0
                    auc_score_2 = 0
                else:
                    # auc_score = roc_auc_score(y_true_array.reshape(-1), y_pred_array.reshape(-1))
                    auc_score_2 = roc_auc_score(y_true_array.reshape(-1), y_pred_prob_array.reshape(-1))
                success_rate = (success / len(y_pred_array)) * 100.00
                avg_loss = sum_loss/len(y_pred_array)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score:{auc_score_2}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        

        additional_score_str = ""
        full_y_pred_prob_array = np.stack([1 - y_pred_prob_array.reshape(-1), y_pred_prob_array.reshape(-1)], axis=1)
        for metric_name in metrics_maps:
            curr_score = metrics_maps[metric_name](y_true_array.reshape(-1),full_y_pred_prob_array)
            additional_score_str += metric_name + ": " + str(curr_score) + " "
        print(additional_score_str)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = additional_score_str)
        # Print information
        
        # if exp_y_pred_arr is not None:
        #     nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
        #     print(nonzero_ids[0])
          
        self.dqn.policy_net.train()
        self.dqn.target_net.train()



    def run(self):
        # exp_pred_array = self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        if self.valid_dataset is not None:
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        # if self.valid_dataset is not None:
        #     self.test_epoch_ls(valid_loader, 0)    
        self.test_epoch_ls(test_loader, 0, feat_embedding=self.test_feat_embeddings)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(1, self.epochs + 1):
                self.train_epoch(i, train_loader)
                torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
                torch.save(self.dqn.target_net.state_dict(), os.path.join(self.work_dir, "target_net_" + str(i)))
                torch.save(self.dqn.memory, os.path.join(self.work_dir, "memory_" + str(i)))
                # self.test_epoch(i)
                if self.valid_dataset is not None:
                    self.test_epoch_ls(valid_loader, i, feat_embedding=self.valid_feat_embeddings)    
                self.test_epoch_ls(test_loader, i, feat_embedding=self.test_feat_embeddings)
                torch.cuda.empty_cache() 

            # self.test_epoch_ls(test_loader, i)



if __name__ == "__main__":
    seed = 10
    set_seed(seed)
    train_data_path = "synthetic_dataset.pd"#"simple_df"
    test_data_path = "synthetic_dataset.pd"
    test_data_path = "synthetic_test_dataset.pd"#"simple_df"
    precomputed_path = "synthetic_precomputed.npy"#"ehr_precomputed.npy"
    # train_data_path = "simple_df"
    # test_data_path = "simple_df"
    # precomputed_path = "ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 0.0005
    learning_rate = 0.001
    mem_sample_size = 16
    batch_size = 256
    batch_size = 8
    gamma = 0.999
    epsilon = 0.1
    epsilon_falloff = 0.9
    target_update = 10
    epochs = 100
    program_max_len = 4
    # program_max_len = 12
    patient_max_appts = 1
    provenance = "difftopkproofs"
    latent_size = 0
    # latent_size = 80
    is_log = True
    dropout_p = 0
    work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    if is_log:
        log_folder = os.path.join(work_dir,'logs_new/')
        log_path = './logs/'+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        logging.basicConfig(filename=log_path,
                filemode='a',
                format='%(message)s',
                level=logging.DEBUG)
        logging.info("EHR Explanation Synthesis\n Seed: {}, train_path: {}, test_path: {}, precomputed_path: {}, mem_cap: {}, learning_rate: {}, batch_\
        size: {}, gamma: {}, epsilon: {}, epsilon_falloff: {}, target_update: {}, epochs: {}, prog_max_len: {}, pat_max_appt: {}, latent_size: {}, dropout: {}".format(
            seed, train_data_path,test_data_path,precomputed_path,replay_memory_capacity,learning_rate,batch_size,gamma,epsilon,epsilon_falloff,target_update,epochs,
            program_max_len,patient_max_appts,latent_size, dropout_p))

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_dataset = EHRDataset(data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    test_dataset = EHRDataset(data = test_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    precomputed = np.load(precomputed_path, allow_pickle=True).item()
    lang = Language(data=train_data, precomputed=precomputed, lang=synthetic_lang)
    feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    train_dataset.rescale_data(feat_range_mappings) 
    test_dataset.rescale_data(feat_range_mappings) 
    model = "mlp"
    # model="transformer"

    category_count = None
    numeric_count = None
    
    numeric_count, category_count = obtain_numeric_categorical_value_count(train_dataset)

    trainer = Trainer6_2(lang=lang, train_dataset=train_dataset,valid_dataset=None,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed, work_dir=log_folder, numeric_count=numeric_count, category_count=category_count, model=model
                        )
    # trainer.dqn.policy_net.load_state_dict(torch.load(os.path.join(log_folder, "policy_net_backup")))
    # trainer.dqn.target_net.load_state_dict(torch.load(os.path.join(log_folder, "policy_net_backup")))

    trainer = Trainer3_2(lang=lang, train_dataset=train_dataset,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=mem_sample_size, seed=seed
                        )

    # batch_size=mem_sample_size
    # trainer = Trainer4(lang=lang, train_dataset=train_dataset,
    #                     test_dataset = test_dataset,
    #                     replay_memory_capacity=replay_memory_capacity,
    #                     learning_rate=learning_rate, batch_size=batch_size, 
    #                     gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
    #                     target_update=target_update, epochs=epochs, provenance=provenance,
    #                     program_max_len=program_max_len, patient_max_appts=patient_max_appts,
    #                     latent_size=latent_size, is_log = is_log, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, seed=seed
    #                     )
    
    
    # trainer = Trainer3_2(lang=lang, train_dataset=train_dataset,
    #                     test_dataset = test_dataset,
    #                     replay_memory_capacity=replay_memory_capacity,
    #                     learning_rate=learning_rate, batch_size=batch_size, 
    #                     gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
    #                     target_update=target_update, epochs=epochs, provenance=provenance,
    #                     program_max_len=program_max_len, patient_max_appts=patient_max_appts,
    #                     latent_size=latent_size, is_log = is_log, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, seed=seed, mem_sample_size=mem_sample_size
    #                     )
    
    # trainer = Trainer(lang=lang, train_dataset=train_dataset,
    #                     test_dataset = test_dataset,
    #                     replay_memory_capacity=replay_memory_capacity,
    #                     learning_rate=learning_rate, batch_size=batch_size, 
    #                     gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
    #                     target_update=target_update, epochs=epochs, provenance=provenance,
    #                     program_max_len=program_max_len, patient_max_appts=patient_max_appts,
    #                     latent_size=latent_size, is_log = is_log, dropout_p=dropout_p)
    # # , feat_range_mappings=feat_range_mappings, seed=seed
    # #                     )
    
    trainer.run()

