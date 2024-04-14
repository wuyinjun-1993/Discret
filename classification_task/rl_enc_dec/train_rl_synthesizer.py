from rl_synthesizer import DQN
from rl_synthesizer_2 import DQN2, RLSynthesizerNetwork2
from rl_synthesizer_3 import DQN3, Transition
import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import synthetic_lang
import logging
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import operator
import sys
import os
from torch.utils.data import DataLoader
import torch
import argparse

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
        y = torch.tensor(m, device=DEVICE, dtype=torch.float)
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.tensor(i, device=DEVICE, dtype=torch.float) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.tensor([0]*len(X[0]), device=DEVICE, dtype=torch.float) ]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y

    @staticmethod
    def collate_fn(data):
        all_other_pats_ls = [data[idx][0][0] for idx in range(len(data))]
        all_x_pd_ls = [data[idx][0][1] for idx in range(len(data))]
        all_x_ls = [data[idx][0][2] for idx in range(len(data))]
        y_ls = [data[idx][1] for idx in range(len(data))]
        
        return (all_other_pats_ls, all_x_pd_ls, all_x_ls), y_ls
        

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
        iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        for episode_i, val in iterator:
            (all_other_pats, X_pd, X), y = val
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
                transition = Transition(X,program, atom, next_program, reward)
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
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats

            # Print information
            success_rate = (success / (episode_i + 1)) * 100.00
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        for i in range(1, self.epochs + 1):
            self.train_epoch(i)
            self.test_epoch(i)



# class Trainer:
#     def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log):
#         self.dqn = DQN(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size)
#         self.epsilon = epsilon
#         self.epsilon_falloff = epsilon_falloff
#         self.epochs = epochs
#         self.train_dataset = train_dataset
#         self.valid_dataset = valid_dataset
#         self.test_dataset = test_dataset
#         self.lang = lang
#         self.target_update = target_update
#         self.program_max_len = program_max_len
#         self.is_log = is_log
#         if self.is_log:
#             self.logger = logging.getLogger()


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

#     def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
#         satisfy_bool=True
#         for atom in atom_ls:
#             curr_bool = lang.evaluate_atom_on_sample(atom, X)
#             satisfy_bool = satisfy_bool & curr_bool
#         return satisfy_bool

#     def check_program_constraint(self, prog: list) -> bool:
#         return len(prog) < self.program_max_len
    
#     def train_epoch(self, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
#         # for episode_i, val in iterator:
#         #     (all_other_pats, X_pd, X), y = val
#         iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
#         # pos_count = np.sum(self.train_dataset.data["label"] == 1)
#         # neg_count = np.sum(self.train_dataset.data["label"] == 0)
#         # sample_weights = torch.ones(len(self.train_dataset.data))
#         # sample_weights[np.array(self.train_dataset.data["label"]) == 1] = neg_count/(neg_count + pos_count)
#         # sample_weights[np.array(self.train_dataset.data["label"]) == 0] = pos_count/(neg_count + pos_count)
#         # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(self.train_dataset.data), replacement=True)
#         # iterator = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, collate_fn = EHRDataset.collate_fn)
#         # episode_i = 0
#         # for val in iterator:
#         all_rand_ids = torch.randperm(len(self.train_dataset))
#         for episode_i, sample_idx in iterator:
#             (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
#             program = []
#             program_str = []
#             program_atom_ls = []
#             while True: # episode
#                 atom = self.dqn.predict_atom(features=X, program=program, epsilon=self.epsilon)
#                 program_atom_ls.append(atom)
#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
#                 next_all_other_pats_valid = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, self.valid_dataset.data)
                
#                 next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
#                 #check constraints
#                 x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
#                 db_cons_valid = self.check_db_constrants(next_all_other_pats_valid, y=int(y[0])) #entropy
#                 #derive reward
#                 reward = db_cons if x_cons else 0 # NOTE: these become part of reward
#                 done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
#                 #record transition in buffer
#                 if done:
#                     next_program = None
#                 transition = Transition(X,X_pd, program, atom, next_program, reward)
#                 self.dqn.observe_transition(transition)
#                 #update model
#                 loss = self.dqn.optimize_model()
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

#     def test_epoch(self, epoch, valid=False):
#         y_true_ls = []
#         y_pred_ls = []
#         success, failure, sum_loss = 0, 0, 0.
#         if valid:
#             iterator = tqdm(enumerate(self.valid_dataset), desc="Testing Synthesizer", total=len(self.valid_dataset))
#         else:
#             iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
#         for episode_i, val in iterator:
#             (all_other_pats, X_pd, X), y = val
#             program = []
#             program_str = []
#             program_atom_ls = []
#             while True: # episode
#                 atom = self.dqn.predict_atom(features=X, program=program, epsilon=0)
#                 #apply new atom
#                 program_atom_ls.append(atom)
#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
#                 next_all_other_pats_valid = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, self.valid_dataset.data)
#                 # next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
#                 next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
#                 #check constraints
#                 x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 y_pred = self.get_test_decision_from_db(next_all_other_pats)
#                 db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
#                 #derive reward
#                 done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
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

#             y_true_array = np.array(y_true_ls, dtype = float)
#             y_pred_array = np.array(y_pred_ls, dtype = float)
#             y_pred_array[y_pred_array < 0] = 0.5
#             # auc_score = roc_auc_score(y_true_array, y_pred_array)
#             if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#             #     recall = 0
#             #     f1 = 0
#                 auc_score= 0
#             else:
#                 auc_score = roc_auc_score(y_true_array, y_pred_array)
#             # if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#             #     recall = 0
#             #     f1 = 0
#             # else:
#             #     recall = recall_score(y_true_array, y_pred_array)
#             #     f1 = f1_score(y_true_array, y_pred_array)
#             # y_true_array = y_true_array[y_pred_array >= 0]
#             # y_pred_array = y_pred_array[y_pred_array >= 0]
#             # recall = recall_score(y_true_array, y_pred_array)
#             # f1 = f1_score(y_true_array, y_pred_array)
#             # Print information
#             # recall = recall_score(np.array(y_true_ls), np.array(y_pred_ls))
#             # f1 = f1_score(np.array(y_true_ls), np.array(y_pred_ls))
#             success_rate = (success / (episode_i + 1)) * 100.00
#             avg_loss = sum_loss/(episode_i+1)
#             desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc_score: {auc_score}"
#             iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
#     def run(self):
#         self.test_epoch(0)
#         for i in range(1, self.epochs + 1):
#             self.train_epoch(i)
#             self.test_epoch(i, valid=True)
#             self.test_epoch(i)
            
class Trainer2:
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, feat_range_mappings):
        self.dqn = DQN2(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, feat_range_mappings=feat_range_mappings)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
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

    def identify_op(self, X:pd, atom:dict):
        selected_feat_val = list(X[atom["num_feat"]])[0]
        constant = atom[atom["num_feat"]]
        if selected_feat_val > constant:
            atom_op = operator.__ge__
        elif selected_feat_val < constant:
            atom_op = operator.__le__
        else:
            atom_op = operator.__eq__
            
        return atom_op



    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def obtain_constants_on_dataset(self, data, program, atom, x):
        
        all_conds = None
        for prev_atom in program:
            feat = prev_atom["num_feat" if "num_feat" in prev_atom else "cat_feat"]
            op = prev_atom["num_op" if "num_op" in prev_atom else "cat_op"]
            target_const = prev_atom[feat]
            cond = op(data[feat], target_const)
            if all_conds is None:
                all_conds = (cond)
            else:
                all_conds = all_conds & (cond)            
        if all_conds is not None:
            patient_matches = data.loc[all_conds,"PAT_ID"].unique()
            remaining_data = data[data['PAT_ID'].isin(patient_matches)]
        else:
            remaining_data = data
        feats = np.array(remaining_data[atom["num_feat"]])
        labels = np.array(remaining_data["label"])
        clf = DecisionTreeClassifier(random_state=0, max_depth=1)
        clf.fit(feats.reshape(len(feats), 1), labels)
        threshold = clf.tree_.threshold[0]
        if np.array(x[atom["num_feat"]])[0] >= threshold:
            atom["num_op"] = operator.__ge__
        else:
            atom["num_op"] = operator.__le__
        atom[atom["num_feat"]] = threshold
        return atom
        # clf.decision_path(np.array(x[atom["num_feat"]]).reshape(len(x), 1))
    
    def check_redundancy(self, atom_ls, atom, atom_op):
        has_redundancy=False
        for existing_atom in atom_ls:
            if atom["num_feat"] == existing_atom["num_feat"]:
                if existing_atom[atom["num_feat"]] >= atom[atom["num_feat"]] and existing_atom["num_op"] == op.__ge__ and atom_op == op.__ge__:
                    has_redundancy=True
                    break
                
                if existing_atom[atom["num_feat"]] <= atom[atom["num_feat"]] and existing_atom["num_op"] == op.__le__ and atom_op == op.__le__:
                    has_redundancy=True
                    break
                    
                    
        return has_redundancy
    
    # 200 features
    #aggregation synthesis
    #  ppo learning 
    # constants learning => continual space
    # steps by sptes => taking steps by episilon
    # ppo to work
    # decision tree ????
    # better accuracy?
    # global explanations 
    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
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
        for episode_i, sample_idx in iterator:
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # all_other_pats = all_other_pats[0]
            # X_pd = X_pd[0]
            # X = X[0]
            # y = y[0]
            program_atom_ls = []
            program = []
            program_str = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, program=program, epsilon=self.epsilon)
                # self.obtain_constants_on_dataset(all_other_pats, program, atom, X_pd)
                #apply new atom
                
                atom_op = self.identify_op(X_pd, atom)
                # has_redundancy = self.check_redundancy(program_atom_ls, atom, atom_op)
                # if has_redundancy:
                #     continue
                
                
                next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                atom["num_op"] = atom_op
                
                
                next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                
                program_atom_ls.append(atom)
                
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                next_all_other_pats_valid = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, self.valid_dataset.data)


                #check constraints
                x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
                db_cons_valid = self.check_db_constrants(next_all_other_pats_valid, y=int(y[0])) #entropy
                #derive reward
                # reward = db_cons*(1-np.abs(db_cons_valid - db_cons)) if x_cons else 0 # NOTE: these become part of reward
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
            episode_i += 1
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff
        print("epsilon value::", self.epsilon)

    def test_epoch(self, epoch, valid=False):
        success, failure, sum_loss = 0, 0, 0.
        if valid:
            iterator = tqdm(enumerate(self.valid_dataset), desc="Testing Synthesizer", total=len(self.valid_dataset))
        else:
            iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls = []
        y_pred_ls = []
        for episode_i, val in iterator:
            (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            program_atom_ls = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, program=program, epsilon=0)
                #apply new atom
                atom_op = self.identify_op(X_pd, atom)
                
                # has_redundancy = self.check_redundancy(program_atom_ls, atom, atom_op)
                # if has_redundancy:
                #     continue
                next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                atom["num_op"] = atom_op
                next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                program_atom_ls.append(atom)

                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                #check constraints
                x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                y_pred = self.get_test_decision_from_db(next_all_other_pats)
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

            # Print information
            y_true_array = np.array(y_true_ls, dtype=float)
            y_pred_array = np.array(y_pred_ls, dtype=float)
            y_pred_array[y_pred_array < 0] = 0.5
            if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
            #     recall = 0
            #     f1 = 0
                auc_score= 0
            else:
                auc_score = roc_auc_score(y_true_array, y_pred_array)
            #     recall = recall_score(y_true_array, y_pred_array)
            #     f1 = f1_score(y_true_array, y_pred_array)
            
            # y_true_array = y_true_array[y_pred_array >= 0]
            # y_pred_array = y_pred_array[y_pred_array >= 0]
            
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
            self.test_epoch(i, valid=True)
            self.test_epoch(i)


class Trainer3:
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, feat_range_mappings, work_dir, discretize_feat_value_count=20):
        self.dqn = DQN3(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, feat_range_mappings=feat_range_mappings)#, discretize_feat_value_count=discretize_feat_value_count)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.is_log = is_log
        self.work_dir = work_dir
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

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def obtain_constants_on_dataset(self, data, program, atom, x):
        
        all_conds = None
        for prev_atom in program:
            feat = prev_atom["num_feat" if "num_feat" in prev_atom else "cat_feat"]
            op = prev_atom["num_op" if "num_op" in prev_atom else "cat_op"]
            target_const = prev_atom[feat]
            cond = op(data[feat], target_const)
            if all_conds is None:
                all_conds = (cond)
            else:
                all_conds = all_conds & (cond)            
        if all_conds is not None:
            patient_matches = data.loc[all_conds,"PAT_ID"].unique()
            remaining_data = data[data['PAT_ID'].isin(patient_matches)]
        else:
            remaining_data = data
        feats = np.array(remaining_data[atom["num_feat"]])
        labels = np.array(remaining_data["label"])
        clf = DecisionTreeClassifier(random_state=0, max_depth=1)
        clf.fit(feats.reshape(len(feats), 1), labels)
        threshold = clf.tree_.threshold[0]
        if np.array(x[atom["num_feat"]])[0] >= threshold:
            atom["num_op"] = operator.__ge__
        else:
            atom["num_op"] = operator.__le__
        atom[atom["num_feat"]] = threshold
        return atom
        # clf.decision_path(np.array(x[atom["num_feat"]]).reshape(len(x), 1))
    
    def check_redundancy(self, atom_ls, atom, atom_op):
        has_redundancy=False
        for existing_atom in atom_ls:
            if atom["num_feat"] == existing_atom["num_feat"]:
                if existing_atom[atom["num_feat"]] >= atom[atom["num_feat"]] and existing_atom["num_op"] == op.__ge__ and atom_op == op.__ge__:
                    has_redundancy=True
                    break
                
                if existing_atom[atom["num_feat"]] <= atom[atom["num_feat"]] and existing_atom["num_op"] == op.__le__ and atom_op == op.__le__:
                    has_redundancy=True
                    break
                    
                    
        return has_redundancy
    
    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
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
        for episode_i, sample_idx in iterator:
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # all_other_pats = all_other_pats[0]
            # X_pd = X_pd[0]
            # X = X[0]
            # y = y[0]
            program_atom_ls = []
            program = []
            program_str = []
            while True: # episode
                atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=self.epsilon)
                # self.obtain_constants_on_dataset(all_other_pats, program, atom, X_pd)
                #apply new atom
                
                atom_ls = self.identify_op(X_pd, atom)
                # has_redundancy = self.check_redundancy(program_atom_ls, atom, atom_op)
                # if has_redundancy:
                #     continue
                
                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)
                
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                next_all_other_pats_valid = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, self.valid_dataset.data)


                #check constraints
                x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
                db_cons_valid = self.check_db_constrants(next_all_other_pats_valid, y=int(y[0])) #entropy
                #derive reward
                # reward = db_cons*(1-np.abs(db_cons_valid - db_cons)) if x_cons else 0 # NOTE: these become part of reward
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
            success_rate = (success / (episode_i + 1)) * 100.00
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            iterator.set_description(desc)
            episode_i += 1
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff
        print("epsilon value::", self.epsilon)

    def test_epoch(self, epoch, valid=False):
        success, failure, sum_loss = 0, 0, 0.
        if valid:
            iterator = tqdm(enumerate(self.valid_dataset), desc="Testing Synthesizer", total=len(self.valid_dataset))
        else:
            iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls = []
        y_pred_ls = []
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 5:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    #apply new atom
                    # atom_op = self.identify_op(X_pd, atom)

                    atom_ls = self.identify_op(X_pd, atom)
                    # has_redundancy = self.check_redundancy(program_atom_ls, atom, atom_op)
                    # if has_redundancy:
                    #     continue
                    
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    
                    # has_redundancy = self.check_redundancy(program_atom_ls, atom, atom_op)
                    # if has_redundancy:
                    #     continue
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # atom["num_op"] = atom_op
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    # program_atom_ls.append(atom)

                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    #check constraints
                    x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)
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

                # Print information
                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)
                #     recall = recall_score(y_true_array, y_pred_array)
                #     f1 = f1_score(y_true_array, y_pred_array)
                
                # y_true_array = y_true_array[y_pred_array >= 0]
                # y_pred_array = y_pred_array[y_pred_array >= 0]
                
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        self.test_epoch(0)
        for i in range(1, self.epochs + 1):
            
            self.train_epoch(i)
            torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
            self.test_epoch(i, valid=True)
            self.test_epoch(i)



def split_train_valid_set(train_data, valid_ratio=0.2):
    total_count = len(train_data)
    random_train_ids = torch.randperm(total_count)
    valid_ids = random_train_ids[0:int(total_count*valid_ratio)]
    train_ids = random_train_ids[int(total_count*valid_ratio):]
    
    valid_data = train_data.iloc[valid_ids]
    train_data = train_data.iloc[train_ids]
    
    return train_data, valid_data


def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.data.columns)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings



def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="std of the initial phi table")
    parser.add_argument('--batch_size', type=int, default=16, help="std of the initial phi table")
    parser.add_argument('--gamma', type=float, default=0.999, help="std of the initial phi table")
    parser.add_argument('--epsilon', type=float, default=0.2, help="std of the initial phi table")
    parser.add_argument('--epsilon_falloff', type=float, default=0.9, help="std of the initial phi table")
    parser.add_argument('--dropout_p', type=float, default=0.1, help="std of the initial phi table")
    parser.add_argument('--target_update', type=int, default=20, help="std of the initial phi table")
    parser.add_argument('--epochs', type=int, default=100, help="std of the initial phi table")
    parser.add_argument('--program_max_len', type=int, default=4, help="std of the initial phi table")
    parser.add_argument('--patient_max_appts', type=int, default=1, help="std of the initial phi table")
    parser.add_argument('--latent_size', type=int, default=80, help="std of the initial phi table")
    parser.add_argument('--new', action='store_true', help='specifies what features to extract')
    # discretize_feat_value_count
    parser.add_argument('--discretize_feat_value_count', type=int, default=10, help="std of the initial phi table")

    # parser.add_argument('--input', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--output', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--log_path', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--cache_path', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--task_name', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--dataset', type=str, default="physionet", choices=["physionet", "mimic3"], help="dataset name")
    # parser.add_argument('--model_type', type=str, default="mTan", choices=["mTan", "csdi", "saits"], help="std of the initial phi table")
    
    
    # parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
    #                     choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    # parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
    #                     choices=['all', 'len', 'all_but_len'])

    # parser.add_argument('--imputed', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--do_train', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--classify_task', action='store_true', help='specifies what features to extract')


    args = parser.parse_args()

    return args





if __name__ == "__main__":
    args = parse_args()
    seed = 0
    work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    train_data_path = os.path.join(work_dir, "synthetic_dataset.pd")#"simple_df"
    test_data_path = os.path.join(work_dir, "synthetic_test_dataset.pd")#"simple_df"
    precomputed_path = os.path.join(work_dir, "synthetic_precomputed.npy")#"ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_falloff = args.epsilon_falloff
    target_update = args.target_update
    epochs = args.epochs
    program_max_len = args.program_max_len
    patient_max_appts = args.patient_max_appts
    provenance = "difftopkproofs"
    latent_size = args.latent_size
    is_log = True
    dropout_p = 0.1

    if is_log:
        if len(sys.argv) >= 2 and sys.argv[1] == "new":
            log_folder = os.path.join(work_dir,'logs_new/')
            os.makedirs(os.path.join(work_dir,'logs_new/'), exist_ok=True)
            log_path = os.path.join(work_dir,'logs_new/')+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        else:
            log_folder = os.path.join(work_dir,'logs_old/')
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
    torch.manual_seed(seed)
    random.seed(seed)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    train_data, valid_data = split_train_valid_set(train_data, valid_ratio=0.2)
    
    
    train_dataset = EHRDataset(data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    feat_range_mappings = obtain_feat_range_mappings(train_dataset)    
    valid_dataset = EHRDataset(data=valid_data, other_data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    test_dataset = EHRDataset(data = test_data, other_data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    # precomputed = np.load(precomputed_path, allow_pickle=True).item()
    precomputed=dict()
    for attr in list(train_data.columns):
        if not attr == "PAT_ID" and not attr == "label":
            precomputed[attr] = [1]
    
    lang = Language(data=train_data, precomputed=precomputed, lang=synthetic_lang)
    # if len(sys.argv) >= 2 and sys.argv[1] == "new":
    if args.new:
        print("use new pipeline")
        trainer = Trainer3(lang=lang, train_dataset=train_dataset,valid_dataset=valid_dataset,
                            test_dataset = test_dataset,
                            replay_memory_capacity=replay_memory_capacity,
                            learning_rate=learning_rate, batch_size=batch_size, 
                            gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                            target_update=target_update, epochs=epochs, provenance=provenance,
                            program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                            latent_size=latent_size, is_log = is_log, feat_range_mappings = feat_range_mappings, work_dir=log_folder#, discretize_feat_value_count=args.discretize_feat_value_count
                            )
        
        # trainer.dqn.policy_net.load_state_dict(torch.load(os.path.join(log_folder, "policy_net_backup_2")))
        # trainer.dqn.target_net.load_state_dict(torch.load(os.path.join(log_folder, "policy_net_backup_2")))
        
    else:
        print("use old pipeline")
        # trainer = Trainer(lang=lang, train_dataset=train_dataset,valid_dataset=valid_dataset,
        #                     test_dataset = test_dataset,
        #                     replay_memory_capacity=replay_memory_capacity,
        #                     learning_rate=learning_rate, batch_size=batch_size, 
        #                     gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
        #                     target_update=target_update, epochs=epochs, provenance=provenance,
        #                     program_max_len=program_max_len, patient_max_appts=patient_max_appts,
        #                     latent_size=latent_size, is_log = is_log
        #                     )
        trainer = Trainer(lang=lang, train_dataset=train_dataset,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, dropout_p=args.dropout_p
                        )
    
    trainer.run()

