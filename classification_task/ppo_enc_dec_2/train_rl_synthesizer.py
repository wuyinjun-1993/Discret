from rl_synthesizer import *
from rl_synthesizer_debug_2 import *
from rl_synthesizer_debug_3 import *
from rl_synthesizer_debug import *

import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import synthetic_lang
import logging
from datetime import datetime
import operator
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


class EHRDataset(Dataset):
    def __init__(self, data, drop_cols,patient_max_appts, balance):
        self.data = data
        self.patient_max_appts = patient_max_appts
        self.drop_cols = drop_cols
        self.patient_ids = self.data['PAT_ID'].unique().tolist()
        if balance:
            most = self.data['label'].value_counts().max()
            for label in self.data['label'].unique():
                match  = self.data.loc[self.data['label'] == label]['PAT_ID'].to_list()
                samples = [random.choice(match) for _ in range(most-len(match))]
                self.patient_ids.extend(samples)
        # random.shuffle(self.patient_ids)

    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[idx]]
        m = [appts['label'].max()]
        y = torch.FloatTensor(m)
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i).to(DEVICE) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.FloatTensor([0]*len(X[0])).to(DEVICE)]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y
    
    
    
        
    @staticmethod
    def collate_fn(data):
        all_other_pats_ls = [item[0][0] for item in data]
        X_pd_ls = [item[0][1] for item in data]
        X_ls = [item[0][2][0].view(1,-1) for item in data]
        X_tensor = torch.cat(X_ls)
        y_ls = [item[1].view(1,-1) for item in data]
        y_tensor = torch.cat(y_ls)
        return (all_other_pats_ls, X_pd_ls, X_tensor), y_tensor

class EHRDeathsOnlyDataset(Dataset):
    def __init__(self, data, drop_cols,patient_max_appts):
        self.data = data
        self.drop_cols = drop_cols
        self.patient_max_appts = patient_max_appts
        self.patient_ids = sorted(self.data['PAT_ID'].unique())
        self.deaths_only = []
        for pat_id in sorted(self.data['PAT_ID'].unique()):
            appts = self.data.loc[self.data['PAT_ID'] == pat_id]
            y = torch.FloatTensor([appts['label'].max()])
            if y[0] == 1:
                self.deaths_only.append(pat_id)
    def __len__(self):
        return len(self.deaths_only)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.deaths_only[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.deaths_only[idx]]
        y = torch.FloatTensor([appts['label'].max()])
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.FloatTensor([0]*len(X[0]))]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y

def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.data.columns)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings


class Trainer:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, timesteps_per_batch, n_updates_per_iteration, clip):
        self.PPO = PPO_debug(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, n_updates_per_iteration=n_updates_per_iteration, clip=clip)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.timesteps_per_batch = timesteps_per_batch
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
        t = 0
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        all_rand_ids = torch.randperm(len(self.train_dataset))
        for episode_i, sample_idx in iterator:
        # for episode_i, val in iterator:
            ep_rews = []
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # (all_other_pats, X_pd, X), y = val
            program = []
            program_str = []
            while t < self.timesteps_per_batch:
                atom_idx, atom_pred = self.PPO.predict_atom(features=X, program=program, train=True)
                atom = self.PPO.idx_to_atom(atom_idx)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
                next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
                #derive reward
                reward = db_cons if x_cons else 0 # NOTE: these become part of reward
                batch_obs.append((X, program))
                ep_rews.append(reward)
                batch_acts.append(atom_idx)
                atom_probs = self.PPO.idx_to_logs(atom_pred, atom_idx)
                batch_log_probs.append(atom_probs)

                done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check

                

                #increment t
                t += 1

                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats
            
            batch_rews.append(ep_rews)
            batch_lens.append(len(program))

            if t == self.timesteps_per_batch:
                batch_rtgs = self.PPO.compute_rtgs(batch_rews=batch_rews)
                batch = (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)
                sum_loss += self.PPO.learn(batch=batch)
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = [], [], [], [], [], []
                t = 0

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
                atom_idx, _ = self.PPO.predict_atom(features=X, program=program, train=False)
                atom = self.PPO.idx_to_atom(atom_idx)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
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
            # desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        for i in range(1, self.epochs + 1):
            self.train_epoch(i)
            self.test_epoch(i)


class Trainer2:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, timesteps_per_batch, n_updates_per_iteration, clip, feat_range_mappings):
        self.PPO = PPO_debug2(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, n_updates_per_iteration=n_updates_per_iteration, clip=clip, feat_range_mappings=feat_range_mappings)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.timesteps_per_batch = timesteps_per_batch
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


    def train_epoch(self, epoch):
        t = 0
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        all_rand_ids = torch.randperm(len(self.train_dataset))
        for episode_i, sample_idx in iterator:
            ep_rews = []

            # (all_other_pats, X_pd, X), y = val
            (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            program_str = []
            program_atom_ls = []
            while t < self.timesteps_per_batch:
                atom_idx, atom_pred = self.PPO.predict_atom(features=X, X_pd=X_pd, program=program, train=True)
                atom = self.PPO.idx_to_atom(atom_idx)
                atom_ls = self.identify_op(X_pd, atom)
                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.PPO.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)
                #apply new atom
                # next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                batch_obs.append((X_pd, X, program))
                ep_rews.append(reward)
                batch_acts.append(atom_idx)
                atom_probs = self.PPO.idx_to_logs(atom_pred, atom_idx)
                batch_log_probs.append(atom_probs)

                done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check

                

                #increment t
                t += 1

                #update next step
                if done: #stopping condition
                    if reward > 0.5: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats = next_all_other_pats
            
            batch_rews.append(ep_rews)
            batch_lens.append(len(program))

            if t == self.timesteps_per_batch:
                batch_rtgs = self.PPO.compute_rtgs(batch_rews=batch_rews)
                batch = (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)
                sum_loss += self.PPO.learn(batch=batch)
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = [], [], [], [], [], []
                t = 0

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
                atom_idx, _ = self.PPO.predict_atom(features=X, X_pd=X_pd, program=program, train=False)
                atom = self.PPO.idx_to_atom(atom_idx)
                atom_ls = self.identify_op(X_pd, atom)
                next_program = program.copy()
                next_program_str = program_str.copy()
                for new_atom in atom_ls:
                    next_program = next_program + [self.PPO.atom_to_vector(new_atom)]
                    # atom["num_op"] = atom_op
                    
                    
                    next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
                    program_atom_ls.append(new_atom)

                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
                # next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
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
            # desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
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
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, timesteps_per_batch, n_updates_per_iteration, clip, feat_range_mappings):
        self.PPO = PPO_debug3(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, n_updates_per_iteration=n_updates_per_iteration, clip=clip, feat_range_mappings=feat_range_mappings)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.timesteps_per_batch = timesteps_per_batch
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

    def train_epoch(self, train_loader, epoch):
        t = 0
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        # all_rand_ids = torch.randperm(len(self.train_dataset))
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))

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
            program = []
            program_str = [[] for _ in range(len(X_pd_ls))]
            program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            
            X_pd_full = pd.concat(X_pd_ls)

            ep_rews = []

            # (all_other_pats, X_pd, X), y = val
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            # program = []
            # program_str = []
            # program_atom_ls = []

            prev_reward = np.zeros(len(X))

            for arr_idx in range(len(col_op_ls)):
                (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                atom_ls = self.PPO.predict_atom(features=X, X_pd_ls=X_pd_full, program=program, col=col, op=op, train=True)
                next_program = program.copy()
                
                curr_vec_ls = self.PPO.atom_to_vector_ls0(atom_ls, col, op)

                next_program.append(curr_vec_ls)

                curr_atom_str_ls = self.lang.atom_to_str_ls0(atom_ls, col, op, pred_v_key)
               
                next_program_str = program_str.copy()
                
                for vec_idx in range(len(curr_vec_ls)):
                    # vec = curr_vec_ls[vec_idx]
                    atom_str = curr_atom_str_ls[vec_idx]
                    
                    next_program_str[vec_idx].append(atom_str)

                next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset0(atom_ls, all_other_pats_ls, col, op, pred_v_key)

                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                
                reward = db_cons
                
                batch_obs.append((X, X_pd_full, program, col, op))
                ep_rews.append(reward - prev_reward)
                batch_acts.append(atom_ls)
                atom_probs = atom_ls[pred_prob_key] #self.PPO.idx_to_logs(atom_pred, atom_idx)
                # atom_log_probs = atom_probs[torch.tensor(list(range(atom_probs.shape[0]))), atom_ls[pred_prob_id]]
                atom_log_probs = self.PPO.idx_to_logs(atom_probs, atom_ls[pred_prob_id])
                batch_log_probs.append(atom_log_probs)
                done = (col == last_col) and (op == last_op)
                
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(reward > 0.5)
                    failure += np.sum(reward <= 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
                
                
            # while t < self.timesteps_per_batch:
            #     atom_idx, atom_pred = self.PPO.predict_atom(features=X, X_pd=X_pd, program=program, train=True)
            #     atom = self.PPO.idx_to_atom(atom_idx)
            #     atom_ls = self.identify_op(X_pd, atom)
            #     next_program = program.copy()
            #     next_program_str = program_str.copy()
            #     for new_atom in atom_ls:
            #         next_program = next_program + [self.PPO.atom_to_vector(new_atom)]
            #         # atom["num_op"] = atom_op
                    
                    
            #         next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                    
            #         program_atom_ls.append(new_atom)
            #     #apply new atom
            #     # next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
            #     next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
            #     # next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
            #     # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
            #     #check constraints
            #     # x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
            #     prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
            #     db_cons = self.check_db_constrants(next_all_other_pats, y=int(y[0])) #entropy
            #     #derive reward
            #     reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
            #     batch_obs.append((X_pd_full, X, program))
            #     ep_rews.append(reward)
            #     batch_acts.append(atom_idx)
            #     atom_probs = self.PPO.idx_to_logs(atom_pred, atom_idx)
            #     batch_log_probs.append(atom_probs)

            #     done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check

                

            #     #increment t
            #     t += 1

            #     #update next step
            #     if done: #stopping condition
            #         if reward > 0.5: success += 1
            #         else: failure += 1
            #         break
            #     else:
            #         program = next_program
            #         program_str = next_program_str
            #         all_other_pats = next_all_other_pats
            
            batch_rews.append(ep_rews)
            batch_lens.append(len(program))

            t += 1

            if t == self.timesteps_per_batch:
                batch_rtgs = self.PPO.compute_rtgs(batch_rews=batch_rews)
                batch = (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)
                sum_loss += self.PPO.learn(batch=batch)
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = [], [], [], [], [], []
                t = 0

            # Print information
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{(episode_i + 1)*self.batch_size} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    # def test_epoch(self, test_loader, epoch):
    #     success, failure, sum_loss = 0, 0, 0.
    #     # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
    #     y_true_ls=[]
    #     y_pred_ls=[]
        
    #     col_list = list(self.train_dataset.data.columns)
        
    #     op_ls = list([operator.__le__, operator.__ge__])
        
    #     col_op_ls = []
        
    #     last_col = None

    #     last_op = None
        
    #     iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        
    #     self.PPO.actor.eval()
    #     self.PPO.critic.eval()
        
    #     for col in col_list:
    #         if col == "PAT_ID" or col == "label":
    #             continue
    #         last_col = col
    #         for op in op_ls:
    #             col_op_ls.append((col, op))
    #             last_op = op
        
    #     with torch.no_grad():
        
    #         for episode_i, val in iterator:
    #             (all_other_pats, X_pd, X), y = val
    #             program = []
    #             program_str = []
    #             program_atom_ls = []
    #             while True: # episode
    #                 atom_idx, _ = self.PPO.predict_atom(features=X, X_pd=X_pd, program=program, train=False)
    #                 atom = self.PPO.idx_to_atom(atom_idx)
    #                 atom_ls = self.identify_op(X_pd, atom)
    #                 next_program = program.copy()
    #                 next_program_str = program_str.copy()
    #                 for new_atom in atom_ls:
    #                     next_program = next_program + [self.PPO.atom_to_vector(new_atom)]
    #                     # atom["num_op"] = atom_op
                        
                        
    #                     next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
    #                     program_atom_ls.append(new_atom)

    #                 #apply new atom
    #                 next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(atom_ls, all_other_pats)
    #                 # next_program = program.copy() + [self.PPO.atom_to_vector(atom)]
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
    #             # desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
    #             desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
    #             iterator.set_description(desc)
    #     if self.is_log:
    #         self.logger.log(level=logging.DEBUG, msg = desc)
            
    #     self.PPO.actor.train()
    #     self.PPO.critic.train()
    
    def test_epoch(self, test_loader, epoch, exp_y_pred_arr = None):
        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        y_pred_prob_ls=[]
        self.PPO.actor.eval()
        self.PPO.critic.eval()
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
                    # features, X_pd, program, train,col, op
                    atom_ls = self.PPO.predict_atom(features=X, X_pd_ls=X_pd_full, program=program, train=True, col=col, op=op)
                    
                    next_program = program.copy()
                    
                    curr_vec_ls = self.PPO.atom_to_vector_ls0(atom_ls, col, op)

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
          
        self.PPO.actor.train()
        self.PPO.critic.train()
    def run(self):

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)

        self.test_epoch(test_loader, 0)
        for i in range(1, self.epochs + 1):
            self.train_epoch(train_loader, i)
            self.test_epoch(test_loader, i)




if __name__ == "__main__":
    args = parse_args()
    seed = 0
    train_data_path = "synthetic_dataset.pd"#"simple_df"
    test_data_path = "synthetic_test_dataset.pd"#"simple_df"
    precomputed_path = "synthetic_precomputed.npy"#"ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 0.005
    batch_size = 16
    gamma = 0.95
    epsilon = 0.9
    epsilon_falloff = 0.9
    target_update = 20
    epochs = 100
    # with trainer2
    program_max_len = 4
    # with trainer
    # program_max_len = 2
    patient_max_appts = 1
    provenance = "difftopkproofs"
    latent_size = 20
    is_log = True
    timesteps_per_batch = 10
    n_updates_per_iteration = 2
    clip = 0.2

    if is_log:
        log_path = './ppo_logs/'+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        logging.basicConfig(filename=log_path,
                filemode='a',
                format='%(message)s',
                level=logging.DEBUG)
        logging.info("PPO Explanation Synthesis\n Seed: {}, train_path: {}, test_path: {}, precomputed_path: {}, mem_cap: {}, learning_rate: {}, batch_\
        size: {}, gamma: {}, epsilon: {}, epsilon_falloff: {}, target_update: {}, epochs: {}, prog_max_len: {}, pat_max_appt: {}, latent_size: {}".format(
            seed, train_data_path,test_data_path,precomputed_path,replay_memory_capacity,learning_rate,batch_size,gamma,epsilon,epsilon_falloff,target_update,epochs,
            program_max_len,patient_max_appts,latent_size))

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_dataset = EHRDataset(data=train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    test_dataset = EHRDataset(data = test_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    precomputed = np.load(precomputed_path, allow_pickle=True).item()
    lang = Language(data=train_data, precomputed=precomputed, lang=synthetic_lang)
    feat_range_mappings = obtain_feat_range_mappings(train_dataset)   
    trainer = Trainer3(lang=lang, train_dataset=train_dataset,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, timesteps_per_batch = timesteps_per_batch,
                        n_updates_per_iteration=n_updates_per_iteration, clip=clip, feat_range_mappings=feat_range_mappings
                        ) 
    # trainer = Trainer(lang=lang, train_dataset=train_dataset,
    #                     test_dataset = test_dataset,
    #                     replay_memory_capacity=replay_memory_capacity,
    #                     learning_rate=learning_rate, batch_size=batch_size, 
    #                     gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
    #                     target_update=target_update, epochs=epochs, provenance=provenance,
    #                     program_max_len=program_max_len, patient_max_appts=patient_max_appts,
    #                     latent_size=latent_size, is_log = is_log, timesteps_per_batch = timesteps_per_batch,
    #                     n_updates_per_iteration=n_updates_per_iteration, clip=clip
    #                     )
    
    
    trainer.run()

