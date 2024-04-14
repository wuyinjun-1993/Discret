from rl_synthesizer_global import *
# from rl_synthesizer_2 import *
# from rl_synthesizer_3 import Transition
import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import synthetic_lang
import logging
from datetime import datetime
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import operator

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
        random.shuffle(self.patient_ids)

    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[idx]]
        m = [appts['label'].max()]
        y = torch.tensor(m, device=DEVICE, dtype=torch.float)
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.tensor(i, device=DEVICE, dtype=torch.float) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.tensor([0]*len(X[0]), device=DEVICE, dtype=torch.float) ]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y

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




# class Trainer:
#     def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, dropout_p):
#         self.dqn = DQN(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p)
#         self.epsilon = epsilon
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
    
#     def train_epoch(self, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
#         for episode_i, val in iterator:
#             (all_other_pats, X_pd, X), y = val
#             program = []
#             program_str = []
#             while True: # episode
#                 atom = self.dqn.predict_atom(features=X, program=program, epsilon=self.epsilon)
#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
#                 next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
#                 #check constraints
#                 x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 db_cons = self.check_db_constrants(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
#                 #derive reward
#                 reward = db_cons if x_cons else 0 # NOTE: these become part of reward
#                 done = atom["formula"] == "end" or not prog_cons or not x_cons # NOTE: Remove reward check
#                 #record transition in buffer
#                 if done:
#                     next_program = None
#                 transition = Transition(X,program, atom, next_program, reward)
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

#     def test_epoch(self, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
#         y_true_ls=[]
#         y_pred_ls=[]
#         for episode_i, val in iterator:
#             (all_other_pats, X_pd, X), y = val
#             program = []
#             program_str = []
#             while True: # episode
#                 atom = self.dqn.predict_atom(features=X, program=program, epsilon=0)
#                 #apply new atom
#                 next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
#                 next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                 next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
#                 #check constraints
#                 x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
#                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                 y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
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
#         self.test_epoch(0)
#         for i in range(1, self.epochs + 1):
#             self.train_epoch(i)
#             self.test_epoch(i)


class Trainer_global:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, global_program_max_len, patient_max_appts, latent_size, is_log, dropout_p, feat_range_mappings):
        self.dqn = DQN_global(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, global_program_max_len=global_program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size,dropout_p=dropout_p, feat_range_mappings=feat_range_mappings)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        self.target_update = target_update
        self.program_max_len = program_max_len
        self.global_program_max_len = global_program_max_len
        self.is_log = is_log
        self.batch_size = batch_size
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
    
    def identify_op(self, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = [atom[k][1], atom[k][2]]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = [atom[k][1], atom[k][2]]#atom[k][1][1]
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

    def sample_a_mini_batch(self, remaining_data, class_idx):
        curr_class_random_data = remaining_data.loc[remaining_data["label"] == class_idx]
        rand_ids = torch.randperm(len(curr_class_random_data))
        
        rand_batch_ids = rand_ids[0:self.batch_size].numpy()
        
        full_rand_batch_data = curr_class_random_data.iloc[rand_batch_ids]
        
        rand_batch_data = full_rand_batch_data.drop(self.train_dataset.drop_cols, axis=1)
        
        return torch.from_numpy(np.array(rand_batch_data)).float()[:,0:2], full_rand_batch_data
        

    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        # iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
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
        # for episode_i, sample_idx in iterator:
        #     # (all_other_pats, X_pd, X), y = val
        #     (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
        class_to_global_program_mappings = dict()
        class_to_global_program_str_mappings = dict()
        class_to_global_program_atom_ls_mappings = dict()
        for class_idx in [0,1]:
            global_program = []
            global_program_str = []
            global_program_atom_ls = []
            remaining_data = self.train_dataset.data
            # all_other_pats = self.train_dataset.data
            while len(global_program) < global_program_max_len: # episode
                program = []
                program_str = []
                # program_atom_ls = []
                global_program.append(program)
                global_program_str.append(program_str)
                global_program_atom_ls.append([])
                
                program_hx = None
                origin_data_size = len(remaining_data)
                # while True:
                while len(global_program_atom_ls[-1]) < self.program_max_len:
                    sampled_features, rand_batch_df = self.sample_a_mini_batch(remaining_data, class_idx)
                    atom = self.dqn.predict_atom(class_idx, sampled_features, program=global_program, epsilon=self.epsilon)
                    atom_ls = self.identify_op(atom)

                    next_global_program = global_program.copy()
                    next_global_program_str = global_program_str.copy()
                    for new_atom in atom_ls:
                        next_global_program[-1] = next_global_program[-1] + [self.dqn.atom_to_vector(class_idx, new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_global_program_str[-1] = next_global_program_str[-1] + [self.lang.atom_to_str(new_atom)]                    
                        
                        global_program_atom_ls[-1].append(new_atom)
                    
                    
                    satisfied_pat_ids = self.lang.evaluate_union_atom_ls_on_dataset(global_program_atom_ls, remaining_data)
                    satisfied_pat_data = remaining_data[remaining_data['PAT_ID'].isin(satisfied_pat_ids)]
                    next_remaining_data = remaining_data[~remaining_data['PAT_ID'].isin(satisfied_pat_ids)]
                    
                    
                    mb_satisfied_pat_ids = self.lang.evaluate_union_atom_ls_on_dataset(global_program_atom_ls, rand_batch_df)
                    
                    if len(satisfied_pat_data) > 0:
                        reward = 1.0*np.sum(np.array(satisfied_pat_data["label"]) ==class_idx)/len(satisfied_pat_data)*np.sum(np.array(satisfied_pat_data["label"]) ==class_idx)/np.sum(np.array(remaining_data["label"]) == class_idx)    
                        # reward = reward*len(mb_satisfied_pat_ids)/len(rand_batch_df)
                        # reward = 1.0*np.sum(np.array(satisfied_pat_data["label"]) ==class_idx)/np.sum(np.array(self.train_dataset.data["label"]) == class_idx) #*np.sum(np.array(satisfied_pat_data["label"]) ==class_idx)/np.sum(np.array(remaining_data["label"]) == class_idx)    
                    else:
                        reward = 0
                    
                    
                    transition = Transition(global_program, sampled_features, atom, next_global_program, reward)
                    # transition = Transition(global_program, next_global_program, reward)
                    self.dqn.observe_transition(class_idx, transition)
                    #update model
                    loss = self.dqn.optimize_model(class_idx)
                    sum_loss += loss
                        
                    global_program = next_global_program
                    global_program_str = next_global_program_str
                    remaining_data = next_remaining_data
                    # global_program_atom_ls = program_atom_ls
                    
                
                curr_all_satisfied_pat_ids = self.lang.evaluate_union_atom_ls_on_dataset(global_program_atom_ls, self.train_dataset.data)
                # remaining_data = self.lang.evaluate_atom_ls_on_dataset_for_remaining_data(global_program_atom_ls, remaining_data)
                remaining_data = self.train_dataset.data[~self.train_dataset.data['PAT_ID'].isin(curr_all_satisfied_pat_ids)]
                
                new_data_size = len(remaining_data)
                
                # if new_data_size >= origin_data_size:
                #     del global_program_atom_ls[-1]
                #     del global_program_str[-1]
                #     del global_program[-1]
                
                if len(remaining_data) <= 0:
                    break
                
                if np.sum(np.array(remaining_data["label"] == class_idx)) <= 0:
                    break
                
                #apply new atom                 
                
                
                
                
                
                # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                #check constraints
                # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                # db_cons = self.check_db_constrants(next_all_other_pats, y=class_idx) #entropy
                # #derive reward
                # reward = db_cons*len(next_all_other_pats)/np.sum(self.train_dataset.data["label"] == class_idx)# if x_cons else 0 # NOTE: these become part of reward
                # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                # #record transition in buffer
                # if done:
                #     next_program = None
                
                #update next step
                # if done: #stopping condition
                #     if reward > 0.5: success += 1
                #     else: failure += 1
                #     break
                # else:
                #     program = next_program
                #     program_str = next_program_str
                #     all_other_pats = next_all_other_pats

                # get remaining data that has not satisfied the existing rules
                # all_other_pats = self.lang.evaluate_atom_ls_on_dataset_for_remaining_data(program_atom_ls, self.train_dataset.data)
                
                # if len(global_program_atom_ls)>3:
                #     break

            class_to_global_program_mappings[class_idx]=global_program
            class_to_global_program_str_mappings[class_idx]=global_program_str
            class_to_global_program_atom_ls_mappings[class_idx]=global_program_atom_ls
            # Print information
            # success_rate = (success / (episode_i + 1)) * 100.0
            # avg_loss = sum_loss/(episode_i+1)
            # desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)"
            # iterator.set_description(desc)
        # if self.is_log:
        #     self.logger.log(level=logging.DEBUG, msg = desc)
        
        self.class_to_global_program_mappings = class_to_global_program_mappings
        self.class_to_global_program_str_mappings = class_to_global_program_str_mappings
        self.class_to_global_program_atom_ls_mappings = class_to_global_program_atom_ls_mappings
        return sum_loss

    def test_epoch(self, dataset):
        success, failure, sum_loss = 0, 0, 0.
        # iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        # for episode_i, val in iterator:
        # iterator = tqdm(enumerate(range(len(self.train_dataset))), desc="Training Synthesizer", total=len(self.train_dataset))
        
        all_rand_ids = torch.randperm(len(self.train_dataset))
        
        dataset.data["pred_labels"] = np.ones(len(dataset.data))*(-1)
        for class_idx in [0,1]:
            

            global_program = self.class_to_global_program_mappings[class_idx]
            global_program_str = self.class_to_global_program_str_mappings[class_idx]
            global_program_atom_ls = self.class_to_global_program_atom_ls_mappings[class_idx]
            matched_pat_ids = self.lang.evaluate_union_atom_ls_on_dataset(global_program_atom_ls, dataset.data)

            dataset.data.loc[dataset.data['PAT_ID'].isin(matched_pat_ids), "pred_labels"] = class_idx

        y_pred_ls = np.array(dataset.data["pred_labels"]).astype(float)
        y_true_ls = np.array(dataset.data["label"]).astype(float)

        y_pred_ls[y_pred_ls == -1] = 0.5

        auc_score = roc_auc_score(y_true_ls, y_pred_ls)

        accuracy = np.mean(y_true_ls==y_pred_ls)

        return auc_score, accuracy
            
            

    # def test_epoch(self, epoch):
    #     success, failure, sum_loss = 0, 0, 0.
    #     iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
    #     y_true_ls=[]
    #     y_pred_ls=[]
    #     self.dqn.policy_net.eval()
    #     self.dqn.target_net.eval()
    #     with torch.no_grad():
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
    #                 x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
    #                 prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
    #                 y_pred = self.get_test_decision_from_db(next_all_other_pats) if x_cons else -1
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
    #     if self.is_log:
    #         self.logger.log(level=logging.DEBUG, msg = desc)
    def run(self):
        # self.test_epoch(self.test_dataset)

        iterator = tqdm(enumerate(range(self.epochs)), desc="Training Synthesizer", total=self.epochs)

        for i, epoch in iterator:
            train_loss = self.train_epoch(epoch)
            train_acc, train_accuracy = self.test_epoch(self.train_dataset)
            test_acc, test_accuracy = self.test_epoch(self.test_dataset)

            desc = f"[Test Epoch {epoch}] Avg Loss: {train_loss}, train auc score: {train_acc}, train accuracy:{train_accuracy}, test auc score: {test_acc}, test accuracy: {test_accuracy}"
            # Update the target net
            if epoch % self.target_update == 0:
                self.dqn.update_target()
                self.epsilon *= self.epsilon_falloff
            iterator.set_description(desc)

def obtain_feat_range_mappings(train_dataset):
    cln_names = list(train_dataset.data.columns)
    feat_range_mappings = dict()
    for cln in cln_names:
        max_val = train_dataset.data[cln].max()
        min_val = train_dataset.data[cln].min()
        feat_range_mappings[cln] = [min_val, max_val]

    return feat_range_mappings

if __name__ == "__main__":
    seed = 0
    train_data_path = "synthetic_dataset.pd"#"simple_df"
    test_data_path = "synthetic_test_dataset.pd"#"simple_df"
    precomputed_path = "synthetic_precomputed.npy"#"ehr_precomputed.npy"
    # train_data_path = "simple_df"
    # test_data_path = "simple_df"
    # precomputed_path = "ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 0.0005
    batch_size = 16
    gamma = 0.999
    epsilon = 0.2
    epsilon_falloff = 0.9
    target_update = 10
    epochs = 1000
    program_max_len = 4
    global_program_max_len = 4
    patient_max_appts = 1
    provenance = "difftopkproofs"
    latent_size = 40
    is_log = True
    dropout_p = 0.1

    if is_log:
        log_path = './logs/'+datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt'
        logging.basicConfig(filename=log_path,
                filemode='a',
                format='%(message)s',
                level=logging.DEBUG)
        logging.info("EHR Explanation Synthesis\n Seed: {}, train_path: {}, test_path: {}, precomputed_path: {}, mem_cap: {}, learning_rate: {}, batch_\
        size: {}, gamma: {}, epsilon: {}, epsilon_falloff: {}, target_update: {}, epochs: {}, prog_max_len: {}, pat_max_appt: {}, latent_size: {}, dropout: {}".format(
            seed, train_data_path,test_data_path,precomputed_path,replay_memory_capacity,learning_rate,batch_size,gamma,epsilon,epsilon_falloff,target_update,epochs,
            program_max_len,patient_max_appts,latent_size, dropout_p))

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
    trainer = Trainer_global(lang=lang, train_dataset=train_dataset,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, global_program_max_len=global_program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings
                        )
    trainer.run()

