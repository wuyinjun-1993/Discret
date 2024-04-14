import rl_synthesizer
import torch
from rl_synthesizer import *
import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import ehr_lang, synthetic_lang

class EHRDataset(Dataset):
    def __init__(self, data, drop_cols,patient_max_appts):
        self.data = data
        self.patient_max_appts = patient_max_appts
        self.drop_cols = drop_cols
        self.patient_ids = sorted(self.data['PAT_ID'].unique())
    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[idx]]
        m = [appts['label'].max()]
        y = torch.FloatTensor(m)
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.FloatTensor([0]*len(X[0]))]*(len(X)-self.patient_max_appts))
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
            y = torch.FloatTensor([appts['label'].max()])
            if y[0] == 1:
                self.deaths_only.append(pat_id)
    def __len__(self):
        return len(self.deaths_only)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[self.deaths_only[idx]]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[self.deaths_only[idx]]]
        y = torch.FloatTensor([appts['label'].max()])
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        #zero pad
        X.extend([torch.FloatTensor([0]*len(X[0]))]*(len(X)-self.patient_max_appts))
        return (all_other_pats, X_pd, X), y




class Trainer:
    def __init__(self, lang:Language, dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, epsilon_update):
        self.dqn = DQN(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts)
        self.epsilon = epsilon
        self.epsilon_falloff = epsilon_falloff
        self.epochs = epochs
        self.dataset = dataset
        self.lang = lang
        self.epsilon_update = epsilon_update
        self.target_update = target_update
        self.program_max_len = program_max_len

    def check_db_constrants(self, data: pd.DataFrame, alpha: float = 0.6, min_same=5, y_train: int = None) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data['label'] == y_train]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        if same < min_same:  # or same/total < alpha:
            return 0
        return same / total

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def train_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.dataset), desc="Training Synthesizer", total=len(self.dataset))
        for episode_i, val in iterator:
            (all_other_pats, X_pd, X), y = val
            program = []
            while True: # episode
                random_action = random.random() < self.epsilon
                if random_action:
                    atom = self.dqn.random_atom(program)
                else:
                    atom = self.dqn.predict_atom(features=X, program=program)
                #apply new atom
                next_all_other_pats = self.lang.evaluate_atom_on_dataset(atom, all_other_pats)
                next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                #check constraints
                x_cons = self.check_x_constraint(X_pd, atom, lang) #is e(r)?
                prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                db_cons = self.check_db_constrants(next_all_other_pats, y_train=int(y[0])) #entropy
                #derive reward
                reward = db_cons if x_cons else 0 # NOTE: these become part of reward
                done = atom["formula"] == "end" or not prog_cons or not x_cons or reward > 0.8 # NOTE: Remove reward check
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
                    if reward > 0.8: success += 1
                    else: failure += 1
                    break
                else:
                    program = next_program
                    all_other_pats = next_all_other_pats

            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            success_rate = (success / (episode_i + 1)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)")
        self.epsilon *= self.epsilon_falloff


    def run(self):
        for i in range(1, self.epochs + 1):
            self.train_epoch(i)


if __name__ == "__main__":
    seed = 0
    data_path = "synthetic_dataset.pd"
    precomputed_path = "synthetic_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 0.0001
    batch_size = 1
    gamma = 0.999
    epsilon = 0.9
    epsilon_falloff = 0.9
    target_update = 20
    epochs = 10
    program_max_len = 1
    patient_max_appts = 1
    epsilon_update = 100
    provenance = "difftopkproofs"

    torch.manual_seed(seed)
    random.seed(0)
    data = pd.read_csv(data_path)
    dataset = EHRDataset(data=data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts)
    precomputed = np.load(precomputed_path, allow_pickle=True).item()
    lang = Language(data=data, precomputed=precomputed, lang=synthetic_lang)
    trainer = Trainer(lang=lang, dataset=dataset, replay_memory_capacity=replay_memory_capacity, 
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        epsilon_update=epsilon_update
                        )
    trainer.run()

