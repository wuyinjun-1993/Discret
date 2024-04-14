from rl_synthesizer import *
import random
from create_language import *
from ehr_lang import *
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import synthetic_lang
import logging
from datetime import datetime

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




class Trainer:
    def __init__(self, lang:Language, train_dataset, test_dataset, replay_memory_capacity, learning_rate, batch_size, gamma, epsilon, epsilon_falloff, epochs, target_update, provenance, program_max_len, patient_max_appts, latent_size, is_log, timesteps_per_batch, n_updates_per_iteration, clip):
        self.PPO = PPO(lang=lang, replay_memory_capacity=replay_memory_capacity, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, provenance=provenance,  program_max_len=program_max_len, patient_max_appts=patient_max_appts,latent_size=latent_size, n_updates_per_iteration=n_updates_per_iteration, clip=clip)
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
        iterator = tqdm(enumerate(self.train_dataset), desc="Training Synthesizer", total=len(self.train_dataset))
        for episode_i, val in iterator:
            ep_rews = []

            (all_other_pats, X_pd, X), y = val
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
                db_cons = self.check_db_constrants(next_all_other_pats, y=int(y.clone().detach()[0])) #entropy
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
                    if y.clone().detach()[0] == y_pred: success += 1
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


if __name__ == "__main__":
    seed = 0
    train_data_path = "synthetic_dataset.pd"#"simple_df"
    test_data_path = "synthetic_test_dataset.pd"#"simple_df"
    precomputed_path = "synthetic_precomputed.npy"#"ehr_precomputed.npy"
    replay_memory_capacity = 5000
    learning_rate = 1e-5
    batch_size = 16
    gamma = 0.95
    epsilon = 0.9
    epsilon_falloff = 0.9
    target_update = 20
    epochs = 100
    program_max_len = 2
    patient_max_appts = 1
    provenance = "difftopkproofs"
    latent_size = 20
    is_log = True
    timesteps_per_batch = 50
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
    trainer = Trainer(lang=lang, train_dataset=train_dataset,
                        test_dataset = test_dataset,
                        replay_memory_capacity=replay_memory_capacity,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        gamma=gamma,epsilon=epsilon, epsilon_falloff=epsilon_falloff,
                        target_update=target_update, epochs=epochs, provenance=provenance,
                        program_max_len=program_max_len, patient_max_appts=patient_max_appts,
                        latent_size=latent_size, is_log = is_log, timesteps_per_batch = timesteps_per_batch,
                        n_updates_per_iteration=n_updates_per_iteration, clip=clip
                        )
    trainer.run()

