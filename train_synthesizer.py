from cgi import test
from sre_constants import SUCCESS
import pandas as pd
from neural_synthesizer import *
from rl_enc_dec.ehr_lang import *
from rl_enc_dec.create_language import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

def check_db_constrants(data:pd.DataFrame, alpha:float = 0.6, min_same = 5, y_train: int = None) -> float:
    if len(data) == 0:
        return 0
    same = data.loc[data['label'] == y_train]["PAT_ID"].nunique()
    total = data['PAT_ID'].nunique()
    if same < min_same: # or same/total < alpha:
        return 0
    return same/total

def check_x_constraint(X:pd.DataFrame, atom:dict, lang:EHRLang) -> bool:
    return lang.evaluate_atom_on_sample(atom, X)

def check_program_constraint(prog: list, max_len = 10) -> bool:
    return len(prog) < max_len

class EHRDataset(Dataset):
    def __init__(self, data, drop_cols):
        self.data = data
        self.drop_cols = drop_cols
        self.patient_ids = sorted(self.data['PAT_ID'].unique())
    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        appts = self.data.loc[self.data['PAT_ID'] == self.patient_ids[idx]]
        all_other_pats = self.data.loc[self.data['PAT_ID'] != self.patient_ids[idx]]
        y = torch.FloatTensor([appts['label'].max()])
        X_pd = appts.drop(self.drop_cols, axis=1)
        X = [torch.FloatTensor(i) for i in X_pd.to_numpy(dtype=np.float64)]
        return (all_other_pats, X_pd, X), y

def train_loop(dataset, synthesizer:SynthesizerNetwork, loss_fn, optimizer, lang: EHRLang):
    success_progs = []
    for _, val in tqdm(enumerate(dataset), desc="Training Synthesizer", total=len(dataset)):
        (all_other_appts, X_pd, X), y = val
        program = []
        program_interpretable = []
        while True:
            if len(program) == 0:
                pred = synthesizer(X, [torch.FloatTensor([0]*synthesizer.ATOM_VEC_LENGTH)])
            else:
                program.sort(key = lambda a: a[1])
                pred = synthesizer(X, program)
            atom = synthesizer.prediction_to_atom(pred=pred)
            program_interpretable.append(atom)
            all_other_appts = lang.evaluate_atom_on_dataset(atom, all_other_appts)
            program.append(synthesizer.atom_to_vector(atom))
            #reward constraints
            x_cons = check_x_constraint(X_pd, atom, lang)
            prog_cons = check_program_constraint(program)
            db_cons = check_db_constrants(all_other_appts, y_train=int(y[0]))
            reward = db_cons if x_cons and prog_cons else 0
            #optim
            losses = []
            for v in pred.values():
                label = torch.zeros_like(v)
                if reward:
                    label[torch.argmax(v)] == reward
                losses.append(loss_fn(v, label))
            loss_sum = sum(losses)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if reward > 0.8:
                success_progs.append([program_interpretable, X_pd])
            if atom["formula"] == "end" or not prog_cons or not x_cons:
                break
    print(len(success_progs))
    np.save("successes", success_progs, allow_pickle=True)


def test_loop(dataloader, model, loss_fn):
    pass

if __name__ == "__main__":
    learning_rate = 1e-3
    datapath = "dataset.pd"
    epochs = 5
    data=pd.read_csv(datapath)
    dataset = EHRDataset(data=data, drop_cols=DROP_FEATS)
    precomputed= np.load("precomputed.npy", allow_pickle=True).item()
    lang = EHRLang(data=data, precomputed=precomputed)
    synthesizer = SynthesizerNetwork(lang=lang)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(synthesizer.parameters(), lr=learning_rate)
    for _ in range(epochs):
        train_loop(dataset, synthesizer, loss_fn, optimizer, lang)