from unicodedata import bidirectional
import torch
from torch import nn
from rl_enc_dec.create_language import *
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, num_output_classes),
            nn.Softmax(dim=0),
        )
        self.to(device=device)

    def forward(self, x):
        return self.linear_relu_stack(x)

class SynthesizerNetwork(nn.Module):
    def __init__(self, lang):
        super(SynthesizerNetwork, self).__init__()
        self.lang = lang
        self.grammar_num_to_token = {i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_num = {i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        for decision, options_dict in self.lang.syntax.items():
            start = self.ATOM_VEC_LENGTH
            for option in list(options_dict.keys()):
                self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        num_features = len(self.lang.syntax["num_feat"]) + len(self.lang.syntax["cat_feat"])
        program_vec_size = self.ATOM_VEC_LENGTH
        prog_latent_size = self.ATOM_VEC_LENGTH
        feat_latent_size = num_features
        
        self.prog_lstm = nn.LSTM(program_vec_size, prog_latent_size, bidirectional=True)
        self.feat_lstm = nn.LSTM(num_features, feat_latent_size, bidirectional=True)
        self.prog_hidden_to_latent = nn.Linear(prog_latent_size*2, prog_latent_size)
        self.feat_hidden_to_latent = nn.Linear(feat_latent_size*2, feat_latent_size)
        self.token_nets = {i:TokenNetwork(feat_latent_size + prog_latent_size, len(v)) for i,v in self.lang.syntax.items()}
        
        self.to(device=device)

    def prediction_to_atom(self, pred:dict):
        return {i:self.grammar_num_to_token[i][torch.argmax(v).item()] for i,v in pred.items()}

    def vector_to_atom(self, pred:list):
        atom = {}
        for i,v in enumerate(pred):
            if v == 1:
                decision, option = self.grammar_pos_to_token[i]
                atom[decision] = option
        return atom

    def atom_to_vector(self, pred:dict):
        one_hot_pos = []
        for token, token_val in pred.items():
            one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
        ret = [0]*self.ATOM_VEC_LENGTH
        for i in one_hot_pos:
            ret[i] = 1
        return torch.FloatTensor(ret)


    def mask_grammar_net_pred(self, program, token, token_pred_out):
        if token in ["num_feat", "cat_feat"]:
            start, end = self.one_hot_token_bounds[token]
            for atom in program:
                atom_token_slice = atom[start:end]
                mask =  torch.logical_not(atom_token_slice)
                token_pred_out = token_pred_out * mask.int().float()
        return token_pred_out
        

    def random_forward(self, program) -> dict:
        ret = {}
        queue = ["formula"]
        while queue:
            token = queue.pop()
            pred = torch.norm(torch.rand(len(self.grammar_num_to_token[token])))
            pred = self.mask_grammar_net_pred(program, token, pred)
            pred_val = self.grammar_num_to_token[token][torch.argmax(pred).item()]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = pred
        return ret


    def forward(self, features, program):
        feat_in = torch.cat(features).view(len(features),1,-1)
        feat_out, (h_feat, c_feat) = self.feat_lstm(feat_in)
        h_feat = h_feat.view(-1,)
        feat_latent = self.feat_hidden_to_latent(h_feat)
        program_in = torch.cat(program).view(len(program),1,-1)
        prog_out, (h_prog, c_prog) = self.prog_lstm(program_in)
        h_prog = h_prog.view(-1,)
        prog_latent = self.prog_hidden_to_latent(h_prog)
        latent = torch.cat((feat_latent,prog_latent),dim=-1)
        ret = {}
        queue = ["formula"]
        while queue:
            token = queue.pop(0)
            pred = self.token_nets[token](latent)
            pred = self.mask_grammar_net_pred(program, token, pred)
            pred_val = self.grammar_num_to_token[token][torch.argmax(pred).item()]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = pred
        return ret

if __name__ == "__main__":
    data = data=pd.read_csv("dataset.pd")
    precomputed= np.load("precomputed.npy", allow_pickle=True).item()
    lang = EHRLang(data=data, precomputed=precomputed)
    synth = SynthesizerNetwork(num_features = 200, lang = lang)
    a = synth([torch.randn(1,200,1) for _ in range(5)], [torch.randn(1,synth.ATOM_VEC_LENGTH,1) for _ in range(5)])
    prog = synth.vector_to_atom(a)
    print(prog)