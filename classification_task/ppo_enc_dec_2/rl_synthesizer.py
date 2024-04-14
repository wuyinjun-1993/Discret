import torch
import logging
from torch import nn, optim
from create_language import *
import numpy as np
import random
from functools import reduce
from collections import namedtuple, deque
DEVICE = "cpu"
print(f"Using {DEVICE} device")

Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_output_classes),
            nn.Softmax(dim=0),
        )
        self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size=input_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size, feat_max_size, prog_max_size, dropout_p=0):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feat_max_size=feat_max_size
        self.prog_max_size=prog_max_size
        self.dropout_p = dropout_p

        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.feat_attn = nn.Linear(self.hidden_size * 2, self.feat_max_size)
        self.prog_attn = nn.Linear(self.hidden_size * 2, self.prog_max_size)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, feat_outputs, prog_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        hidden = hidden.view(1,1,-1)
        embedded = self.dropout(embedded)

        feat_attn_weights = nn.functional.softmax(
            self.feat_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        prog_attn_weights = nn.functional.softmax(
            self.prog_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)


        feat_attn_applied = torch.bmm(feat_attn_weights.unsqueeze(0),
                                 feat_outputs.unsqueeze(0))

        prog_attn_applied = torch.bmm(prog_attn_weights.unsqueeze(0),
                                 prog_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], feat_attn_applied[0], prog_attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    def initInput(self):
        return torch.zeros(1, 1, self.input_size, device=DEVICE)

class AttnCriticNN(nn.Module):
    def __init__(self, hidden_size, input_size, feat_max_size, prog_max_size, dropout_p=0):
        super(AttnCriticNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feat_max_size=feat_max_size
        self.prog_max_size=prog_max_size
        self.dropout_p = dropout_p

        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.feat_attn = nn.Linear(self.hidden_size * 2, self.feat_max_size)
        self.prog_attn = nn.Linear(self.hidden_size * 2, self.prog_max_size)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size*2, 1)

    def forward(self, input, hidden, feat_outputs, prog_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        hidden = hidden.view(1,1,-1)
        embedded = self.dropout(embedded)

        feat_attn_weights = nn.functional.softmax(
            self.feat_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        prog_attn_weights = nn.functional.softmax(
            self.prog_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)


        feat_attn_applied = torch.bmm(feat_attn_weights.unsqueeze(0),
                                 feat_outputs.unsqueeze(0))

        prog_attn_applied = torch.bmm(prog_attn_weights.unsqueeze(0),
                                 prog_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], feat_attn_applied[0], prog_attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = nn.functional.relu(output)

        out_and_hidden  = torch.cat((output,hidden)).view(-1)
        ret = self.out(out_and_hidden)

        return ret

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    def initInput(self):
        return torch.zeros(1, 1, self.input_size, device=DEVICE)



class Actor(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size):
        super(Actor, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        self.patient_max_appts=patient_max_appts
        self.grammar_num_to_token_val = {i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
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


        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len)

        self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})

        
        self.to(device=DEVICE)

    def idx_to_atom(self, idx: dict):
        return {i:self.grammar_num_to_token_val[i][v] for i,v in idx.items()}
    
    def idx_to_atom(self, idx:dict):
        return {i:self.grammar_num_to_token_val[i][v] for i,v in idx.items()}

    def vector_to_atom(self, pred:list):
        atom = {}
        for i,v in enumerate(pred):
            if v == 1:
                decision, option = self.grammar_pos_to_token[i]
                atom[decision] = option
        return atom

    def atom_to_vector(self, atom:dict):
        one_hot_pos = []
        for token, token_val in atom.items():
            one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
        ret = [0]*self.ATOM_VEC_LENGTH
        for i in one_hot_pos:
            ret[i] = 1
        return torch.FloatTensor(ret)

    def mask_grammar_net_pred(self, program, token, token_pred_out):
        # if not any(program[0]) and token == "formula":
        #     end_index = self.grammar_token_to_pos[(token, "end")]
        #     #TODO
        if token in ["num_feat", "cat_feat"]:
            start, end = self.one_hot_token_bounds[token]
            for atom in program:
                mask =  torch.logical_not(atom[start:end])
                token_pred_out = token_pred_out * mask.int().float()
        return token_pred_out
        


    def forward(self, features, program, queue, train):
        hx = self.feat_gru.initHidden()
        feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        for ei in range(len(features)):
            feat_out, hx = self.feat_gru(features[ei], hx)
            feat_encoder_outputs[ei] = feat_out[0,0]
        hx = hx.view(1,1,-1)
        prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        for ei in range(len(program)):
            prog_out, hx = self.prog_gru(program[ei], hx)
            prog_encoder_outputs[ei] = prog_out[0,0]

        ret = {}
        ret_preds = {}
        decoder_input = self.decoder.initInput()
        while queue:
            token = queue.pop(0)
            if token in ret:
                continue
            decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
            decoder_output = decoder_output.view(-1)
            pred = self.token_nets[token](decoder_output)
            pred = self.mask_grammar_net_pred(program, token, pred)
            if train:
                dist = torch.distributions.Categorical(pred)
                action = int(dist.sample())
            else:
                action = torch.argmax(pred).item()
            pred_val = self.grammar_num_to_token_val[token][action]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = action
            ret_preds[token] = pred
            decoder_input = self.atom_to_vector(self.idx_to_atom(ret))
        return ret, ret_preds

class Critic(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size):
        super(Critic, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        self.patient_max_appts=patient_max_appts
        self.grammar_num_to_token_val = {i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
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


        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        self.decoder = AttnCriticNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len)
        



    def forward(self, features, program):
        hx = self.feat_gru.initHidden()
        feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        for ei in range(len(features)):
            feat_out, hx = self.feat_gru(features[ei], hx)
            feat_encoder_outputs[ei] = feat_out[0,0]
        hx = hx.view(1,1,-1)
        prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        for ei in range(len(program)):
            prog_out, hx = self.prog_gru(program[ei], hx)
            prog_encoder_outputs[ei] = prog_out[0,0]

        decoder_input = self.decoder.initInput()
        ret = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
        return ret



class PPO:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, n_updates_per_iteration, clip):
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip = clip

        self.n_updates_per_iteration = n_updates_per_iteration

        self.actor = Actor(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size)
        self.critic = Critic(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size)
    
        self.actor_optimizer = optim.Adam(self.actor.parameters(), learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), learning_rate)

        self.first_prog_embed = torch.FloatTensor([0]*self.actor.ATOM_VEC_LENGTH)

    def flatten_probs(self, probs: dict):
        return reduce((lambda x,y: x*y), list(probs.values()))
    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, requires_grad=True)
        return batch_rtgs
    
    def learn(self, batch):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = batch
        loss = 0
        batch_probs_num = torch.tensor([self.flatten_probs(i) for i in batch_log_probs], dtype = torch.float, requires_grad=True)
        V, _ = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)
        A_k = batch_rtgs - V.clone().detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        for _ in range(self.n_updates_per_iteration):
            _, curr_log_probs = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)

            curr_probs_num = torch.tensor([self.flatten_probs(i) for i in curr_log_probs], dtype = torch.float, requires_grad=True)

            ratios = torch.exp(curr_probs_num - batch_probs_num) #is exp needed?

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            actor_loss = nn.MSELoss()(torch.zeros(size=actor_loss.shape), actor_loss)
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optim.step()

            loss += abs(critic_loss.clone().detach()) + abs(actor_loss.clone().detach())


        return loss


    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.actor.atom_to_vector(atom)

    def vector_to_atom(self, vec):
        return self.actor.vector_to_atom(vec)
    
    def idx_to_atom(self, idx):
        return self.actor.idx_to_atom(idx)

    #turns network Grammar Networks idx and turns them into an atom
    def idx_to_atom(self, idx:dict):
        return self.actor.idx_to_atom(idx)

    def idx_to_logs(self, pred:dict, idx:dict):
        logs =  {}
        for token,index in idx.items():
            p = pred[token]
            dist = torch.distributions.Categorical(p)
            logs[token] = dist.log_prob(torch.tensor(index))
        return logs


    def predict_atom(self, features, program, train):
        if len(program) == 0:
            program = [self.first_prog_embed]
        atom_idx, atom_preds = self.actor(features, program, ["formula"], train = train)
        return atom_idx, atom_preds

    def evaluate(self, batch_obs, batch_acts):
        V = torch.tensor([self.critic(f, p) for f,p in batch_obs], dtype = torch.float, requires_grad=True).squeeze()
        batch_eval_probs = []
        for obs, act in zip(batch_obs, batch_acts):
            _, preds = self.actor(*obs, list(act.keys()), train=False)
            atom_probs = self.idx_to_logs(preds, act)
            batch_eval_probs.append(atom_probs)
        return V, batch_eval_probs
    

    

