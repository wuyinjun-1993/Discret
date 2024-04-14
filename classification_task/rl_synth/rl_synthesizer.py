import torch
from torch import nn, optim
from rl_synth.create_language import *
import numpy as np
import random
from collections import namedtuple, deque
import scallopy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        #TODO: adjustable latent size and layers
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, num_output_classes),
            nn.Softmax(dim=0),
        )
        self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

class RLSynthesizerNetwork(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts):
        super(RLSynthesizerNetwork, self).__init__()
        self.lang = lang
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
        feat_latent_size = num_features
        program_vec_size = self.ATOM_VEC_LENGTH
        prog_latent_size = self.ATOM_VEC_LENGTH
        
        self.prog_lstm = nn.LSTM(input_size=program_vec_size, hidden_size=prog_latent_size, bidirectional=True)
        self.feat_lstm = nn.LSTM(input_size=num_features, hidden_size=feat_latent_size, bidirectional=True)
        self.prog_hidden_to_latent = nn.Linear(prog_latent_size*2, prog_latent_size)
        self.feat_hidden_to_latent = nn.Linear(feat_latent_size*2, feat_latent_size)
        self.token_nets = nn.ModuleDict({i:TokenNetwork(feat_latent_size + prog_latent_size, len(v)) for i,v in self.lang.syntax.items()})

        
        self.to(device=DEVICE)

    def prediction_to_atom(self, pred:dict):
        return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}

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
        

    def random_atom(self, program) -> dict:
        ret = {}
        queue = ["formula"]
        while queue:
            token = queue.pop()
            pred = torch.rand(len(self.grammar_num_to_token_val[token]))
            pred = torch.nn.functional.softmax(pred, dim=-1)
            pred = self.mask_grammar_net_pred(program, token, pred)
            pred_val = self.grammar_num_to_token_val[token][torch.argmax(pred).item()]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = pred
        return ret


    def forward(self, features, program, queue):
        feat_in = torch.cat(features).view(len(features), 1, -1)
        feat_out, (h_feat, c_feat) = self.feat_lstm(feat_in)
        h_feat = h_feat.view(-1,)
        feat_latent = self.feat_hidden_to_latent(h_feat)
        program_in = torch.cat(program).view(len(program), 1, -1)
        prog_out, (h_prog, c_prog) = self.prog_lstm(program_in)
        h_prog = h_prog.view(-1,)
        prog_latent = self.prog_hidden_to_latent(h_prog)
        latent = torch.cat((feat_latent,prog_latent),dim=-1)
        ret = {}
        while queue:
            token = queue.pop(0)
            if token in ret:
                continue
            pred = self.token_nets[token](latent)
            pred = self.mask_grammar_net_pred(program, token, pred)
            argmax = torch.argmax(pred).item()
            pred_val = self.grammar_num_to_token_val[token][argmax]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = pred
        return ret


class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

class DQN:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts):
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = RLSynthesizerNetwork(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts)
        self.target_net = RLSynthesizerNetwork(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(replay_memory_capacity)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), learning_rate)

        self.first_prog_embed = torch.FloatTensor([0]*self.policy_net.ATOM_VEC_LENGTH)#torch.randn(self.policy_net.ATOM_VEC_LENGTH, requires_grad=True)

    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.policy_net.atom_to_vector(atom)

    def vector_to_atom(self, vec):
        return self.policy_net.vector_to_atom(vec)

    #turns network Grammar Networks predictions and turns them into an atom
    def prediction_to_atom(self, pred:dict):
        return self.policy_net.prediction_to_atom(pred)

    def random_atom(self, program):
        #TODO
        if len(program) == 0:
            pred = self.policy_net.random_atom(program = [torch.FloatTensor([0]*self.policy_net.ATOM_VEC_LENGTH)])
        else:
            pred = self.policy_net.random_atom(program = program)
        return self.policy_net.prediction_to_atom(pred)

    def predict_atom(self, features, program):
        if len(program) == 0:
            pred = self.policy_net(features, [self.first_prog_embed], ["formula"])
        else:
            #program.sort()
            pred = self.policy_net(features, program, ["formula"])
        return self.policy_net.prediction_to_atom(pred)
    
    #predicts the best atom to add to the program from the given next state, and provides the maximal tensors which produced that decision
    #uses target net!!!
    def predict_next_state_with_tensor_info(self, features, program):
        if len(program) == 0:
            pred = self.target_net(features, [self.first_prog_embed], ["formula"])
        else:
            #program.sort()
            pred = self.target_net(features, program, ["formula"])
        max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items()}
        return self.target_net.prediction_to_atom(pred), max_tensors

    #takes a state,action (where action is an atom) pair and returns prediction tensors which are generated when picking the same tokens from the given atom
    def get_state_action_prediction_tensors(self, features, program, atom):
        queue = list(atom.keys())
        if len(program) == 0:
            pred = self.policy_net(features, [self.first_prog_embed], queue)
        else:
            #program.sort()
            pred = self.policy_net(features, program, queue)

        tensor_indeces = {token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        atom_prediction_tensors = {token:pred[token][tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        return atom_prediction_tensors
    
    #takes an atom, and the maximal tensors used to produce it, and returns a Q value
    def get_atom_Q_value(self, atom:dict, atom_prediction_tensors: dict):
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.FloatTensor([[1]])
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                op = atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                op = atom_prediction_tensors["cat_op"]
            constant = atom_prediction_tensors[feat_name]
        Q = formula*feat*op*constant
        return Q[0]

    
    def observe_transition(self, transition: Transition):
        self.memory.push(transition)

 
    def optimize_model(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.program, transition.action) for transition in batch]
        reward_batch = torch.FloatTensor([[transition.reward] for transition in batch])

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors(f,p,a)) for f,p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 1])
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Return loss
        return loss.detach()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


