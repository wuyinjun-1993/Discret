import torch
import logging
from torch import nn, optim
from create_language import *
import numpy as np
import random
from collections import namedtuple, deque
DEVICE = "cuda"
print(f"Using {DEVICE} device")

Transition = namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))
discretize_feat_value_count=10
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


class TokenNetwork_regression(nn.Module):
    def __init__(self, input_size):
        super(TokenNetwork_regression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 2)
        )
        self.to(device=DEVICE)

    def forward(self, x):
        return torch.clamp(self.linear_relu_stack(x), min=0,max=1)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size=input_size
        self.gru = nn.GRU(input_size, hidden_size)
        # self.gru = nn.Linear(input_size, hidden_size)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        # output = self.gru(input)
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
        self.feat_attn = nn.Linear(self.hidden_size, self.feat_max_size)
        self.prog_attn = nn.Linear(self.hidden_size, self.prog_max_size)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    # def forward(self, input, hidden, feat_outputs, prog_outputs):
    #     embedded = self.embedding(input).view(1, 1, -1)
    #     hidden = hidden.view(1,1,-1)
    #     embedded = self.dropout(embedded)

    #     feat_attn_weights = nn.functional.softmax(
    #         self.feat_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

    #     prog_attn_weights = nn.functional.softmax(
    #         self.prog_attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)


    #     feat_attn_applied = torch.bmm(feat_attn_weights.unsqueeze(0),
    #                              feat_outputs.unsqueeze(0))

    #     prog_attn_applied = torch.bmm(prog_attn_weights.unsqueeze(0),
    #                              prog_outputs.unsqueeze(0))

    #     output = torch.cat((embedded[0], feat_attn_applied[0], prog_attn_applied[0]), 1)
    #     output = self.attn_combine(output).unsqueeze(0)

    #     # output = nn.functional.relu(output)
    #     # output, hidden = self.gru(output, hidden)

    #     return output, hidden

    def forward(self, input, feat_outputs, prog_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        # hidden = hidden.view(1,1,-1)
        # embedded = self.dropout(embedded)
        embedded = embedded.view(1,1,-1)
        feat_attn_weights = nn.functional.softmax(
            self.feat_attn(embedded[0]), dim=1)

        prog_attn_weights = nn.functional.softmax(
            self.prog_attn(embedded[0]), dim=1)


        feat_attn_applied = torch.bmm(feat_attn_weights.unsqueeze(0),
                                 feat_outputs.unsqueeze(0))

        prog_attn_applied = torch.bmm(prog_attn_weights.unsqueeze(0),
                                 prog_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], feat_attn_applied[0], prog_attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        # output = nn.functional.relu(output)
        # output, hidden = self.gru(output, hidden)

        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    def initInput(self):
        return torch.zeros(1, 1, self.input_size, device=DEVICE)



class RLSynthesizerNetwork3(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size, feat_range_mappings):
        super(RLSynthesizerNetwork3, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        self.patient_max_appts=patient_max_appts
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        
        
        for i,v in self.lang.syntax.items():
            if not i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            else:
                self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))
        
        
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        # for decision, options_dict in self.lang.syntax.items():
        #     if decision == "num_op":
        #         continue
        #     start = self.ATOM_VEC_LENGTH
        #     for option in list(options_dict.keys()):
        #         if not decision in self.lang.syntax["num_feat"]:
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #         else:
        #             self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #             self.ATOM_VEC_LENGTH += discretize_feat_value_count
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        for decision, options_dict in self.lang.syntax.items():
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not decision in self.lang.syntax["num_feat"]:
                for option in list(options_dict.keys()):        
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:
                self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        self.feat_range_mappings = feat_range_mappings
        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items() if i in self.lang.syntax["num_feat"]})

        net_maps = {}
        for i,v in self.lang.syntax.items():
            if i == "num_op":
                continue
            if not i in self.lang.syntax["num_feat"]:
                net_maps[i] = TokenNetwork(latent_size, len(v))
            else:
                net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
                net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        
        self.to(device=DEVICE)

    def prediction_to_atom(self, pred:dict):
        res = {}
        for i,v in pred.items():
            if not i in self.lang.syntax["num_feat"]:
                res[i] = self.grammar_num_to_token_val[i][torch.argmax(v).item()]
            else:
                res[i] = [[v[0][0].item(), v[0][1].item()],v[1]]
        # return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
        return res

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
            if token.endswith("_prob"):
                continue

            if not token in self.lang.syntax["num_feat"]:
                one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
            else:
                # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
        ret = [0.0]*self.ATOM_VEC_LENGTH
        for i in one_hot_pos:
            if type(i) is tuple:
                ret[i[0]] = i[1]
                # ret[i[0] + 1] = i[1][1]
            else:
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
            token = queue.pop(0)
            pred = torch.rand(len(self.grammar_num_to_token_val[token]))
            pred = torch.nn.functional.softmax(pred, dim=-1)
            pred = self.mask_grammar_net_pred(program, token, pred)
            pred_val = self.grammar_num_to_token_val[token][torch.argmax(pred).item()]
            queue.extend(self.lang.syntax[token][pred_val])
            ret[token] = pred
        return ret


    def forward(self, features, X_pd, program, queue, epsilon):
        hx = self.feat_gru.initHidden()
        feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        for ei in range(len(features)):
            feat_out, hx = self.feat_gru(features[ei], hx)
            # feat_out = self.feat_gru(features[ei])
            feat_encoder_outputs[ei] = feat_out[0,0]
        hx = hx.view(1,1,-1)
        prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        for ei in range(len(program)):
            prog_out, hx = self.prog_gru(program[ei], hx)
            # prog_out = self.prog_gru(program[ei])
            prog_encoder_outputs[ei] = prog_out[0,0]

        ret = {}
        decoder_input = self.decoder.initInput()
        while queue:
            token = queue.pop(0)
            if token in ret:
                continue
            if token == "num_op":
                continue
            decoder_output = self.decoder(decoder_input, feat_encoder_outputs, prog_encoder_outputs)
            decoder_output = decoder_output.view(-1)
            if token in self.lang.syntax["num_feat"]:
                # if np.random.rand() < epsilon:
                #     sub_keys=["_ub", "_lb"]
                # else:
                sub_keys=["_lb", "_ub"]
                pred_ls = []
                for key in sub_keys:
                    if np.random.rand() < epsilon:
                        pred = torch.rand(len(self.grammar_num_to_token_val[token]))
                    else:
                        pred = self.token_nets[token+key](decoder_output.view(-1))    
                    pred = self.mask_grammar_net_pred(program, token, pred)
                    pred_ls.append(pred)
            else:
                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[token]))
                else:
                    pred = self.token_nets[token](decoder_output.view(-1))
                pred = self.mask_grammar_net_pred(program, token, pred)
            if not token in self.lang.syntax["num_feat"]:
                argmax = torch.argmax(pred).item()
                pred_val = self.grammar_num_to_token_val[token][argmax]
                # if not token == "num_feat":
                queue.extend(self.lang.syntax[token][pred_val])
                ret[token] = pred
            else:

                range_max = self.feat_range_mappings[token][1]
                range_min = self.feat_range_mappings[token][0]

                feat_val = list(X_pd[token])[0]

                argmax = torch.argmax(pred_ls[0]).item()


                pred_lb = (feat_val - range_min)*argmax/discretize_feat_value_count + range_min

                argmax = torch.argmax(pred_ls[1]).item()

                pred_ub = (range_max - feat_val)*argmax/discretize_feat_value_count + feat_val

                ret[token] = [[pred_lb, pred_ub],pred_ls]

                break
                # ret[token]["max"] = pred_ub
            # else:
            #     print()
            
            decoder_input = self.atom_to_vector(self.prediction_to_atom(ret))
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

class DQN3:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, feat_range_mappings):
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang

        self.policy_net = RLSynthesizerNetwork3(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, feat_range_mappings=feat_range_mappings)
        self.target_net = RLSynthesizerNetwork3(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, feat_range_mappings=feat_range_mappings)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(replay_memory_capacity)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), learning_rate)

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

    def predict_atom(self, features, X_pd, program, epsilon):
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], ["formula"], epsilon)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, ["formula"], epsilon)
        return self.policy_net.prediction_to_atom(pred)
    
    #predicts the best atom to add to the program from the given next state, and provides the maximal tensors which produced that decision
    #uses target net!!!
    def predict_next_state_with_tensor_info(self, features, data, program):
        if len(program) == 0:
            pred = self.target_net(features, data, [self.first_prog_embed], ["formula"], 0)
        else:
            #program.sort()
            pred = self.target_net(features, data, program, ["formula"], 0)
        max_tensors = dict()
        for token, token_val in pred.items():
            if not token in self.lang.syntax["num_feat"]:
                max_tensors[token] = torch.max(token_val).reshape((1,1))
            else:
                max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
        
        # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        
        return self.target_net.prediction_to_atom(pred), max_tensors

    #takes a state,action (where action is an atom) pair and returns prediction tensors which are generated when picking the same tokens from the given atom
    def get_state_action_prediction_tensors(self, features, X_pd, program, atom):
        queue = list(atom.keys())
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], queue, 0)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, queue, 0)

        tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        for token, token_val in atom.items():
            if token == "num_op" or token.endswith("_prob"):
                continue

            if token not in self.lang.syntax["num_feat"]:
                # if not token.endswith("_prob"):
                    tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
            else:
                tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
            # else:
            #     tensor_indeces[token] = 0
        atom_prediction_tensors = {}
        for token, tensor_idx in tensor_indeces.items():
            if token not in self.lang.syntax["num_feat"]:
                atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
            else:
                atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
        # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        return atom_prediction_tensors
    
    #takes an atom, and the maximal tensors used to produce it, and returns a Q value
    def get_atom_Q_value(self, atom:dict, atom_prediction_tensors: dict):
        op=1
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.FloatTensor([[1]])
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                # op = atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                # op = atom_prediction_tensors["cat_op"]
            constant = atom_prediction_tensors[feat_name]
        Q = formula*feat*op*constant[0]*constant[1]
        return Q[0]

    
    def observe_transition(self, transition: Transition):
        self.memory.push(transition)

 
    def optimize_model(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.FloatTensor([[transition.reward] for transition in batch])

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 1])
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values[non_final_mask], expected_state_action_values[non_final_mask])#*sum(count_ls)/count_ls[label.long().item()]
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Return loss
        return loss.detach()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


