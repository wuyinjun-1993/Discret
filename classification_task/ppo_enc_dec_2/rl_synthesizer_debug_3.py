import torch
import logging
from torch import nn, optim
from create_language import *
import numpy as np
import random
from functools import reduce
from collections import namedtuple, deque
import operator



DEVICE = "cpu"


print(f"Using {DEVICE} device")

discretize_feat_value_count=10

Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
# class TokenNetwork(nn.Module):
#     def __init__(self, input_size, num_output_classes):
#         super(TokenNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_size, num_output_classes),
#             nn.Softmax(dim=0),
#         )
#         self.to(device=DEVICE)

#     def forward(self, x):
#         return self.linear_relu_stack(x)

col_key = "col"

pred_prob_key = "pred_probs"
        
pred_v_key = "pred_v"

pred_prob_id = "pred_id"
        
op_key = "op"

class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_output_classes),
            nn.Softmax(dim=-1)
            # nn.ReLU(),
        )
        self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

class TokenNetwork2(nn.Module):
    def __init__(self, input_size, latent_size):
        super(TokenNetwork2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            # nn.Linear(latent_size, int(latent_size)),
            # nn.ReLU(),
            # nn.Linear(int(latent_size), num_output_classes),
            # nn.Softmax(dim=-1),
        )
        self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

class TokenNetwork3(nn.Module):
    def __init__(self, input_size, latent_size):
        super(TokenNetwork3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1),
            # nn.ReLU(),
            # nn.Linear(int(latent_size), num_output_classes),
            # nn.Softmax(dim=-1),
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



class Actor2(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size, feat_range_mappings):
        super(Actor2, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        self.patient_max_appts=patient_max_appts
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        # for decision, options_dict in self.lang.syntax.items():
        #     start = self.ATOM_VEC_LENGTH
        #     for option in list(options_dict.keys()):
        #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        # for decision, options_dict in self.lang.syntax.items():
        #     # if decision == "num_op":
        #     #     continue
        #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
        #         continue
        #     start = self.ATOM_VEC_LENGTH


        #     if not decision in self.lang.syntax["num_feat"]:
        #         for option in list(options_dict.keys()):        
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #     else:
        #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
                continue
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
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1


        self.feat_range_mappings = feat_range_mappings
        for i,v in self.lang.syntax.items():
            if not i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            else:
                self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}

        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH

        self.embedding = TokenNetwork2(full_input_size, latent_size)

        for i,v in self.lang.syntax.items():
            if i == "num_op":
                continue
            # if i in self.lang.syntax["num_feat"]:
            #     continue
            
            # if not i == "num_feat":
            #     # net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
            if not i in self.lang.syntax["num_feat"]:
                net_maps[i] = TokenNetwork(latent_size, len(v))
            else:
                net_maps["constant"] = TokenNetwork(latent_size, discretize_feat_value_count)
        # for i,v in self.lang.syntax.items():
        #     if i == "num_op":
        #         continue
        #     if not i in self.lang.syntax["num_feat"]:
        #         net_maps[i] = TokenNetwork(latent_size, len(v))
        #     else:
        #         net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
        #         net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
        #         # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
        #         # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)

        self.first_prog_embed = torch.tensor([0]*self.ATOM_VEC_LENGTH, device =DEVICE, dtype=torch.float)
        self.to(device=DEVICE)

    # def idx_to_atom(self, idx: dict):
    #     return {i:self.grammar_num_to_token_val[i][v] for i,v in idx.items()}
    
    def idx_to_atom(self, idx:dict):
        res = dict()
        for i,v in idx.items():
            if not (i in self.lang.syntax["num_feat"] or i == "num_op"):
                res[i] = self.grammar_num_to_token_val[i][v]
            else:
                res[i] = v
        return res
        # return {i:self.grammar_num_to_token_val[i][v] for i,v in idx.items()}

    def vector_to_atom(self, pred:list):
        atom = {}
        for i,v in enumerate(pred):
            if v == 1:
                decision, option = self.grammar_pos_to_token[i]
                atom[decision] = option
        return atom

    def atom_to_vector_ls0(self, atom_ls, col, op):
        ret_tensor_ls = []
        pred_v_arr = atom_ls[pred_v_key]
        
        ret_tensor_ls = torch.zeros([len(pred_v_arr), self.ATOM_VEC_LENGTH])
        
        ret_tensor_ls[:,self.grammar_token_to_pos[("num_op", op)]]=1
        
        ret_tensor_ls[:,self.grammar_token_to_pos[col]]=1
        
        ret_tensor_ls[:, self.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr).view(-1)
    
        return ret_tensor_ls

    def atom_to_vector(self, atom:dict):
        one_hot_pos = []
        for token, token_val in atom.items():
            if token.endswith("_prob"):
                continue
            # one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
            if not token in self.lang.syntax["num_feat"]:
                one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
            else:
                # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
        ret = [0]*self.ATOM_VEC_LENGTH
        for i in one_hot_pos:
            if type(i) is tuple:
                ret[i[0]] = i[1]
                # ret[i[0] + 1] = i[1][1]
            else:
                ret[i] = 1
            # ret[i] = 1
        return torch.FloatTensor(ret).to(DEVICE)

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
        


    def forward2(self, X_pd, features, program, queue, train, eval=False):
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
        if not eval:
            while queue:
                token = queue.pop(0)
                if token in ret:
                    continue

                if token == "num_op":
                    continue
                
                decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                decoder_output = decoder_output.view(-1)

                
                if token in self.lang.syntax["num_feat"]:
                    # if np.random.rand() < epsilon:
                    #     sub_keys=["_ub", "_lb"]
                    # else:
                    sub_keys=["_lb", "_ub"]
                    pred_ls = []
                    for key in sub_keys:
                        pred = self.token_nets[token+key](decoder_output.view(-1))    
                        pred = self.mask_grammar_net_pred(program, token, pred)
                        pred_ls.append(pred)
                else:
                    pred = self.token_nets[token](decoder_output.view(-1))
                    pred = self.mask_grammar_net_pred(program, token, pred)

                if not token in self.lang.syntax["num_feat"]:
                    if train:
                        dist = torch.distributions.Categorical(pred)
                        action = int(dist.sample())
                    else:
                        action = torch.argmax(pred).item()
                else:
                    range_max = self.feat_range_mappings[token][1]
                    range_min = self.feat_range_mappings[token][0]

                    feat_val = list(X_pd[token])[0]

                    if train:
                        dist = torch.distributions.Categorical(pred_ls[0])
                        argmax1 = int(dist.sample())
                        # argmax = torch.argmax(pred_ls[0]).item()
                    else:
                        argmax1 = torch.argmax(pred_ls[0]).item()


                    pred_lb = (feat_val - range_min)*argmax1/(discretize_feat_value_count-1) + range_min

                    if train:
                        dist = torch.distributions.Categorical(pred_ls[1])
                        argmax2 = int(dist.sample())
                        # argmax = torch.argmax(pred_ls[1]).item()
                    else:
                        argmax2 = torch.argmax(pred_ls[1]).item()

                    pred_ub = (range_max - feat_val)*argmax2/(discretize_feat_value_count-1) + feat_val

                    ret[token] = [[pred_lb, pred_ub],[argmax1, argmax2]]
                    ret_preds[token] = pred_ls

                    break

                pred_val = self.grammar_num_to_token_val[token][action]
                queue.extend(self.lang.syntax[token][pred_val])
                ret[token] = action
                ret_preds[token] = pred
                decoder_input = self.atom_to_vector(self.idx_to_atom(ret))
                
        else:
            for token in queue:
                decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                decoder_output = decoder_output.view(-1)
                if token in self.lang.syntax["num_feat"]:
                    # if np.random.rand() < epsilon:
                    #     sub_keys=["_ub", "_lb"]
                    # else:
                    sub_keys=["_lb", "_ub"]
                    pred_ls = []
                    ret_preds[token] = []
                    for idx in range(len(sub_keys)):
                        key = sub_keys[idx]
                        pred = self.token_nets[token+key](decoder_output.view(-1))    
                        # pred = self.mask_grammar_net_pred(program, token, pred)
                        pred_ls.append(pred)
                        ret_preds[token].append(pred)
                    break
                else:
                    pred = self.token_nets[token](decoder_output.view(-1))
                    # act_val_id = self.grammar_token_val_to_num[act_val]
                    ret_preds[token] = pred
                    # pred = self.mask_grammar_net_pred(program, token, pred)
                    ret[token] = queue[token]
                    decoder_input = self.atom_to_vector(self.idx_to_atom(ret))
                
        return ret, ret_preds
    
    def forward(self, features,X_pd_full, program, col, op, train, eval=False, selected_idx = None):
        # features = [f.to(DEVICE) for f in features]
        # hx = self.feat_gru.initHidden()
        # feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        # for ei in range(len(features)):
        #     feat_out, hx = self.feat_gru(features[ei].view(1,1,-1), hx)
        #     feat_encoder_outputs[ei] = feat_out[0,0]
        # feat_encoder_outputs = feat_encoder_outputs.unsqueeze(0)
        # hx = hx.view(1,1,-1)
        # prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        # for ei in range(len(program)):
        #     prog_out, hx = self.prog_gru(program[ei].view(1,1,-1), hx)
        #     prog_encoder_outputs[ei] = prog_out[0,0]
        # prog_encoder_outputs = prog_encoder_outputs.unsqueeze(0)

        features = features.to(DEVICE)
        pat_count = features.shape[0]
        if len(program) == 0:
            program = [self.first_prog_embed]
        
        hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*program[0].shape[-1], device=DEVICE)
        hx[:,0:features[0].shape[0]] = features
        hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = torch.cat(program,dim=-1)
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        
        hx_out = self.embedding(hx)
        # if train:
        #     dist = torch.distributions.Categorical(pred_ls[0])
        #     argmax1 = int(dist.sample())
        # if np.random.rand() < epsilon:
        #     pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        # else:
            # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        pred = self.token_nets["constant"](hx_out.view(pat_count, -1))    
        

            # pred_lb = pred_lb*(range_max - range_min) + range_min

        feat_val = np.array(X_pd_full[col])

        if selected_idx is None:
            if train:
                dist = torch.distributions.Categorical(pred)
                argmax = dist.sample().cpu()
            else:
                argmax = torch.argmax(pred,dim=-1).cpu()
        else:
            argmax = selected_idx

        if op == operator.__ge__:

            pred_v = (feat_val)*(argmax.numpy()/(discretize_feat_value_count-1))
        else:
            pred_v = (1 - feat_val)*(argmax.numpy()/(discretize_feat_value_count-1)) + feat_val
        
        ret[col_key] = col

        if eval:
            ret[pred_prob_key] = pred
        else:
            ret[pred_prob_key] = pred.data.cpu()
        
        ret[pred_v_key] = pred_v
        
        ret[op_key] = op        
        
        ret[pred_prob_id] = argmax
        
        return ret
    
    # def forward(self, X_pd_ls, features, program, queue, train, eval=False):
        
    #     features = features.to(DEVICE)
    #     pat_count = features.shape[0]
    #     X_pd_full = pd.concat(X_pd_ls)
        
    #     # hx = self.feat_gru.initHidden()
    #     # feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
    #     # for ei in range(len(features)):
    #     #     feat_out, hx = self.feat_gru(features[ei], hx)
    #     #     feat_encoder_outputs[ei] = feat_out[0,0]
    #     # hx = hx.view(1,1,-1)
    #     # prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
    #     # for ei in range(len(program)):
    #     #     prog_out, hx = self.prog_gru(program[ei], hx)
    #     #     prog_encoder_outputs[ei] = prog_out[0,0]

    #     ret = {}
    #     ret_preds = {}
    #     decoder_input = self.decoder.initInput()
    #     if not eval:
    #         while queue:
    #             token = queue.pop(0)
    #             if token in ret:
    #                 continue

    #             if token == "num_op":
    #                 continue
                
    #             # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
    #             # decoder_output = decoder_output.view(-1)

                
    #             if token in self.lang.syntax["num_feat"]:
    #                 # if np.random.rand() < epsilon:
    #                 #     sub_keys=["_ub", "_lb"]
    #                 # else:
    #                 sub_keys=["_lb", "_ub"]
    #                 pred_ls = []
    #                 for key in sub_keys:
    #                     pred = self.token_nets[token+key](hx.view(-1))    
    #                     pred = self.mask_grammar_net_pred(program, token, pred)
    #                     pred_ls.append(pred)
    #             else:
    #                 pred = self.token_nets[token](hx.view(-1))
    #                 pred = self.mask_grammar_net_pred(program, token, pred)

    #             if not token in self.lang.syntax["num_feat"]:
    #                 if train:
    #                     dist = torch.distributions.Categorical(pred)
    #                     action = int(dist.sample())
    #                 else:
    #                     action = torch.argmax(pred).item()
    #             else:
    #                 range_max = self.feat_range_mappings[token][1]
    #                 range_min = self.feat_range_mappings[token][0]

    #                 feat_val = list(X_pd[token])[0]

    #                 if train:
    #                     dist = torch.distributions.Categorical(pred_ls[0])
    #                     argmax1 = int(dist.sample())
    #                     # argmax = torch.argmax(pred_ls[0]).item()
    #                 else:
    #                     argmax1 = torch.argmax(pred_ls[0]).item()


    #                 pred_lb = (feat_val - range_min)*argmax1/(discretize_feat_value_count-1) + range_min

    #                 if train:
    #                     dist = torch.distributions.Categorical(pred_ls[1])
    #                     argmax2 = int(dist.sample())
    #                     # argmax = torch.argmax(pred_ls[1]).item()
    #                 else:
    #                     argmax2 = torch.argmax(pred_ls[1]).item()

    #                 pred_ub = (range_max - feat_val)*argmax2/(discretize_feat_value_count-1) + feat_val

    #                 ret[token] = [[pred_lb, pred_ub],[argmax1, argmax2]]
    #                 ret_preds[token] = pred_ls

    #                 break

    #             pred_val = self.grammar_num_to_token_val[token][action]
    #             queue.extend(self.lang.syntax[token][pred_val])
    #             ret[token] = action
    #             ret_preds[token] = pred
    #             decoder_input = self.atom_to_vector(self.idx_to_atom(ret))
                
    #     else:
    #         for token in queue:
    #             # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
    #             # decoder_output = decoder_output.view(-1)
    #             if token in self.lang.syntax["num_feat"]:
    #                 # if np.random.rand() < epsilon:
    #                 #     sub_keys=["_ub", "_lb"]
    #                 # else:
    #                 sub_keys=["_lb", "_ub"]
    #                 pred_ls = []
    #                 ret_preds[token] = []
    #                 for idx in range(len(sub_keys)):
    #                     key = sub_keys[idx]
    #                     pred = self.token_nets[token+key](hx.view(-1))    
    #                     # pred = self.mask_grammar_net_pred(program, token, pred)
    #                     pred_ls.append(pred)
    #                     ret_preds[token].append(pred)
    #                     break
    #             else:
    #                 pred = self.token_nets[token](hx.view(-1))
    #                 # act_val_id = self.grammar_token_val_to_num[act_val]
    #                 ret_preds[token] = pred
    #                 # pred = self.mask_grammar_net_pred(program, token, pred)
    #             ret[token] = queue[token]
    #             # decoder_input = self.atom_to_vector(self.idx_to_atom(ret))
                
    #     return ret, ret_preds

class Critic2(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size):
        super(Critic2, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        self.patient_max_appts=patient_max_appts
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        
        for i,v in self.lang.syntax.items():
            if not i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            else:
                self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))
        # for decision, options_dict in self.lang.syntax.items():
        #     start = self.ATOM_VEC_LENGTH
        #     for option in list(options_dict.keys()):
        #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        # for decision, options_dict in self.lang.syntax.items():
        #     # if decision == "num_op":
        #     #     continue
        #     start = self.ATOM_VEC_LENGTH


        #     if not decision in self.lang.syntax["num_feat"]:
        #         for option in list(options_dict.keys()):        
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #     else:
        #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
                continue
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
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1


        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        self.decoder = AttnCriticNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len)
        # self.embedding = TokenNetwork3(full_input_size, latent_size)
        self.embedding = nn.Linear(full_input_size,1)
        # self.decoder = nn.Linear(latent_size,1)
        self.first_prog_embed = torch.tensor([0]*self.ATOM_VEC_LENGTH, device =DEVICE, dtype=torch.float)
        self.to(DEVICE)

    def forward(self, features, program):
        
#         hx = self.feat_gru.initHidden()
#         feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
#         for ei in range(len(features)):
#             feat_out, hx = self.feat_gru(features[ei], hx)
#             feat_encoder_outputs[ei] = feat_out[0,0]
#         hx = hx.view(1,1,-1)
#         prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
#         for ei in range(len(program)):
#             prog_out, hx = self.prog_gru(program[ei], hx)
#             prog_encoder_outputs[ei] = prog_out[0,0]
# # 
#         decoder_input = self.decoder.initInput()
#         ret = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
        # ret = self.decoder(hx)
        if len(program) == 0:
            program = [self.first_prog_embed]
        hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*self.ATOM_VEC_LENGTH, device=DEVICE)
        hx[:,0:features[0].shape[0]] = features
        if len(program) > 0:
            hx[:, features[0].shape[0]:len(program)*self.ATOM_VEC_LENGTH+features[0].shape[0]] = torch.cat(program,dim=-1)
        ret = self.embedding(hx)
        return ret



class PPO_debug3:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, n_updates_per_iteration, clip,feat_range_mappings):
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip = clip
        self.lang = lang
        self.n_updates_per_iteration = n_updates_per_iteration

        self.actor = Actor2(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, feat_range_mappings=feat_range_mappings)
        self.critic = Critic2(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size)
    
        self.actor_optimizer = optim.Adam(self.actor.parameters(), learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), learning_rate)

        self.first_prog_embed = torch.tensor([0]*self.actor.ATOM_VEC_LENGTH, device =DEVICE, dtype=torch.float)

    def flatten_probs(self, probs: dict):
        all_val_list = []
        
        for val in list(probs.values()):
            if type(val) is list:
                all_val_list.extend(val)
            else:
                all_val_list.append(val)
        
        return reduce((lambda x,y: x+y), all_val_list)
    
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
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=DEVICE)
        return batch_rtgs
    
    def learn(self, batch):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = batch
        loss = 0
        # batch_probs_num = torch.tensor([self.flatten_probs(i) for i in batch_log_probs], dtype = torch.float, requires_grad=True)
        # batch_probs_num = torch.cat([self.flatten_probs(i).view(1,-1) for i in batch_log_probs]).squeeze()
        batch_probs_num = torch.stack(batch_log_probs,dim=0)
        # batch_probs_num = 
        V, _ = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)
        A_k = batch_rtgs - V.clone().detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        for t_idx in range(self.n_updates_per_iteration):
            V, curr_log_probs = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)


            # curr_probs_num = torch.cat([self.flatten_probs(i).view(1,-1) for i in curr_log_probs]).squeeze()
            curr_probs_num = torch.stack(curr_log_probs, dim=0)
            # curr_probs_num = torch.tensor([self.flatten_probs(i) for i in curr_log_probs], dtype = torch.float, requires_grad=True)

            ratios = torch.exp(curr_probs_num - batch_probs_num.clone().detach()) #is exp needed?

            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()
            # actor_loss = nn.MSELoss()(torch.zeros(size=actor_loss.shape), actor_loss)
            # actor_loss = nn.MSELoss()(torch.zeros_like(actor_loss), actor_loss)

            self.actor_optimizer.zero_grad()
            
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            # 
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            self.critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optim.step()

            

            # V, _ = self.evaluate(batch_obs=batch_obs, batch_acts=batch_acts)

            loss += abs(critic_loss.clone().detach()) + abs(actor_loss.clone().detach())


        return loss

    def atom_to_vector_ls0(self, atom, col, op):
        return self.actor.atom_to_vector_ls0(atom, col, op)

    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.actor.atom_to_vector(atom)

    def vector_to_atom(self, vec):
        return self.actor.vector_to_atom(vec)
    
    # def idx_to_atom(self, idx):
    #     return self.actor.idx_to_atom(idx)

    #turns network Grammar Networks idx and turns them into an atom
    def idx_to_atom(self, idx:dict):
        return self.actor.idx_to_atom(idx)

    def idx_to_logs(self, pred:dict, idx:dict):
        
        atom_log_probs = pred[torch.tensor(list(range(pred.shape[0]))), idx]
        
        # logs =  {}
        # for token,index in idx.items():
        #     p_ls = pred[token]
            
        #     if token in self.lang.syntax["num_feat"]:
        #         logs[token] = []
        #         for idx in range(len(p_ls)):
        #             dist = torch.distributions.Categorical(p_ls[idx])
        #             logs[token].append(dist.log_prob(torch.tensor(index[1][idx]).to(DEVICE)))
            
        #     else:
        #         dist = torch.distributions.Categorical(p_ls)
        #         logs[token] = dist.log_prob(torch.tensor(index).to(DEVICE))
        return torch.log(atom_log_probs)


    def predict_atom(self, features, X_pd_ls, program, train,col, op):
        if len(program) == 0:
            program = [self.first_prog_embed]
        # features,X_pd_full, program, train, col, op, eval=False
        atom_preds = self.actor.forward(features,X_pd_ls, program, train = train, col = col, op = op)
        return atom_preds

    def evaluate(self, batch_obs, batch_acts):
        # V = torch.tensor([self.critic(f, p) for _, f,p in batch_obs], dtype = torch.float, requires_grad=True).squeeze()
        V = torch.stack([self.critic(f, p) for f,_,p,_,_ in batch_obs]).squeeze(-1)
        batch_eval_probs = []
        for obs, act in zip(batch_obs, batch_acts):
            preds = self.actor.forward(*obs, train=False, eval=True, selected_idx=act[pred_prob_id])
            atom_probs = self.idx_to_logs(preds[pred_prob_key], act[pred_prob_id])
            batch_eval_probs.append(atom_probs)
        return V, batch_eval_probs
    

    

