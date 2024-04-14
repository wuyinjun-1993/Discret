import torch
import logging
from torch import nn, optim
from create_language import *
import numpy as np
import random
from collections import namedtuple, deque
DEVICE = "cuda"
print(f"Using {DEVICE} device")
discretize_feat_value_count=10

Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_output_classes),
            nn.Softmax(dim=-1),
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
        # input = input.view(1,1,-1)
        output, o_hidden = self.gru(input, hidden)
        del hidden
        return output, o_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size, feat_max_size, prog_max_size, dropout_p):
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
        embedded = self.embedding(input)#.view(1, 1, -1)
        # hidden = hidden.view(1,1,-1)
        # embedded = self.dropout(embedded)

        # feat_attn_weights = nn.functional.softmax(
        #     self.feat_attn(torch.cat((embedded[:,0], hidden[:,0].repeat(embedded.shape[0], 1)), 1)), dim=1)

        # prog_attn_weights = nn.functional.softmax(
        #     self.prog_attn(torch.cat((embedded[:,0], hidden[:,0].repeat(embedded.shape[0], 1)), 1)), dim=1)
        
        feat_attn_weights = nn.functional.softmax(
            self.feat_attn(torch.cat((embedded[:,0], hidden.view(embedded.shape[0], -1)), 1)), dim=1)

        prog_attn_weights = nn.functional.softmax(
            self.prog_attn(torch.cat((embedded[:,0], hidden.view(embedded.shape[0], -1)), 1)), dim=1)


        feat_attn_applied = torch.bmm(feat_attn_weights.unsqueeze(1),
                                 feat_outputs)

        prog_attn_applied = torch.bmm(prog_attn_weights.unsqueeze(1),
                                 prog_outputs)

        output = torch.cat((embedded, feat_attn_applied, prog_attn_applied), -1)
        final_output = self.attn_combine(output).unsqueeze(0)
        del embedded, feat_attn_applied, prog_attn_applied, output
        # output = nn.functional.relu(output)
        # output, hidden = self.gru(output, hidden)

        return final_output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
    
    def initInput(self):
        return torch.zeros(1, 1, self.input_size, device=DEVICE)



class RLSynthesizerNetwork2(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size, dropout_p, feat_range_mappings):
        super(RLSynthesizerNetwork2, self).__init__()
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
        self.feat_range_mappings = feat_range_mappings
        for i,v in self.lang.syntax.items():
            if not i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            else:
                self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

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


        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
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

    # def prediction_to_atom(self, pred:dict):
    #     return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
    def prediction_to_atom(self, pred:dict):
        res = {}
        for i,v in pred.items():
            if not i in self.lang.syntax["num_feat"]:
                res[i] = self.grammar_num_to_token_val[i][torch.argmax(v).item()]
            else:
                res[i] = [[v[0][0].item(), v[0][1].item()],v[1]]
        # return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
        return res

    def prediction_to_atom_ls(self, pred:dict):
        res = {}
        for i,v in pred.items():
            # if not i in self.lang.syntax["num_feat"]:
            if not type(i) is tuple:
                if v.shape[1] <= 1:
                    res[i] = self.grammar_num_to_token_val[i][torch.argmax(v[0]).item()]
                else:
                    argmax_ids = torch.argmax(v, dim=1)
                    argmax_unique_ids = torch.unique(argmax_ids)
                    token_sample_id_maps = dict()
                    for unique_id in argmax_unique_ids:
                        sample_ids = torch.nonzero(argmax_ids == unique_id)
                        selected_token = self.grammar_num_to_token_val[i][unique_id.item()]
                        token_sample_id_maps[selected_token] = sample_ids
                    res[i] = token_sample_id_maps
            else:
                res[i] = v
        # return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
        return res

    def prediction_to_atom_ls2(self, batch_size, pred:dict):
        res_ls = [{} for _ in range(batch_size)]
        for i,v in pred.items():
            # if not i in self.lang.syntax["num_feat"]:
            
            if not type(i) is tuple:
                if v.shape[1] <= 1:
                    for res_idx in range(len(res_ls)):
                        res = res_ls[res_idx]
                        res[i] = self.grammar_num_to_token_val[i][torch.argmax(v[res_idx]).item()]
                else:
                    argmax_ids = torch.argmax(v, dim=1)
                    argmax_unique_ids = torch.unique(argmax_ids)
                    # token_sample_id_maps = dict()
                    for unique_id in argmax_unique_ids:
                        sample_ids = torch.nonzero(argmax_ids == unique_id)
                        selected_token = self.grammar_num_to_token_val[i][unique_id.item()]
                        # token_sample_id_maps[selected_token] = sample_ids
                        for sample_id in sample_ids:
                            res_ls[sample_id][i] = selected_token
            # else:
            #     res[i] = v

        # return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
        return res_ls

    def vector_to_atom(self, pred:list):
        atom = {}
        for i,v in enumerate(pred):
            if v == 1:
                decision, option = self.grammar_pos_to_token[i]
                atom[decision] = option
        return atom

    # def atom_to_vector(self, atom:dict):
    #     one_hot_pos = []
    #     for token, token_val in atom.items():
    #         one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
    #     ret = [0]*self.ATOM_VEC_LENGTH
    #     for i in one_hot_pos:
    #         ret[i] = 1
    #     return torch.tensor(ret, device=DEVICE, dtype=torch.float)

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
        return torch.FloatTensor(ret).to(DEVICE)
    
    def atom_to_vector_ls(self, atom_ls:dict):
        ret_tensor_ls = []
        for atom in atom_ls:
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
            ret_tensor_ls.append(torch.FloatTensor(ret))
        return ret_tensor_ls

    def mask_grammar_net_pred(self, program, token, token_pred_out):
        # if not any(program[0]) and token == "formula":
        #     end_index = self.grammar_token_to_pos[(token, "end")]
        #     #TODO
        if token in ["num_feat", "cat_feat"]:
            start, end = self.one_hot_token_bounds[token]
            for atom in program:
                mask =  torch.logical_not(atom[start:end]).int().float()
                token_pred_out = token_pred_out * mask
        return token_pred_out
    
    def mask_grammar_net_pred_ls(self, program, token, token_pred_out):
        # if not any(program[0]) and token == "formula":
        #     end_index = self.grammar_token_to_pos[(token, "end")]
        #     #TODO
        if token in ["num_feat", "cat_feat"]:
            start, end = self.one_hot_token_bounds[token]
            for atom in program:
                mask =  torch.logical_not(atom[:,start:end]).int().float()
                mask = mask.to(DEVICE)
                token_pred_out = token_pred_out * mask
                del mask
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


    def forward(self, features,X_pd, program, queue, epsilon, eval=False):
        features = [f.to(DEVICE) for f in features]
        hx = self.feat_gru.initHidden()
        feat_encoder_outputs = torch.zeros(self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        for ei in range(len(features)):
            feat_out, hx = self.feat_gru(features[ei].view(1,1,-1), hx)
            feat_encoder_outputs[ei] = feat_out[0,0]
        feat_encoder_outputs = feat_encoder_outputs.unsqueeze(0)
        hx = hx.view(1,1,-1)
        prog_encoder_outputs = torch.zeros(self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        for ei in range(len(program)):
            prog_out, hx = self.prog_gru(program[ei].view(1,1,-1), hx)
            prog_encoder_outputs[ei] = prog_out[0,0]
        prog_encoder_outputs = prog_encoder_outputs.unsqueeze(0)

        ret = {}
        decoder_input = self.decoder.initInput()
        
        while queue:
            hx = hx.view(1,1,-1)
            decoder_input = decoder_input.view(1,1,-1)
            token = queue.pop(0)
            if token in ret:
                continue
            if token == "num_op":
                continue
            if token in self.lang.syntax["num_feat"]:
                # if np.random.rand() < epsilon:
                #     sub_keys=["_ub", "_lb"]
                # else:
                sub_keys=["_lb", "_ub"]
                pred_ls = []
                for key in sub_keys:
                    if np.random.rand() < epsilon:
                        pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                    else:
                        decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                        decoder_output = decoder_output.view(-1)
                        pred = self.token_nets[token+key](decoder_output.view(-1))    
                    pred = self.mask_grammar_net_pred(program, token, pred)
                    pred_ls.append(pred)
            else:
                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                else:
                    decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                    decoder_output = decoder_output.view(-1)
                    # print(hx)
                    pred = self.token_nets[token](decoder_output)
            if not token in self.lang.syntax["num_feat"]:
                pred = self.mask_grammar_net_pred(program, token, pred)
                argmax = torch.argmax(pred).item()
                pred_val = self.grammar_num_to_token_val[token][argmax]
                if not eval:
                    queue.extend(self.lang.syntax[token][pred_val])
                ret[token] = pred
            else:
                range_max = self.feat_range_mappings[token][1]
                range_min = self.feat_range_mappings[token][0]

                feat_val = list(X_pd[token])[0]

                argmax = torch.argmax(pred_ls[0]).item()


                pred_lb = (feat_val - range_min)*(argmax/(discretize_feat_value_count-1)) + range_min

                argmax = torch.argmax(pred_ls[1]).item()

                pred_ub = (range_max - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

                ret[token] = [[pred_lb, pred_ub],pred_ls]

                break
            decoder_input = self.atom_to_vector(self.prediction_to_atom(ret))
            
        del features
        return ret


    def forward_ls(self, features,X_pd_ls, program, queue, epsilon, eval=False, replay=False):
        features = features.to(DEVICE)
        pat_count = features.shape[0]
        X_pd_full = pd.concat(X_pd_ls)
        hx = self.feat_gru.initHidden()
        hx = hx.repeat(1, pat_count,1)
        # feat_encoder_outputs = torch.zeros(pat_count, self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
        # for ei in range(len(features)):
        feat_out, hx = self.feat_gru(features.view(1, features.shape[0], -1), hx)
        # feat_encoder_outputs[:,0] = feat_out[0,0]
        hx = hx.view(1,pat_count,-1)
        prog_encoder_outputs = torch.zeros(pat_count, self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
        if len(program) > 0:
            input_program = torch.stack(program, dim=0)
            input_program = input_program.to(DEVICE)
            prog_out, hx = self.prog_gru(input_program, hx)
            prog_encoder_outputs[:, 0:len(program)] = prog_out.permute((1,0,2))
            del prog_out, input_program
        # for ei in range(len(program)):
        #     sub_program = program[ei].view(1, program[ei].shape[0], -1).to(DEVICE)
        #     prog_out, hx = self.prog_gru(sub_program, hx)
        #     prog_encoder_outputs[:, ei] = prog_out.view(prog_encoder_outputs[:, ei].shape)
        #     del prog_out, sub_program
            
        ret = {}
        decoder_input = self.decoder.initInput()
        decoder_input = decoder_input.repeat(pat_count, 1, 1)
        while queue:
            token = queue.pop(0)
            
            decoder_output, hx = self.decoder(decoder_input, hx, feat_out.squeeze(0).unsqueeze(1), prog_encoder_outputs)
            decoder_output = decoder_output.view(len(X_pd_ls), -1)
            if type(token) is tuple:
            # if token in self.lang.syntax["num_feat"]:
                # if np.random.rand() < epsilon:
                #     sub_keys=["_ub", "_lb"]
                # else:
                hx = hx.squeeze(0)
                pred_probs_ls_map = dict()
                pred_lb_map = dict()
                pred_ub_map = dict()
                token_sample_id_maps = dict()
                pred_interval_id_maps = dict()

                if not eval:
                # if True:
                    for token_key in token:
                        
                        token_sample_ids = pred_val_idx_maps[token_key]
                        # token_key  = token_key[0]
                        sub_keys=["_lb", "_ub"]
                        pred_ls = []
                        # for key in sub_keys:
                        for key in sub_keys:
                            if np.random.rand() < epsilon:
                                pred = torch.rand([len(token_sample_ids), len(self.grammar_num_to_token_val[token_key[0]])], device=DEVICE)
                            else:
                                pred = self.token_nets[token_key[0]+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                            pred = self.mask_grammar_net_pred_ls(program, token, pred)
                            if not replay:
                                pred_ls.append(pred.data.cpu())
                                del pred
                            else:
                                pred_ls.append(pred)
                            
                            
                        range_max = self.feat_range_mappings[token_key[0]][1]
                        range_min = self.feat_range_mappings[token_key[0]][0]

                        feat_val = np.array(X_pd_full.iloc[token_sample_ids.view(-1).cpu().numpy()][token_key[0]])

                        argmax1 = torch.argmax(pred_ls[0], dim=1)


                        pred_lb = (feat_val - range_min)*(argmax1.cpu().numpy()/(discretize_feat_value_count-1)) + range_min

                        argmax2 = torch.argmax(pred_ls[1], dim=1)

                        pred_ub = (range_max - feat_val)*(argmax2.cpu().numpy()/(discretize_feat_value_count-1)) + feat_val

                        # print(pred_lb)

                        # print(feat_val)

                        # print(pred_ub)

                        # print()

                        pred_probs_ls_map[token_key[0]] = pred_ls
                        pred_lb_map[token_key[0]] = pred_lb
                        pred_ub_map[token_key[0]] = pred_ub
                        pred_interval_id_maps[token_key[0]] = [argmax1.cpu(), argmax2.cpu()]
                        token_sample_id_maps[token_key[0]] = token_sample_ids.cpu()
                        # pred_probs_ls[0].extend(pred_ls[0])
                        # pred_probs_ls[1].extend(pred_ls[1])
                        # pred_lb_ls.append(pred_lb)
                        # pred_ub_ls.append(pred_ub)

                else:
                    for token_key in token:
                        # token_sample_ids = pred_val_idx_maps[token_key]
                        # token_key  = token_key[0]
                        sub_keys=["_lb", "_ub"]
                        pred_ls = []
                        # for key in sub_keys:
                        for key in sub_keys:
                            # if np.random.rand() < epsilon:
                            #     pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token_key[0]])])
                            # else:
                            pred = self.token_nets[token_key[0]+key](decoder_output.view(pat_count, -1))    
                            pred = self.mask_grammar_net_pred_ls(program, token, pred)
                            if not replay:
                                pred_ls.append(pred.data.cpu())
                                del pred
                            else:
                                pred_ls.append(pred)
                            
                            pred_probs_ls_map[token_key[0]] = pred_ls
                ret[token] = [[pred_lb_map, pred_ub_map], pred_probs_ls_map, token_sample_id_maps, pred_interval_id_maps]

                break
                            
            # if not token in self.lang.syntax["num_feat"]:
            else:
                if token in ret:
                    continue
                if token == "num_op":
                    continue
                    
                if np.random.rand() < epsilon:
                    pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token])], device=DEVICE)
                else:
                    # print(hx)
                    pred = self.token_nets[token](decoder_output.view(pat_count, -1))
                pred = self.mask_grammar_net_pred_ls(program, token, pred)
                if not eval:
                    argmax = torch.argmax(pred, dim=1)
                    unique_argmax_val = argmax.unique()
                    pred_val_idx_maps = dict()
                    for argmax_val in unique_argmax_val:
                        argmax_val_idx = torch.nonzero(argmax == argmax_val)
                        
                        pred_val = self.grammar_num_to_token_val[token][argmax_val.item()]
                        
                        pred_val_idx_maps[tuple(self.lang.syntax[token][pred_val])] = argmax_val_idx
                    del argmax
                
                    if not token == "num_feat":                
                        queue.extend(list(pred_val_idx_maps.keys())[0])
                    else:
                        queue.append(tuple(list(pred_val_idx_maps.keys())))
                        
                # else:
                #     print()
                # else:
                #     # if len(pred_val_idx_maps) == 1:  
                #     if not token == "num_feat":              
                #         queue.extend(list(pred_val_idx_maps.keys())[0])
                #     else:

                #         queue.append(tuple(self.lang.syntax[token]))
                if not replay:
                    ret[token] = pred.data.cpu()
                    del pred
                else:
                    ret[token] = pred
            
            del decoder_input, decoder_output
            decoder_input = torch.stack(self.atom_to_vector_ls(self.prediction_to_atom_ls2(pat_count, ret)))
            decoder_input = decoder_input.view(pat_count,1,-1).to(DEVICE)#.repeat(pat_count, 1, 1)
        del features, prog_encoder_outputs, hx, decoder_input, decoder_output, feat_out
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

class DQN2:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0):
        self.mem_sample_size = mem_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang
        torch.manual_seed(seed)
        self.policy_net = RLSynthesizerNetwork2(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings)
        torch.manual_seed(seed)
        self.target_net = RLSynthesizerNetwork2(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, dropout_p = 0, feat_range_mappings=feat_range_mappings)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.memory = ReplayMemory(replay_memory_capacity)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), learning_rate)

        self.first_prog_embed = torch.tensor([0]*self.policy_net.ATOM_VEC_LENGTH, device=DEVICE, dtype=torch.float)#torch.randn(self.policy_net.ATOM_VEC_LENGTH, requires_grad=True)

    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.policy_net.atom_to_vector(atom)

    def atom_to_vector_ls(self, atom:dict):
        return self.policy_net.atom_to_vector_ls(atom)

    def vector_to_atom(self, vec):
        return self.policy_net.vector_to_atom(vec)

    #turns network Grammar Networks predictions and turns them into an atom
    def prediction_to_atom(self, pred:dict):
        return self.policy_net.prediction_to_atom(pred)

    def random_atom(self, program):
        #TODO
        if len(program) == 0:
            pred = self.policy_net.random_atom(program = [torch.tensor([0]*self.policy_net.ATOM_VEC_LENGTH, device=DEVICE, dtype=torch.float)])
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
    
    def predict_atom_ls(self, features, X_pd_ls, program, epsilon):
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls),1)
            pred = self.policy_net.forward_ls(features, X_pd_ls, [init_program], ["formula"], epsilon)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls(features, X_pd_ls, program, ["formula"], epsilon)
        return self.policy_net.prediction_to_atom_ls(pred)
    
    #predicts the best atom to add to the program from the given next state, and provides the maximal tensors which produced that decision
    #uses target net!!!
    # def predict_next_state_with_tensor_info(self, features, program):
    #     if len(program) == 0:
    #         pred = self.target_net(features, [self.first_prog_embed], ["formula"], 0)
    #     else:
    #         #program.sort()
    #         pred = self.target_net(features, program, ["formula"], 0)
    #     max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items()}
    #     return self.target_net.prediction_to_atom(pred), max_tensors

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
    
    def predict_next_state_with_tensor_info_ls(self, features, data, program):
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data),1)
            pred = self.target_net.forward_ls(features, data, [init_program], ["formula"], 0, replay=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls(features, data, program, ["formula"], 0, replay=True)
        max_tensors = dict()
        for token, token_val in pred.items():
            # if not token in self.lang.syntax["num_feat"]:
            if not type(token) is tuple:
                max_tensors[token] = torch.max(token_val, dim=1)[0].reshape((len(data),1))
            else:
                
                max_tensors[token] = [torch.zeros(len(data), device = DEVICE), torch.zeros(len(data), device = DEVICE)]
                pred_val = pred[token]
                for token_key in token:
                    
                    token_key = token_key[0]
                    lb_probs = pred_val[1][token_key][0]
                    ub_probs = pred_val[1][token_key][1]
                    sample_ids = token_val[2][token_key].view(-1)
                    sample_cln_id_ls = token_val[3][token_key]
                    val = lb_probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[0].view(-1)]
                    max_tensors[token][0][sample_ids] = val
                    del val
                    val = ub_probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[1].view(-1)]      
                    max_tensors[token][1][sample_ids] = val
                    del val
                # print()
                # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
        
        # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        return_pred = self.target_net.prediction_to_atom_ls(pred)
        del pred
        return return_pred, max_tensors

    #takes a state,action (where action is an atom) pair and returns prediction tensors which are generated when picking the same tokens from the given atom
    # def get_state_action_prediction_tensors(self, features, program, atom):
    #     queue = list(atom.keys())
    #     if len(program) == 0:
    #         pred = self.policy_net(features, [self.first_prog_embed], queue, 0)
    #     else:
    #         #program.sort()
    #         pred = self.policy_net(features, program, queue, 0)

    #     tensor_indeces = {token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
    #     atom_prediction_tensors = {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
    #     return atom_prediction_tensors
    
    def get_state_action_prediction_tensors(self, features, X_pd, program, atom):
        queue = list(atom.keys())
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], queue, 0, eval=True)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, queue, 0, eval=True)

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

    def get_state_action_prediction_tensors_ls(self, features, X_pd, program, atom):
        queue = list(atom.keys())
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            pred = self.policy_net.forward_ls(features, X_pd, [init_program], queue, 0, eval=True, replay=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls(features, X_pd, program, queue, 0, eval=True, replay=True)

        tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        atom_prediction_tensors = {}
        for token, token_val in atom.items():
            # if token == "num_op" or token.endswith("_prob"):
            #     continue

            # if token not in self.lang.syntax["num_feat"]:
            if not type(token) is tuple:
                # if not token.endswith("_prob"):
                    # tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
                    if not type(token_val) is dict:
                        tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val]
                        val = pred[token][:,tensor_idx].reshape((len(X_pd),1))
                        atom_prediction_tensors[token] = val
                        del val
                    else:
                        atom_prediction_tensors[token] = torch.zeros(len(X_pd), device = DEVICE)
                        for token_val_key in token_val:
                            token_val_sample_ids = token_val[token_val_key]
                            tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val_key]
                            val = pred[token][token_val_sample_ids,tensor_idx]
                            atom_prediction_tensors[token][token_val_sample_ids] = val
                            del val
                        
            else:
                atom_prediction_tensors[token] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
                pred_val = pred[token]
                for token_key in token:
                    
                    token_key = token_key[0]
                    lb_probs = pred_val[1][token_key][0]
                    ub_probs = pred_val[1][token_key][1]
                    sample_ids = token_val[2][token_key].view(-1)
                    sample_cln_id_ls = token_val[3][token_key]
                    val = lb_probs[sample_ids.view(-1), sample_cln_id_ls[0].view(-1)]
                    atom_prediction_tensors[token][0][sample_ids] = val
                    del val
                    val = ub_probs[sample_ids.view(-1), sample_cln_id_ls[1].view(-1)]
                    atom_prediction_tensors[token][1][sample_ids] = val
                    del val


                # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
            # else:
            #     tensor_indeces[token] = 0
        
        # for token, tensor_idx in tensor_indeces.items():
        #     if token not in self.lang.syntax["num_feat"]:
        #         atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
        #     else:
        #         atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
        # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        del pred
        return atom_prediction_tensors
    
    #takes an atom, and the maximal tensors used to produce it, and returns a Q value
    def get_atom_Q_value(self, atom:dict, atom_prediction_tensors: dict):
        formula = atom_prediction_tensors["formula"]
        if atom["formula"] == "end":
            one = torch.tensor([[1]], dtype=torch.float,device=DEVICE)
            feat, op, constant = one, one, one
        else:
            if "num_feat" in atom:
                feat_name = atom["num_feat"]
                feat = atom_prediction_tensors["num_feat"]
                op = 1#atom_prediction_tensors["num_op"]
            else:
                feat_name = atom["cat_feat"]
                feat = atom_prediction_tensors["cat_feat"]
                op = 1#atom_prediction_tensors["cat_op"]
            constant = atom_prediction_tensors[feat_name]
        Q = formula*feat*op*constant[0]*constant[1]
        return Q[0]

    def get_atom_Q_value_ls(self, atom:dict, atom_prediction_tensors: dict):
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
            constant = atom_prediction_tensors[tuple([tuple([item]) for item in list(feat_name.keys())])]
        # feat = feat.to(DEVICE)
        # formula = formula.to(DEVICE)
        Q = formula.view(-1)*feat.view(-1)*op*constant[0].view(-1)*constant[1].view(-1)
        return Q
    
    def observe_transition(self, transition: Transition):
        self.memory.push(transition)

 
    def optimize_model(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.tensor([[transition.reward] for transition in batch], device=DEVICE, requires_grad=True, dtype=torch.float)

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 1], device=DEVICE, dtype=torch.float)
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        # for i in range(len(batch)):
        #     print(batch[i].data)
        #     print("program::")
        #     for pid in range(len(batch[i].program)):
        #         print(batch[i].program[pid])
        # print("loss::", loss)
        # print("expected_state_action_values::", expected_state_action_values)
        # print("next_state_values::", next_state_values)
        # print("reward_batch::", reward_batch)
        # print("state_action_values::", state_action_values)

        # Return loss
        return loss.detach()
    
    def optimize_model_ls(self):
        if len(self.memory) < self.mem_sample_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.mem_sample_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.stack([torch.from_numpy(transition.reward).view(-1) for transition in batch]).to(DEVICE)

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors_ls(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
        state_action_values = state_action_values.to(DEVICE)
        
        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.mem_sample_size, self.batch_size], dtype=torch.float, device=DEVICE)
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value_ls(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final
            del next_state_values_non_final
        next_state_values = next_state_values.to(DEVICE)
        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        # for item in non_final_samples:
        #     del item
        # for  item in state_action_batch:
        #     del item
        # for item in state_action_pred:
        #     del item
        # for item in next_state_pred_non_final:
        #     del item
        # non_final_samples.clear()
        # state_action_pred.clear()
        # state_action_batch.clear()
        # non_final_samples.clear()
        # del state_action_values, expected_state_action_values, next_state_values, reward_batch, state_action_pred, next_state_pred_non_final, non_final_mask
        # del non_final_samples, batch, state_action_batch
        # for i in range(len(batch)):
        #     print(batch[i].data)
        #     print("program::")
        #     for pid in range(len(batch[i].program)):
        #         print(batch[i].program[pid])
        # print(batch[0].data)
        # print(batch[1].data)
        # print("loss::", loss)
        # print("expected_state_action_values::", expected_state_action_values)
        # print("next_state_values::", next_state_values)
        # print("reward_batch::", reward_batch)
        # print("state_action_values::", state_action_values)
        # Return loss
        return_loss = loss.detach()
        del loss
        return return_loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


