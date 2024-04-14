import torch
from collections import namedtuple, deque
import numpy as np
import random
import operator
import sys,os
from torch import nn, optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from feat_encoder.ft_transformer0 import FTTransformer

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from enc_dec import DEVICE, TokenNetwork3, pred_v_key, mask_atom_representation1, down_weight_removed_feats, col_id_key, prev_prog_key, col_probs_key, col_Q_key, op_key, op_id_key, col_key, min_Q_val

range_key = "selected_range"

range_tensor_key = "selected_range_tensor"

def atom_to_vector_ls0_main(net, atom_ls):
    ret_tensor_ls = []
    # pred_v_arr = atom_ls[pred_v_key]
    range_id_tensor = atom_ls[range_tensor_key].to(DEVICE)
    
    col_id_tensor = atom_ls[col_id_key]
    
    # op_id_tensor = atom_ls[op_id_key]
        
    ret_tensor_ls = torch.zeros([len(col_id_tensor), net.topk_act, net.ATOM_VEC_LENGTH]).to(DEVICE)
            
    sample_id_tensor = torch.arange(len(ret_tensor_ls),device=DEVICE)
    
    for k in range(net.topk_act):
        ret_tensor_ls[sample_id_tensor,k, net.num_start_pos + col_id_tensor[:,k]]=1
        ret_tensor_ls[sample_id_tensor,k, net.num_feat_len]=range_id_tensor[sample_id_tensor, k, 0]
        ret_tensor_ls[sample_id_tensor,k, net.num_feat_len + 1]=range_id_tensor[sample_id_tensor, k, 1]
        # ret_tensor_ls[sample_id_tensor,k, net.op_start_pos + op_id_tensor[:,k]]=1
    
    # ret_tensor_ls[:, :, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)

    return ret_tensor_ls

def mask_atom_representation_medical(topk_act, num_feat_len, program_ls, feat_pred_logit, init=False):
    
    # op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    # op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    program_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    if not init:
        for program in program_ls:
            program_mat += program[:, :, 0:-2].to(DEVICE)
    
    # if not init:
    #     for program in program_ls:
    #         op1_feat_occur_mat += program[:,:, op_start_pos:op_start_pos+1].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    #         op2_feat_occur_mat += program[:,:, op_start_pos+1:op_start_pos+2].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    # else:
    #     for program in program_ls:
    #         op1_feat_occur_mat += program[:,:, op_start_pos:op_start_pos+1].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    #         op2_feat_occur_mat += program[:,:, op_start_pos+1:op_start_pos+2].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    
    # op1_feat_occur_mat = torch.sum(op1_feat_occur_mat,dim=1)
    
    # op2_feat_occur_mat = torch.sum(op2_feat_occur_mat, dim=1)
    
    # feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
    
    feat_pred_probs = torch.softmax(feat_pred_logit, dim=-1) + 1e-6

    feat_pred_Q = torch.tanh(feat_pred_logit)

    if not init:
        feat_pred_probs = feat_pred_probs*(program_mat < 1).float()

        feat_pred_Q = feat_pred_Q*(program_mat < 1).float() + (min_Q_val)*(program_mat >= 1).float()
    
    return feat_pred_probs, feat_pred_Q, program_mat


def identify_range(selected_ranges, feat_val):
    for curr_range in selected_ranges:
        if curr_range[0] != curr_range[1]:
            if feat_val> curr_range[0] and feat_val <= curr_range[1]:
                break
        else:
            if feat_val >= curr_range[0] and feat_val <= curr_range[1]:
                break
    return curr_range

def normalize_range(selected_ranges_ls, selected_cols_ls, feat_range_mappings):
    normalized_selected_ranges_ls = []
    for idx in range(len(selected_ranges_ls)):
        selected_ranges = selected_ranges_ls[idx]
        selected_cols = selected_cols_ls[idx]
        curr_normalized_selected_ranges_ls = []
        for sub_idx in range(len(selected_ranges)):
            col = selected_cols[sub_idx]
            range_min, range_max = selected_ranges[sub_idx]
            feat_min, feat_max = feat_range_mappings[col]
            if np.isnan(range_min) or np.isinf(range_min):
                normalized_range_min = -1
            else:
                normalized_range_min = (range_min-feat_min)/(feat_max - feat_min)
            

            if np.isnan(range_max) or np.isinf(range_max):
                normalized_range_max = 2
            else:
                normalized_range_max = (range_max-feat_min)/(feat_max - feat_min)
                
            curr_normalized_selected_ranges_ls.append([normalized_range_min, normalized_range_max])
            
        
        normalized_selected_ranges_ls.append(curr_normalized_selected_ranges_ls)
        
    return torch.tensor(normalized_selected_ranges_ls, dtype=torch.float)

def forward_main1(net, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False):
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            if net.feat_group_names is None:
                output_num = net.num_feat_len
            else:
                output_num = net.feat_group_num

            if init:
                selected_feat_logit = torch.rand([pat_count, output_num], device=DEVICE)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, output_num], device=DEVICE)

        else:
            selected_feat_logit = net.feat_selector(hx)
    
    else:
        selected_feat_logit = net.feat_selector(hx)

    selected_feat_probs, selected_Q_feat, program_feat_mask = mask_atom_representation_medical(net.topk_act, net.num_feat_len, program, selected_feat_logit, init=init)

    # if net.feat_group_names is None:
    #     mask_atom_representation_medical(net.topk_act, net.num_feat_len, program, selected_feat_logit, init=init)
    #     selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feat_len, net.op_start_pos, program, selected_feat_logit, init=init)
    # else:
    #     mask_atom_representation_medical(net.topk_act, net.num_feat_len, net.op_start_pos, program, selected_feat_logit, init=init)
    #     selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.feat_group_num, net.op_start_pos, program, selected_feat_logit, init=init)

    if len(net.removed_feat_ls) > 0:
        selected_Q_feat, selected_feat_probs = down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, net.removed_feat_ls)

    if not eval:
        if not is_ppo:
            if init:
                _, selected_feat_col = torch.topk(selected_feat_probs, k=net.topk_act, dim=-1)
            else:
                # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
                _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=net.topk_act, dim=-1)
        else:
            if train:
                selected_feat_col = torch.multinomial(selected_feat_probs.view(len(selected_feat_probs),-1), net.topk_act, replacement=False)
            else:
                # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
                _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=net.topk_act, dim=-1)

    else:
        selected_feat_col = atom[col_id_key]

    selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
    prev_program_ids = None
    curr_selected_feat_col = None

    if init:
        selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, net.topk_act, 1)
        for k in range(net.topk_act):
            selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        hx = torch.cat([hx.unsqueeze(1).repeat(1, net.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
    else:
        if not eval:
            if net.feat_group_names is None:
                prev_program_ids = torch.div(selected_feat_col, net.num_feat_len, rounding_mode='floor')
                # prev_program_ids = (selected_feat_col//net.num_feat_len)
                curr_selected_feat_col = selected_feat_col%net.num_feat_len
            else:
                prev_program_ids = torch.div(selected_feat_col, net.feat_group_num, rounding_mode='floor')
                # prev_program_ids = (selected_feat_col//net.num_feat_len)
                curr_selected_feat_col = selected_feat_col%net.feat_group_num
        else:
            prev_program_ids = atom[prev_prog_key]
            curr_selected_feat_col = atom[col_id_key]
        new_hx = []
        seq_ids = torch.arange(pat_count)
        for k in range(net.topk_act):
            selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], selected_feat_probs[seq_ids,prev_program_ids[:,k]]*selected_feat_col_onehot[seq_ids, prev_program_ids[:,k]]],dim=-1))

        hx = torch.stack(new_hx, dim=1)    

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
    #         pred = torch.rand([pat_count, net.topk_act, discretize_feat_value_count], device=DEVICE)
    #     else:
    #         pred = net.token_nets["constant"](hx)
    # else:
    #     pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
    #         selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=DEVICE)
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_ranges_ls = []
    if init:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            # if net.feat_group_names is None:
            selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
            selected_ranges_ls.append([net.feat_bound_mappings[col] for col in selected_col_ls[-1]])
            # else:
            #     selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
    else:
        curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            # if net.feat_group_names is None:
            selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] for k in range(len(curr_selected_feat_col_ls[idx]))])
            selected_ranges_ls.append([net.feat_bound_mappings[col] for col in selected_col_ls[-1]])
            # else:
            #     selected_col_ls.append([net.feat_group_names[curr_selected_feat_col_ls[idx][k]][0] for k in range(len(curr_selected_feat_col_ls[idx]))])

    feat_val = []
    final_selected_range_ls = []
    for idx in range(len(selected_col_ls)):
        feat_val.append(torch.tensor([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(DEVICE))
        final_selected_range_ls.append([identify_range(selected_ranges_ls[idx][k], feat_val[k]) for k in range(len(selected_ranges_ls[idx]))])
        
        

    feat_val = torch.stack(feat_val, dim=0)

    
    # if not eval:
    #     pred_out_mask =  mask_atom_representation_for_op0(net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, init=init)
    # else:
    #     pred_out_mask =  mask_atom_representation_for_op0(net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,init=init)

    # pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    # pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    # if net.prefer_smaller_range:
        
    #     regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
    #     regularized_coeff = torch.stack(regularized_coeff, dim=1)
    #     pred_probs_vals = pred_probs_vals*regularized_coeff
    #     pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    # if not eval:
    #     if not is_ppo:
    #         selected_op = torch.argmax(selected_op_probs, dim=-1)
    #     else:
    #         if train:
    #             dist = torch.distributions.Categorical(selected_op_probs)
    #             selected_op = dist.sample()
    #         else:
    #             selected_op = torch.argmax(selected_op_probs, dim=-1)
    # else:
    #     selected_op = atom[op_id_key]
    
    # selected_op_onehot = torch.zeros_like(selected_op_probs)
    # for k in range(net.topk_act):
    #     selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
    # hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
     
    ret = {}
    

    

        # pred_lb = pred_lb*(range_max - range_min) + range_min
    # if not eval:
        


    #     # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

    #     # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
    #     if not is_ppo:
    #         argmax = torch.argmax(pred_probs_vals,dim=-1)
            
    #         pred_v = argmax/(discretize_feat_value_count-1)
    #         # # __ge__
    #         # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
    #         # # __le__
    #         # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
    #     else:
    #         if not net.continue_act:
    #             if train:
    #                 dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
    #                 argmax = dist.sample()
    #             else:
    #                 argmax = torch.argmax(pred, dim=-1)

    #             # argmax = argmax.cpu().numpy()

    #             pred_v = argmax/(discretize_feat_value_count-1)
    #             # # __ge__
    #             # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
    #             # # __le__
    #             # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
    #         else:
    #             pred = torch.clamp(pred, min=1e-6, max=1)
    #             if train:
                    
    #                 dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

    #                 # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
    #                 argmax = torch.clamp(dist.sample(), min=0, max=1)
    #             else:
    #                 argmax = pred[:,:,0]
    #             # argmax = argmax.cpu().numpy()

    #             pred_v = argmax/(discretize_feat_value_count-1)

    #             # # __ge__
    #             # pred_v1 = (feat_val)*(argmax)
    #             # # __le__
    #             # pred_v2 = (1 - feat_val)*(argmax) + feat_val
    #     pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
    #     pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        
    #     selected_op = (pred_v <= feat_val).type(torch.long)

    #     selected_op_ls = []
            
    #     selected_op_id_ls = selected_op.cpu().tolist()

    #     for idx in range(len(selected_op_id_ls)):
    #         selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    if eval:
        # ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        # ret[op_probs_key] = selected_op_probs
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred_probs_vals
        # else:
        #     ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        # ret[pred_Q_key] = pred_Q_vals.data#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        # ret[op_probs_key] = selected_op_probs.data
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred.data
        # else:
        #     ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

        # ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        # ret[op_key] = selected_op_ls        
        
        # ret[op_id_key] = selected_op.data
        ret[range_key] = final_selected_range_ls
        ret[range_tensor_key] = normalize_range(final_selected_range_ls, selected_col_ls, net.feat_range_mappings)
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

    
    
    return ret



class RLSynthesizerNetwork_mlp0(nn.Module):
    def init_without_feat_groups(self, lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=1, continue_act=False):
        super(RLSynthesizerNetwork_mlp0, self).__init__()
        self.topk_act=topk_act
        self.lang = lang
        self.program_max_len=program_max_len
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        self.continue_act = continue_act
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
            # else:
            #     self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
            #     self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            # if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
            if not (decision in self.lang.syntax["num_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            # if not decision in self.lang.syntax["num_feat"]:
            #     for option in list(options_dict.keys()):        
            #         if self.op_start_pos < 0:
            #             self.op_start_pos = self.ATOM_VEC_LENGTH
                    
            #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
            #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
            #         self.ATOM_VEC_LENGTH += 1
            # else:
            if self.num_start_pos < 0:
                self.num_start_pos = self.ATOM_VEC_LENGTH
            
            self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
            self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 2

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        self.num_feat_len = num_feat_len
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        self.num_features = num_features
        self.full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.feat_to_num_mappings = dict()
        # self.op_to_num_mappings = dict()
        # # feat_idx = 0
        # # for col in self.lang.syntax["num_feat"]:
        # for feat_idx in range(len(self.column_ls)):
        #     self.feat_to_num_mappings[self.column_ls[feat_idx]] = feat_idx
        #     # feat_idx += 1
        # op_idx = 0
        
        # # self.op_list = list(self.lang.syntax["num_op"].keys())
        # self.op_list=[operator.__le__, operator.__ge__]
        
        # for op_idx in range(len(self.op_list)):
        #     self.op_to_num_mappings[self.op_list[op_idx]] = op_idx
        #     # op_idx += 1
        self.op_num = len(self.lang.syntax["num_op"])

        # for i,v in self.lang.syntax.items():
        #     if i == "num_op":
        #         continue
        #     # if i in self.lang.syntax["num_feat"]:
        #     #     continue
            
        #     # if not i == "num_feat":
        #     #     # net_maps[i] = TokenNetwork(latent_size, len(v))
        #     #     net_maps[i] = TokenNetwork(latent_size, len(v))
        #     # else:
        #     #     net_maps[i] = TokenNetwork(latent_size, len(v))
        #     #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
        #     #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                
        #     #     net_maps[i] = TokenNetwork(latent_size, len(v))
        #     #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
        #     #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
        #     # if not i in self.lang.syntax["num_feat"]:
        #     #     net_maps[i] = TokenNetwork(latent_size, len(v))
        #     # else:
        #     if latent_size > 0:
        #         if not continue_act:
        #             net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, discretize_feat_value_count)
        #         else:
        #             net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, 1)
        #     else:
        #         if not continue_act:
        #             net_maps["constant"] = nn.Linear(full_input_size + self.num_feat_len, discretize_feat_value_count)
        #         else:
        #             net_maps["constant"] = nn.Linear(full_input_size + self.num_feat_len, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        # self.token_nets = nn.ModuleDict(net_maps)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(self.full_input_size, latent_size, self.num_feat_len)
            # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(self.full_input_size, self.num_feat_len)
            # self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)
    
    # def init_with_feat_groups(self, lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=1, continue_act=False, feat_group_names=None):
    #     super(RLSynthesizerNetwork_mlp0, self).__init__()
    #     self.topk_act=topk_act
    #     self.lang = lang
    #     self.program_max_len=program_max_len
    #     self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
    #     self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
    #     self.grammar_token_to_pos = {}
    #     self.grammar_pos_to_token = {}
    #     self.ATOM_VEC_LENGTH = 0
    #     self.one_hot_token_bounds = {}
    #     self.continue_act = continue_act
    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     start = self.ATOM_VEC_LENGTH
    #     #     for option in list(options_dict.keys()):
    #     #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.feat_range_mappings = feat_range_mappings
    #     for i,v in self.lang.syntax.items():
    #         if not i in self.lang.syntax["num_feat"]:
    #             self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
    #             self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
    #     #     else:
    #     #         self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
    #     #         self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

    #     self.op_start_pos = -1
    #     self.num_start_pos = -1

    #     decision = "num_op"
    #     start = self.ATOM_VEC_LENGTH


    #     for option in list(self.lang.syntax[decision].keys()):        
    #         if self.op_start_pos < 0:
    #             self.op_start_pos = self.ATOM_VEC_LENGTH
            
    #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #         self.ATOM_VEC_LENGTH += 1
    #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

    #     for group_idx in range(len(feat_group_names)):
    #         start = self.ATOM_VEC_LENGTH
    #         if self.num_start_pos < 0:
    #                 self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #         self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
    #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
    #         self.ATOM_VEC_LENGTH += 1
    #         self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)
        
        
        
    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
    #     #         continue
    #     #     # if decision == "num_op":
    #     #     #     continue
    #     #     start = self.ATOM_VEC_LENGTH


    #     #     if not decision in self.lang.syntax["num_feat"]:
    #     #         for option in list(options_dict.keys()):        
    #     #             if self.op_start_pos < 0:
    #     #                 self.op_start_pos = self.ATOM_VEC_LENGTH
                    
    #     #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #             self.ATOM_VEC_LENGTH += 1
    #     #     else:
    #     #         if self.num_start_pos < 0:
    #     #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #     #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
    #     self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
    #     self.ATOM_VEC_LENGTH += 1

    #     # self.column_ls = list(self.lang.syntax["num_feat"].keys())

    #     num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
    #     cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
    #     num_features = num_feat_len+cat_feat_len
    #     self.num_feat_len = num_features
    #     self.feat_group_num = len(feat_group_names)
        
    #     # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
    #     # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

    #     # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

    #     # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
    #     net_maps = {}
    #     full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
    #     # self.feat_to_num_mappings = dict()
    #     # self.op_to_num_mappings = dict()
    #     # # feat_idx = 0
    #     # # for col in self.lang.syntax["num_feat"]:
    #     # for feat_idx in range(len(self.column_ls)):
    #     #     self.feat_to_num_mappings[self.column_ls[feat_idx]] = feat_idx
    #     #     # feat_idx += 1
    #     # op_idx = 0
        
    #     # # self.op_list = list(self.lang.syntax["num_op"].keys())
    #     # self.op_list=[operator.__le__, operator.__ge__]
        
    #     # for op_idx in range(len(self.op_list)):
    #     #     self.op_to_num_mappings[self.op_list[op_idx]] = op_idx
    #     #     # op_idx += 1
    #     self.op_num = len(self.lang.syntax["num_op"])

    #     for i,v in self.lang.syntax.items():
    #         if i == "num_op":
    #             continue
    #         # if i in self.lang.syntax["num_feat"]:
    #         #     continue
            
    #         # if not i == "num_feat":
    #         #     # net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         # if latent_size > 0:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, 1)
    #         # else:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, 1)
                    
    #         # if latent_size > 0:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, 1)
    #         # else:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, 1)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

    #     # self.token_nets = nn.ModuleDict(net_maps)
        
        
    #     if latent_size > 0:
    #         self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.feat_group_num)
    #         # self.op_selector = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, self.op_num)
    #     else:
    #         self.feat_selector = nn.Linear(full_input_size, self.feat_group_num)
    #         # self.op_selector = nn.Linear(full_input_size + self.feat_group_num, self.op_num)
        
    #     self.to(device=DEVICE)

    def __init__(self, lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=1, continue_act=False, feat_group_names=None, removed_feat_ls=None, prefer_smaller_range=False, prefer_smaller_range_coeff=0.5, args = None):
        self.feat_group_names = feat_group_names
        self.removed_feat_ls = removed_feat_ls
        self.prefer_smaller_range=prefer_smaller_range
        self.prefer_smaller_range_coeff = prefer_smaller_range_coeff
        # if self.prefer_smaller_range:
        #     self.selected_vals = torch.tensor([k/(discretize_feat_value_count-1) for k in range(discretize_feat_value_count)]).to(DEVICE)
        # if feat_group_names is None:
        self.feat_bound_mappings = args.feat_bound_mappings
        self.init_without_feat_groups(lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=topk_act, continue_act=continue_act)
        # else:
        #     self.init_with_feat_groups(lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=topk_act, continue_act=continue_act, feat_group_names=feat_group_names)

    # def prediction_to_atom(self, pred:dict):
    #     return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
    def prediction_to_atom(self, pred:dict):
        res = {}
        for i,v in pred.items():
            if not self.get_prefix(i) in self.lang.syntax["num_feat"]:
                res[i] = self.grammar_num_to_token_val[i][torch.argmax(v).item()]
            else:
                # res[i] = [self.grammar_num_to_token_val[i.split("_")[0]][torch.argmax(v[1]).item()], v[0]]
                res[i] = v
                # res[i] = [[v[0][0], v[0][1]],v[1]]
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

    def prediction_to_atom_ls3(self, batch_size, atom):
        atom1 = []
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size

                for selected_item in atom[k][2]:
                    sample_ids = atom[k][2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        atom1[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][selected_item][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        return atom1

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
            else:
                for selected_item in v[2]:
                    sample_ids = v[2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        res_ls[sample_ids[sample_id_id].item()][selected_item] = v[0][selected_item][sample_id_id]

                # res[i] = v

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

            if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
            else:
                # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                if token.endswith("_lb"):
                    one_hot_pos.append(self.grammar_token_to_pos[("num_op", operator.__ge__)])
                    # one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val[0]))
                    # one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
                elif token.endswith("_ub"):
                    one_hot_pos.append(self.grammar_token_to_pos[("num_op", operator.__le__)])
                if type(token_val) is tuple or type(token_val) is list:
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val[0]))
                else:
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val))
                    # one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
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
                # if token.endswith("_prob"):
                #     continue

                if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                    one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
                else:
                    # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], atom[token]))
            ret = [0.0]*self.ATOM_VEC_LENGTH
            for i in one_hot_pos:
                if type(i) is tuple:
                    ret[i[0]] = i[1]
                    # ret[i[0] + 1] = i[1][1]
                else:
                    ret[i] = 1
            ret_tensor_ls.append(torch.FloatTensor(ret))
        return ret_tensor_ls

    def atom_to_vector_ls0(self, atom_ls):
        return atom_to_vector_ls0_main(self, atom_ls)
        # ret_tensor_ls = []
        # pred_v_arr = atom_ls[pred_v_key]
        
        # col_id_tensor = atom_ls[col_id_key]
        
        # op_id_tensor = atom_ls[op_id_key]
        
        
        
        # ret_tensor_ls = torch.zeros([len(pred_v_arr), self.ATOM_VEC_LENGTH])
        
        # # ret_tensor_ls[:,self.grammar_token_to_pos[("num_op", op)]]=1
        
        # sample_id_tensor = torch.arange(len(ret_tensor_ls),device=DEVICE)
        
        # ret_tensor_ls[sample_id_tensor,self.num_start_pos + col_id_tensor]=1
        
        # ret_tensor_ls[sample_id_tensor,self.op_start_pos + op_id_tensor]=1
        
        # ret_tensor_ls[:, self.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr).view(-1)
        
        # for atom in atom_ls:    
        #     one_hot_pos = []
        #     one_hot_pos.append(self.grammar_token_to_pos[("num_op", op)])
        #     one_hot_pos.append(self.grammar_token_to_pos[col])
        #     one_hot_pos.append((self.ATOM_VEC_LENGTH-1, atom[pred_v_key]))
        #     # for token, token_val in atom.items():
        #     #     # if token.endswith("_prob"):
        #     #     #     continue

        #     #     if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
        #     #         one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
        #     #     else:
        #     #         # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
        #     #         one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], atom[token]))
        #     ret = [0.0]*self.ATOM_VEC_LENGTH
        #     for i in one_hot_pos:
        #         if type(i) is tuple:
        #             ret[i[0]] = i[1]
        #             # ret[i[0] + 1] = i[1][1]
        #         else:
        #             ret[i] = 1
        #     ret_tensor_ls.append(torch.FloatTensor(ret))
        # return ret_tensor_ls

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

    def get_prefix(self, token):
        if token.endswith("_lb"):
            return token.split("_lb")[0]
        elif token.endswith("_ub"):
            return token.split("_ub")[0]
        
        else:
            return token

    def mask_atom_representation(self, program_ls, feat_pred_logit):
        
        op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
        op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
        
        for program in program_ls:
            op1_feat_occur_mat += program[:,self.op_start_pos:self.op_start_pos+1].to(DEVICE)*program[:,self.op_start_pos+2:-1].to(DEVICE)
            op2_feat_occur_mat += program[:,self.op_start_pos+1:self.op_start_pos+2].to(DEVICE)*program[:,self.op_start_pos+2:-1].to(DEVICE)
        
        feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
        
        feat_pred_probs = torch.softmax(feat_pred_logit, dim=-1) + 1e-6

        feat_pred_Q = torch.tanh(feat_pred_logit)

        feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2).float()

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (min_Q_val)*(feat_occur_count_mat >= 2).float()
        
        return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat
    
    def mask_atom_representation_for_op(self, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, op_pred_logic):
        
        assert len(torch.unique(op1_feat_occur_mat)) <= 2
        assert len(torch.unique(op2_feat_occur_mat)) <= 2
        
        # op_pred_probs[:,0] = op_pred_probs[:,0] * (1 - (op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx]))
        # op_pred_probs[:,1] = op_pred_probs[:,1] * (1 - (op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]))
        
        op_pred_probs = torch.softmax(op_pred_logic ,dim=-1) + 1e-6

        mask  =torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]],dim=-1)

        op_pred_probs = op_pred_probs*(1-mask)
        
        op_pred_Q = torch.tanh(op_pred_logic)

        op_pred_Q = op_pred_Q*(1-mask) + (min_Q_val)*mask


        assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
        
        return op_pred_probs, op_pred_Q
        
        
    def forward_ls0(self, features,X_pd_full, program, atom, epsilon=0, eval=False, existing_atom=None, init=False, is_ppo=False, train=False):
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
        
        
        total_feat_prog_array_len =features[0].shape[0] + self.program_max_len*program[0].shape[-1]

        # selecting feature
        
        concat_program_tensor = torch.cat(program,dim=-1)
        
        # hx = torch.zeros(features.shape[0], total_feat_prog_array_len, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor.view(concat_program_tensor.shape[0],-1)
        
        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([features.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:features[0].shape[0]] = features
            hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)


        # if not eval:
        #     if np.random.rand() < epsilon and not is_ppo:
        #         if self.feat_group_names is None:
        #             output_num = self.num_feat_len
        #         else:
        #             output_num = self.feat_group_num

        #         if init:
        #             selected_feat_logit = torch.rand([pat_count, output_num], device=DEVICE)
        #         else:
        #             selected_feat_logit = torch.rand([pat_count,self.topk_act, output_num], device=DEVICE)
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         # selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
        #     else:
        #         selected_feat_logit = self.feat_selector(hx)
        # else:
        #     selected_feat_logit = self.feat_selector(hx)
        
        return forward_main1(self, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train)

class RLSynthesizerNetwork_transformer0(nn.Module):
    def init_without_feat_groups2(self,lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False):
        super(RLSynthesizerNetwork_transformer0, self).__init__()
        self.topk_act = topk_act
        self.lang = lang
        self.program_max_len=program_max_len
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        self.continue_act = continue_act
        # self.removed_feat_ls = removed_feat_ls
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
            # else:
            #     self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
            #     self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not decision in self.lang.syntax["num_feat"]:
                for option in list(options_dict.keys()):        
                    if self.op_start_pos < 0:
                        self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:
                if self.num_start_pos < 0:
                    self.num_start_pos = self.ATOM_VEC_LENGTH
                
                self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        self.num_feat_len = num_feat_len
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.feat_to_num_mappings = dict()
        # self.op_to_num_mappings = dict()
        # # feat_idx = 0
        # # for col in self.lang.syntax["num_feat"]:
        # for feat_idx in range(len(self.column_ls)):
        #     self.feat_to_num_mappings[self.column_ls[feat_idx]] = feat_idx
        #     # feat_idx += 1
        # op_idx = 0
        
        # # self.op_list = list(self.lang.syntax["num_op"].keys())
        # self.op_list=[operator.__le__, operator.__ge__]
        
        # for op_idx in range(len(self.op_list)):
        #     self.op_to_num_mappings[self.op_list[op_idx]] = op_idx
        #     # op_idx += 1
        self.op_num = len(self.lang.syntax["num_op"])

        # tf_latent = 30
        
        if not has_embeddings:
            self.input_embedding = FTTransformer(
                    categories = category_count,      # tuple containing the number of unique values within each category
                    num_continuous = numeric_count,                # number of continuous values
                    dim = tf_latent_size,                           # dimension, paper set at 32
                    dim_out = 1,                        # binary prediction, but could be anything
                    depth = 6,                          # depth, paper recommended 6
                    heads = 8,                          # heads, paper recommends 8
                    attn_dropout = 0.1,                 # post-attention dropout
                    ff_dropout = 0.1                    # feed forward dropout
                )

            if pretrained_model_path is not None:
                self.input_embedding.load_state_dict(torch.load(pretrained_model_path))

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
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, discretize_feat_value_count)
            # if latent_size > 0:
            #     if not continue_act:
            #         net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, latent_size, discretize_feat_value_count)
            #     else:
            #         net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, latent_size, 1)
            # else:
            #     if not continue_act:
            #         net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, discretize_feat_value_count)
            #     else:
            #         net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        # self.token_nets = nn.ModuleDict(net_maps)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.num_feat_len)
            # self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)
            # self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


        # self.token_nets = nn.ModuleDict(net_maps)
        
        # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

        # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)


    # def init_with_feat_groups2(self,lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False, feat_group_names=None):
    #     super(RLSynthesizerNetwork_transformer0, self).__init__()
    #     self.topk_act = topk_act
    #     self.lang = lang
    #     self.program_max_len=program_max_len
    #     self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
    #     self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
    #     self.grammar_token_to_pos = {}
    #     self.grammar_pos_to_token = {}
    #     self.ATOM_VEC_LENGTH = 0
    #     self.one_hot_token_bounds = {}
    #     self.continue_act = continue_act
    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     start = self.ATOM_VEC_LENGTH
    #     #     for option in list(options_dict.keys()):
    #     #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.feat_range_mappings = feat_range_mappings
    #     for i,v in self.lang.syntax.items():
    #         if not i in self.lang.syntax["num_feat"]:
    #             self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
    #             self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
    #         # else:
    #         #     self.grammar_num_to_token_val[i] = list(range(discretize_feat_value_count))
    #         #     self.grammar_token_val_to_num[i] = list(range(discretize_feat_value_count))

    #     self.op_start_pos = -1
    #     self.num_start_pos = -1

    #     decision = "num_op"
    #     start = self.ATOM_VEC_LENGTH


    #     for option in list(self.lang.syntax[decision].keys()):        
    #         if self.op_start_pos < 0:
    #             self.op_start_pos = self.ATOM_VEC_LENGTH
            
    #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #         self.ATOM_VEC_LENGTH += 1
    #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

    #     for group_idx in range(len(feat_group_names)):
    #         start = self.ATOM_VEC_LENGTH
    #         if self.num_start_pos < 0:
    #                 self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #         self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
    #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
    #         self.ATOM_VEC_LENGTH += 1
    #         self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)


    #     # for group_idx in range(len(feat_group_names)):
    #     #     start = self.ATOM_VEC_LENGTH
    #     #     if self.num_start_pos < 0:
    #     #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #     #     self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
    #     #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
    #     #     self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)

    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     start = self.ATOM_VEC_LENGTH


    #     #     for option in list(options_dict.keys()):        
    #     #         if self.op_start_pos < 0:
    #     #             self.op_start_pos = self.ATOM_VEC_LENGTH
                
    #     #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
    #     #         continue
    #     #     # if decision == "num_op":
    #     #     #     continue
    #     #     start = self.ATOM_VEC_LENGTH


    #     #     if not decision in self.lang.syntax["num_feat"]:
    #     #         for option in list(options_dict.keys()):        
    #     #             if self.op_start_pos < 0:
    #     #                 self.op_start_pos = self.ATOM_VEC_LENGTH
                    
    #     #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #             self.ATOM_VEC_LENGTH += 1
    #     #     else:
    #     #         if self.num_start_pos < 0:
    #     #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #     #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
    #     self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
    #     self.ATOM_VEC_LENGTH += 1

    #     # self.column_ls = list(self.lang.syntax["num_feat"].keys())

    #     # num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
    #     # cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
    #     num_features = len(feat_group_names)# num_feat_len+cat_feat_len
    #     self.num_feat_len = num_features
    #     self.feat_group_num = len(feat_group_names)
        
    #     # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
    #     # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

    #     # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

    #     # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
    #     net_maps = {}
    #     full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
    #     # self.feat_to_num_mappings = dict()
    #     # self.op_to_num_mappings = dict()
    #     # # feat_idx = 0
    #     # # for col in self.lang.syntax["num_feat"]:
    #     # for feat_idx in range(len(self.column_ls)):
    #     #     self.feat_to_num_mappings[self.column_ls[feat_idx]] = feat_idx
    #     #     # feat_idx += 1
    #     # op_idx = 0
        
    #     # # self.op_list = list(self.lang.syntax["num_op"].keys())
    #     # self.op_list=[operator.__le__, operator.__ge__]
        
    #     # for op_idx in range(len(self.op_list)):
    #     #     self.op_to_num_mappings[self.op_list[op_idx]] = op_idx
    #     #     # op_idx += 1
    #     self.op_num = len(self.lang.syntax["num_op"])

    #     # tf_latent = 30
        
    #     if not has_embeddings:
    #         self.input_embedding = FTTransformer(
    #                 categories = category_count,      # tuple containing the number of unique values within each category
    #                 num_continuous = numeric_count,                # number of continuous values
    #                 dim = tf_latent_size,                           # dimension, paper set at 32
    #                 dim_out = 1,                        # binary prediction, but could be anything
    #                 depth = 6,                          # depth, paper recommended 6
    #                 heads = 8,                          # heads, paper recommends 8
    #                 attn_dropout = 0.1,                 # post-attention dropout
    #                 ff_dropout = 0.1                    # feed forward dropout
    #             )

    #         if pretrained_model_path is not None:
    #             self.input_embedding.load_state_dict(torch.load(pretrained_model_path))

    #     for i,v in self.lang.syntax.items():
    #         if i == "num_op":
    #             continue
    #         # if i in self.lang.syntax["num_feat"]:
    #         #     continue
            
    #         # if not i == "num_feat":
    #         #     # net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, discretize_feat_value_count)
    #         # if latent_size > 0:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, 1)
    #         # else:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, 1)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

    #     # self.token_nets = nn.ModuleDict(net_maps)
        
        
    #     if latent_size > 0:
    #         self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.feat_group_num)
    #         # self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, self.op_num)
    #     else:
    #         self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.feat_group_num)
    #         # self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, self.op_num)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


    #     # self.token_nets = nn.ModuleDict(net_maps)
        
    #     # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

    #     # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
    #     self.to(device=DEVICE)



    def __init__(self, lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False, feat_group_names=None, removed_feat_ls=None, prefer_smaller_range=False, prefer_smaller_range_coeff=0.5):
        self.feat_group_names = feat_group_names
        self.prefer_smaller_range = prefer_smaller_range
        self.prefer_smaller_range_coeff = prefer_smaller_range_coeff
        self.removed_feat_ls = removed_feat_ls
        # if self.prefer_smaller_range:
        #     self.selected_vals = torch.tensor([k/(discretize_feat_value_count-1) for k in range(discretize_feat_value_count)]).to(DEVICE)
        # self.method_two = method_two
        # if not method_two:
        #     if feat_group_names is None:
        #         self.init_without_feat_groups(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act)
        #     else:
        #         self.init_with_feat_groups(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act, feat_group_names=feat_group_names)
        # else:
        # if feat_group_names is None:
        self.init_without_feat_groups2(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act)
        # else:
        #     self.init_with_feat_groups2(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act, feat_group_names=feat_group_names)
    # def prediction_to_atom(self, pred:dict):
    #     return {i:self.grammar_num_to_token_val[i][torch.argmax(v).item()] for i,v in pred.items()}
    def prediction_to_atom(self, pred:dict):
        res = {}
        for i,v in pred.items():
            if not self.get_prefix(i) in self.lang.syntax["num_feat"]:
                res[i] = self.grammar_num_to_token_val[i][torch.argmax(v).item()]
            else:
                # res[i] = [self.grammar_num_to_token_val[i.split("_")[0]][torch.argmax(v[1]).item()], v[0]]
                res[i] = v
                # res[i] = [[v[0][0], v[0][1]],v[1]]
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

    def prediction_to_atom_ls3(self, batch_size, atom):
        atom1 = []
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size

                for selected_item in atom[k][2]:
                    sample_ids = atom[k][2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        atom1[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][selected_item][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        return atom1

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
            else:
                for selected_item in v[2]:
                    sample_ids = v[2][selected_item]
                    for sample_id_id in range(len(sample_ids)):
                        res_ls[sample_ids[sample_id_id].item()][selected_item] = v[0][selected_item][sample_id_id]

                # res[i] = v

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

            if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
            else:
                # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                if token.endswith("_lb"):
                    one_hot_pos.append(self.grammar_token_to_pos[("num_op", operator.__ge__)])
                    # one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val[0]))
                    # one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
                elif token.endswith("_ub"):
                    one_hot_pos.append(self.grammar_token_to_pos[("num_op", operator.__le__)])
                if type(token_val) is tuple or type(token_val) is list:
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val[0]))
                else:
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], token_val))
                    # one_hot_pos.append((self.grammar_token_to_pos[token], atom[token]))
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
                # if token.endswith("_prob"):
                #     continue

                if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                    one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
                else:
                    # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
                    one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], atom[token]))
            ret = [0.0]*self.ATOM_VEC_LENGTH
            for i in one_hot_pos:
                if type(i) is tuple:
                    ret[i[0]] = i[1]
                    # ret[i[0] + 1] = i[1][1]
                else:
                    ret[i] = 1
            ret_tensor_ls.append(torch.FloatTensor(ret))
        return ret_tensor_ls

    def atom_to_vector_ls0(self, atom_ls):
        ret_tensor_ls = []
        pred_v_arr = atom_ls[pred_v_key]
        
        col_id_tensor = atom_ls[col_id_key]
        
        op_id_tensor = atom_ls[op_id_key]
        
        
        
        ret_tensor_ls = torch.zeros([len(pred_v_arr), self.topk_act, self.ATOM_VEC_LENGTH])
        
        # ret_tensor_ls[:,self.grammar_token_to_pos[("num_op", op)]]=1
        
        sample_id_tensor = torch.arange(len(ret_tensor_ls),device=DEVICE)
        
        for k in range(self.topk_act):
            ret_tensor_ls[sample_id_tensor,k, self.num_start_pos + col_id_tensor[:,k]]=1
            ret_tensor_ls[sample_id_tensor,k, self.op_start_pos + op_id_tensor[:,k]]=1
        
        ret_tensor_ls[:, :, self.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)
        
        # for atom in atom_ls:    
        #     one_hot_pos = []
        #     one_hot_pos.append(self.grammar_token_to_pos[("num_op", op)])
        #     one_hot_pos.append(self.grammar_token_to_pos[col])
        #     one_hot_pos.append((self.ATOM_VEC_LENGTH-1, atom[pred_v_key]))
        #     # for token, token_val in atom.items():
        #     #     # if token.endswith("_prob"):
        #     #     #     continue

        #     #     if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
        #     #         one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
        #     #     else:
        #     #         # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
        #     #         one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], atom[token]))
        #     ret = [0.0]*self.ATOM_VEC_LENGTH
        #     for i in one_hot_pos:
        #         if type(i) is tuple:
        #             ret[i[0]] = i[1]
        #             # ret[i[0] + 1] = i[1][1]
        #         else:
        #             ret[i] = 1
        #     ret_tensor_ls.append(torch.FloatTensor(ret))
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

    def get_prefix(self, token):
        if token.endswith("_lb"):
            return token.split("_lb")[0]
        elif token.endswith("_ub"):
            return token.split("_ub")[0]
        
        else:
            return token



    def forward_ls0(self, input_data,X_pd_full, program, atom, epsilon=0, init=False, eval=False, existing_atom=None, is_ppo=False, train=False):
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
        
        if type(input_data) is not torch.Tensor:
            features, feature_embedding = input_data
            features = features.to(DEVICE)
            feature_embedding = feature_embedding.to(DEVICE)
        else:
            features = input_data
            features = features.to(DEVICE)
            feature_embedding = self.input_embedding(None, features, return_embedding=True)
        
        
        pat_count = features.shape[0]
        
        total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]

        concat_program_tensor = torch.cat(program,dim=-1)

        # if len(program) == 1 and len(program.shape) == 2:
        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([feature_embedding.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:feature_embedding[0].shape[0]] = feature_embedding
            hx[:, feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]+feature_embedding[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([feature_embedding.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:feature_embedding[0].shape[0]] = feature_embedding.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]+feature_embedding[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)

        


        # 
        
        # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]*self.topk_act

        # # selecting feature
        
        
        
        # hx = torch.zeros(feature_embedding.shape[0], total_feat_prog_array_len, device=DEVICE)
        # hx[:,0:feature_embedding[0].shape[0]] = feature_embedding
        # hx[:, feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]*self.topk_act+feature_embedding[0].shape[0]] = concat_program_tensor.view(concat_program_tensor.shape[0], -1)

        # if not eval:
        #     if np.random.rand() < epsilon and not is_ppo:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         if self.feat_group_names is None:
        #             output_num = self.num_feat_len
        #         else:
        #             output_num = self.feat_group_num

        #         if init:
        #             selected_feat_logit = torch.rand([pat_count, output_num], device=DEVICE)
        #         else:
        #             selected_feat_logit = torch.rand([pat_count,self.topk_act, output_num], device=DEVICE)

        #         # if init:
        #         #     selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
        #         # else:
        #         #     selected_feat_logit = torch.rand([pat_count,self.topk_act, self.num_feat_len], device=DEVICE)
        #     else:
        #         selected_feat_logit = self.feat_selector(hx)
        
        # else:
        #     selected_feat_logit = self.feat_selector(hx)
        # if not self.method_two:
        #     return forward_main(self, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train)
        # else:
        return forward_main0(self, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train)
        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = self.mask_atom_representation(program, selected_feat_logit)
        
        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(self.topk_act, self.num_feat_len, self.op_start_pos, program, selected_feat_logit, init=init)

        # # selected_Q_feat = selected_feat_probs

        # # selected_feat_probs = selected_feat_probs
        # if not eval:
        #     # if self.topk_act == 1:
        #     #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        #     # else:
        #     if init:
        #         _, selected_feat_col = torch.topk(selected_feat_probs, k=self.topk_act, dim=-1)
        #     else:
        #         # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        #         _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=self.topk_act, dim=-1)
        # else:
        #     selected_feat_col = atom[col_id_key]

        # selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
        # prev_program_ids = None
        # curr_selected_feat_col = None

        # if init:
        #     selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, self.topk_act, 1)
        #     for k in range(self.topk_act):
        #         selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        #     hx = torch.cat([hx.unsqueeze(1).repeat(1, self.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
        # else:
        #     if not eval:
        #         prev_program_ids = (selected_feat_col//self.num_feat_len)
        #         curr_selected_feat_col = selected_feat_col%self.num_feat_len
        #     else:
        #         prev_program_ids = atom[prev_prog_key]
        #         curr_selected_feat_col = atom[col_id_key]
        #     new_hx = []
        #     seq_ids = torch.arange(pat_count)
        #     for k in range(self.topk_act):
        #         selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
        #         new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], selected_feat_probs[seq_ids,prev_program_ids[:,k]]*selected_feat_col_onehot[seq_ids, prev_program_ids[:,k]]],dim=-1))

        #     hx = torch.stack(new_hx, dim=1)
        #     # op1_feat_occur_mat = torch.stack([op1_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
        #     # op2_feat_occur_mat = torch.stack([op2_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
        # # else:

        # #     selected_Q_feat = torch.tanh(self.feat_selector(hx))

        # #     # selected_feat_probs = torch.zeros([pat_count, selected_Q_feat.shape[-1]], device = DEVICE)

        # #     # selected_feat_probs[torch.arange(pat_count, device=DEVICE), atom[col_id_key]]=1

        # #     selected_feat_col = atom[col_id_key]
        
        # # selecting op

        # # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len, device=DEVICE)
        # # hx[:,0:features[0].shape[0]] = features
        # # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        

        # if not eval:
        #     if np.random.rand() < epsilon:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         selected_op_logit = torch.rand([pat_count, self.topk_act, self.op_num], device=DEVICE)
        #     else:
        #         selected_op_logit = self.op_selector(hx)
        
        # else:
        #     selected_op_logit = self.op_selector(hx)
        
        # # selected_op_probs, selected_Q_op = self.mask_atom_representation_for_op(op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
        # if not eval:
        #     selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, prev_program_ids, curr_selected_feat_col,  init=init)
        # else:
        #     selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, atom[prev_prog_key], atom[col_id_key],  init=init)

        #     # selected_op_probs = selected_op_probs, dim=-1)

        #     #  = selected_op_probs
        # if not eval:
        #     selected_op = torch.argmax(selected_op_probs, dim=-1)
        # else:
        #     # selected_Q_op = torch.tanh(self.op_selector(hx))

        #     # selected_op_probs = torch.zeros([pat_count, selected_Q_op.shape[-1]], device = DEVICE)

        #     # selected_op_probs[torch.arange(pat_count, device=DEVICE), atom[op_id_key]]=1

        #     selected_op = atom[op_id_key]
        
        # selected_op_onehot = torch.zeros_like(selected_op_probs)
        # for k in range(self.topk_act):
        #     selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
        
        
        # # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len + self.op_num, device=DEVICE)
        # # hx[:,0:features[0].shape[0]] = features
        # # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        # # hx[:, total_feat_prog_array_len+ self.num_feat_len:total_feat_prog_array_len + self.num_feat_len + self.op_num] = selected_op_probs
        # # hx = torch.cat([features, concat_program_tensor, selected_feat_probs, selected_op_probs], dim=-1)
        # hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
        
        # # feat_encoder = torch.zeros(self.num_feat_len, device = DEVICE)
        # # feat_encoder[self.feat_to_num_mappings[col]] = 1
        # # op_encoder = torch.zeros(self.op_num, device = DEVICE)
        # # op_encoder[self.op_to_num_mappings[op]] = 1

        # # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = feat_encoder
        
        
        # # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # # hx[0:features[0].shape[0]] = features[0]
        # # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        # ret = {}
        
        
        
        # if not eval:
        #     if np.random.rand() < epsilon:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         pred = torch.rand([pat_count, self.topk_act, discretize_feat_value_count], device=DEVICE)
        #     else:
        #         # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         pred = self.token_nets["constant"](hx)
        # else:
        #     pred = self.token_nets["constant"](hx)

        

        #     # pred_lb = pred_lb*(range_max - range_min) + range_min
        # if not eval:
        #     selected_col_ls = []
        #     if init:
        #         selected_feat_col_ls = selected_feat_col.cpu().tolist()

                

        #         for idx in range(len(selected_feat_col_ls)):
        #             selected_col_ls.append([self.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
        #     else:
        #         curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        #         for idx in range(len(prev_program_ids)):
        #             selected_col_ls.append([self.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] for k in range(len(curr_selected_feat_col_ls[idx]))])
            
        #     selected_op_ls = []
            
        #     selected_op_id_ls = selected_op.cpu().tolist()

        #     for idx in range(len(selected_op_id_ls)):
        #         selected_op_ls.append([self.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] for k in range(len(selected_op_id_ls[idx]))])


        #     feat_val = []
        #     for idx in range(len(selected_col_ls)):
        #         feat_val.append(np.array([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))]))

        #     feat_val = np.stack(feat_val, axis=0)


        #     selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        #     op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
            
            
        #     argmax = torch.argmax(pred,dim=-1).cpu().numpy()

        #     # __ge__
        #     pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        #     # __le__
        #     pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

        #     if self.lang.precomputed is not None:
        #         pred_v1, pred_v2 = find_nearest_thres_vals(self.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        #     # if self.op_list[0] == operator.__ge__:     
            
                
        #     pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
        # # else:
        #     # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

        # # if op == operator.__ge__:

        # #     pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # # else:
        # #     pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        
        # if init:
        #     ret[col_id_key] = selected_feat_col
        # else:
        #     ret[col_id_key] = curr_selected_feat_col

        # if eval:
        #     ret[pred_Q_key] = pred
        #     ret[col_Q_key] = selected_Q_feat
        #     ret[op_Q_key] = selected_Q_op
        #     ret[prev_prog_key] = prev_program_ids
        # else:
        #     ret[pred_Q_key] = pred.data
        #     ret[col_Q_key] = selected_Q_feat.data
        #     ret[op_Q_key] = selected_Q_op.data
        
        #     ret[pred_v_key] = pred_v
        
        #     ret[op_key] = selected_op_ls        
            
        #     ret[op_id_key] = selected_op
            
        #     ret[col_key] = selected_col_ls
        #     ret[prev_prog_key] = prev_program_ids

        
        
        # return ret
 