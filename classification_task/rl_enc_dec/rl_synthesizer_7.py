import torch
import logging
from torch import nn, optim
from create_language import *
import numpy as np
import random
from collections import namedtuple, deque
import operator
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from feat_encoder.ft_transformer0 import FTTransformer

DEVICE = "cuda"
print(f"Using {DEVICE} device")
discretize_feat_value_count=10

Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))


col_key = "col"

col_id_key = "col_id"

col_Q_key = "col_probs"

pred_Q_key = "pred_probs"
        
pred_v_key = "pred_v"

op_id_key = "op_id"
        
op_key = "op"

op_Q_key = "op_probs"

prev_prog_key = "prev_prog"

def mask_atom_representation_for_op(topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, op_pred_logic):
        
    assert len(torch.unique(op1_feat_occur_mat)) <= 2
    assert len(torch.unique(op2_feat_occur_mat)) <= 2
    
    # op_pred_probs[:,0] = op_pred_probs[:,0] * (1 - (op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    # op_pred_probs[:,1] = op_pred_probs[:,1] * (1 - (op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    
    op_pred_probs = torch.softmax(op_pred_logic ,dim=-1) + 1e-6

    mask_ls = []

    for k in range(topk_act):
        mask_ls.append(torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx[:,k]], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx[:,k]]],dim=-1))

    # mask = mask.unsqueeze(1).repeat(1,self.topk_act, 1)
    mask = torch.stack(mask_ls, dim=1)

    op_pred_probs = op_pred_probs*(1-mask)
    
    op_pred_Q = torch.tanh(op_pred_logic)

    op_pred_Q = op_pred_Q*(1-mask) + (-2)*mask


    assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
    
    return op_pred_probs, op_pred_Q

def mask_atom_representation_for_op1(topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, op_pred_logic, prev_program_ids, curr_selected_feat_col, init=False):
        
    assert len(torch.unique(op1_feat_occur_mat)) <= 2
    assert len(torch.unique(op2_feat_occur_mat)) <= 2
    
    # op_pred_probs[:,0] = op_pred_probs[:,0] * (1 - (op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    # op_pred_probs[:,1] = op_pred_probs[:,1] * (1 - (op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    
    op_pred_probs = torch.softmax(op_pred_logic ,dim=-1) + 1e-6

    op_pred_Q = torch.tanh(op_pred_logic)

    if not init:
        mask_ls = []

        for k in range(topk_act):
            mask_ls.append(torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),prev_program_ids[:,k],curr_selected_feat_col[:,k]], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),prev_program_ids[:,k],curr_selected_feat_col[:,k]]],dim=-1))

    # mask = mask.unsqueeze(1).repeat(1,self.topk_act, 1)
        mask = torch.stack(mask_ls, dim=1)

        op_pred_probs = op_pred_probs*(1-mask)
        
        

        op_pred_Q = op_pred_Q*(1-mask) + (-2)*mask


    assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
    
    return op_pred_probs, op_pred_Q

def mask_atom_representation(topk_act, num_feat_len, op_start_pos, program_ls, feat_pred_logit):
    
    op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    
    for program in program_ls:
        op1_feat_occur_mat += program[:,:, op_start_pos:op_start_pos+1].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
        op2_feat_occur_mat += program[:,:, op_start_pos+1:op_start_pos+2].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    
    op1_feat_occur_mat = torch.sum(op1_feat_occur_mat,dim=1)
    
    op2_feat_occur_mat = torch.sum(op2_feat_occur_mat, dim=1)
    
    feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
    
    feat_pred_probs = torch.softmax(feat_pred_logit, dim=-1) + 1e-6

    feat_pred_Q = torch.tanh(feat_pred_logit)

    feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2).float()

    feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (-2)*(feat_occur_count_mat >= 2).float()
    
    return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat

def mask_atom_representation1(topk_act, num_feat_len, op_start_pos, program_ls, feat_pred_logit, init=False):
    
    op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    
    if not init:
        for program in program_ls:
            op1_feat_occur_mat += program[:,:, op_start_pos:op_start_pos+1].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
            op2_feat_occur_mat += program[:,:, op_start_pos+1:op_start_pos+2].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    # else:
    #     for program in program_ls:
    #         op1_feat_occur_mat += program[:,:, op_start_pos:op_start_pos+1].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    #         op2_feat_occur_mat += program[:,:, op_start_pos+1:op_start_pos+2].to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
    
    # op1_feat_occur_mat = torch.sum(op1_feat_occur_mat,dim=1)
    
    # op2_feat_occur_mat = torch.sum(op2_feat_occur_mat, dim=1)
    
    feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
    
    feat_pred_probs = torch.softmax(feat_pred_logit, dim=-1) + 1e-6

    feat_pred_Q = torch.tanh(feat_pred_logit)

    if not init:
        feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2).float()

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (-2)*(feat_occur_count_mat >= 2).float()
    
    return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat

# def find_nearest_thres_vals(precomputed, selected_col_ls, pred_v1, pred_v2):
#     # if self.lang.precomputed is not None:
#     selected_col_thres_ls = [precomputed[col] for col in selected_col_ls]
#     new_pred_v1 = [selected_col_thres_ls[idx][np.abs(pred_v1[idx] - selected_col_thres_ls[idx]).argmin()] for idx in range(len(pred_v1))]
#     new_pred_v2 = [selected_col_thres_ls[idx][np.abs(pred_v2[idx] - selected_col_thres_ls[idx]).argmin()] for idx in range(len(pred_v2))]
#     pred_v1 = np.array(new_pred_v1)
#     pred_v2 = np.array(new_pred_v2)
#     return pred_v1, pred_v2

def find_nearest_thres_vals(precomputed, selected_col_ls, pred_v1, pred_v2):
    # if self.lang.precomputed is not None:
    pred_v1_ls = []
    pred_v2_ls = []
    for idx in range(len(selected_col_ls)):
        # for sub_idx in range(len(selected_col_ls[idx])):
            # selected_col_ls[idx][sub_idx]
        selected_col_thres_ls = [precomputed[selected_col_ls[idx][sub_idx]] for sub_idx in range(len(selected_col_ls[idx]))]
        new_pred_v1 = [selected_col_thres_ls[sub_idx][np.abs(pred_v1[idx][sub_idx] - selected_col_thres_ls[sub_idx]).argmin()] for sub_idx in range(len(pred_v1[idx]))]
        new_pred_v2 = [selected_col_thres_ls[sub_idx][np.abs(pred_v2[idx][sub_idx] - selected_col_thres_ls[sub_idx]).argmin()] for sub_idx in range(len(pred_v2[idx]))]


    
        new_pred_v1 = np.array(new_pred_v1)
        new_pred_v2 = np.array(new_pred_v2)

        pred_v1_ls.append(new_pred_v1)
        pred_v2_ls.append(new_pred_v2)
    return np.stack(pred_v1_ls, axis=0), np.stack(pred_v2, axis=0)


def atom_to_vector_ls0_main(net, atom_ls):
    ret_tensor_ls = []
    pred_v_arr = atom_ls[pred_v_key]
    
    col_id_tensor = atom_ls[col_id_key]
    
    op_id_tensor = atom_ls[op_id_key]
    
    
    
    ret_tensor_ls = torch.zeros([len(pred_v_arr), net.topk_act, net.ATOM_VEC_LENGTH])
            
    sample_id_tensor = torch.arange(len(ret_tensor_ls),device=DEVICE)
    
    for k in range(net.topk_act):
        ret_tensor_ls[sample_id_tensor,k, net.num_start_pos + col_id_tensor[:,k]]=1
        ret_tensor_ls[sample_id_tensor,k, net.op_start_pos + op_id_tensor[:,k]]=1
    
    ret_tensor_ls[:, :, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)

    return ret_tensor_ls


def forward_main(net, hx, eval, epsilon, program, selected_feat_logit, atom, pat_count, X_pd_full, init=False):
    selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feat_len, net.op_start_pos, program, selected_feat_logit, init=init)

    # selected_Q_feat = selected_feat_probs

    # selected_feat_probs = selected_feat_probs
    if not eval:
        # if self.topk_act == 1:
        #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        # else:
        if init:
            _, selected_feat_col = torch.topk(selected_feat_probs, k=net.topk_act, dim=-1)
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
            prev_program_ids = (selected_feat_col//net.num_feat_len)
            curr_selected_feat_col = selected_feat_col%net.num_feat_len
        else:
            prev_program_ids = atom[prev_prog_key]
            curr_selected_feat_col = atom[col_id_key]
        new_hx = []
        seq_ids = torch.arange(pat_count)
        for k in range(net.topk_act):
            selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], selected_feat_probs[seq_ids,prev_program_ids[:,k]]*selected_feat_col_onehot[seq_ids, prev_program_ids[:,k]]],dim=-1))

        hx = torch.stack(new_hx, dim=1)
        # op1_feat_occur_mat = torch.stack([op1_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
        # op2_feat_occur_mat = torch.stack([op2_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
    # else:

    #     selected_Q_feat = torch.tanh(self.feat_selector(hx))

    #     # selected_feat_probs = torch.zeros([pat_count, selected_Q_feat.shape[-1]], device = DEVICE)

    #     # selected_feat_probs[torch.arange(pat_count, device=DEVICE), atom[col_id_key]]=1

    #     selected_feat_col = atom[col_id_key]
    
    # selecting op

    # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len, device=DEVICE)
    # hx[:,0:features[0].shape[0]] = features
    # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
    # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
    

    if not eval:
        if np.random.rand() < epsilon:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=DEVICE)
        else:
            selected_op_logit = net.op_selector(hx)
    
    else:
        selected_op_logit = net.op_selector(hx)
    
    # selected_op_probs, selected_Q_op = self.mask_atom_representation_for_op(op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
    if not eval:
        selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, prev_program_ids, curr_selected_feat_col,  init=init)
    else:
        selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, atom[prev_prog_key], atom[col_id_key],  init=init)

        # selected_op_probs = selected_op_probs, dim=-1)

        #  = selected_op_probs
    if not eval:
        selected_op = torch.argmax(selected_op_probs, dim=-1)
    else:
        # selected_Q_op = torch.tanh(self.op_selector(hx))

        # selected_op_probs = torch.zeros([pat_count, selected_Q_op.shape[-1]], device = DEVICE)

        # selected_op_probs[torch.arange(pat_count, device=DEVICE), atom[op_id_key]]=1

        selected_op = atom[op_id_key]
    
    selected_op_onehot = torch.zeros_like(selected_op_probs)
    for k in range(net.topk_act):
        selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
    
    
    # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len + self.op_num, device=DEVICE)
    # hx[:,0:features[0].shape[0]] = features
    # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
    # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
    # hx[:, total_feat_prog_array_len+ self.num_feat_len:total_feat_prog_array_len + self.num_feat_len + self.op_num] = selected_op_probs
    # hx = torch.cat([features, concat_program_tensor, selected_feat_probs, selected_op_probs], dim=-1)
    hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
    
    # feat_encoder = torch.zeros(self.num_feat_len, device = DEVICE)
    # feat_encoder[self.feat_to_num_mappings[col]] = 1
    # op_encoder = torch.zeros(self.op_num, device = DEVICE)
    # op_encoder[self.op_to_num_mappings[op]] = 1

    # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = feat_encoder
    
    
    # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
    # hx[0:features[0].shape[0]] = features[0]
    # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
    
    ret = {}
    
    # hx_out = self.embedding(hx)
    
    
    if not eval:
        if np.random.rand() < epsilon:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, discretize_feat_value_count], device=DEVICE)
        else:
            # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    

        # pred_lb = pred_lb*(range_max - range_min) + range_min
    if not eval:
        selected_col_ls = []
        if init:
            selected_feat_col_ls = selected_feat_col.cpu().tolist()

            

            for idx in range(len(selected_feat_col_ls)):
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
        else:
            curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
            for idx in range(len(prev_program_ids)):
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] for k in range(len(curr_selected_feat_col_ls[idx]))])
        
        selected_op_ls = []
        
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] for k in range(len(selected_op_id_ls[idx]))])


        feat_val = []
        for idx in range(len(selected_col_ls)):
            feat_val.append(np.array([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))]))

        feat_val = np.stack(feat_val, axis=0)


        selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        argmax = torch.argmax(pred,dim=-1).cpu().numpy()

        # __ge__
        pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # __le__
        pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

        if net.lang.precomputed is not None:
            pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    # else:
        # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

    # if op == operator.__ge__:

    #     pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
    # else:
    #     pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
    
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    if eval:
        ret[pred_Q_key] = torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
    else:
        ret[pred_Q_key] = torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        ret[op_Q_key] = selected_Q_op.data
    
        ret[pred_v_key] = pred_v
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op
        
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

    
    
    return ret


class TokenNetwork(nn.Module):
    def __init__(self, input_size, num_output_classes):
        super(TokenNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, num_output_classes),
            nn.Tanh()
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
    def __init__(self, input_size, latent_size, output_size):
        super(TokenNetwork3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, int(output_size)),
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
        # self.gru = nn.RNN(input_size, hidden_size)

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
        # self.gru = nn.RNN(self.hidden_size, self.hidden_size)

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



class RLSynthesizerNetwork7(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size, dropout_p, feat_range_mappings, topk_act=1):
        super(RLSynthesizerNetwork7, self).__init__()
        self.topk_act=topk_act
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
        
        # self.embedding2 = TokenNetwork(latent_size, discretize_feat_value_count)
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
            if latent_size > 0:
                net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feat_len + self.op_num, latent_size, discretize_feat_value_count)
            else:
                net_maps["constant"] = nn.Linear(full_input_size + self.num_feat_len + self.op_num, discretize_feat_value_count)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.num_feat_len)
            self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.num_feat_len)
            self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)

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


    def forward(self, features,X_pd, program, queue, epsilon, eval=False, existing_atom=None):
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

        hx = torch.zeros(features[0].shape[0] + self.program_max_len*program[0].shape[0], device=DEVICE)
        hx[0:features[0].shape[0]] = features[0]
        hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        decoder_input = self.decoder.initInput()
        hx_out = self.embedding(hx)
        existing_atom_to_added=dict()
        while queue:
            hx_out = hx_out.view(1,1,-1)
            # decoder_input = decoder_input.view(1,1,-1)
                        
            token = queue.pop(0)
            if token in ret:
                continue
            if token == "num_op":
                continue
            if self.get_prefix(token) in self.lang.syntax["num_feat"]:
                # if np.random.rand() < epsilon:
                #     sub_keys=["_ub", "_lb"]
                # else:

                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[self.get_prefix(token)]), device=DEVICE)
                else:
                    pred = self.token_nets["constant"](hx_out.view(-1)) 

                # sub_keys=["_lb", "_ub"]
                # pred_ls = []
                # for key in sub_keys:
                #     if np.random.rand() < epsilon:
                #         pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                #     else:
                #         # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                #         # decoder_output = decoder_output.view(-1)
                #         # pred = self.token_nets["num_feat"+key](hx.view(-1))    
                #         pred = self.token_nets[token+key](hx_out.view(-1))    
                #     pred = self.mask_grammar_net_pred(program, token, pred)
                #     pred_ls.append(pred)
            else:
                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                else:
                    # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                    # decoder_output = decoder_output.view(-1)
                    # print(hx)
                    pred = self.token_nets[token](hx_out)
            if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                pred = self.mask_grammar_net_pred(program, token, pred)
                argmax = torch.argmax(pred).item()
                pred_val = self.grammar_num_to_token_val[token][argmax]
                if not eval:
                    if not token == "num_feat":
                        queue.extend(self.lang.syntax[token][pred_val])
                    else:
                        queue.extend([self.lang.syntax[token][pred_val][0] + "_lb", self.lang.syntax[token][pred_val][0] + "_ub"])
                ret[token] = pred
            else:
                feat_val = list(X_pd[self.get_prefix(token)])[0]
                if token.endswith("_lb"):
                    argmax = torch.argmax(pred).item()

                    pred_lb = (feat_val)*(argmax/(discretize_feat_value_count-1))

                    ret[token] = [pred_lb,pred]
                elif token.endswith("_ub"):

                    # range_max = self.feat_range_mappings[token][1]
                    # range_min = self.feat_range_mappings[token][0]

                    

                    

                    # pred_lb = pred_lb*(range_max - range_min) + range_min

                    argmax = torch.argmax(pred).item()

                    pred_ub = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

                    # pred_ub = pred_ub*(range_max - range_min) + range_min
                    # pred_ub = (range_max - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

                    ret[token] = [pred_ub,pred]

                    break
                
            # if eval:
            #     existing_atom_to_added[token] = existing_atom[token]
            #     decoder_input = self.atom_to_vector(self.prediction_to_atom(existing_atom_to_added))

            # else:
            decoder_input = self.atom_to_vector(self.prediction_to_atom(ret))
            
            curr_program = program.copy()
            curr_program.append(decoder_input)
            hx[features[0].shape[0]:len(curr_program)*curr_program[0].shape[0]+features[0].shape[0]] = torch.cat(curr_program)
            hx_out = self.embedding(hx)
            
        del features
        return ret

    def forward_ls0_backup(self, features,X_pd_full, program, queue, epsilon, col, op, eval=False, existing_atom=None):
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
        
        
        hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*program[0].shape[-1], device=DEVICE)
        hx[:,0:features[0].shape[0]] = features
        hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = torch.cat(program,dim=-1)
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        
        hx_out = self.embedding(hx)
        
        if np.random.rand() < epsilon:
            pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        else:
            # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            pred = self.token_nets["constant"](hx_out.view(pat_count, -1))    
        

            # pred_lb = pred_lb*(range_max - range_min) + range_min

        feat_val = np.array(X_pd_full[col])


        argmax = torch.argmax(pred,dim=-1).cpu().numpy()

        if op == operator.__ge__:

            pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        else:
            pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        ret[col_key] = col

        if eval:
            ret[pred_Q_key] = pred
        else:
            ret[pred_Q_key] = pred.data.cpu()
        
        ret[pred_v_key] = pred_v
        
        ret[op_key] = op        
        
        return ret
        
        # decoder_input = self.decoder.initInput()
        
        # existing_atom_to_added=dict()
        # while queue:
        #     hx_out = hx_out.view(1,1,-1)
        #     # decoder_input = decoder_input.view(1,1,-1)
                        
        #     token = queue.pop(0)
        #     if token in ret:
        #         continue
        #     if token == "num_op":
        #         continue
        #     if self.get_prefix(token) in self.lang.syntax["num_feat"]:
        #         # if np.random.rand() < epsilon:
        #         #     sub_keys=["_ub", "_lb"]
        #         # else:

        #         if np.random.rand() < epsilon:
        #             pred = torch.rand(len(self.grammar_num_to_token_val[self.get_prefix(token)]), device=DEVICE)
        #         else:
        #             pred = self.token_nets["constant"](hx_out.view(-1)) 

        #         # sub_keys=["_lb", "_ub"]
        #         # pred_ls = []
        #         # for key in sub_keys:
        #         #     if np.random.rand() < epsilon:
        #         #         pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
        #         #     else:
        #         #         # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
        #         #         # decoder_output = decoder_output.view(-1)
        #         #         # pred = self.token_nets["num_feat"+key](hx.view(-1))    
        #         #         pred = self.token_nets[token+key](hx_out.view(-1))    
        #         #     pred = self.mask_grammar_net_pred(program, token, pred)
        #         #     pred_ls.append(pred)
        #     else:
        #         if np.random.rand() < epsilon:
        #             pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
        #         else:
        #             # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
        #             # decoder_output = decoder_output.view(-1)
        #             # print(hx)
        #             pred = self.token_nets[token](hx_out)
        #     if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
        #         pred = self.mask_grammar_net_pred(program, token, pred)
        #         argmax = torch.argmax(pred).item()
        #         pred_val = self.grammar_num_to_token_val[token][argmax]
        #         if not eval:
        #             if not token == "num_feat":
        #                 queue.extend(self.lang.syntax[token][pred_val])
        #             else:
        #                 queue.extend([self.lang.syntax[token][pred_val][0] + "_lb", self.lang.syntax[token][pred_val][0] + "_ub"])
        #         ret[token] = pred
        #     else:
        #         feat_val = list(X_pd[self.get_prefix(token)])[0]
        #         if token.endswith("_lb"):
        #             argmax = torch.argmax(pred).item()

        #             pred_lb = (feat_val)*(argmax/(discretize_feat_value_count-1))

        #             ret[token] = [pred_lb,pred]
        #         elif token.endswith("_ub"):

        #             # range_max = self.feat_range_mappings[token][1]
        #             # range_min = self.feat_range_mappings[token][0]

                    

                    

        #             # pred_lb = pred_lb*(range_max - range_min) + range_min

        #             argmax = torch.argmax(pred).item()

        #             pred_ub = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

        #             # pred_ub = pred_ub*(range_max - range_min) + range_min
        #             # pred_ub = (range_max - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

        #             ret[token] = [pred_ub,pred]

        #             break
                
        #     # if eval:
        #     #     existing_atom_to_added[token] = existing_atom[token]
        #     #     decoder_input = self.atom_to_vector(self.prediction_to_atom(existing_atom_to_added))

        #     # else:
        #     decoder_input = self.atom_to_vector(self.prediction_to_atom(ret))
            
        #     curr_program = program.copy()
        #     curr_program.append(decoder_input)
        #     hx[features[0].shape[0]:len(curr_program)*curr_program[0].shape[0]+features[0].shape[0]] = torch.cat(curr_program)
        #     hx_out = self.embedding(hx)
            
        # del features
        # return ret

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

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (-2)*(feat_occur_count_mat >= 2).float()
        
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

        op_pred_Q = op_pred_Q*(1-mask) + (-2)*mask


        assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
        
        return op_pred_probs, op_pred_Q
        
        
    def forward_ls0(self, features,X_pd_full, program, atom, epsilon, eval=False, existing_atom=None, init=False):
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


        if not eval:
            if np.random.rand() < epsilon:
                if init:
                    selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
                else:
                    selected_feat_logit = torch.rand([pat_count,self.topk_act, self.num_feat_len], device=DEVICE)
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                # selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
            else:
                selected_feat_logit = self.feat_selector(hx)
        else:
            selected_feat_logit = self.feat_selector(hx)
        
        return forward_main(self, hx, eval, epsilon, program, selected_feat_logit, atom, pat_count, X_pd_full, init=init)

        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = self.mask_atom_representation(program, selected_feat_logit)
        
        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation(self.topk_act, self.num_feat_len, self.op_start_pos, program, selected_feat_logit)

        # # selected_Q_feat = selected_feat_probs

        # # selected_feat_probs = selected_feat_probs
        # if not eval:
        #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        #     # _, selected_feat_col = torch.topk(selected_feat_probs, self.topk_act, dim=-1)
        # else:
        #     selected_feat_col = atom[col_id_key]
        # selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
        # selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), selected_feat_col]=1
        # # else:

        # #     selected_Q_feat = torch.tanh(self.feat_selector(hx))

        # #     selected_feat_probs = torch.zeros([pat_count, selected_Q_feat.shape[-1]], device = DEVICE)

        # #     selected_feat_probs[torch.arange(pat_count, device=DEVICE), atom[col_id_key]]=1

        # #     selected_feat_col = atom[col_id_key]
        
        # # selecting op

        # # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len, device=DEVICE)
        # # hx[:,0:features[0].shape[0]] = features
        # # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs

        # hx = torch.cat([hx, selected_feat_probs*selected_feat_col_onehot], dim=-1)

        # if not eval:
        #     if np.random.rand() < epsilon:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         selected_op_logit = torch.rand([pat_count, self.op_num], device=DEVICE)
        #     else:
        #         selected_op_logit = self.op_selector(hx)
        # else:
        #     selected_op_logit = self.op_selector(hx)

        # selected_op_probs, selected_Q_op = self.mask_atom_representation_for_op(op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
        # # selected_op_probs, selected_Q_op = mask_atom_representation_for_op(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
        
        # # selected_op_probs = selected_op_probs, dim=-1)

        # #  = selected_op_probs
        # if not eval:
        #     selected_op = torch.argmax(selected_op_probs, dim=-1)
        # else:
        #     selected_op = atom[op_id_key]
        
        # selected_op_onehot = torch.zeros_like(selected_op_probs)
        # selected_op_onehot[torch.arange(len(selected_op_probs)), selected_op]=1
        # # else:
        # #     selected_Q_op = torch.tanh(self.op_selector(hx))

        # #     selected_op_probs = torch.zeros([pat_count, selected_Q_op.shape[-1]], device = DEVICE)

        # #     selected_op_probs[torch.arange(pat_count, device=DEVICE), atom[op_id_key]]=1

        # #     selected_op = atom[op_id_key]
        
        
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
        
        # # hx_out = self.embedding(hx)
        
        
        # if not eval:
        #     if np.random.rand() < epsilon:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         pred = torch.rand([pat_count, discretize_feat_value_count], device=DEVICE)
        #     else:
        #         # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
        #         pred = torch.tanh(self.token_nets["constant"](hx.view(pat_count, -1)))
        # else:
        #     pred = torch.tanh(self.token_nets["constant"](hx.view(pat_count, -1)))

        

        #     # pred_lb = pred_lb*(range_max - range_min) + range_min

        # if not eval:

        #     selected_feat_col_ls = selected_feat_col.cpu().tolist()

        #     selected_op_id_ls = selected_op.cpu().tolist()

        #     selected_col_ls = [self.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx]] for idx in range(len(selected_feat_col_ls))]

        #     selected_op_ls = [self.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx]] for idx in range(len(selected_op_id_ls))]

        #     feat_val = np.array([X_pd_full.iloc[idx][selected_col_ls[idx]] for idx in range(len(selected_col_ls))])



        #     selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).view(-1,1)

        #     op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
            
            
        #     argmax = torch.argmax(pred,dim=-1).cpu().numpy()

        #     # __ge__
        #     pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        #     # __le__
        #     pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        #     # if self.op_list[0] == operator.__ge__:     
        #     if self.lang.precomputed is not None:
        #         pred_v1, pred_v2 = find_nearest_thres_vals(self.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
            
        #     pred_v = pred_v1*op_val[:,1] + pred_v2*op_val[:,0]
        # # else:
        #     # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

        # # if op == operator.__ge__:

        # #     pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # # else:
        # #     pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        

        # ret[col_id_key] = selected_feat_col

        # if eval:
        #     ret[pred_Q_key] = pred
        #     ret[col_Q_key] = selected_Q_feat
        #     ret[op_Q_key] = selected_Q_op
        # else:
        #     ret[pred_Q_key] = pred.data
        #     ret[col_Q_key] = selected_Q_feat.data
        #     ret[op_Q_key] = selected_Q_op.data
        
        #     ret[pred_v_key] = pred_v
            
        #     ret[op_key] = selected_op_ls        
            
        #     ret[op_id_key] = selected_op
            
        #     ret[col_key] = selected_col_ls

        
        
        # return ret

    # def forward_ls(self, features,X_pd_ls, program, queue, epsilon, eval=False, replay=False, existing_atom=None):
    #     features = features.to(DEVICE)
    #     pat_count = features.shape[0]
    #     X_pd_full = pd.concat(X_pd_ls)
    #     # hx = self.feat_gru.initHidden()
    #     # hx = hx.repeat(1, pat_count,1)
    #     # # feat_encoder_outputs = torch.zeros(pat_count, self.patient_max_appts, self.feat_gru.hidden_size, device=DEVICE)
    #     # # for ei in range(len(features)):
    #     # feat_out, hx = self.feat_gru(features.view(1, features.shape[0], -1), hx)
    #     # # feat_encoder_outputs[:,0] = feat_out[0,0]
    #     # hx = hx.view(1,pat_count,-1)
    #     # prog_encoder_outputs = torch.zeros(pat_count, self.program_max_len, self.prog_gru.hidden_size, device=DEVICE)
    #     # if len(program) > 0:
    #     #     input_program = torch.stack(program, dim=0)
    #     #     input_program = input_program.to(DEVICE)
    #     #     prog_out, hx = self.prog_gru(input_program, hx)
    #     #     prog_encoder_outputs[:, 0:len(program)] = prog_out.permute((1,0,2))
    #     #     del prog_out, input_program
        
    #     hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*program[0].shape[-1], device=DEVICE)
    #     hx[:,0:features[0].shape[0]] = features
    #     hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = torch.cat(program,dim=-1)
        
    #     # hx = torch.zeros(features.shape[0], features[0].shape[0], device=DEVICE)
    #     # hx[:,0:features[0].shape[0]] = features
    #     # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = torch.cat(program,dim=-1)

        
    #     # for ei in range(len(program)):
    #     #     sub_program = program[ei].view(1, program[ei].shape[0], -1).to(DEVICE)
    #     #     prog_out, hx = self.prog_gru(sub_program, hx)
    #     #     prog_encoder_outputs[:, ei] = prog_out.view(prog_encoder_outputs[:, ei].shape)
    #     #     del prog_out, sub_program
    #     hx_out = self.embedding(hx)
    #     ret = {}
    #     # decoder_input = self.decoder.initInput()
    #     # decoder_input = decoder_input.repeat(pat_count, 1, 1)
    #     existing_atom_to_added=dict()
    #     while queue:
    #         token = queue.pop(0)
            
    #         # decoder_output, hx = self.decoder(decoder_input, hx, feat_out.squeeze(0).unsqueeze(1), prog_encoder_outputs)
    #         # decoder_output = decoder_output.view(len(X_pd_ls), -1)
    #         if type(token) is tuple:
    #         # if token in self.lang.syntax["num_feat"]:
    #             # if np.random.rand() < epsilon:
    #             #     sub_keys=["_ub", "_lb"]
    #             # else:
    #             # hx = hx.squeeze(0)
    #             pred_probs_ls_map = dict()
    #             pred_val_map = dict()
    #             # pred_ub_map = dict()
    #             # token_sample_id_maps = dict()
    #             pred_interval_id_maps = dict()

    #             # if not eval:
    #             # if True:
    #             # for token_key in token:
                    
    #             #     token_sample_ids = pred_val_idx_maps[self.get_prefix(token_key)]
    #                 # token_key  = token_key[0]
    #             sub_keys=["_lb", "_ub"]
    #             # pred_ls = []
    #             # for key in sub_keys:
    #             # for key in sub_keys:
    #             if np.random.rand() < epsilon:
    #                 pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token_key[0]])], device=DEVICE)
    #             else:
    #                 # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
    #                 # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
    #                 # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
    #                 pred = self.token_nets["constant"](hx_out.view(pat_count, -1))    
    #             pred = self.mask_grammar_net_pred_ls(program, token, pred)
    #             # if not replay:
    #             #     pred_ls.append(pred.data.cpu())
    #             #     del pred
    #             # else:
    #             #     pred_ls.append(pred)
                    
                    
    #             # range_max = self.feat_range_mappings[token_key[0]][1]
    #             # range_min = self.feat_range_mappings[token_key[0]][0]

    #             # feat_val = np.array(X_pd_full.iloc[token_sample_ids.view(-1).cpu().numpy()][token_key[0]])
    #             for token_key in token:
    #                 token_sample_ids = pred_val_idx_maps[self.get_prefix(token_key)]
    #                 feat_val = np.array(X_pd_full.iloc[token_sample_ids.view(-1).cpu().numpy()][token_key[0]])

    #                 if token_key.endswith("_lb"):

    #                     argmax = torch.argmax(pred, dim=1)

    #                     pred_val = (feat_val)*(argmax.cpu().numpy()/(discretize_feat_value_count-1))
                    
    #                 # pred_lb = pred_lb*(range_max - range_min) + range_min
    #                 elif token_key.endswith("_ub"):
    #                     argmax = torch.argmax(pred, dim=1)

    #                     pred_val = (1 - feat_val)*(argmax.cpu().numpy()/(discretize_feat_value_count-1)) + feat_val


    #             # pred_ub = pred_ub*(range_max - range_min) + range_min
    #             # pred_ub = (range_max - feat_val)*(argmax2.cpu().numpy()/(discretize_feat_value_count-1)) + feat_val

    #             # print(pred_lb)

    #             # print(feat_val)

    #             # print(pred_ub)

    #             # print()

    #             pred_probs_ls_map[token_key] = pred
    #             pred_val_map[token_key] = pred_val
    #             # pred_ub_map[token_key] = pred_ub
    #             pred_interval_id_maps[token_key] = argmax.cpu()
    #                 # token_sample_id_maps[token_key] = token_sample_ids.cpu()
    #                 # pred_probs_ls[0].extend(pred_ls[0])
    #                 # pred_probs_ls[1].extend(pred_ls[1])
    #                 # pred_lb_ls.append(pred_lb)
    #                 # pred_ub_ls.append(pred_ub)

    #         # else:
    #             # for token_key in token:
    #             #     # token_sample_ids = pred_val_idx_maps[token_key]
    #             #     # token_key  = token_key[0]
    #             #     # sub_keys=["_lb", "_ub"]
    #             #     # pred_ls = []
    #             #     # # for key in sub_keys:
    #             #     # for key in sub_keys:
    #             #         # if np.random.rand() < epsilon:
    #             #         #     pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token_key[0]])])
    #             #         # else:
    #             #         # pred = self.token_nets["num_feat"+key](decoder_output.view(pat_count, -1))    
    #             #         # pred = self.token_nets["num_feat"+key](hx.view(pat_count, -1))    
    #             #     pred = self.token_nets["constant"](hx_out.view(pat_count, -1))   
    #             #     pred = self.mask_grammar_net_pred_ls(program, token, pred)
    #                     # if not replay:
    #                     #     pred_ls.append(pred.data.cpu())
    #                     #     del pred
    #                     # else:
    #                     #     pred_ls.append(pred)
    #             if eval:
    #                 for token_key in token:
                        
    #                     # token_sample_ids = pred_val_idx_maps[self.get_prefix(token_key)]
    #                     # token_key  = token_key[0]
    #                     sub_keys=["_lb", "_ub"]
    #                     # pred_ls = []
    #                     # for key in sub_keys:
    #                     # for key in sub_keys:
    #                     # if np.random.rand() < epsilon:
    #                     #     pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token_key[0]])], device=DEVICE)
    #                     # else:
    #                         # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
    #                         # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
    #                     pred = self.token_nets["constant"](hx_out.view(pat_count, -1))    
    #                     pred = self.mask_grammar_net_pred_ls(program, token, pred)
    #                     if not replay:
    #                         pred_probs_ls_map[token_key] = pred.data.cpu()
    #                         del pred
    #                     else:
    #                         pred_probs_ls_map[token_key] = pred
                        
    #                     # if token_key.endswith("_lb"):
    #                     #     print()

    #             ret[token] = [pred_val_map, pred_probs_ls_map, pred_val_idx_maps, pred_interval_id_maps]

    #             if token_key.endswith("_ub"):
    #                 break
                            
    #         # if not token in self.lang.syntax["num_feat"]:
    #         else:
    #             if token in ret:
    #                 continue
    #             if token == "num_op":
    #                 continue
                    
    #             if np.random.rand() < epsilon:
    #                 pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[token])], device=DEVICE)
    #             else:
    #                 # print(hx)
    #                 # pred = self.token_nets[token](decoder_output.view(pat_count, -1))
    #                 pred = self.token_nets[token](hx_out.view(pat_count, -1))
    #             pred = self.mask_grammar_net_pred_ls(program, token, pred)
                
    #             argmax = torch.argmax(pred, dim=1)
    #             unique_argmax_val = argmax.unique()
    #             pred_val_idx_maps = dict()
    #             for argmax_val in unique_argmax_val:
    #                 argmax_val_idx = torch.nonzero(argmax == argmax_val)
                    
    #                 pred_val = self.grammar_num_to_token_val[token][argmax_val.item()]
                    
    #                 if not token == "num_feat":     
    #                     pred_val_idx_maps[tuple(self.lang.syntax[token][pred_val])] = argmax_val_idx
    #                 else:
    #                     pred_val_idx_maps[self.lang.syntax[token][pred_val][0]] = argmax_val_idx
    #             del argmax

    #             if not eval:
    #                 if not token == "num_feat":                
    #                     queue.extend(list(pred_val_idx_maps.keys())[0])
    #                 else:
    #                     lb_token_list = [next_token[0] + "_lb" for next_token in list(pred_val_idx_maps.keys())]
    #                     ub_token_list = [next_token[0] + "_ub" for next_token in list(pred_val_idx_maps.keys())]
    #                     # queue.append(tuple(list(pred_val_idx_maps.keys())))
    #                     queue.append(tuple(lb_token_list))
    #                     queue.append(tuple(ub_token_list))
                        
    #             # else:
    #             #     print()
    #             # else:
    #             #     # if len(pred_val_idx_maps) == 1:  
    #             #     if not token == "num_feat":              
    #             #         queue.extend(list(pred_val_idx_maps.keys())[0])
    #             #     else:

    #             #         queue.append(tuple(self.lang.syntax[token]))
    #             if not replay:
    #                 ret[token] = pred.data.cpu()
    #                 del pred
    #             else:
    #                 ret[token] = pred
            
    #     #     del decoder_input, decoder_output
    #         # if not type(token) is tuple:
    #         # if eval:
    #         #     existing_atom_to_added[token] = existing_atom[token]
    #         #     decoder_input = torch.stack(self.atom_to_vector_ls(self.prediction_to_atom_ls2(pat_count, existing_atom_to_added)))

    #         # else:
    #         decoder_input = torch.stack(self.atom_to_vector_ls(self.prediction_to_atom_ls2(pat_count, ret)))
    #         # else:
    #         #     decoder_input = torch.stack(self.atom_to_vector_ls(self.prediction_to_atom_ls3(pat_count, ret)))
    #         decoder_input = decoder_input.view(pat_count,-1).to(DEVICE)#.repeat(pat_count, 1, 1)
    #         curr_program = program.copy()
    #         curr_program.append(decoder_input)
    #         del hx, hx_out
    #         hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*program[0].shape[-1], device=DEVICE)
    #         hx[:,0:features[0].shape[0]] = features
    #         hx[:, features[0].shape[0]:len(curr_program)*curr_program[0].shape[-1]+features[0].shape[0]] = torch.cat(curr_program,dim=-1).clone()

    #         # hx[features[0].shape[0]:len(curr_program)*curr_program[0].shape[0]+features[0].shape[0]] = torch.cat(curr_program)
    #         hx_out = self.embedding(hx.detach())
    #     # del features, prog_encoder_outputs, hx, decoder_input, decoder_output, feat_out
    #     return ret


class RLSynthesizerNetwork7_pretrained(nn.Module):
    def __init__(self, lang,  program_max_len, patient_max_appts, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1):
        super(RLSynthesizerNetwork7_pretrained, self).__init__()
        self.topk_act = topk_act
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
        
        # self.embedding2 = TokenNetwork(latent_size, discretize_feat_value_count)
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
            if latent_size > 0:
                net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, latent_size, discretize_feat_value_count)
            else:
                net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, discretize_feat_value_count)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.num_feat_len)
            self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)
            self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


        # self.token_nets = nn.ModuleDict(net_maps)
        # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

        # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)

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


    def forward(self, features,X_pd, program, queue, epsilon, eval=False, existing_atom=None):
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

        hx = torch.zeros(features[0].shape[0] + self.program_max_len*program[0].shape[0], device=DEVICE)
        hx[0:features[0].shape[0]] = features[0]
        hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        decoder_input = self.decoder.initInput()
        hx_out = self.embedding(hx)
        existing_atom_to_added=dict()
        while queue:
            hx_out = hx_out.view(1,1,-1)
            # decoder_input = decoder_input.view(1,1,-1)
                        
            token = queue.pop(0)
            if token in ret:
                continue
            if token == "num_op":
                continue
            if self.get_prefix(token) in self.lang.syntax["num_feat"]:
                # if np.random.rand() < epsilon:
                #     sub_keys=["_ub", "_lb"]
                # else:

                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[self.get_prefix(token)]), device=DEVICE)
                else:
                    pred = self.token_nets["constant"](hx_out.view(-1)) 

                # sub_keys=["_lb", "_ub"]
                # pred_ls = []
                # for key in sub_keys:
                #     if np.random.rand() < epsilon:
                #         pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                #     else:
                #         # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                #         # decoder_output = decoder_output.view(-1)
                #         # pred = self.token_nets["num_feat"+key](hx.view(-1))    
                #         pred = self.token_nets[token+key](hx_out.view(-1))    
                #     pred = self.mask_grammar_net_pred(program, token, pred)
                #     pred_ls.append(pred)
            else:
                if np.random.rand() < epsilon:
                    pred = torch.rand(len(self.grammar_num_to_token_val[token]), device=DEVICE)
                else:
                    # decoder_output, hx = self.decoder(decoder_input, hx, feat_encoder_outputs, prog_encoder_outputs)
                    # decoder_output = decoder_output.view(-1)
                    # print(hx)
                    pred = self.token_nets[token](hx_out)
            if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
                pred = self.mask_grammar_net_pred(program, token, pred)
                argmax = torch.argmax(pred).item()
                pred_val = self.grammar_num_to_token_val[token][argmax]
                if not eval:
                    if not token == "num_feat":
                        queue.extend(self.lang.syntax[token][pred_val])
                    else:
                        queue.extend([self.lang.syntax[token][pred_val][0] + "_lb", self.lang.syntax[token][pred_val][0] + "_ub"])
                ret[token] = pred
            else:
                feat_val = list(X_pd[self.get_prefix(token)])[0]
                if token.endswith("_lb"):
                    argmax = torch.argmax(pred).item()

                    pred_lb = (feat_val)*(argmax/(discretize_feat_value_count-1))

                    ret[token] = [pred_lb,pred]
                elif token.endswith("_ub"):

                    # range_max = self.feat_range_mappings[token][1]
                    # range_min = self.feat_range_mappings[token][0]

                    

                    

                    # pred_lb = pred_lb*(range_max - range_min) + range_min

                    argmax = torch.argmax(pred).item()

                    pred_ub = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

                    # pred_ub = pred_ub*(range_max - range_min) + range_min
                    # pred_ub = (range_max - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

                    ret[token] = [pred_ub,pred]

                    break
                
            # if eval:
            #     existing_atom_to_added[token] = existing_atom[token]
            #     decoder_input = self.atom_to_vector(self.prediction_to_atom(existing_atom_to_added))

            # else:
            decoder_input = self.atom_to_vector(self.prediction_to_atom(ret))
            
            curr_program = program.copy()
            curr_program.append(decoder_input)
            hx[features[0].shape[0]:len(curr_program)*curr_program[0].shape[0]+features[0].shape[0]] = torch.cat(curr_program)
            hx_out = self.embedding(hx)
            
        del features
        return ret

    def forward_ls0_backup(self, features,X_pd_full, program, queue, epsilon, col, op, eval=False, existing_atom=None):
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
        
        
        hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*program[0].shape[-1], device=DEVICE)
        hx[:,0:features[0].shape[0]] = features
        hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = torch.cat(program,dim=-1)
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        
        hx_out = self.embedding(hx)
        
        if np.random.rand() < epsilon:
            pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        else:
            # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
            pred = self.token_nets["constant"](hx_out.view(pat_count, -1))    
        

            # pred_lb = pred_lb*(range_max - range_min) + range_min

        feat_val = np.array(X_pd_full[col])


        argmax = torch.argmax(pred,dim=-1).cpu().numpy()

        if op == operator.__ge__:

            pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        else:
            pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        ret[col_key] = col

        if eval:
            ret[pred_Q_key] = pred
        else:
            ret[pred_Q_key] = pred.data.cpu()
        
        ret[pred_v_key] = pred_v
        
        ret[op_key] = op        
        
        return ret
        
    # def mask_atom_representation(self, program_ls, feat_pred_probs):
        
    #     op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
    #     op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
        
    #     for program in program_ls:
    #         op1_feat_occur_mat += program[:,self.op_start_pos:self.op_start_pos+1].to(DEVICE)*program[:,self.op_start_pos+2:-1].to(DEVICE)
    #         op2_feat_occur_mat += program[:,self.op_start_pos+1:self.op_start_pos+2].to(DEVICE)*program[:,self.op_start_pos+2:-1].to(DEVICE)
        
    #     feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
        
    #     feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2).float()
        
    #     return feat_pred_probs, op1_feat_occur_mat, op2_feat_occur_mat

    
    # def mask_atom_representation_for_op(self, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, op_pred_probs):
        
    #     assert len(torch.unique(op1_feat_occur_mat)) <= 2
    #     assert len(torch.unique(op2_feat_occur_mat)) <= 2
        
    #     # op_pred_probs[:,0] = op_pred_probs[:,0] * (1 - (op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    #     # op_pred_probs[:,1] = op_pred_probs[:,1] * (1 - (op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]))
        
    #     op_pred_probs = op_pred_probs*(1-torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]],dim=-1))
        
    #     assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
        
    #     # op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
    #     # op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feat_len], device = DEVICE)
        
    #     # for program in program_ls:
    #     #     op1_feat_occur_mat += program[self.op_start_pos:self.op_start_pos+1].view(-1,1)*program[self.op_start_pos+1:-1]
    #     #     op2_feat_occur_mat += program[self.op_start_pos+1:self.op_start_pos+2].view(-1,1)*program[self.op_start_pos+1:-1]
        
    #     # feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
        
    #     # feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2)
        
    #     # return feat_pred_probs, op1_feat_occur_mat, op2_feat_occur_mat
    #     return op_pred_probs
    

    def forward_ls2(self, input_data,X_pd_full, program, atom, epsilon, eval=False, existing_atom=None):
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
        
        # 
        
        total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]*self.topk_act

        # selecting feature
        
        concat_program_tensor = torch.cat(program,dim=-1)
        
        hx = torch.zeros(feature_embedding.shape[0], total_feat_prog_array_len, device=DEVICE)
        hx[:,0:feature_embedding[0].shape[0]] = feature_embedding
        hx[:, feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]*self.topk_act+feature_embedding[0].shape[0]] = concat_program_tensor.view(concat_program_tensor.shape[0], -1)

        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
            else:
                selected_feat_logit = self.feat_selector(hx)
        
        else:
            selected_feat_logit = self.feat_selector(hx)
        
        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = self.mask_atom_representation(program, selected_feat_logit)
        
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation(self.topk_act, self.num_feat_len, self.op_start_pos, program, selected_feat_logit)

        # selected_Q_feat = selected_feat_probs

        # selected_feat_probs = selected_feat_probs
        if not eval:
            # if self.topk_act == 1:
            #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
            # else:
                _, selected_feat_col = torch.topk(selected_feat_probs, k=self.topk_act, dim=-1)
        else:
            selected_feat_col = atom[col_id_key]

        selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
        selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, self.topk_act, 1)
        for k in range(self.topk_act):
            selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        # else:

        #     selected_Q_feat = torch.tanh(self.feat_selector(hx))

        #     # selected_feat_probs = torch.zeros([pat_count, selected_Q_feat.shape[-1]], device = DEVICE)

        #     # selected_feat_probs[torch.arange(pat_count, device=DEVICE), atom[col_id_key]]=1

        #     selected_feat_col = atom[col_id_key]
        
        # selecting op

        # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        hx = torch.cat([hx.unsqueeze(1).repeat(1, self.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)

        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                selected_op_logit = torch.rand([pat_count, self.topk_act, self.op_num], device=DEVICE)
            else:
                selected_op_logit = self.op_selector(hx)
        
        else:
            selected_op_logit = self.op_selector(hx)
        
        # selected_op_probs, selected_Q_op = self.mask_atom_representation_for_op(op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
        
        selected_op_probs, selected_Q_op =  mask_atom_representation_for_op(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)

            # selected_op_probs = selected_op_probs, dim=-1)

            #  = selected_op_probs
        if not eval:
            selected_op = torch.argmax(selected_op_probs, dim=-1)
        else:
            # selected_Q_op = torch.tanh(self.op_selector(hx))

            # selected_op_probs = torch.zeros([pat_count, selected_Q_op.shape[-1]], device = DEVICE)

            # selected_op_probs[torch.arange(pat_count, device=DEVICE), atom[op_id_key]]=1

            selected_op = atom[op_id_key]
        
        selected_op_onehot = torch.zeros_like(selected_op_probs)
        for k in range(self.topk_act):
            selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
        
        
        # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len + self.op_num, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        # hx[:, total_feat_prog_array_len+ self.num_feat_len:total_feat_prog_array_len + self.num_feat_len + self.op_num] = selected_op_probs
        # hx = torch.cat([features, concat_program_tensor, selected_feat_probs, selected_op_probs], dim=-1)
        hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
        
        # feat_encoder = torch.zeros(self.num_feat_len, device = DEVICE)
        # feat_encoder[self.feat_to_num_mappings[col]] = 1
        # op_encoder = torch.zeros(self.op_num, device = DEVICE)
        # op_encoder[self.op_to_num_mappings[op]] = 1

        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = feat_encoder
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        
        # hx_out = self.embedding(hx)
        
        
        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                pred = torch.rand([pat_count, self.topk_act, discretize_feat_value_count], device=DEVICE)
            else:
                # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                pred = self.token_nets["constant"](hx)
        else:
            pred = self.token_nets["constant"](hx)

        

            # pred_lb = pred_lb*(range_max - range_min) + range_min
        if not eval:
        
            selected_feat_col_ls = selected_feat_col.cpu().tolist()

            selected_op_id_ls = selected_op.cpu().tolist()

            selected_col_ls = []

            for idx in range(len(selected_feat_col_ls)):
                selected_col_ls.append([self.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
            
            selected_op_ls = []
            
            for idx in range(len(selected_op_id_ls)):
                selected_op_ls.append([self.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] for k in range(len(selected_op_id_ls[idx]))])


            feat_val = []
            for idx in range(len(selected_col_ls)):
                feat_val.append(np.array([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))]))

            feat_val = np.stack(feat_val, axis=0)


            selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

            op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
            
            
            argmax = torch.argmax(pred,dim=-1).cpu().numpy()

            # __ge__
            pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # __le__
            pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

            if self.lang.precomputed is not None:
                pred_v1, pred_v2 = find_nearest_thres_vals(self.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
            # if self.op_list[0] == operator.__ge__:     
            
                
            pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
        # else:
            # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

        # if op == operator.__ge__:

        #     pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # else:
        #     pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        

        ret[col_id_key] = selected_feat_col

        if eval:
            ret[pred_Q_key] = pred
            ret[col_Q_key] = selected_Q_feat
            ret[op_Q_key] = selected_Q_op
        else:
            ret[pred_Q_key] = pred.data
            ret[col_Q_key] = selected_Q_feat.data
            ret[op_Q_key] = selected_Q_op.data
        
            ret[pred_v_key] = pred_v
        
            ret[op_key] = selected_op_ls        
            
            ret[op_id_key] = selected_op
            
            ret[col_key] = selected_col_ls

        
        
        return ret

    def forward_ls0(self, input_data,X_pd_full, program, atom, epsilon, init=False, eval=False, existing_atom=None):
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

        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                if init:
                    selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
                else:
                    selected_feat_logit = torch.rand([pat_count,self.topk_act, self.num_feat_len], device=DEVICE)
            else:
                selected_feat_logit = self.feat_selector(hx)
        
        else:
            selected_feat_logit = self.feat_selector(hx)
        
        # selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = self.mask_atom_representation(program, selected_feat_logit)
        
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(self.topk_act, self.num_feat_len, self.op_start_pos, program, selected_feat_logit, init=init)

        # selected_Q_feat = selected_feat_probs

        # selected_feat_probs = selected_feat_probs
        if not eval:
            # if self.topk_act == 1:
            #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
            # else:
            if init:
                _, selected_feat_col = torch.topk(selected_feat_probs, k=self.topk_act, dim=-1)
            else:
                # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
                _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=self.topk_act, dim=-1)
        else:
            selected_feat_col = atom[col_id_key]

        selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
        prev_program_ids = None
        curr_selected_feat_col = None

        if init:
            selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, self.topk_act, 1)
            for k in range(self.topk_act):
                selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
            hx = torch.cat([hx.unsqueeze(1).repeat(1, self.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
        else:
            if not eval:
                prev_program_ids = (selected_feat_col//self.num_feat_len)
                curr_selected_feat_col = selected_feat_col%self.num_feat_len
            else:
                prev_program_ids = atom[prev_prog_key]
                curr_selected_feat_col = atom[col_id_key]
            new_hx = []
            seq_ids = torch.arange(pat_count)
            for k in range(self.topk_act):
                selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
                new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], selected_feat_probs[seq_ids,prev_program_ids[:,k]]*selected_feat_col_onehot[seq_ids, prev_program_ids[:,k]]],dim=-1))

            hx = torch.stack(new_hx, dim=1)
            # op1_feat_occur_mat = torch.stack([op1_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
            # op2_feat_occur_mat = torch.stack([op2_feat_occur_mat[seq_ids, prev_program_ids[:,k]] for k in range(self.topk_act)],dim=1)
        # else:

        #     selected_Q_feat = torch.tanh(self.feat_selector(hx))

        #     # selected_feat_probs = torch.zeros([pat_count, selected_Q_feat.shape[-1]], device = DEVICE)

        #     # selected_feat_probs[torch.arange(pat_count, device=DEVICE), atom[col_id_key]]=1

        #     selected_feat_col = atom[col_id_key]
        
        # selecting op

        # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        

        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                selected_op_logit = torch.rand([pat_count, self.topk_act, self.op_num], device=DEVICE)
            else:
                selected_op_logit = self.op_selector(hx)
        
        else:
            selected_op_logit = self.op_selector(hx)
        
        # selected_op_probs, selected_Q_op = self.mask_atom_representation_for_op(op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit)
        if not eval:
            selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, prev_program_ids, curr_selected_feat_col,  init=init)
        else:
            selected_op_probs, selected_Q_op =  mask_atom_representation_for_op1(self.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, selected_op_logit, atom[prev_prog_key], atom[col_id_key],  init=init)

            # selected_op_probs = selected_op_probs, dim=-1)

            #  = selected_op_probs
        if not eval:
            selected_op = torch.argmax(selected_op_probs, dim=-1)
        else:
            # selected_Q_op = torch.tanh(self.op_selector(hx))

            # selected_op_probs = torch.zeros([pat_count, selected_Q_op.shape[-1]], device = DEVICE)

            # selected_op_probs[torch.arange(pat_count, device=DEVICE), atom[op_id_key]]=1

            selected_op = atom[op_id_key]
        
        selected_op_onehot = torch.zeros_like(selected_op_probs)
        for k in range(self.topk_act):
            selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
        
        
        # hx = torch.zeros(features.shape[0], total_feat_prog_array_len + self.num_feat_len + self.op_num, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor
        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = selected_feat_probs
        # hx[:, total_feat_prog_array_len+ self.num_feat_len:total_feat_prog_array_len + self.num_feat_len + self.op_num] = selected_op_probs
        # hx = torch.cat([features, concat_program_tensor, selected_feat_probs, selected_op_probs], dim=-1)
        hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
        
        # feat_encoder = torch.zeros(self.num_feat_len, device = DEVICE)
        # feat_encoder[self.feat_to_num_mappings[col]] = 1
        # op_encoder = torch.zeros(self.op_num, device = DEVICE)
        # op_encoder[self.op_to_num_mappings[op]] = 1

        # hx[:, total_feat_prog_array_len:total_feat_prog_array_len + self.num_feat_len] = feat_encoder
        
        
        # hx = torch.zeros(features[0].shape[0], device=DEVICE)# + self.program_max_len*program[0].shape[0], device=DEVICE)
        # hx[0:features[0].shape[0]] = features[0]
        # hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
        
        ret = {}
        
        # hx_out = self.embedding(hx)
        
        
        if not eval:
            if np.random.rand() < epsilon:
                # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
                pred = torch.rand([pat_count, self.topk_act, discretize_feat_value_count], device=DEVICE)
            else:
                # pred = self.token_nets["num_feat"+key](decoder_output[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                # pred = self.token_nets["num_feat"+key](hx[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                # pred = self.token_nets["constant"](hx_out[token_sample_ids.view(-1)].view(len(token_sample_ids), -1))    
                pred = self.token_nets["constant"](hx)
        else:
            pred = self.token_nets["constant"](hx)

        

            # pred_lb = pred_lb*(range_max - range_min) + range_min
        if not eval:
            selected_col_ls = []
            if init:
                selected_feat_col_ls = selected_feat_col.cpu().tolist()

                

                for idx in range(len(selected_feat_col_ls)):
                    selected_col_ls.append([self.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
            else:
                curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
                for idx in range(len(prev_program_ids)):
                    selected_col_ls.append([self.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] for k in range(len(curr_selected_feat_col_ls[idx]))])
            
            selected_op_ls = []
            
            selected_op_id_ls = selected_op.cpu().tolist()

            for idx in range(len(selected_op_id_ls)):
                selected_op_ls.append([self.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] for k in range(len(selected_op_id_ls[idx]))])


            feat_val = []
            for idx in range(len(selected_col_ls)):
                feat_val.append(np.array([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))]))

            feat_val = np.stack(feat_val, axis=0)


            selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

            op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
            
            
            argmax = torch.argmax(pred,dim=-1).cpu().numpy()

            # __ge__
            pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # __le__
            pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val

            if self.lang.precomputed is not None:
                pred_v1, pred_v2 = find_nearest_thres_vals(self.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
            # if self.op_list[0] == operator.__ge__:     
            
                
            pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
        # else:
            # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

        # if op == operator.__ge__:

        #     pred_v = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # else:
        #     pred_v = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        
        
        if init:
            ret[col_id_key] = selected_feat_col
        else:
            ret[col_id_key] = curr_selected_feat_col

        if eval:
            ret[pred_Q_key] = torch.tanh(pred)
            ret[col_Q_key] = selected_Q_feat
            ret[op_Q_key] = selected_Q_op
            ret[prev_prog_key] = prev_program_ids
        else:
            ret[pred_Q_key] = torch.tanh(pred).data
            ret[col_Q_key] = selected_Q_feat.data
            ret[op_Q_key] = selected_Q_op.data
        
            ret[pred_v_key] = pred_v
        
            ret[op_key] = selected_op_ls        
            
            ret[op_id_key] = selected_op
            
            ret[col_key] = selected_col_ls
            ret[prev_prog_key] = prev_program_ids

        
        
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

class DQN7:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, tf_latent_size, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0, numeric_count=None, category_count=None, has_embeddings=False, model="mlp", pretrained_model_path=None, topk_act=1):
        self.mem_sample_size = mem_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang
        torch.manual_seed(seed)
        self.topk_act = topk_act
        if model == "mlp":
            self.policy_net = RLSynthesizerNetwork7(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, topk_act=topk_act)
        else:
            self.policy_net = RLSynthesizerNetwork7_pretrained(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, tf_latent_size=tf_latent_size, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=pretrained_model_path, topk_act=topk_act)
        
        torch.manual_seed(seed)
        if model == "mlp":
            self.target_net = RLSynthesizerNetwork7(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, dropout_p = 0, feat_range_mappings=feat_range_mappings, topk_act=topk_act)
        else:
            self.target_net = RLSynthesizerNetwork7_pretrained(lang=lang, program_max_len=program_max_len,patient_max_appts=patient_max_appts,latent_size=latent_size, tf_latent_size=tf_latent_size, dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=pretrained_model_path, topk_act=topk_act)

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

    def atom_to_vector_ls0(self, atom):
        return self.policy_net.atom_to_vector_ls0(atom)

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
        return self.policy_net.prediction_to_atom(pred), pred
    
    def predict_atom_ls(self, features, X_pd_ls, program, epsilon):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)
            pred = self.policy_net.forward_ls0(features, X_pd_ls, [init_program], ["formula"], epsilon, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features, X_pd_ls, program, ["formula"], epsilon)
        # return self.policy_net.prediction_to_atom_ls(pred), pred
        return pred
    
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
            if not self.policy_net.get_prefix(token) in self.lang.syntax["num_feat"]:
                max_tensors[token] = torch.max(token_val).reshape((1,1))
            else:
                if token.endswith("_ub"):
                    # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
                    max_tensors[self.policy_net.get_prefix(token)] = torch.max(token_val[1]).reshape(1,1)
        
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
                if not "pred_score" in max_tensors:
                    max_tensors["pred_score"] = [torch.zeros(len(data), device = DEVICE), torch.zeros(len(data), device = DEVICE)]
                pred_val = pred[token]
                for token_key in token:
                    
                    # token_key = token_key[0]
                    probs = pred_val[1][token_key]
                    # ub_probs = pred_val[1][token_key][1]
                    sample_ids = token_val[2][token_key].view(-1)
                    sample_cln_id_ls = token_val[3][token_key]
                    val = probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[0].view(-1)]
                    if token_key.endswith("_lb"):
                        max_tensors["pred_score"][0][sample_ids] = val
                    elif token_key.endswith("_ub"):
                        max_tensors["pred_score"][1][sample_ids] = val
                    del val
                    # val = ub_probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[1].view(-1)]      
                    # max_tensors[token][1][sample_ids] = val
                    # del val
                # print()
                # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
        
        # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        return_pred = self.target_net.prediction_to_atom_ls(pred)
        del pred
        return return_pred, max_tensors
    
    
    def predict_next_state_with_tensor_info_ls0(self, features, data, state):
        program = state
        
        if len(state) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            pred = self.target_net.forward_ls0(features, data, [init_program], ["formula"], 0, eval=False, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls0(features, data, program, ["formula"], 0, eval=False)
            
        max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)

        max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

        # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

        max_tensors += max_col_tensors + max_op_tensors

        max_tensors = max_tensors/3
        # max_tensors = dict()
        # for token, token_val in pred.items():
        #     # if not token in self.lang.syntax["num_feat"]:
        #     if not type(token) is tuple:
        #         max_tensors[token] = torch.max(token_val, dim=1)[0].reshape((len(data),1))
        #     else:
        #         if not "pred_score" in max_tensors:
        #             max_tensors["pred_score"] = [torch.zeros(len(data), device = DEVICE), torch.zeros(len(data), device = DEVICE)]
        #         pred_val = pred[token]
        #         for token_key in token:
                    
        #             # token_key = token_key[0]
        #             probs = pred_val[1][token_key]
        #             # ub_probs = pred_val[1][token_key][1]
        #             sample_ids = token_val[2][token_key].view(-1)
        #             sample_cln_id_ls = token_val[3][token_key]
        #             val = probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[0].view(-1)]
        #             if token_key.endswith("_lb"):
        #                 max_tensors["pred_score"][0][sample_ids] = val
        #             elif token_key.endswith("_ub"):
        #                 max_tensors["pred_score"][1][sample_ids] = val
        #             del val
        #             # val = ub_probs[torch.tensor(list(range(len(sample_ids)))), sample_cln_id_ls[1].view(-1)]      
        #             # max_tensors[token][1][sample_ids] = val
        #             # del val
        #         # print()
        #         # max_tensors[token] = [torch.max(token_val[1][0]).reshape((1,1)), torch.max(token_val[1][1]).reshape((1,1))]
        
        # # max_tensors = {token:torch.max(token_val).reshape((1,1)) for token, token_val in pred.items() if not token in self.lang.syntax["num_feat"]}
        # return_pred = self.target_net.prediction_to_atom_ls(pred)
        # del pred
        # return return_pred, max_tensors
        return max_tensors.to(DEVICE)

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
    
    def get_state_action_prediction_tensors(self, features, X_pd, program, atom_ls):
        atom, origin_atom = atom_ls
        queue = list(atom.keys())
        if len(program) == 0:
            pred = self.policy_net(features, X_pd, [self.first_prog_embed], queue, 0, eval=True, existing_atom=origin_atom)
        else:
            #program.sort()
            pred = self.policy_net(features, X_pd, program, queue, 0, eval=True, existing_atom=origin_atom)

        tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        for token, token_val in atom.items():
            if token == "num_op" or token.endswith("_prob"):
                continue

            if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
                # if not token.endswith("_prob"):
                    tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
            else:
                # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
                tensor_indeces[token] = torch.argmax(atom[token][1]).item()
            # else:
            #     tensor_indeces[token] = 0
        atom_prediction_tensors = {}
        for token, tensor_idx in tensor_indeces.items():
            if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
                atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
            else:
                if token.endswith("_ub"):
                    atom_prediction_tensors[self.policy_net.get_prefix(token)] = pred[token][1][tensor_idx].view(-1)
                # atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
        # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        return atom_prediction_tensors

    def get_state_action_prediction_tensors_ls(self, features, X_pd, program, atom_pair):
        atom = atom_pair[0]
        origin_atom = atom_pair[1]
        queue = list(atom.keys())
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            pred = self.policy_net.forward_ls(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls(features, X_pd, program, queue, 0, eval=True, replay=True, existing_atom=origin_atom)

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
                if not "pred_score" in atom_prediction_tensors:
                    atom_prediction_tensors["pred_score"] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
                # atom_prediction_tensors[token] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
                pred_val = pred[token]
                for token_key in token:
                    
                    # token_key = token_key[0]
                    # lb_probs = pred_val[1][token_key][0]
                    probs = pred_val[1][token_key]
                    sample_ids = token_val[2][token_key].view(-1)
                    sample_cln_id_ls = token_val[3][token_key]
                    val = probs[sample_ids.view(-1), sample_cln_id_ls.view(-1)]
                    if token_key.endswith("_lb"):
                        atom_prediction_tensors["pred_score"][0][sample_ids] = val
                    elif token_key.endswith("_ub"):
                        atom_prediction_tensors["pred_score"][1][sample_ids] = val
                    del val
                    # val = ub_probs[sample_ids.view(-1), sample_cln_id_ls[1].view(-1)]
                    # atom_prediction_tensors[token][1][sample_ids] = val
                    # del val


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
    
    def get_state_action_prediction_tensors_ls0(self, features, X_pd, state, atom):
        # atom = atom_pair[0]
        # origin_atom = atom_pair[1]
        queue = list(atom.keys())
        
        program = state
        
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
            # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            pred = self.policy_net.forward_ls0(features,X_pd, [init_program], atom, 0, eval=True, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features,X_pd, program, atom, 0, eval=True)
            # pred = self.policy_net.forward_ls(features, X_pd, state, queue, 0, eval=True, replay=True, existing_atom=origin_atom)

        # tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        # atom_prediction_tensors = {}
        tensor_indeces = atom[pred_Q_key].argmax(-1)
        
        x_idx = torch.tensor(list(range(len(X_pd))))
        
        atom_prediction_tensors_ls = []
        for k in range(tensor_indeces.shape[-1]):
            atom_prediction_tensors_ls.append(pred[pred_Q_key][x_idx, k, tensor_indeces[:,k]])
        atom_prediction_tensors = torch.stack(atom_prediction_tensors_ls, dim=1) #atom_prediction_tensors/tensor_indeces.shape[-1]

        # col_tensor_indices = atom[col_Q_key].argmax(-1)
        # _,col_tensor_indices = torch.topk(atom[col_Q_key], k = self.topk_act, dim=-1)
        
        _,col_tensor_indices = torch.topk(atom[col_Q_key].view(len(atom[col_Q_key]),-1), k=self.topk_act, dim=-1)


        col_prediction_Q_tensor_ls = []
        
        for k in range(self.topk_act):
            col_prediction_Q_tensor_ls.append(pred[col_Q_key].view(len(pred[col_Q_key]), -1)[x_idx, col_tensor_indices[:,k]])
        
        col_prediction_Q_tensor = torch.stack(col_prediction_Q_tensor_ls, dim=1)
        # col_prediction_Q_tensor_ls = []
        # for k in range(col_tensor_indices.shape[-1]):
        #     col_prediction_Q_tensor_ls += pred[col_Q_key][x_idx, col_tensor_indices[:,k]]
        # col_prediction_Q_tensor = pred[col_Q_key][x_idx, col_tensor_indices]
        # col_prediction_Q_tensor = col_prediction_Q_tensor/col_tensor_indices.shape[-1]
        
        op_tensor_indices = atom[op_Q_key].argmax(-1)

        op_prediction_Q_tensor_ls = []
        for k in range(op_tensor_indices.shape[-1]):
            op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
        op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
        # op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

        assert torch.sum(atom_prediction_tensors < -1) + torch.sum(col_prediction_Q_tensor < -1) + torch.sum(op_prediction_Q_tensor < -1) == 0

        atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3

        # tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
        # for token, token_val in atom.items():
        #     if token == "num_op" or token.endswith("_prob"):
        #         continue

        #     if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
        #         # if not token.endswith("_prob"):
        #             tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
        #     else:
        #         # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
        #         tensor_indeces[token] = torch.argmax(atom[token][1]).item()
        #     # else:
        #     #     tensor_indeces[token] = 0
        # atom_prediction_tensors = {}
        # for token, tensor_idx in tensor_indeces.items():
        #     if self.policy_net.get_prefix(token) not in self.lang.syntax["num_feat"]:
        #         atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
        #     else:
        #         if token.endswith("_ub"):
        #             atom_prediction_tensors[self.policy_net.get_prefix(token)] = pred[token][1][tensor_idx].view(-1)
                # atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))

        # for token, token_val in atom.items():
        #     # if token == "num_op" or token.endswith("_prob"):
        #     #     continue

        #     # if token not in self.lang.syntax["num_feat"]:
        #     if not type(token) is tuple:
        #         # if not token.endswith("_prob"):
        #             # tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
        #             if not type(token_val) is dict:
        #                 tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val]
        #                 val = pred[token][:,tensor_idx].reshape((len(X_pd),1))
        #                 atom_prediction_tensors[token] = val
        #                 del val
        #             else:
        #                 atom_prediction_tensors[token] = torch.zeros(len(X_pd), device = DEVICE)
        #                 for token_val_key in token_val:
        #                     token_val_sample_ids = token_val[token_val_key]
        #                     tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val_key]
        #                     val = pred[token][token_val_sample_ids,tensor_idx]
        #                     atom_prediction_tensors[token][token_val_sample_ids] = val
        #                     del val
                        
        #     else:
        #         if not "pred_score" in atom_prediction_tensors:
        #             atom_prediction_tensors["pred_score"] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
        #         # atom_prediction_tensors[token] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
        #         pred_val = pred[token]
        #         for token_key in token:
                    
        #             # token_key = token_key[0]
        #             # lb_probs = pred_val[1][token_key][0]
        #             probs = pred_val[1][token_key]
        #             sample_ids = token_val[2][token_key].view(-1)
        #             sample_cln_id_ls = token_val[3][token_key]
        #             val = probs[sample_ids.view(-1), sample_cln_id_ls.view(-1)]
        #             if token_key.endswith("_lb"):
        #                 atom_prediction_tensors["pred_score"][0][sample_ids] = val
        #             elif token_key.endswith("_ub"):
        #                 atom_prediction_tensors["pred_score"][1][sample_ids] = val
        #             del val
        #             # val = ub_probs[sample_ids.view(-1), sample_cln_id_ls[1].view(-1)]
        #             # atom_prediction_tensors[token][1][sample_ids] = val
        #             # del val


        #         # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
        #     # else:
        #     #     tensor_indeces[token] = 0
        
        # # for token, tensor_idx in tensor_indeces.items():
        # #     if token not in self.lang.syntax["num_feat"]:
        # #         atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
        # #     else:
        # #         atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
        # # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
        # del pred
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
        # Q = formula*feat*op*constant[0]*constant[1]
        Q = constant
        return Q[0]
    
    def get_atom_Q_value2(self, atom:dict, atom_prediction_tensors: dict):
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
        # Q = formula*feat*op*constant[0]*constant[1]
        # Q = constant[0]*constant[1]
        # return Q[0]
        return torch.cat([constant[0].view(-1), constant[1].view(-1)])

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
            # constant = atom_prediction_tensors[tuple([tuple([item]) for item in list(feat_name.keys())])]
            constant = atom_prediction_tensors["pred_score"]
        # feat = feat.to(DEVICE)
        # formula = formula.to(DEVICE)
        # Q = formula.view(-1)*feat.view(-1)*op*
        # Q = constant[0].view(-1)*
        Q = constant[1].view(-1)
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
        state_action_pred = [(a[0],self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
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
        loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))
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
    
    def optimize_model2(self):
        if len(self.memory) < self.batch_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.batch_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.tensor([transition.reward for transition in batch], device=DEVICE, requires_grad=True, dtype=torch.float)

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors(f,d, p,a)) for f,d, p,a in state_action_batch]
        state_action_values = torch.stack([self.get_atom_Q_value2(a,t) for a,t in state_action_pred])

        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.batch_size, 2], device=DEVICE, dtype=torch.float)
        if len(next_state_pred_non_final) > 0:
            next_state_values_non_final = torch.stack([self.get_atom_Q_value2(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values[non_final_mask] = next_state_values_non_final

        # Prepare the loss function
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute the loss
        loss = self.criterion(state_action_values[:,1:2].repeat(1,2), expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

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
    
    def optimize_model_ls0(self):
        if len(self.memory) < self.mem_sample_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.mem_sample_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.stack([torch.from_numpy(transition.reward) for transition in batch]).to(DEVICE)

        #get Q value for (s,a)
        state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0(f,d, p,a)) for f,d, p,a in state_action_batch]
        # state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
        state_action_values = torch.stack([t for a,t in state_action_pred])
        state_action_values = state_action_values.to(DEVICE)
        
        #get Q value for (s', max_a')
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.mem_sample_size, self.batch_size, self.topk_act], dtype=torch.float, device=DEVICE)
        if len(next_state_pred_non_final) > 0:
            # next_state_values_non_final = torch.stack([self.get_atom_Q_value_ls(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
            next_state_values_non_final = torch.stack(next_state_pred_non_final)
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
        
        return_loss = loss.detach()
        del loss
        return return_loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


