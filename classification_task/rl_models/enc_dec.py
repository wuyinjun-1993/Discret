import torch
from collections import namedtuple, deque
import numpy as np
import random
import operator
import sys,os
from torch import nn, optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from feat_encoder.ft_transformer0 import FTTransformer


# Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
Transition = namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))

min_Q_val = -1.01

col_key = "col"

col_id_key = "col_id"

col_Q_key = "col_Q"

pred_Q_key = "pred_Q"

op_Q_key = "op_Q"

col_probs_key = "col_probs"

pred_probs_key = "pred_probs"

op_probs_key = "op_probs"

pred_v_key = "pred_v"

op_id_key = "op_id"
        
op_key = "op"

select_num_feat_key = "select_num_feat_key"



prev_prog_key = "prev_prog"

outbound_key = "outbound"

further_sel_mask_key = "further_sel_mask"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# discretize_feat_value_count=20

def encoding_program_init(net):
    net.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in net.lang.syntax.items()}
    net.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in net.lang.syntax.items()}
    net.grammar_token_to_pos = {}
    net.grammar_pos_to_token = {}
    net.ATOM_VEC_LENGTH = 0
    net.one_hot_token_bounds = {}
    # for decision, options_dict in net.lang.syntax.items():
    #     start = net.ATOM_VEC_LENGTH
    #     for option in list(options_dict.keys()):
    #         net.grammar_token_to_pos[(decision,option)] = net.ATOM_VEC_LENGTH
    #         net.grammar_pos_to_token[net.ATOM_VEC_LENGTH] = (decision, option)
    #         net.ATOM_VEC_LENGTH += 1
    #     net.one_hot_token_bounds[decision] = (start, net.ATOM_VEC_LENGTH)
    for i,v in net.lang.syntax.items():
        if not i in net.lang.syntax["num_feat"]:
            net.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
            net.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
        else:
            net.grammar_num_to_token_val[i] = list(range(net.discretize_feat_value_count))
            net.grammar_token_val_to_num[i] = list(range(net.discretize_feat_value_count))

    net.op_start_pos = -1
    net.num_start_pos = -1

    for decision, options_dict in net.lang.syntax.items():
        if not (decision == "num_op" or decision in net.lang.syntax["num_feat"]):
            continue
        # if decision == "num_op":
        #     continue
        start = net.ATOM_VEC_LENGTH


        if not decision in net.lang.syntax["num_feat"]:
            for option in list(options_dict.keys()):        
                if net.op_start_pos < 0:
                    net.op_start_pos = net.ATOM_VEC_LENGTH
                
                net.grammar_token_to_pos[(decision,option)] = net.ATOM_VEC_LENGTH
                net.grammar_pos_to_token[net.ATOM_VEC_LENGTH] = (decision, option)
                net.ATOM_VEC_LENGTH += 1
        else:
            if net.num_start_pos < 0:
                net.num_start_pos = net.ATOM_VEC_LENGTH
            
            net.grammar_token_to_pos[decision] = net.ATOM_VEC_LENGTH
            net.grammar_pos_to_token[net.ATOM_VEC_LENGTH] = decision
            net.ATOM_VEC_LENGTH += 1
        net.one_hot_token_bounds[decision] = (start, net.ATOM_VEC_LENGTH)
    net.grammar_token_to_pos[pred_v_key] = net.ATOM_VEC_LENGTH
    net.one_hot_token_bounds[pred_v_key] = (start, net.ATOM_VEC_LENGTH)
    net.grammar_pos_to_token[net.ATOM_VEC_LENGTH] = pred_v_key
    net.ATOM_VEC_LENGTH += 1



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

    op_pred_Q = op_pred_Q*(1-mask) + (min_Q_val)*mask


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
        
        

        op_pred_Q = op_pred_Q*(1-mask) + (min_Q_val)*mask


    assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
    
    return op_pred_probs, op_pred_Q

# def determine_pred_val_by_ids(argmax, feat_val):
#     pred_v = 1/(discretize_feat_value_count-1)*argmax
    
#     ops = (feat_val < pred_v).type(torch.long)

#     return ops

def determine_pred_ids_by_val(net, feat_val, pred, op1_mask, op2_mask, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max):
    
    if selected_feat_point_tensors is None:
        pred_candidate_vals = torch.zeros_like(pred)
        for k in range(net.discretize_feat_value_count):
            pred_candidate_vals[:, :, k] = 1/(net.discretize_feat_value_count-1)*k
    else:
        pred_candidate_vals = selected_feat_point_tensors

    # pred_candidate_vals[:, :, -1] += 1e-5
    # if selected_feat_point_tensors_max is None:
    #     large_ids = (feat_val >= 1).nonzero()
    #     pred_candidate_vals[large_ids[:,0],large_ids[:,1],-1] = feat_val[feat_val >= 1] + 1e-5
    # else:
    #     large_ids = (feat_val >= selected_feat_point_tensors_max).nonzero()
    #     pred_candidate_vals[large_ids[:,0],large_ids[:,1],-1] = feat_val[feat_val >= selected_feat_point_tensors_max] + 1e-5
    
    # if selected_feat_point_tensors_min is None:
    #     small_ids = (feat_val <= 0).nonzero()
    #     pred_candidate_vals[small_ids[:,0], small_ids[:,1],0] = feat_val[feat_val <= 0] - 1e-5
    # else:
    #     small_ids = (feat_val <= selected_feat_point_tensors_min).nonzero()
    #     pred_candidate_vals[small_ids[:,0], small_ids[:,1],0] = feat_val[feat_val <= selected_feat_point_tensors_min] - 1e-5
    # op1: feat_val < candidate_vals
    pred_out_mask = (pred_candidate_vals > feat_val.unsqueeze(-1))*(1-op1_mask.unsqueeze(-1)) + (pred_candidate_vals <= feat_val.unsqueeze(-1))*(1-op2_mask.unsqueeze(-1))

    return (pred_out_mask >= 1).type(torch.float)#, torch.logical_or(feat_val < selected_feat_point_tensors_min, feat_val > selected_feat_point_tensors_max)


# net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, init=init
def mask_atom_representation_for_op0(net, topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=False):
        
    assert len(torch.unique(op1_feat_occur_mat)) <= 2
    assert len(torch.unique(op2_feat_occur_mat)) <= 2
    
    # op_pred_probs[:,0] = op_pred_probs[:,0] * (1 - (op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    # op_pred_probs[:,1] = op_pred_probs[:,1] * (1 - (op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),selected_feat_idx]))
    
    # op_pred_probs = torch.softmax(op_pred_logic ,dim=-1) + 1e-6

    # op_pred_Q = torch.tanh(op_pred_logic)
    pred_out_mask = torch.ones_like(pred)
    

    if not init:
        mask_ls = []

        for k in range(topk_act):
            mask_ls.append(torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=DEVICE),prev_program_ids[:,k],curr_selected_feat_col[:,k]], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=DEVICE),prev_program_ids[:,k],curr_selected_feat_col[:,k]]],dim=-1))

    # mask = mask.unsqueeze(1).repeat(1,self.topk_act, 1)
        mask = torch.stack(mask_ls, dim=1)

        # pred_vals = pred

        # op_pred_probs = op_pred_probs*(1-mask)
        pred_out_mask = determine_pred_ids_by_val(net, feat_val, pred, mask[:,:,0], mask[:,:,1], selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max)
        


        # op_pred_Q = op_pred_Q*(1-mask) + (min_Q_val)*mask
    return pred_out_mask

    # assert len(torch.nonzero(torch.sum(op_pred_probs, dim=-1) == 0)) == 0
    
    # return op_pred_probs, op_pred_Q


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

    feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (min_Q_val)*(feat_occur_count_mat >= 2).float()
    
    return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat

def mask_feat_by_nan_time_range(X_pd_full, op1_feat_occur_mat, op2_feat_occur_mat):
    for idx in range(len(X_pd_full)):    
        nan_val_ls =  X_pd_full[idx]
        # nan_val_count_ls = torch.from_numpy(selected_feat_vals.isna().sum(axis=0).values)
        # nan_val_ls = torch.from_numpy(selected_feat_vals.values.astype(float))
        # nan_val_count_ls = (torch.sum(nan_val_ls != nan_val_ls, dim =0) == nan_val_ls.shape[0]).nonzero().view(-1)
        
        nan_val_count_ls = (nan_val_ls != nan_val_ls).nonzero().view(-1)
        
        op1_feat_occur_mat[idx, :, nan_val_count_ls] = 1
        op2_feat_occur_mat[idx, :, nan_val_count_ls] = 1
    
    return op1_feat_occur_mat, op2_feat_occur_mat


def mask_atom_representation1(X_pd_full, topk_act, num_feat_len, op_start_pos, program_ls, outbound_mask_ls, feat_pred_logit, init=False):
    
    op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = DEVICE)
    
    
    for idx in range(len(program_ls)):
        program = program_ls[idx]
        if not init:
            outbound_mask = outbound_mask_ls[idx].type(torch.float).unsqueeze(-1)
            op1_feat_occur_mat += ((program[:,:, op_start_pos:op_start_pos+1] + outbound_mask)>=1).type(torch.float).to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
            op2_feat_occur_mat += ((program[:,:, op_start_pos+1:op_start_pos+2] + outbound_mask)>=1).type(torch.float).to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE)
        op1_feat_occur_mat, op2_feat_occur_mat = mask_feat_by_nan_time_range(X_pd_full, op1_feat_occur_mat, op2_feat_occur_mat)
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

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (min_Q_val)*(feat_occur_count_mat >= 2).float()
    else:
        feat_pred_probs = feat_pred_probs*(feat_occur_count_mat[:,0] < 2).float()

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat[:,0] < 2).float() + (min_Q_val)*(feat_occur_count_mat[:,0] >= 2).float()
    
    further_selection_masks = torch.ones(feat_occur_count_mat.shape[0]).bool()#(torch.sum(feat_occur_count_mat.view(feat_occur_count_mat.shape[0], -1) < 2, dim=-1) >= topk_act)
    return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks

def mask_atom_representation1_2(topk_act, num_feat_len, op_start_pos, program_ls, outbound_mask_ls, feat_pred_logit, init=False):
    
    op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], len(program_ls), num_feat_len], device = DEVICE)
    op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], len(program_ls), num_feat_len], device = DEVICE)
    
    if not init:
        for idx in range(len(program_ls)):
            program = program_ls[idx]
            outbound_mask = outbound_mask_ls[idx].type(torch.float).unsqueeze(-1)
            op1_feat_occur_mat[:,idx] = torch.sum(((program[:,:, op_start_pos:op_start_pos+1] + outbound_mask)>=1).type(torch.float).to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE), dim=1)
            op2_feat_occur_mat[:,idx] = torch.sum(((program[:,:, op_start_pos+1:op_start_pos+2] + outbound_mask)>=1).type(torch.float).to(DEVICE)*program[:,:,op_start_pos+2:-1].to(DEVICE), dim=1)
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

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (min_Q_val)*(feat_occur_count_mat >= 2).float()
    
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
        ret_tensor_ls[(op_id_tensor[:,k] < 2),k, net.op_start_pos + op_id_tensor[(op_id_tensor[:,k] < 2),k]]=1
        # if torch.sum(op_id_tensor[:,k] == 2) > 0:
        #     print()
        
        ret_tensor_ls[(op_id_tensor[:,k] == 2),k, net.op_start_pos]=1
        ret_tensor_ls[(op_id_tensor[:,k] == 2),k, net.op_start_pos + 1]=1
        ret_tensor_ls[(op_id_tensor[:,k] < 2).cpu(), k, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)[(op_id_tensor[:,k] < 2).cpu(), k]
    # ret_tensor_ls[:, :, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)

    return ret_tensor_ls

def vector_ls_to_str0_main(net, program):
    # sample_id_tensor = torch.arange(len(program),device=DEVICE)
    # all_cols = []
    # all_ops = []
    # all_constants = []
    
    idx_ls = program.nonzero()
    
    op_ls = []
    col_ls = []
    col_prob_ls = []
    op_prob_ls = []
    for idx in idx_ls:
        idx = idx.item()
        token = net.grammar_pos_to_token[idx]
        if idx >= net.num_start_pos and idx < net.ATOM_VEC_LENGTH-1:
            col_ls.append(token)
            col_prob_ls.append(program[idx].item())

        if idx < net.num_start_pos:
            op_ls.append(token[1])
            op_prob_ls.append(program[idx].item())
        
        if idx == net.ATOM_VEC_LENGTH-1:
            pred_const = program[idx].item()

    curr_op = None
    # if len(op_ls) >= 2:
    #     curr_op = net.lang.lang.NON_STR_REP[operator.__eq__]
    # else:
    #     curr_op = net.lang.lang.NON_STR_REP[op_ls[0]]
    max_op_id = np.argmax(np.array(op_prob_ls))
    curr_op = net.lang.lang.NON_STR_REP[op_ls[max_op_id]]

    max_col_id = np.argmax(np.array(col_prob_ls))
    col = col_ls[max_col_id]
    max_range = net.feat_range_mappings[col][1]
    min_range = net.feat_range_mappings[col][0]
    # pred_const = (max_range - min_range)*pred_const + min_range


    return col + curr_op# + str(pred_const)


def forward_main(net, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False):
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            if net.feat_group_names is None:
                output_num = net.num_feats
            else:
                output_num = net.feat_group_num

            if init:
                selected_feat_logit = torch.rand([pat_count, output_num], device=DEVICE)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, output_num], device=DEVICE)

            # if init:
            #     selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
            # else:
            #     selected_feat_logit = torch.rand([pat_count,self.topk_act, self.num_feat_len], device=DEVICE)
        else:
            selected_feat_logit = net.feat_selector(hx)
    
    else:
        selected_feat_logit = net.feat_selector(hx)

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feats, net.op_start_pos, program, selected_feat_logit, init=init)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.feat_group_num, net.op_start_pos, program, selected_feat_logit, init=init)

    # selected_Q_feat = selected_feat_probs

    # selected_feat_probs = selected_feat_probs
    if not eval:
        # if self.topk_act == 1:
        #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        # else:
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
            # dist = torch.distributions.Categorical(selected_feat_probs)
            # dist.sample()
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
            prev_program_ids = torch.div(selected_feat_col, net.num_feats, rounding_mode='floor')
            curr_selected_feat_col = selected_feat_col%net.num_feats
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
        if np.random.rand() < epsilon and not is_ppo:
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
        if not is_ppo:
            selected_op = torch.argmax(selected_op_probs, dim=-1)
        else:
            if train:
                dist = torch.distributions.Categorical(selected_op_probs)
                selected_op = dist.sample()
            else:
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
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=DEVICE)
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
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))])
                else:
                    selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
        else:
            curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
            for idx in range(len(prev_program_ids)):
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] for k in range(len(curr_selected_feat_col_ls[idx]))])
                else:
                    selected_col_ls.append([net.feat_group_names[curr_selected_feat_col_ls[idx][k]][0] for k in range(len(curr_selected_feat_col_ls[idx]))])
        
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
        
        if not is_ppo:
            argmax = torch.argmax(pred,dim=-1).cpu().numpy()
            
            # __ge__
            pred_v1 = (feat_val)*(argmax/(net.discretize_feat_value_count-1))
            # __le__
            pred_v2 = (1 - feat_val)*(argmax/(net.discretize_feat_value_count-1)) + feat_val
        else:
            if not net.continue_act:
                if train:
                    dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = dist.sample()
                else:
                    argmax = torch.argmax(pred, dim=-1)

                argmax = argmax.cpu().numpy()
                # __ge__
                pred_v1 = (feat_val)*(argmax/(net.discretize_feat_value_count-1))
                # __le__
                pred_v2 = (1 - feat_val)*(argmax/(net.discretize_feat_value_count-1)) + feat_val
            else:
                pred = torch.clamp(pred, min=1e-6, max=1)
                if train:
                    
                    dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

                    # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = torch.clamp(dist.sample(), min=0, max=1)
                else:
                    argmax = pred[:,:,0]
                argmax = argmax.cpu().numpy()

                # __ge__
                pred_v1 = (feat_val)*(argmax)
                # __le__
                pred_v2 = (1 - feat_val)*(argmax) + feat_val

        

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
        
        ret[col_probs_key] = selected_feat_probs
        ret[op_probs_key] = selected_op_probs
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        ret[op_probs_key] = selected_op_probs.data
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred.data
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

        ret[pred_v_key] = pred_v
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op
        
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

    
    
    return ret

def down_weight_features_not_abnormal(selected_feat_probs, abnormal_feature_indicator):
    sample_count = abnormal_feature_indicator.shape[0]
    
    normal_feature_indicator = torch.zeros_like(abnormal_feature_indicator)
    has_abnormal_indicator = torch.sum(abnormal_feature_indicator, dim=-1) > 0
    if len(has_abnormal_indicator.shape) == 1:
        normal_feature_indicator[has_abnormal_indicator] = ~abnormal_feature_indicator[has_abnormal_indicator]
    else:
        for k in range(has_abnormal_indicator.shape[1]):
            normal_feature_indicator[has_abnormal_indicator[:,k].nonzero().view(-1),k] = ~abnormal_feature_indicator[has_abnormal_indicator[:,k].nonzero().view(-1),k]
    # normal_feature_indicator = normal_feature_indicator.bool()
    # if len(selected_Q_feat.shape) == 3:
    #     # adjusted_bias2 = adjusted_bias2 + abnormal_feature_indicator.view(sample_count, 1,-1)*0.5
    #     selected_feat_probs = selected_feat_probs - adjusted_bias2
    # else:
        # adjusted_bias2 = adjusted_bias2 + abnormal_feature_indicator.view(sample_count,-1)*0.5
    selected_feat_probs[normal_feature_indicator] = selected_feat_probs[normal_feature_indicator]*0.2
    selected_feat_probs[abnormal_feature_indicator] = selected_feat_probs[abnormal_feature_indicator]*2
    
    return selected_feat_probs

    
def down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, removed_feat_ls):
    adjusted_bias2 = torch.zeros_like(selected_feat_probs)
    adjusted_bias = torch.zeros_like(selected_Q_feat)

    feat_id_tensor = []

    for feat in removed_feat_ls:
        if feat in net.grammar_token_val_to_num["num_feat"]:
            feat_id = net.grammar_token_val_to_num["num_feat"][feat]
            feat_id_tensor.append(feat_id)

    feat_id_tensor = torch.tensor(feat_id_tensor).to(DEVICE)
    feat_id_tensor_onehot = torch.sum(torch.nn.functional.one_hot(feat_id_tensor, selected_Q_feat.shape[-1]), dim=0)

    if len(selected_Q_feat.shape) == 3:
        
        adjusted_bias = adjusted_bias + feat_id_tensor_onehot.view(1, 1,-1)*0.8 
        adjusted_bias2 = adjusted_bias2 + feat_id_tensor_onehot.view(1, 1,-1)*0.5

        selected_Q_feat = selected_Q_feat - adjusted_bias
        selected_feat_probs = selected_feat_probs - adjusted_bias2
    else:
        adjusted_bias = adjusted_bias + feat_id_tensor_onehot.view(1,-1)*0.8 
        adjusted_bias2 = adjusted_bias2 + feat_id_tensor_onehot.view(1,-1)*0.5

        selected_Q_feat = selected_Q_feat - adjusted_bias
        selected_feat_probs = selected_feat_probs - adjusted_bias2

    return selected_Q_feat, selected_feat_probs

def forward_main0(net, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False, X_pd_full2=None):
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            if net.feat_group_names is None:
                output_num = net.num_feats
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

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feats, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.feat_group_num, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)

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
                prev_program_ids = torch.div(selected_feat_col, net.num_feats, rounding_mode='floor')
                curr_selected_feat_col = selected_feat_col%net.num_feats
            else:
                prev_program_ids = torch.div(selected_feat_col, net.feat_group_num, rounding_mode='floor')
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

    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=DEVICE)
        else:
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
    #         selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=DEVICE)
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_feat_point_ls = []
    selected_num_col_boolean_ls = []
    if init:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k]< net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len]  for k in range(len(selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    else:
        selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k] < net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len] for k in range(len(selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    
    selected_num_feat_tensors_bool = torch.tensor(selected_num_col_boolean_ls).to(DEVICE)            
    if net.feat_bound_point_ls is not None:
        selected_feat_point_tensors = torch.tensor(selected_feat_point_ls,dtype=torch.float).to(DEVICE)
        
        selected_feat_point_tensors_min = selected_feat_point_tensors[:,:,0]
        selected_feat_point_tensors_max = selected_feat_point_tensors[:,:,-1]
    else:
        selected_feat_point_tensors = None
        selected_feat_point_tensors_min = None
        selected_feat_point_tensors_max = None
    feat_val = []
    for idx in range(len(selected_col_ls)):
        # curr_feat_val = torch.tensor([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(DEVICE)
        curr_feat_val = torch.tensor([X_pd_full2[idx][selected_feat_col_ls[idx][k]].item() for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(DEVICE)
        feat_val.append(curr_feat_val)

    feat_val = torch.stack(feat_val, dim=0)


    if not eval:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)
    else:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)

    pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    if net.prefer_smaller_range:
        
        regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
        regularized_coeff = torch.stack(regularized_coeff, dim=1)
        pred_probs_vals = pred_probs_vals*regularized_coeff
        pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    # pred_Q_vals = pred_Q_vals*selected_num_feat_tensors_bool + 0*(1-selected_num_feat_tensors_bool)
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
    if not eval:
        


        # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        if not is_ppo:
            argmax = torch.argmax(pred_probs_vals,dim=-1)
            
            
            if net.feat_bound_point_ls is None:
                pred_v = argmax/(net.discretize_feat_value_count-1)
            else:
                pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                
            # pred_v = pred_v*selected_cat_feat_tensors + feat_val*(1-selected_cat_feat_tensors)
            # # __ge__
            # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # # __le__
            # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        else:
            if not net.continue_act:
                if train:
                    dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = dist.sample()
                else:
                    argmax = torch.argmax(pred, dim=-1)

                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                # # __ge__
                # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
            else:
                pred = torch.clamp(pred, min=1e-6, max=1)
                if train:
                    
                    dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

                    # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = torch.clamp(dist.sample(), min=0, max=1)
                else:
                    argmax = pred[:,:,0]
                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)

                # # __ge__
                # pred_v1 = (feat_val)*(argmax)
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax) + feat_val
                
        if net.feat_bound_point_ls is None:
            outbound_mask = (feat_val >= 1)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= 0))
            # pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
            # pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        else:
            outbound_mask = (feat_val >= selected_feat_point_tensors_max)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= selected_feat_point_tensors_min))
            
            # pred_v[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] = feat_val[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] + 1e-5
            # pred_v[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] = feat_val[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] - 1e-5
        
        # if len(torch.nonzero(selected_num_feat_tensors_bool == 0)) > 0:
        #     print()
        
        pred_v = pred_v*selected_num_feat_tensors_bool + feat_val*(1-selected_num_feat_tensors_bool)
        
        selected_op = (pred_v <= feat_val).type(torch.long)*selected_num_feat_tensors_bool + 2*(1-selected_num_feat_tensors_bool)

        selected_op_ls = []
            
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] if selected_op_id_ls[idx][k] < 2 else net.grammar_num_to_token_val['cat_op'][0] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    ret[select_num_feat_key] = selected_num_feat_tensors_bool

    if eval:
        ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        # ret[op_probs_key] = selected_op_probs
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred_probs_vals
        else:
            ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = pred_Q_vals.data#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        # ret[op_probs_key] = selected_op_probs.data
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred.data
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

        ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op.data
        
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

        ret[outbound_key] = outbound_mask.cpu()
    
    return ret

def select_sub_tensors_for_each_conjunction(net, full_data, full_selected_ids):
    selected_data = []
    for k in range(net.topk_act):
        curr_selected_data = full_data[torch.arange(len(full_data)), full_selected_ids[:,k].view(-1)]
        selected_data.append(curr_selected_data)
    
    selected_data_tensor = torch.stack(selected_data, dim=1)
    return selected_data_tensor

def forward_main0_opt(net, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False, abnormal_info=None):
    if abnormal_info is not None:
        abnormal_feature_indicator, activated_indicator = abnormal_info
        activated_indicator = activated_indicator.to(DEVICE)
        abnormal_feature_indicator = abnormal_feature_indicator.to(DEVICE)
        
    else:
        abnormal_feature_indicator, activated_indicator = None, None
    
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            if net.feat_group_names is None:
                output_num = net.num_feats
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

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks = mask_atom_representation1(X_pd_full, net.topk_act, net.num_feats, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks = mask_atom_representation1(X_pd_full, net.topk_act, net.feat_group_num, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)

    if len(net.removed_feat_ls) > 0:
        selected_Q_feat, selected_feat_probs = down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, net.removed_feat_ls)
    if abnormal_feature_indicator is not None:
        if init:
            selected_feat_probs = down_weight_features_not_abnormal(selected_feat_probs, abnormal_feature_indicator)
        else:
            selected_feat_probs = down_weight_features_not_abnormal(selected_feat_probs, abnormal_feature_indicator.unsqueeze(1).repeat(1, net.topk_act, 1))
    
    if not eval:
        selected_feat_col = torch.ones([len(selected_feat_probs), net.topk_act], dtype=torch.long, device=DEVICE)*(-1)
        if not is_ppo:
            if init:
                _, sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks], k=net.topk_act, dim=-1)
            else:
                # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
                _,sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), k=net.topk_act, dim=-1)
        else:
            if train:
                sub_selected_feat_col = torch.multinomial(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), net.topk_act, replacement=False)
            else:
                # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
                _,sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), k=net.topk_act, dim=-1)
        selected_feat_col[further_selection_masks] = sub_selected_feat_col
    else:
        selected_feat_col = atom[col_id_key]
        further_selection_masks = atom[further_sel_mask_key]

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
            curr_selected_feat_col = torch.zeros_like(selected_feat_col)
            if net.feat_group_names is None:
                prev_program_ids = torch.div(selected_feat_col[further_selection_masks], net.num_feats, rounding_mode='floor')
                sub_curr_selected_feat_col = selected_feat_col[further_selection_masks]%net.num_feats
            else:
                prev_program_ids = torch.div(selected_feat_col[further_selection_masks], net.feat_group_num, rounding_mode='floor')
                sub_curr_selected_feat_col = selected_feat_col[further_selection_masks]%net.feat_group_num
            curr_selected_feat_col[further_selection_masks] = sub_curr_selected_feat_col
        else:
            prev_program_ids = atom[prev_prog_key]
            curr_selected_feat_col = atom[col_id_key]
        new_hx = []
        seq_ids = torch.arange(pat_count)
        for k in range(net.topk_act):
            # selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            selected_feat_col_onehot[further_selection_masks, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            curr_added_hx = selected_feat_probs[further_selection_masks,prev_program_ids[:,k]]*selected_feat_col_onehot[further_selection_masks, prev_program_ids[:,k]]
            curr_new_hx = torch.zeros([len(selected_feat_probs), curr_added_hx.shape[-1]], dtype=torch.float, device=DEVICE)
            curr_new_hx[further_selection_masks] = curr_added_hx
            new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], curr_new_hx],dim=-1))

        hx = torch.stack(new_hx, dim=1)    

    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=DEVICE)
        else:
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
    #         selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=DEVICE)
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_feat_point_ls = []
    selected_num_col_boolean_ls = []
    if init:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            if further_selection_masks[idx]:
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k]< net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len]  for k in range(len(selected_feat_col_ls[idx]))])
                    selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
                else:
                    selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([])
                selected_num_col_boolean_ls.append([])
                selected_feat_point_ls.append([])
    else:
        selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            if further_selection_masks[idx]:
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k] < net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len] for k in range(len(selected_feat_col_ls[idx]))])
                    selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
                else:
                    selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([])
                selected_num_col_boolean_ls.append([])
                selected_feat_point_ls.append([])
    
    selected_num_feat_tensors_bool = torch.tensor(selected_num_col_boolean_ls).to(DEVICE)            
    if net.feat_bound_point_ls is not None:
        selected_feat_point_tensors = torch.tensor(selected_feat_point_ls,dtype=torch.float).to(DEVICE)
        
        selected_feat_point_tensors_min = selected_feat_point_tensors[:,:,0]
        selected_feat_point_tensors_max = selected_feat_point_tensors[:,:,-1]
    else:
        selected_feat_point_tensors = None
        selected_feat_point_tensors_min = None
        selected_feat_point_tensors_max = None
    feat_val = []
    for idx in range(len(selected_col_ls)):
        # feat_val.append(torch.tensor([X_pd_full[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(DEVICE))
        if further_selection_masks[idx]:
            feat_val.append(torch.tensor([X_pd_full[idx][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))], dtype=torch.float).to(DEVICE))
        else:
            feat_val.append(torch.tensor([-1 for _ in range(net.topk_act)], dtype=torch.float).to(DEVICE))

    feat_val = torch.stack(feat_val, dim=0)


    if not eval:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)
    else:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)

    pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    if net.prefer_smaller_range:
        
        regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
        regularized_coeff = torch.stack(regularized_coeff, dim=1)
        pred_probs_vals = pred_probs_vals*regularized_coeff
        pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    
    if abnormal_feature_indicator is not None and activated_indicator is not None:
        if not init:
            activated_indicator_given_cols = select_sub_tensors_for_each_conjunction(net, activated_indicator, curr_selected_feat_col)
        else:
            activated_indicator_given_cols = select_sub_tensors_for_each_conjunction(net, activated_indicator, selected_feat_col)
        # activated_indicator_given_cols = activated_indicator[torch.arange(len(activated_indicator)), selected_feat_col.view(-1)]
        # activated_indicator_given_cols = activated_indicator_given_cols.unsqueeze(1).repeat(1, net.topk_act, 1)
        pred_probs_vals = down_weight_features_not_abnormal(pred_probs_vals, activated_indicator_given_cols)
        # pred_probs_vals = pred_probs_vals*activated_indicator + 1e-5*(1 - activated_indicator)
        
    # pred_Q_vals = pred_Q_vals*selected_num_feat_tensors_bool + 0*(1-selected_num_feat_tensors_bool)
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
    if not eval:
        


        # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        if not is_ppo:
            argmax = torch.argmax(pred_probs_vals,dim=-1)
            
            
            if net.feat_bound_point_ls is None:
                pred_v = argmax/(net.discretize_feat_value_count-1)
            else:
                pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                
            # pred_v = pred_v*selected_cat_feat_tensors + feat_val*(1-selected_cat_feat_tensors)
            # # __ge__
            # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # # __le__
            # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        else:
            if not net.continue_act:
                if train:
                    dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = dist.sample()
                else:
                    argmax = torch.argmax(pred, dim=-1)

                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                # # __ge__
                # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
            else:
                pred = torch.clamp(pred, min=1e-6, max=1)
                if train:
                    
                    dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

                    # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = torch.clamp(dist.sample(), min=0, max=1)
                else:
                    argmax = pred[:,:,0]
                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)

                # # __ge__
                # pred_v1 = (feat_val)*(argmax)
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax) + feat_val
                
        if net.feat_bound_point_ls is None:
            outbound_mask = (feat_val >= 1)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= 0))
            # pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
            # pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        else:
            outbound_mask = (feat_val >= selected_feat_point_tensors_max)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= selected_feat_point_tensors_min))
            
            # pred_v[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] = feat_val[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] + 1e-5
            # pred_v[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] = feat_val[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] - 1e-5
        
        # if len(torch.nonzero(selected_num_feat_tensors_bool == 0)) > 0:
        #     print()
        
        pred_v = pred_v*selected_num_feat_tensors_bool + feat_val*(1-selected_num_feat_tensors_bool)
        
        selected_op = (pred_v <= feat_val).type(torch.long)*selected_num_feat_tensors_bool + 2*(1-selected_num_feat_tensors_bool)

        selected_op_ls = []
            
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] if selected_op_id_ls[idx][k] < 2 else net.grammar_num_to_token_val['cat_op'][0] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    ret[select_num_feat_key] = selected_num_feat_tensors_bool.cpu()

    if eval:
        ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        
        ret[further_sel_mask_key] = further_selection_masks
        # ret[op_probs_key] = selected_op_probs
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred_probs_vals
        else:
            ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = pred_Q_vals.data.cpu()#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data.cpu()
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data.cpu()
        # ret[op_probs_key] = selected_op_probs.data
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred.data.cpu()
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1).data.cpu()

        ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op.data.cpu()
        
        ret[col_key] = selected_col_ls
        if prev_program_ids is not None:
            ret[prev_prog_key] = prev_program_ids.cpu()
        else:
            ret[prev_prog_key] = prev_program_ids

        ret[outbound_key] = outbound_mask.cpu()
        
        ret[further_sel_mask_key] = further_selection_masks.cpu()
    
    return ret


def forward_main0_2(net, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False):
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            if net.feat_group_names is None:
                output_num = net.num_feats
            else:
                output_num = net.feat_group_num

            if True:
                selected_feat_logit = torch.rand([pat_count, output_num], device=DEVICE)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, output_num], device=DEVICE)

        else:
            selected_feat_logit = net.feat_selector(hx)
    
    else:
        selected_feat_logit = net.feat_selector(hx)

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feats, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.feat_group_num, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)

    if len(net.removed_feat_ls) > 0:
        selected_Q_feat, selected_feat_probs = down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, net.removed_feat_ls)

    if not eval:
        if not is_ppo:
            if True:
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

    if True:
        selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, net.topk_act, 1)
        for k in range(net.topk_act):
            selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        hx = torch.cat([hx.unsqueeze(1).repeat(1, net.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
    else:
        if not eval:
            if net.feat_group_names is None:
                prev_program_ids = torch.div(selected_feat_col, net.num_feats, rounding_mode='floor')
                curr_selected_feat_col = selected_feat_col%net.num_feats
            else:
                prev_program_ids = torch.div(selected_feat_col, net.feat_group_num, rounding_mode='floor')
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

    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=DEVICE)
        else:
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
    #         selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=DEVICE)
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_feat_point_ls = []
    selected_num_col_boolean_ls = []
    if True:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k]< net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len]  for k in range(len(selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    else:
        curr_selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][curr_selected_feat_col_ls[idx][k]] if curr_selected_feat_col_ls[idx][k] < net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][curr_selected_feat_col_ls[idx][k] - net.num_feat_len] for k in range(len(curr_selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if curr_selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(curr_selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[curr_selected_feat_col_ls[idx][k]][0] for k in range(len(curr_selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    
    selected_num_feat_tensors_bool = torch.tensor(selected_num_col_boolean_ls).to(DEVICE)            
    if net.feat_bound_point_ls is not None:
        selected_feat_point_tensors = torch.tensor(selected_feat_point_ls,dtype=torch.float).to(DEVICE)
        
        selected_feat_point_tensors_min = selected_feat_point_tensors[:,:,0]
        selected_feat_point_tensors_max = selected_feat_point_tensors[:,:,-1]
    else:
        selected_feat_point_tensors = None
        selected_feat_point_tensors_min = None
        selected_feat_point_tensors_max = None
    feat_val = []
    for idx in range(len(selected_col_ls)):
        feat_val.append(torch.tensor([X_pd_full.iloc[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(DEVICE))

    feat_val = torch.stack(feat_val, dim=0)


    if not eval:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)
    else:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)

    pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    if net.prefer_smaller_range:
        
        regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
        regularized_coeff = torch.stack(regularized_coeff, dim=1)
        pred_probs_vals = pred_probs_vals*regularized_coeff
        pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    # pred_Q_vals = pred_Q_vals*selected_num_feat_tensors_bool + 0*(1-selected_num_feat_tensors_bool)
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
    if not eval:
        


        # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        if not is_ppo:
            argmax = torch.argmax(pred_probs_vals,dim=-1)
            
            
            if net.feat_bound_point_ls is None:
                pred_v = argmax/(net.discretize_feat_value_count-1)
            else:
                pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                
            # pred_v = pred_v*selected_cat_feat_tensors + feat_val*(1-selected_cat_feat_tensors)
            # # __ge__
            # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # # __le__
            # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        else:
            if not net.continue_act:
                if train:
                    dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = dist.sample()
                else:
                    argmax = torch.argmax(pred, dim=-1)

                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                # # __ge__
                # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
            else:
                pred = torch.clamp(pred, min=1e-6, max=1)
                if train:
                    
                    dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

                    # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = torch.clamp(dist.sample(), min=0, max=1)
                else:
                    argmax = pred[:,:,0]
                # argmax = argmax.cpu().numpy()
                if net.feat_bound_point_ls is None:
                    pred_v = argmax/(net.discretize_feat_value_count-1)
                else:
                    pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)

                # # __ge__
                # pred_v1 = (feat_val)*(argmax)
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax) + feat_val
                
        if net.feat_bound_point_ls is None:
            outbound_mask = (feat_val >= 1)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= 0))
            # pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
            # pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        else:
            outbound_mask = (feat_val >= selected_feat_point_tensors_max)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= selected_feat_point_tensors_min))
            
            # pred_v[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] = feat_val[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] + 1e-5
            # pred_v[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] = feat_val[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] - 1e-5
        
        # if len(torch.nonzero(selected_num_feat_tensors_bool == 0)) > 0:
        #     print()
        
        pred_v = pred_v*selected_num_feat_tensors_bool + feat_val*(1-selected_num_feat_tensors_bool)
        
        selected_op = (pred_v <= feat_val).type(torch.long)*selected_num_feat_tensors_bool + 2*(1-selected_num_feat_tensors_bool)

        selected_op_ls = []
            
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] if selected_op_id_ls[idx][k] < 2 else net.grammar_num_to_token_val['cat_op'][0] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if True:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    ret[select_num_feat_key] = selected_num_feat_tensors_bool

    if eval:
        ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        # ret[op_probs_key] = selected_op_probs
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred_probs_vals
        else:
            ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = pred_Q_vals.data#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        # ret[op_probs_key] = selected_op_probs.data
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred.data
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

        ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op.data
        
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

        ret[outbound_key] = outbound_mask.cpu()
    
    return ret

def forward_main2(net, hx, eval, epsilon, program, selected_feat_logit, atom, pat_count, X_pd_full, init=False, is_ppo=False, train=False):
    
    
    
    
    if not eval:
        if np.random.rand() < epsilon and not is_ppo:
            if init:
                selected_feat_logit = torch.rand([pat_count, net.num_feats], device=DEVICE)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, net.num_feats], device=DEVICE)
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            # selected_feat_logit = torch.rand([pat_count, self.num_feat_len], device=DEVICE)
        else:
            selected_feat_logit = net.feat_selector(hx)
    else:
        selected_feat_logit = net.feat_selector(hx)
        
    selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feats, net.op_start_pos, program, selected_feat_logit, init=init)

    # selected_Q_feat = selected_feat_probs

    # selected_feat_probs = selected_feat_probs
    if not eval:
        # if self.topk_act == 1:
        #     selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        # else:
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
            # dist = torch.distributions.Categorical(selected_feat_probs)
            # dist.sample()
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
            prev_program_ids = torch.div(selected_feat_col, net.num_feats, rounding_mode='floor')
            curr_selected_feat_col = selected_feat_col%net.num_feats
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
        if np.random.rand() < epsilon and not is_ppo:
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
        if not is_ppo:
            selected_op = torch.argmax(selected_op_probs, dim=-1)
        else:
            if train:
                dist = torch.distributions.Categorical(selected_op_probs)
                selected_op = dist.sample()
            else:
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
        if np.random.rand() < epsilon and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=DEVICE)
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
        
        if not is_ppo:
            argmax = torch.argmax(pred,dim=-1).cpu().numpy()
            
            # __ge__
            pred_v1 = (feat_val)*(argmax/(net.discretize_feat_value_count-1))
            # __le__
            pred_v2 = (1 - feat_val)*(argmax/(net.discretize_feat_value_count-1)) + feat_val
        else:
            if not net.continue_act:
                if train:
                    dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = dist.sample()
                else:
                    argmax = torch.argmax(pred, dim=-1)

                argmax = argmax.cpu().numpy()
                # __ge__
                pred_v1 = (feat_val)*(argmax/(net.discretize_feat_value_count-1))
                # __le__
                pred_v2 = (1 - feat_val)*(argmax/(net.discretize_feat_value_count-1)) + feat_val
            else:
                pred = torch.clamp(pred, min=1e-6, max=1)
                if train:
                    
                    dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

                    # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
                    argmax = torch.clamp(dist.sample(), min=0, max=1)
                else:
                    argmax = pred[:,:,0]
                argmax = argmax.cpu().numpy()

                # __ge__
                pred_v1 = (feat_val)*(argmax)
                # __le__
                pred_v2 = (1 - feat_val)*(argmax) + feat_val

        

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
        
        ret[col_probs_key] = selected_feat_probs
        ret[op_probs_key] = selected_op_probs
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        ret[op_probs_key] = selected_op_probs.data
        if net.continue_act and is_ppo:
            ret[pred_probs_key] = pred.data
        else:
            ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

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
    
    
class Critic_mlp(nn.Module):
    def __init__(self, lang,  program_max_len, latent_size, topk_act):
        super(Critic_mlp, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        encoding_program_init(self)
        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        self.topk_act = topk_act
        if latent_size > 0:
            self.embedding = TokenNetwork3(full_input_size, latent_size, self.topk_act)
        else:
            self.embedding = nn.Linear(full_input_size,1)
        # self.decoder = nn.Linear(latent_size,1)
        self.first_prog_embed = torch.tensor([0]*self.ATOM_VEC_LENGTH, device =DEVICE, dtype=torch.float)
        self.to(DEVICE)

    def forward(self, features, program, init=False):
        if type(features) is not torch.Tensor:
            features,_ = features
            features = features.to(DEVICE)
        else:
            features = features.to(DEVICE)
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            program = [self.first_prog_embed.unsqueeze(0).repeat(len(features), 1)]
        total_feat_prog_array_len =features[0].shape[0] + self.program_max_len*program[0].shape[-1]
        concat_program_tensor = torch.cat(program,dim=-1)
        
        
        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([features.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:features[0].shape[0]] = features
            hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        # if len(program) == 0:
        #     program = [self.first_prog_embed]

        
        # hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*self.ATOM_VEC_LENGTH, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # if len(program) > 0:
        #     hx[:, features[0].shape[0]:len(program)*self.ATOM_VEC_LENGTH+features[0].shape[0]] = torch.cat(program,dim=-1)
        ret = torch.tanh(self.embedding(hx))

        ret,_ = torch.topk(ret.view(len(ret), -1), k = self.topk_act, dim=-1)

        return ret


class Critic_transformer(nn.Module):
    def __init__(self, lang,  program_max_len, latent_size, topk_act, has_embeddings, category_count, numeric_count, tf_latent_size, pretrained_model_path=None):
        super(Critic_transformer, self).__init__()
        self.lang = lang
        self.program_max_len=program_max_len
        encoding_program_init(self)
        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)
        full_input_size = tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH
        self.topk_act = topk_act

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


        if latent_size > 0:
            self.embedding = TokenNetwork3(full_input_size, latent_size, self.topk_act)
        else:
            self.embedding = nn.Linear(full_input_size,self.topk_act)
        # self.decoder = nn.Linear(latent_size,1)
        self.first_prog_embed = torch.tensor([0]*self.ATOM_VEC_LENGTH, device =DEVICE, dtype=torch.float)
        self.to(DEVICE)

    def forward(self, input_data, program, init=False):
        if type(input_data) is not torch.Tensor:
            _, features = input_data
            features = features.to(DEVICE)
        else:
            features = input_data
            features = features.to(DEVICE)
            features = self.input_embedding(None, features, return_embedding=True)

        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            program = [self.first_prog_embed.unsqueeze(0).repeat(len(features), 1)]
        total_feat_prog_array_len =features[0].shape[0] + self.program_max_len*program[0].shape[-1]
        concat_program_tensor = torch.cat(program,dim=-1)
        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([features.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:features[0].shape[0]] = features
            hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        # if len(program) == 0:
        #     program = [self.first_prog_embed]

        
        # hx = torch.zeros(features.shape[0], features[0].shape[0] + self.program_max_len*self.ATOM_VEC_LENGTH, device=DEVICE)
        # hx[:,0:features[0].shape[0]] = features
        # if len(program) > 0:
        #     hx[:, features[0].shape[0]:len(program)*self.ATOM_VEC_LENGTH+features[0].shape[0]] = torch.cat(program,dim=-1)
        ret = torch.tanh(self.embedding(hx))

        ret,_ = torch.topk(ret.view(len(ret), -1), k = self.topk_act, dim=-1)

        return ret

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

class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        x = torch.sum(x, dim=1, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out

def create_deep_set_net_for_programs(net, ATOM_VEC_LENGTH, latent_size):
    encoder = TokenNetwork3(ATOM_VEC_LENGTH, latent_size, latent_size)
    # decoder = torch.nn.Linear(latent_size, ATOM_VEC_LENGTH)
    decoder = torch.nn.Identity(latent_size)
    net.program_net = InvariantModel(encoder, decoder)


class RLSynthesizerNetwork_mlp(nn.Module):
    def init_without_feat_groups(self, lang, args,  latent_size, num_feat_count, category_sum_count, feat_range_mappings, continue_act=False):
        super(RLSynthesizerNetwork_mlp, self).__init__()
        self.topk_act=args.topk_act
        self.lang = lang
        self.program_max_len=args.program_max_len
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
            if i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))
            else:
                if i in self.lang.syntax["cat_feat"]:
                    self.grammar_num_to_token_val[i] = []
                    self.grammar_token_val_to_num[i] = []
                else:
                    self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                    self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
                
        self.op_start_pos = -1
        self.num_start_pos = -1
        self.cat_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                for option in list(options_dict.keys()):        
                    if self.op_start_pos < 0:
                        self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:                    
                if decision in self.lang.syntax["num_feat"]:
                    if self.num_start_pos < 0:
                        self.num_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
                else:
                    if self.cat_start_pos < 0:
                        self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        self.num_feat_len  = num_feat_count#len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        self.cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = self.num_feat_len+self.cat_feat_len
        self.all_input_feat_len = self.num_feat_len+category_sum_count
        self.num_feats = num_features
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        # full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        full_input_size = self.all_input_feat_len + latent_size# self.ATOM_VEC_LENGTH
        self.full_input_size = full_input_size
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.num_feats)
            # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.num_feats)
            # self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)
    
    def init_with_feat_groups(self, lang, args, latent_size, feat_range_mappings, continue_act=False, feat_group_names=None):
        super(RLSynthesizerNetwork_mlp, self).__init__()
        self.topk_act=args.topk_act
        self.lang = lang
        self.program_max_len=args.program_max_len
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
        #     else:
        #         self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
        #         self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1

        decision = "num_op"
        start = self.ATOM_VEC_LENGTH


        for option in list(self.lang.syntax[decision].keys()):        
            if self.op_start_pos < 0:
                self.op_start_pos = self.ATOM_VEC_LENGTH
            
            self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
            self.ATOM_VEC_LENGTH += 1
        self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        for group_idx in range(len(feat_group_names)):
            start = self.ATOM_VEC_LENGTH
            if self.num_start_pos < 0:
                    self.num_start_pos = self.ATOM_VEC_LENGTH
                
            self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
            self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)
        
        
        
        # for decision, options_dict in self.lang.syntax.items():
        #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
        #         continue
        #     # if decision == "num_op":
        #     #     continue
        #     start = self.ATOM_VEC_LENGTH


        #     if not decision in self.lang.syntax["num_feat"]:
        #         for option in list(options_dict.keys()):        
        #             if self.op_start_pos < 0:
        #                 self.op_start_pos = self.ATOM_VEC_LENGTH
                    
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #     else:
        #         if self.num_start_pos < 0:
        #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
        #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = num_feat_len+cat_feat_len
        self.num_feats = num_features
        self.feat_group_num = len(feat_group_names)
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            # if latent_size > 0:
            #     if not continue_act:
            #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, self.discretize_feat_value_count)
            #     else:
            #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, 1)
            # else:
            #     if not continue_act:
            #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, self.discretize_feat_value_count)
            #     else:
            #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, 1)
                    
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.feat_group_num)
            self.op_selector = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.feat_group_num)
            self.op_selector = nn.Linear(full_input_size + self.feat_group_num, self.op_num)
        
        self.to(device=DEVICE)

    def __init__(self, lang, args, model_config, rl_config, num_feat_count, category_sum_count, feat_range_mappings, continue_act=False, feat_group_names=None, removed_feat_ls=None):
        self.feat_bound_point_ls = args.feat_bound_point_ls
        if self.feat_bound_point_ls is None:
            self.discretize_feat_value_count = rl_config["discretize_feat_value_count"]
        else:
            self.discretize_feat_value_count = len(list(self.feat_bound_point_ls.values())[0])

        print("discrete feat value count::", self.discretize_feat_value_count)
        self.feat_group_names = feat_group_names
        self.removed_feat_ls = removed_feat_ls
        self.prefer_smaller_range=args.prefer_smaller_range
        self.prefer_smaller_range_coeff = args.prefer_smaller_range_coeff
        if self.prefer_smaller_range:
            self.selected_vals = torch.tensor([k/(self.discretize_feat_value_count-1) for k in range(self.discretize_feat_value_count)]).to(DEVICE)
        if feat_group_names is None:
            self.init_without_feat_groups(lang,  args, model_config["latent_size"], num_feat_count, category_sum_count, feat_range_mappings, continue_act=continue_act)
        else:
            self.init_with_feat_groups(lang,  args, model_config["latent_size"], feat_range_mappings, continue_act=continue_act, feat_group_names=feat_group_names)

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

    def vector_ls_to_str0(self, program):
        return vector_ls_to_str0_main(self, program)
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

                    pred_lb = (feat_val)*(argmax/(self.discretize_feat_value_count-1))

                    ret[token] = [pred_lb,pred]
                elif token.endswith("_ub"):

                    # range_max = self.feat_range_mappings[token][1]
                    # range_min = self.feat_range_mappings[token][0]

                    

                    

                    # pred_lb = pred_lb*(range_max - range_min) + range_min

                    argmax = torch.argmax(pred).item()

                    pred_ub = (1 - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

                    # pred_ub = pred_ub*(range_max - range_min) + range_min
                    # pred_ub = (range_max - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

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

    def mask_atom_representation(self, program_ls, feat_pred_logit):
        
        op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feats], device = DEVICE)
        op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feats], device = DEVICE)
        
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
        
        
    def forward_ls0_back(self, features,X_pd_full, program, atom, epsilon=0, eval=False, existing_atom=None, init=False, is_ppo=False, train=False):
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
        
        return forward_main0(self, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train)

    def forward_ls0(self, features,X_pd_full, program, outbound_mask_ls, atom, epsilon=0, eval=False, existing_atom=None, init=False, is_ppo=False, train=False, X_pd_full2=None, abnormal_info=None):
        features,_,_ = features
        features = features.to(DEVICE)
        pat_count = features.shape[0]
        
        
        total_feat_prog_array_len = self.full_input_size#program[0].shape[-1]

        # selecting feature
        
        concat_program_tensor = torch.cat(program,dim=-1)

        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([features.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:features[0].shape[0]] = features
            hx[:, features[0].shape[0]:] = self.program_net(torch.stack(program, dim=1).to(DEVICE)).squeeze(1)
            # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,features[0].shape[0]:] = self.program_net(torch.cat(program, dim=1).to(DEVICE))
            # hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        
        # return forward_main0(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train, X_pd_full2=X_pd_full2)
        return forward_main0_opt(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full2, init=init,is_ppo=is_ppo, train=train)


class RLSynthesizerNetwork_mlp2(nn.Module):
    def init_without_feat_groups(self, lang,  program_max_len, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=1, continue_act=False):
        super(RLSynthesizerNetwork_mlp2, self).__init__()
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
            if i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))
            else:
                if i in self.lang.syntax["cat_feat"]:
                    self.grammar_num_to_token_val[i] = []
                    self.grammar_token_val_to_num[i] = []
                else:
                    self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                    self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
                
        self.op_start_pos = -1
        self.num_start_pos = -1
        self.cat_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH


            if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                for option in list(options_dict.keys()):        
                    if self.op_start_pos < 0:
                        self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:                    
                if decision in self.lang.syntax["num_feat"]:
                    if self.num_start_pos < 0:
                        self.num_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
                else:
                    if self.cat_start_pos < 0:
                        self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        self.num_feat_len  = num_feat_count#len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        self.cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = self.num_feat_len+self.cat_feat_len
        self.all_input_feat_len = self.num_feat_len+category_sum_count
        self.num_feats = num_features
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        # full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        full_input_size = self.all_input_feat_len + latent_size# self.ATOM_VEC_LENGTH
        self.full_input_size = full_input_size
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(full_input_size + self.num_feats, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(full_input_size + self.num_feats, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.num_feats)
            # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(full_input_size, self.num_feats)
            # self.op_selector = nn.Linear(full_input_size + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)
    
    # def init_with_feat_groups(self, lang,  program_max_len, latent_size, dropout_p, feat_range_mappings, topk_act=1, continue_act=False, feat_group_names=None):
    #     super(RLSynthesizerNetwork_mlp, self).__init__()
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
    #     #         self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
    #     #         self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

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
    #     self.num_feats = num_features
    #     self.feat_group_num = len(feat_group_names)
        
    #     # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
    #     # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

    #     # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

    #     # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
    #     net_maps = {}
    #     full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
    #     # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         # if latent_size > 0:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, self.discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num + self.op_num, latent_size, 1)
    #         # else:
    #         #     if not continue_act:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, self.discretize_feat_value_count)
    #         #     else:
    #         #         net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num + self.op_num, 1)
                    
    #         if latent_size > 0:
    #             if not continue_act:
    #                 net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, 1)
    #         else:
    #             if not continue_act:
    #                 net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = nn.Linear(full_input_size + self.feat_group_num, 1)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

    #     self.token_nets = nn.ModuleDict(net_maps)
    #     self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
    #     create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        
    #     if latent_size > 0:
    #         self.feat_selector = TokenNetwork3(full_input_size, latent_size, self.feat_group_num)
    #         self.op_selector = TokenNetwork3(full_input_size + self.feat_group_num, latent_size, self.op_num)
    #     else:
    #         self.feat_selector = nn.Linear(full_input_size, self.feat_group_num)
    #         self.op_selector = nn.Linear(full_input_size + self.feat_group_num, self.op_num)
        
    #     self.to(device=DEVICE)

    def __init__(self, lang,  program_max_len, latent_size, dropout_p,num_feat_count, category_sum_count, feat_range_mappings, topk_act=1, continue_act=False, feat_group_names=None, removed_feat_ls=None, prefer_smaller_range=False, prefer_smaller_range_coeff=0.5, args=None, discretize_feat_value_count=20):
        self.feat_bound_point_ls = args.feat_bound_point_ls
        if self.feat_bound_point_ls is None:
            self.discretize_feat_value_count = discretize_feat_value_count
        else:
            self.discretize_feat_value_count = len(list(self.feat_bound_point_ls.values())[0])
        self.feat_group_names = feat_group_names
        self.removed_feat_ls = removed_feat_ls
        self.prefer_smaller_range=prefer_smaller_range
        self.prefer_smaller_range_coeff = prefer_smaller_range_coeff
        if self.prefer_smaller_range:
            self.selected_vals = torch.tensor([k/(self.discretize_feat_value_count-1) for k in range(self.discretize_feat_value_count)]).to(DEVICE)
        # if feat_group_names is None:
        self.init_without_feat_groups(lang,  program_max_len, latent_size, dropout_p, num_feat_count, category_sum_count, feat_range_mappings, topk_act=topk_act, continue_act=continue_act)
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

    def vector_ls_to_str0(self, program):
        return vector_ls_to_str0_main(self, program)
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

        hx = torch.zeros(features[0].shape[0] + self.program_max_len*program[0].shape[0], device=DEVICE)
        hx[0:features[0].shape[0]] = features[0]
        hx[features[0].shape[0]:len(program)*program[0].shape[0]+features[0].shape[0]] = torch.cat(program)
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

                    pred_lb = (feat_val)*(argmax/(self.discretize_feat_value_count-1))

                    ret[token] = [pred_lb,pred]
                elif token.endswith("_ub"):

                    # range_max = self.feat_range_mappings[token][1]
                    # range_min = self.feat_range_mappings[token][0]

                    

                    

                    # pred_lb = pred_lb*(range_max - range_min) + range_min

                    argmax = torch.argmax(pred).item()

                    pred_ub = (1 - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

                    # pred_ub = pred_ub*(range_max - range_min) + range_min
                    # pred_ub = (range_max - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

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

    def mask_atom_representation(self, program_ls, feat_pred_logit):
        
        op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feats], device = DEVICE)
        op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], self.num_feats], device = DEVICE)
        
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
        
        
    def forward_ls0_back(self, features,X_pd_full, program, atom, epsilon=0, eval=False, existing_atom=None, init=False, is_ppo=False, train=False):
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
        
        return forward_main0(self, hx, eval, epsilon, program, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train)

    def forward_ls0(self, features,X_pd_full, program, outbound_mask_ls, atom, epsilon=0, eval=False, existing_atom=None, init=False, is_ppo=False, train=False):
        # features,_,_ = features
        features = features.to(DEVICE)
        pat_count = features.shape[0]
        
        
        total_feat_prog_array_len = self.full_input_size#program[0].shape[-1]

        # selecting feature
        
        concat_program_tensor = torch.cat(program,dim=-1)

        # if init:
        # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
        hx = torch.zeros([features.shape[0], total_feat_prog_array_len], device=DEVICE)
        hx[:,0:features[0].shape[0]] = features
        if init:
            hx[:, features[0].shape[0]:] = self.program_net(torch.stack(program, dim=1).to(DEVICE)).squeeze(1)
        else:
            hx[:, features[0].shape[0]:] = self.program_net(torch.cat(program, dim=1).to(DEVICE)).squeeze(1)
        # hx[:, features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        # else:
        #     hx = torch.zeros([features.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
        #     hx[:,:,0:features[0].shape[0]] = features.unsqueeze(1).repeat(1,self.topk_act,1)
        #     hx[:,:,features[0].shape[0]:] = self.program_net(torch.cat(program, dim=1).to(DEVICE))
            # hx[:,:,features[0].shape[0]:len(program)*program[0].shape[-1]+features[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        
        return forward_main0_2(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=True,is_ppo=is_ppo, train=train)


class RLSynthesizerNetwork_transformer(nn.Module):
    # def init_without_feat_groups0(self,lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False, args=None):
    #     super(RLSynthesizerNetwork_transformer, self).__init__()
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
    #     # self.removed_feat_ls = removed_feat_ls
    #     # for decision, options_dict in self.lang.syntax.items():
    #     #     start = self.ATOM_VEC_LENGTH
    #     #     for option in list(options_dict.keys()):
    #     #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #     #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #     #         self.ATOM_VEC_LENGTH += 1
    #     #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.feat_range_mappings = feat_range_mappings
    #     for i,v in self.lang.syntax.items():
    #         if i in self.lang.syntax["num_feat"]:
    #             self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
    #             self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))
    #         else:
    #             if i in self.lang.syntax["cat_feat"]:
    #                 self.grammar_num_to_token_val[i] = []
    #                 self.grammar_token_val_to_num[i] = []
    #             else:
    #                 self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
    #                 self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
    #         #     self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
    #         # else:
    #         #     self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
    #         #     self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

    #     self.op_start_pos = -1
    #     self.num_start_pos = -1
    #     self.cat_start_pos = -1

    #     for decision, options_dict in self.lang.syntax.items():
    #         if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
    #             continue
    #         # if decision == "num_op":
    #         #     continue
    #         start = self.ATOM_VEC_LENGTH


    #         # if not decision in self.lang.syntax["num_feat"]:
    #         #     for option in list(options_dict.keys()):        
    #         #         if self.op_start_pos < 0:
    #         #             self.op_start_pos = self.ATOM_VEC_LENGTH
                    
    #         #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #         #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #         #         self.ATOM_VEC_LENGTH += 1
    #         # else:
    #         #     if self.num_start_pos < 0:
    #         #         self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #         #     self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #         #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #         #     self.ATOM_VEC_LENGTH += 1
    #         if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
    #             for option in list(options_dict.keys()):        
    #                 if self.op_start_pos < 0:
    #                     self.op_start_pos = self.ATOM_VEC_LENGTH
                    
    #                 self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #                 self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #                 self.ATOM_VEC_LENGTH += 1
    #         else:                    
    #             if decision in self.lang.syntax["num_feat"]:
    #                 if self.num_start_pos < 0:
    #                     self.num_start_pos = self.ATOM_VEC_LENGTH
                    
    #                 self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #                 self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #                 self.ATOM_VEC_LENGTH += 1
    #             else:
    #                 if self.cat_start_pos < 0:
    #                     self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
    #                 self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #                 self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #                 self.ATOM_VEC_LENGTH += 1
    #         self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
    #     self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
    #     self.ATOM_VEC_LENGTH += 1

    #     # self.column_ls = list(self.lang.syntax["num_feat"].keys())

    #     self.num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
    #     self.cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
    #     num_features = self.num_feat_len+self.cat_feat_len
    #     self.num_feats = num_features
    #     # self.all_input_feat_len = self.num_feat_len+category_sum_count
        
    #     # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
    #     # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

    #     # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

    #     # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
    #     net_maps = {}
    #     full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
    #     # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
    #                 depth = 3,                          # depth, paper recommended 6
    #                 heads = 4,                          # heads, paper recommends 8
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
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, self.discretize_feat_value_count)
    #         if latent_size > 0:
    #             if not continue_act:
    #                 net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, latent_size, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, latent_size, 1)
    #         else:
    #             if not continue_act:
    #                 net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, 1)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

    #     self.token_nets = nn.ModuleDict(net_maps)
    #     self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        
    #     if latent_size > 0:
    #         self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.num_feats)
    #         self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, latent_size, self.op_num)
    #     else:
    #         self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feats)
    #         self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, self.op_num)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


    #     # self.token_nets = nn.ModuleDict(net_maps)
    #     # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
    #     # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

    #     # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
    #     self.to(device=DEVICE)
        
    def init_without_feat_groups2(self,lang, args, latent_size, tf_latent_size, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, continue_act=False):
        super(RLSynthesizerNetwork_transformer, self).__init__()
        self.topk_act = args.topk_act
        self.lang = lang
        self.program_max_len=args.program_max_len
        self.grammar_num_to_token_val = {}#{i:{num:option for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_val_to_num = {}#{i:{option:num for num,option in enumerate(list(v.keys()))} for i,v in self.lang.syntax.items()}
        self.grammar_token_to_pos = {}
        self.grammar_pos_to_token = {}
        self.ATOM_VEC_LENGTH = 0
        self.one_hot_token_bounds = {}
        self.continue_act = continue_act

        self.feat_range_mappings = feat_range_mappings
        for i,v in self.lang.syntax.items():
            if i in self.lang.syntax["num_feat"]:
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))
            else:
                if i in self.lang.syntax["cat_feat"]:
                    self.grammar_num_to_token_val[i] = []
                    self.grammar_token_val_to_num[i] = []
                else:
                    self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
                    self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            # if not i in self.lang.syntax["num_feat"]:
            #     self.grammar_num_to_token_val[i] = {num:option for num,option in enumerate(list(v.keys()))}
            #     self.grammar_token_val_to_num[i] = {option:num for num,option in enumerate(list(v.keys()))}
            # else:
            #     self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
            #     self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1
        self.cat_start_pos = -1

        for decision, options_dict in self.lang.syntax.items():
            if not (decision == "num_op" or decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                continue
            # if decision == "num_op":
            #     continue
            start = self.ATOM_VEC_LENGTH

            if not (decision in self.lang.syntax["num_feat"] or decision in self.lang.syntax["cat_feat"]):
                for option in list(options_dict.keys()):        
                    if self.op_start_pos < 0:
                        self.op_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
                    self.ATOM_VEC_LENGTH += 1
            else:                    
                if decision in self.lang.syntax["num_feat"]:
                    if self.num_start_pos < 0:
                        self.num_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
                else:
                    if self.cat_start_pos < 0:
                        self.cat_start_pos = self.ATOM_VEC_LENGTH
                    
                    self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
                    self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
                    self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        self.num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        self.cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = self.num_feat_len+self.cat_feat_len
        self.num_feats = num_features
        # self.all_input_feat_len = self.num_feat_len+category_sum_count
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
                    depth = 3,                          # depth, paper recommended 6
                    heads = 4,                          # heads, paper recommends 8
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, self.discretize_feat_value_count)
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + latent_size + self.num_feats, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + latent_size + self.num_feats, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(tf_latent_size + latent_size + self.num_feats, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(tf_latent_size + latent_size + self.num_feats, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        # self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        create_deep_set_net_for_programs(self, self.ATOM_VEC_LENGTH, latent_size)
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(tf_latent_size + latent_size, latent_size, self.num_feats)
            # self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(tf_latent_size + latent_size, self.num_feats)
            # self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, self.op_num)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


        # self.token_nets = nn.ModuleDict(net_maps)
        # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

        # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)

    # def init_without_feat_groups(self,lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False):
    #     super(RLSynthesizerNetwork_transformer, self).__init__()
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
    #         else:
    #             self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
    #             self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

    #     self.op_start_pos = -1
    #     self.num_start_pos = -1

    #     for decision, options_dict in self.lang.syntax.items():
    #         if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
    #             continue
    #         # if decision == "num_op":
    #         #     continue
    #         start = self.ATOM_VEC_LENGTH


    #         if not decision in self.lang.syntax["num_feat"]:
    #             for option in list(options_dict.keys()):        
    #                 if self.op_start_pos < 0:
    #                     self.op_start_pos = self.ATOM_VEC_LENGTH
                    
    #                 self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
    #                 self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
    #                 self.ATOM_VEC_LENGTH += 1
    #         else:
    #             if self.num_start_pos < 0:
    #                 self.num_start_pos = self.ATOM_VEC_LENGTH
                
    #             self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
    #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
    #             self.ATOM_VEC_LENGTH += 1
    #         self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
    #     self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
    #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
    #     self.ATOM_VEC_LENGTH += 1

    #     # self.column_ls = list(self.lang.syntax["num_feat"].keys())

    #     num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
    #     cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
    #     num_features = num_feat_len+cat_feat_len
    #     self.num_feats = num_feat_len
        
    #     # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
    #     # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

    #     # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

    #     # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
    #     net_maps = {}
    #     full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
    #     # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #         # if not i in self.lang.syntax["num_feat"]:
    #         #     net_maps[i] = TokenNetwork(latent_size, len(v))
    #         # else:
    #         # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, self.discretize_feat_value_count)
    #         if latent_size > 0:
    #             if not continue_act:
    #                 net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats + self.op_num, latent_size, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats + self.op_num, latent_size, 1)
    #         else:
    #             if not continue_act:
    #                 net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats + self.op_num, self.discretize_feat_value_count)
    #             else:
    #                 net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats + self.op_num, 1)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

    #     self.token_nets = nn.ModuleDict(net_maps)
    #     self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        
    #     if latent_size > 0:
    #         self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.num_feats)
    #         self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, latent_size, self.op_num)
    #     else:
    #         self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feats)
    #         self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feats, self.op_num)
    #             # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
    #             # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
    #             # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


    #     # self.token_nets = nn.ModuleDict(net_maps)
    #     # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
    #     # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

    #     # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
    #     # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
    #     self.to(device=DEVICE)
    # def init_with_feat_groups(self,lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, topk_act=1, continue_act=False, feat_group_names=None):
        super(RLSynthesizerNetwork_transformer, self).__init__()
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
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1

        decision = "num_op"
        start = self.ATOM_VEC_LENGTH


        for option in list(self.lang.syntax[decision].keys()):        
            if self.op_start_pos < 0:
                self.op_start_pos = self.ATOM_VEC_LENGTH
            
            self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
            self.ATOM_VEC_LENGTH += 1
        self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        for group_idx in range(len(feat_group_names)):
            start = self.ATOM_VEC_LENGTH
            if self.num_start_pos < 0:
                    self.num_start_pos = self.ATOM_VEC_LENGTH
                
            self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
            self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)


        # for group_idx in range(len(feat_group_names)):
        #     start = self.ATOM_VEC_LENGTH
        #     if self.num_start_pos < 0:
        #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
        #     self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
        #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
        #     self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)

        # for decision, options_dict in self.lang.syntax.items():
        #     start = self.ATOM_VEC_LENGTH


        #     for option in list(options_dict.keys()):        
        #         if self.op_start_pos < 0:
        #             self.op_start_pos = self.ATOM_VEC_LENGTH
                
        #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        # for decision, options_dict in self.lang.syntax.items():
        #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
        #         continue
        #     # if decision == "num_op":
        #     #     continue
        #     start = self.ATOM_VEC_LENGTH


        #     if not decision in self.lang.syntax["num_feat"]:
        #         for option in list(options_dict.keys()):        
        #             if self.op_start_pos < 0:
        #                 self.op_start_pos = self.ATOM_VEC_LENGTH
                    
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #     else:
        #         if self.num_start_pos < 0:
        #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
        #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        # num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        # cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = len(feat_group_names)# num_feat_len+cat_feat_len
        self.num_feats = num_features
        self.feat_group_num = len(feat_group_names)
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, self.discretize_feat_value_count)
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num + self.op_num, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num + self.op_num, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num + self.op_num, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num + self.op_num, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.feat_group_num)
            self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.feat_group_num)
            self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, self.op_num)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


        # self.token_nets = nn.ModuleDict(net_maps)
        # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

        # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)

    def init_with_feat_groups2(self,lang, args,  latent_size, tf_latent_size, feat_range_mappings, numeric_count, category_count, has_embeddings=False,pretrained_model_path=None, continue_act=False, feat_group_names=None):
        super(RLSynthesizerNetwork_transformer, self).__init__()
        self.topk_act = args.topk_act
        self.lang = lang
        self.program_max_len=args.program_max_len
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
            else:
                self.grammar_num_to_token_val[i] = list(range(self.discretize_feat_value_count))
                self.grammar_token_val_to_num[i] = list(range(self.discretize_feat_value_count))

        self.op_start_pos = -1
        self.num_start_pos = -1

        decision = "num_op"
        start = self.ATOM_VEC_LENGTH


        for option in list(self.lang.syntax[decision].keys()):        
            if self.op_start_pos < 0:
                self.op_start_pos = self.ATOM_VEC_LENGTH
            
            self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
            self.ATOM_VEC_LENGTH += 1
        self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)

        for group_idx in range(len(feat_group_names)):
            start = self.ATOM_VEC_LENGTH
            if self.num_start_pos < 0:
                    self.num_start_pos = self.ATOM_VEC_LENGTH
                
            self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
            self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
            self.ATOM_VEC_LENGTH += 1
            self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)


        # for group_idx in range(len(feat_group_names)):
        #     start = self.ATOM_VEC_LENGTH
        #     if self.num_start_pos < 0:
        #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
        #     self.grammar_token_to_pos[feat_group_names[group_idx][0]] = self.ATOM_VEC_LENGTH
        #     self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = feat_group_names[group_idx][0]
        #     self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[feat_group_names[group_idx][0]] = (start, self.ATOM_VEC_LENGTH)

        # for decision, options_dict in self.lang.syntax.items():
        #     start = self.ATOM_VEC_LENGTH


        #     for option in list(options_dict.keys()):        
        #         if self.op_start_pos < 0:
        #             self.op_start_pos = self.ATOM_VEC_LENGTH
                
        #         self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        # for decision, options_dict in self.lang.syntax.items():
        #     if not (decision == "num_op" or decision in self.lang.syntax["num_feat"]):
        #         continue
        #     # if decision == "num_op":
        #     #     continue
        #     start = self.ATOM_VEC_LENGTH


        #     if not decision in self.lang.syntax["num_feat"]:
        #         for option in list(options_dict.keys()):        
        #             if self.op_start_pos < 0:
        #                 self.op_start_pos = self.ATOM_VEC_LENGTH
                    
        #             self.grammar_token_to_pos[(decision,option)] = self.ATOM_VEC_LENGTH
        #             self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = (decision, option)
        #             self.ATOM_VEC_LENGTH += 1
        #     else:
        #         if self.num_start_pos < 0:
        #             self.num_start_pos = self.ATOM_VEC_LENGTH
                
        #         self.grammar_token_to_pos[decision] = self.ATOM_VEC_LENGTH
        #         self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = decision
        #         self.ATOM_VEC_LENGTH += 1
        #     self.one_hot_token_bounds[decision] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_token_to_pos[pred_v_key] = self.ATOM_VEC_LENGTH
        self.one_hot_token_bounds[pred_v_key] = (start, self.ATOM_VEC_LENGTH)
        self.grammar_pos_to_token[self.ATOM_VEC_LENGTH] = pred_v_key
        self.ATOM_VEC_LENGTH += 1

        # self.column_ls = list(self.lang.syntax["num_feat"].keys())

        # num_feat_len  = len(self.lang.syntax["num_feat"]) if "num_feat" in self.lang.syntax else 0
        # cat_feat_len = len(self.lang.syntax["cat_feat"]) if "cat_feat" in self.lang.syntax else 0
        num_features = len(feat_group_names)# num_feat_len+cat_feat_len
        self.num_feats = num_features
        self.feat_group_num = len(feat_group_names)
        
        # self.prog_gru = EncoderRNN(input_size=self.ATOM_VEC_LENGTH, hidden_size=latent_size)
        # self.feat_gru = EncoderRNN(input_size=num_features, hidden_size=latent_size)

        # self.decoder = AttnDecoderRNN(hidden_size=latent_size, input_size=self.ATOM_VEC_LENGTH, feat_max_size=patient_max_appts, prog_max_size=program_max_len, dropout_p=dropout_p)

        # self.token_nets = nn.ModuleDict({i:TokenNetwork(latent_size, len(v)) for i,v in self.lang.syntax.items()})
        net_maps = {}
        full_input_size = num_features + self.program_max_len*self.ATOM_VEC_LENGTH
        
        # self.embedding2 = TokenNetwork(latent_size, self.discretize_feat_value_count)
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
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            #     net_maps[i + "_lb"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            #     net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
            # if not i in self.lang.syntax["num_feat"]:
            #     net_maps[i] = TokenNetwork(latent_size, len(v))
            # else:
            # net_maps["constant"] = TokenNetwork(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len + self.op_num, self.discretize_feat_value_count)
            if latent_size > 0:
                if not continue_act:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, 1)
            else:
                if not continue_act:
                    net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, self.discretize_feat_value_count)
                else:
                    net_maps["constant"] = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, 1)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)

        self.token_nets = nn.ModuleDict(net_maps)
        self.embedding = TokenNetwork2(full_input_size + self.num_feats + self.op_num, latent_size)
        
        
        if latent_size > 0:
            self.feat_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, latent_size, self.feat_group_num)
            self.op_selector = TokenNetwork3(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, latent_size, self.op_num)
        else:
            self.feat_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.feat_group_num)
            self.op_selector = nn.Linear(tf_latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.feat_group_num, self.op_num)
                # net_maps[i + "_ub"] = TokenNetwork(latent_size, self.discretize_feat_value_count)
                # net_maps[i]["min"] = TokenNetwork_regression(latent_size)
                # net_maps[i]["max"] = TokenNetwork_regression(latent_size)


        # self.token_nets = nn.ModuleDict(net_maps)
        # # self.embedding = TokenNetwork2(full_input_size + self.num_feat_len + self.op_num, latent_size)
        
        # # self.feat_selector = TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # # self.op_selector = TokenNetwork3(full_input_size + self.num_feat_len, latent_size, self.op_num)

        # self.feat_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH, self.num_feat_len)# TokenNetwork2(full_input_size, latent_size, self.num_feat_len)
        # self.op_selector = torch.nn.Linear(latent_size + self.program_max_len*self.ATOM_VEC_LENGTH + self.num_feat_len, self.op_num)
        
        self.to(device=DEVICE)



    def __init__(self, args, lang, model_config, rl_config,  feat_range_mappings, numeric_count, category_count, has_embeddings=False, continue_act=False, feat_group_names=None, removed_feat_ls=None):
        self.feat_group_names = feat_group_names
        self.prefer_smaller_range = args.prefer_smaller_range
        self.prefer_smaller_range_coeff = args.prefer_smaller_range_coeff
        self.removed_feat_ls = removed_feat_ls
        self.latent_size = model_config["latent_size"]
        self.feat_bound_point_ls = args.feat_bound_point_ls
        if self.feat_bound_point_ls is None:
            self.discretize_feat_value_count = rl_config["discretize_feat_value_count"]
        else:
            self.discretize_feat_value_count = len(list(self.feat_bound_point_ls.values())[0])
        print("discrete feat value count::", self.discretize_feat_value_count)
        if self.prefer_smaller_range:
            self.selected_vals = torch.tensor([k/(self.discretize_feat_value_count-1) for k in range(self.discretize_feat_value_count)]).to(DEVICE)
        self.method_two = args.method_two
        # if not method_two:
        #     if feat_group_names is None:
        #         self.init_without_feat_groups(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act)
        #     else:
        #         self.init_with_feat_groups(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act, feat_group_names=feat_group_names)
        # else:
        # if not method_two:
        #     if feat_group_names is None:
        #         self.init_without_feat_groups0(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act)
        #     else:
        #         self.init_with_feat_groups2(lang,  program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=pretrained_model_path, topk_act=topk_act, continue_act=continue_act, feat_group_names=feat_group_names)
        # else:
        if feat_group_names is None:
            self.init_without_feat_groups2(lang,  args, model_config["latent_size"], model_config["tf_latent_size"], feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=model_config["pretrained_model_path"], continue_act=continue_act)
        else:
            self.init_with_feat_groups2(lang,  args, model_config["latent_size"], model_config["tf_latent_size"], feat_range_mappings, numeric_count, category_count, has_embeddings=has_embeddings,pretrained_model_path=model_config["pretrained_model_path"], continue_act=continue_act, feat_group_names=feat_group_names)
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
    
    def vector_ls_to_str0(self, program):
        return vector_ls_to_str0_main(self, program)
        # ret_tensor_ls = []
        # pred_v_arr = atom_ls[pred_v_key]
        
        # col_id_tensor = atom_ls[col_id_key]
        
        # op_id_tensor = atom_ls[op_id_key]
        
        
        
        # ret_tensor_ls = torch.zeros([len(pred_v_arr), self.topk_act, self.ATOM_VEC_LENGTH])
        
        # # ret_tensor_ls[:,self.grammar_token_to_pos[("num_op", op)]]=1
        
        # sample_id_tensor = torch.arange(len(ret_tensor_ls),device=DEVICE)
        
        # for k in range(self.topk_act):
        #     ret_tensor_ls[sample_id_tensor,k, self.num_start_pos + col_id_tensor[:,k]]=1
        #     ret_tensor_ls[sample_id_tensor,k, self.op_start_pos + op_id_tensor[:,k]]=1
        
        # ret_tensor_ls[:, :, self.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)
        
        # # for atom in atom_ls:    
        # #     one_hot_pos = []
        # #     one_hot_pos.append(self.grammar_token_to_pos[("num_op", op)])
        # #     one_hot_pos.append(self.grammar_token_to_pos[col])
        # #     one_hot_pos.append((self.ATOM_VEC_LENGTH-1, atom[pred_v_key]))
        # #     # for token, token_val in atom.items():
        # #     #     # if token.endswith("_prob"):
        # #     #     #     continue

        # #     #     if not self.get_prefix(token) in self.lang.syntax["num_feat"]:
        # #     #         one_hot_pos.append(self.grammar_token_to_pos[(token, token_val)])
        # #     #     else:
        # #     #         # one_hot_pos.append(self.grammar_token_to_pos[token] + torch.argmax(atom[token + "_prob"]))
        # #     #         one_hot_pos.append((self.grammar_token_to_pos[self.get_prefix(token)], atom[token]))
        # #     ret = [0.0]*self.ATOM_VEC_LENGTH
        # #     for i in one_hot_pos:
        # #         if type(i) is tuple:
        # #             ret[i[0]] = i[1]
        # #             # ret[i[0] + 1] = i[1][1]
        # #         else:
        # #             ret[i] = 1
        # #     ret_tensor_ls.append(torch.FloatTensor(ret))
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

                    pred_lb = (feat_val)*(argmax/(self.discretize_feat_value_count-1))

                    ret[token] = [pred_lb,pred]
                elif token.endswith("_ub"):

                    # range_max = self.feat_range_mappings[token][1]
                    # range_min = self.feat_range_mappings[token][0]

                    

                    

                    # pred_lb = pred_lb*(range_max - range_min) + range_min

                    argmax = torch.argmax(pred).item()

                    pred_ub = (1 - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

                    # pred_ub = pred_ub*(range_max - range_min) + range_min
                    # pred_ub = (range_max - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

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

    def forward_ls0(self, input_data,X_pd_full, program, outbound_mask_ls, atom, epsilon=0, init=False, eval=False, existing_atom=None, is_ppo=False, train=False, X_pd_full2=None, abnormal_info=None):
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
        
        # if type(input_data) is not torch.Tensor:
        if len(input_data) == 2:
            features, feature_embedding = input_data
            features = features.to(DEVICE)
            feature_embedding = feature_embedding.to(DEVICE)
        else:
            features, X_num, X_cat = input_data
            X_num = X_num.to(DEVICE)
            X_cat = X_cat.to(DEVICE)
            features = features.to(DEVICE)
            feature_embedding = self.input_embedding(X_cat, X_num, return_embedding=True)
        
        
        pat_count = features.shape[0]
        
        # if not self.method_two:
            
        #     total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]

        #     concat_program_tensor = torch.cat(program,dim=-1)

        #     # if len(program) == 1 and len(program.shape) == 2:
        #     if init:
        #         # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
        #         hx = torch.zeros([feature_embedding.shape[0], total_feat_prog_array_len], device=DEVICE)
        #         hx[:,0:feature_embedding[0].shape[0]] = feature_embedding
        #         hx[:, feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]+feature_embedding[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
        #     else:
        #         hx = torch.zeros([feature_embedding.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
        #         hx[:,:,0:feature_embedding[0].shape[0]] = feature_embedding.unsqueeze(1).repeat(1,self.topk_act,1)
        #         hx[:,:,feature_embedding[0].shape[0]:len(program)*program[0].shape[-1]+feature_embedding[0].shape[0]] = concat_program_tensor#.view(concat_program_tensor.shape[0], -1)
                
        # else:
        total_feat_prog_array_len =feature_embedding[0].shape[0] + self.latent_size

        concat_program_tensor = torch.cat(program,dim=-1)

        # if len(program) == 1 and len(program.shape) == 2:
        if init:
            # total_feat_prog_array_len =feature_embedding[0].shape[0] + self.program_max_len*program[0].shape[-1]
            hx = torch.zeros([feature_embedding.shape[0], total_feat_prog_array_len], device=DEVICE)
            hx[:,0:feature_embedding[0].shape[0]] = feature_embedding
            hx[:, feature_embedding[0].shape[0]:] = self.program_net(torch.stack(program, dim=1).to(DEVICE)).squeeze(1)#.view(concat_program_tensor.shape[0], -1)
        else:
            hx = torch.zeros([feature_embedding.shape[0], self.topk_act, total_feat_prog_array_len], device=DEVICE)
            hx[:,:,0:feature_embedding[0].shape[0]] = feature_embedding.unsqueeze(1).repeat(1,self.topk_act,1)
            hx[:,:,feature_embedding[0].shape[0]:] = self.program_net(torch.cat(program, dim=1).to(DEVICE))#.view(concat_program_tensor.shape[0], -1)

        


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
        return forward_main0_opt(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full2, init=init,is_ppo=is_ppo, train=train, abnormal_info=abnormal_info)
        # return forward_main0(self, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=init,is_ppo=is_ppo, train=train, X_pd_full2=X_pd_full2)
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
        
        # # hx_out = self.embedding(hx)
        
        
        # if not eval:
        #     if np.random.rand() < epsilon:
        #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=DEVICE)
        #         pred = torch.rand([pat_count, self.topk_act, self.discretize_feat_value_count], device=DEVICE)
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
        #     pred_v1 = (feat_val)*(argmax/(self.discretize_feat_value_count-1))
        #     # __le__
        #     pred_v2 = (1 - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val

        #     if self.lang.precomputed is not None:
        #         pred_v1, pred_v2 = find_nearest_thres_vals(self.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        #     # if self.op_list[0] == operator.__ge__:     
            
                
        #     pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
        # # else:
        #     # pred_v = pred_v1*op_val[:,0] + pred_v2*op_val[:,1]

        # # if op == operator.__ge__:

        # #     pred_v = (feat_val)*(argmax/(self.discretize_feat_value_count-1))
        # # else:
        # #     pred_v = (1 - feat_val)*(argmax/(self.discretize_feat_value_count-1)) + feat_val
        
        
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
 