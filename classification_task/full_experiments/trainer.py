import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from rl_models.rl_algorithm import DQN_all, PPO_all, DQN_all2
import logging
import pandas as pd
import numpy as np
import operator
from rl_models.enc_dec import col_id_key, col_key, pred_Q_key, pred_v_key, col_Q_key, prev_prog_key, op_key, Transition, outbound_key
from create_language import *
from tqdm import tqdm
import torch
from classification_task.utils_mortality.metrics import metrics_maps
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from classification_task.datasets.EHR_datasets import *
from rl_models.enc_dec_medical import range_key
from cluster_programs import *
from sklearn.metrics import confusion_matrix
import psutil

def integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls):
        prev_prog_ids = atom_ls[prev_prog_key].cpu()
        curr_col_ids = atom_ls[col_key]
        outbound_mask = atom_ls[outbound_key]
        program = []
        outbound_mask_ls = []
        sample_ids = torch.arange(len(next_program[0]))
        # program length
        for pid in range(len(next_program)):
            program.append(torch.stack([next_program[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])],dim=1))
            outbound_mask_ls.append(torch.stack([next_outbound_mask_ls[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])], dim=-1))
        program.append(curr_vec_ls)
        outbound_mask_ls.append(outbound_mask)
        new_program_col_ls = []
        new_program_str = []
        for idx in range(len(program_col_ls)):
            curr_sample_new_program_col_ls = []
            curr_sample_new_program_str = []
            for k in range(trainer.topk_act):
                curr_new_program_col_ls = []
                curr_new_program_str = []
                # for pid in range(len(program_col_ls[idx])):
                
                #     curr_new_program_col_ls.append(program_col_ls[idx][prev_prog_ids[idx,k].item()][pid])
                #     # [k].append()
                #     curr_new_program_str.append(next_program_str[idx][prev_prog_ids[idx,k].item()][pid])
                curr_new_program_col_ls.extend(program_col_ls[idx][prev_prog_ids[idx,k].item()])
                curr_new_program_str.extend(next_program_str[idx][prev_prog_ids[idx,k].item()])
                
                
                curr_new_program_col_ls.append(curr_col_ids[idx][k])
                curr_new_program_str.append(curr_atom_str_ls[idx][k])
                curr_sample_new_program_col_ls.append(curr_new_program_col_ls)
                curr_sample_new_program_str.append(curr_new_program_str)
            new_program_col_ls.append(curr_sample_new_program_col_ls)
            new_program_str.append(curr_sample_new_program_str)
        return program, new_program_col_ls, new_program_str, outbound_mask_ls


def process_curr_atoms(trainer, atom_ls, program, program_str, all_other_pats_ls, all_other_pats_ids_ls_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=None):
    process = psutil.Process()
    # if not trainer.do_medical:    
    curr_atom_str_ls = trainer.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, trainer.feat_range_mappings, trainer.train_dataset.cat_id_unique_vals_mappings, other_keys=other_keys)
    # else:
    #     curr_atom_str_ls = trainer.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, trainer.feat_range_mappings)
    # outbound_mask_ls = atom_ls[outbound_key]
    
    next_program = program.copy()
    
    next_outbound_mask_ls=outbound_mask_ls.copy()
    
    next_program_str = program_str.copy()
    
    curr_vec_ls = trainer.dqn.atom_to_vector_ls0(atom_ls)

    if len(program) > 0:            
        next_program, program_col_ls, next_program_str, next_outbound_mask_ls = integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)
    else:
        next_program.append(curr_vec_ls)
        next_outbound_mask_ls.append(atom_ls[outbound_key])
        program_col_ls = []
        for vec_idx in range(len(curr_vec_ls)):
            # vec = curr_vec_ls[vec_idx]
            atom_str = curr_atom_str_ls[vec_idx]
            program_sub_col_ls = []
            next_program_str_sub_ls = []
            for k in range(len(atom_ls[col_key][vec_idx])):
                # program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                program_sub_col_ls.append([atom_ls[col_key][vec_idx][k]])
                next_program_str_sub_ls.append([atom_str[k]])
            next_program_str.append(next_program_str_sub_ls)
            program_col_ls.append(program_sub_col_ls)
    # if not trainer.do_medical:
        # print(process.memory_info().rss/(1024*1024*1024))
    next_all_other_pats_ls,_ = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, all_other_pats_ids_ls_ls, col_key, op_key, pred_v_key, other_keys=other_keys)
        # print(process.memory_info().rss/(1024*1024*1024))
    # else:
    #     next_all_other_pats_ls = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, all_other_pats_ids_ls_ls, col_key, range_key)

    return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls


def process_curr_atoms0(trainer, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=None):
    # process = psutil.Process()
    # if not trainer.do_medical:    
    curr_atom_str_ls = trainer.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, trainer.feat_range_mappings, trainer.train_dataset.cat_id_unique_vals_mappings, other_keys=other_keys)
    # else:
    #     curr_atom_str_ls = trainer.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, trainer.feat_range_mappings)
    # outbound_mask_ls = atom_ls[outbound_key]
    
    next_program = program.copy()
    
    next_outbound_mask_ls=outbound_mask_ls.copy()
    
    next_program_str = program_str.copy()
    
    curr_vec_ls = trainer.dqn.atom_to_vector_ls0(atom_ls)

    if len(program) > 0:            
        next_program, program_col_ls, next_program_str, next_outbound_mask_ls = integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)
    else:
        next_program.append(curr_vec_ls)
        next_outbound_mask_ls.append(atom_ls[outbound_key])
        program_col_ls = []
        for vec_idx in range(len(curr_vec_ls)):
            # vec = curr_vec_ls[vec_idx]
            atom_str = curr_atom_str_ls[vec_idx]
            program_sub_col_ls = []
            next_program_str_sub_ls = []
            for k in range(len(atom_ls[col_key][vec_idx])):
                # program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                program_sub_col_ls.append([atom_ls[col_key][vec_idx][k]])
                next_program_str_sub_ls.append([atom_str[k]])
            next_program_str.append(next_program_str_sub_ls)
            program_col_ls.append(program_sub_col_ls)
    # if not trainer.do_medical:
        # print(process.memory_info().rss/(1024*1024*1024))
    next_all_other_pats_ls,_ = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key, other_keys=other_keys)
        # print(process.memory_info().rss/(1024*1024*1024))
    # else:
    #     next_all_other_pats_ls = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_id_key, range_key)

    return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls

def process_curr_atoms0_2(trainer, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=None, return_expr_ls=False):
    # process = psutil.Process()
    # if not trainer.do_medical:    
    curr_atom_str_ls = trainer.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, trainer.feat_range_mappings, trainer.train_dataset.cat_id_unique_vals_mappings, other_keys=other_keys)
    # else:
    #     curr_atom_str_ls = trainer.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, trainer.feat_range_mappings)
    # outbound_mask_ls = atom_ls[outbound_key]
    
    next_program = program.copy()
    
    next_outbound_mask_ls=outbound_mask_ls.copy()
    
    next_program_str = program_str.copy()
    
    curr_vec_ls = trainer.dqn.atom_to_vector_ls0(atom_ls)

    if len(program) > 0:            
        next_program, program_col_ls, next_program_str, next_outbound_mask_ls = integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)
    else:
        next_program.append(curr_vec_ls)
        next_outbound_mask_ls.append(atom_ls[outbound_key])
        program_col_ls = []
        for vec_idx in range(len(curr_vec_ls)):
            # vec = curr_vec_ls[vec_idx]
            atom_str = curr_atom_str_ls[vec_idx]
            program_sub_col_ls = []
            next_program_str_sub_ls = []
            for k in range(len(atom_ls[col_key][vec_idx])):
                # program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                program_sub_col_ls.append([atom_ls[col_key][vec_idx][k]])
                next_program_str_sub_ls.append([atom_str[k]])
            next_program_str.append(next_program_str_sub_ls)
            program_col_ls.append(program_sub_col_ls)
    # if not trainer.do_medical:
        # print(process.memory_info().rss/(1024*1024*1024))
    next_all_other_pats_ls, transformed_expr_ls, = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi2(atom_ls, all_other_pats_ls, col_id_key, op_key, pred_v_key, other_keys=other_keys)
        # print(process.memory_info().rss/(1024*1024*1024))
    # else:
    #     next_all_other_pats_ls = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_id_key, range_key)

    if return_expr_ls:
        return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls, transformed_expr_ls
    else:
        return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls



class Trainer_all:
    # lang=lang, train_dataset=train_dataset, valid_dataset=valid_dataset,test_dataset = test_dataset, train_feat_embeddings=train_feat_embeddings, valid_feat_embeddings=valid_feat_embeddings, test_feat_embeddings=test_feat_embeddings, program_max_len=program_max_len, topk_act=args.num_ors,   learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs,       is_log = args.is_log, dropout_p=args.dropout_p, feat_range_mappings=feat_range_mappings, seed=args.seed, work_dir=work_dir, numeric_count=numeric_count, category_count=category_count , model=args.model, rl_algorithm=args.rl_algorithm,model_config = model_config,rl_config = rl_config
    def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, train_feat_embeddings, valid_feat_embeddings, test_feat_embeddings, feat_range_mappings, args, work_dir, numeric_count=None, category_count=None, category_sum_count=None, model_config=None, rl_config=None, feat_group_names=None, removed_feat_ls=None, id_cln = "id", label_cln="label"):
        self.label_cln = label_cln
        self.id_cln = id_cln
        self.topk_act =args.num_ors
        self.feat_range_mappings = feat_range_mappings
        if args.rl_algorithm == "dqn":
            self.dqn = DQN_all(lang=lang, args = args, rl_config=rl_config, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count,category_sum_count = category_sum_count, has_embeddings=(train_feat_embeddings is not None), model_config = model_config, feat_group_names = feat_group_names, removed_feat_ls=removed_feat_ls)
            self.epsilon = rl_config["epsilon"]
            self.epsilon_falloff = rl_config["epsilon_falloff"]
            self.target_update = rl_config["target_update"]
        else:
            # (self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, tf_latent_size, dropout_p, n_updates_per_iteration, clip,feat_range_mappings, seed=0, numeric_count=None, category_count=None, has_embeddings=False, model="mlp", pretrained_model_path=None, topk_act=1)
            self.dqn = PPO_all(lang=lang, learning_rate=args.learning_rate, batch_size=args.batch_size, gamma=rl_config["gamma"], program_max_len=args.num_ands, dropout_p=args.dropout_p, n_updates_per_iteration = rl_config["n_updates_per_iteration"], clip=rl_config["clip"], feat_range_mappings=feat_range_mappings, seed=args.seed, numeric_count=numeric_count, category_count=category_count, category_sum_count = category_sum_count, has_embeddings=(train_feat_embeddings is not None), model=args.model, topk_act=args.num_ors, continue_act = rl_config["continue_act"], model_config = model_config, feat_group_names = feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = args.prefer_smaller_range, prefer_smaller_range_coeff= args.prefer_smaller_range_coeff, args = args)
            self.timesteps_per_batch = rl_config["timesteps_per_batch"]

        self.rl_algorithm = args.rl_algorithm

        
        self.work_dir = work_dir
        
        self.epochs = args.epochs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lang = lang
        
        self.program_max_len = args.num_ands
        self.is_log = args.is_log
        self.batch_size = args.batch_size
        self.train_feat_embeddings=train_feat_embeddings
        self.valid_feat_embeddings=valid_feat_embeddings
        self.test_feat_embeddings=test_feat_embeddings
        # self.do_medical = args.do_medical
        self.multilabel = (len(torch.unique(train_dataset.labels)) > 2)
        if self.is_log:
            self.logger = logging.getLogger()


    def get_test_decision_from_db(self, data: pd.DataFrame):
        if data.shape[0] == 0:
            return -1
        return data[self.label_cln].value_counts().idxmax()
    
    def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for data in data_ls:
            if len(data) == 0:
                label_ls.append(-1)
                prob_label_ls.append(-1)
                continue
            label = data[self.label_cln].value_counts().idxmax()
            prob_label = np.mean(data[self.label_cln])
            label_ls.append(label)
            prob_label_ls.append(prob_label)
        return label_ls, prob_label_ls
    
    def get_test_decision_from_db_ls_multi(self, data_ls):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for sub_data_ls in data_ls:
            sub_label_ls = []
            sub_prob_label_ls = []
            for data in sub_data_ls:
                if len(data) == 0:
                    # sub_label_ls.append(-1)
                    # sub_prob_label_ls.append(-1)
                    continue
                label = data[self.label_cln].value_counts().idxmax()
                if self.multilabel:
                    prob_label = np.mean(np.array(list(data[self.label_cln])), axis=0)
                else:
                    prob_label = np.mean(data[self.label_cln])
                sub_label_ls.append(label)
                sub_prob_label_ls.append(prob_label)
            if self.multilabel:
                if len(sub_label_ls) <= 0:
                    label_ls.append(np.array([-1]*len(self.train_dataset.data[self.label_cln][0])))
                    prob_label_ls.append(np.array([-1]*len(self.train_dataset.data[self.label_cln][0])))
                else:
                    prob_label = np.mean(np.stack(sub_prob_label_ls), axis=0)
                    prob_label_ls.append(prob_label)
                    label_ls.append((prob_label > 0.5).astype(np.int32))
            else:
                if len(sub_label_ls) <= 0:    
                    label_ls.append(-1)
                    prob_label_ls.append(-1)
                else:
                    
                    prob_label = np.mean(np.array(sub_prob_label_ls))
                    prob_label_ls.append(prob_label)
                    
                    if prob_label == 0.5:
                        label_ls.append(-1)
                    elif prob_label > 0.5:
                        label_ls.append(1)
                    else:
                        label_ls.append(0)
                
            
        return label_ls, prob_label_ls

    def get_test_decision_from_db_ls_multi2(self, data_ls):
        if len(data_ls) == 0:
            return -1
        
        label_ls = []
        prob_label_ls = []
        for i in range(len(data_ls)):
            sub_data_ls = data_ls[i]
            sub_label_ls = []
            sub_prob_label_ls = []
            for j in range(len(sub_data_ls)):
                data = sub_data_ls[j]
                # if self.lang.db_backend == "tql":
                #     curr_label_ls = Query("label_" + str(i) + "_" + str(j), base="sub_data_" + str(i) + "_" + str(j)).project(lambda x, y:y).run(self.lang.db, disable=True)
                #     curr_label_ls = torch.tensor(list(curr_label_ls))
                #     curr_label_ls0 = self.lang.labels[data]
                #     assert torch.norm(curr_label_ls.type(torch.float) - curr_label_ls0.type(torch.float)).item() <= 0
                # elif self.lang.db_backend == "tql_opt":
                #     curr_label_ls = Query("label_" + str(i) + "_" + str(j), base="sub_data_" + str(i) + "_" + str(j)).project(lambda x, y:y).run(self.lang.db, disable=True)
                #     curr_label_ls0 = self.lang.labels[data]
                #     assert torch.norm(curr_label_ls[0].type(torch.float) - curr_label_ls0.type(torch.float)).item() <= 0
                curr_label_ls = self.lang.labels[data]
                if len(curr_label_ls) == 0:
                    # sub_label_ls.append(-1)
                    # sub_prob_label_ls.append(-1)
                    continue
                n_labels = len(torch.unique(self.lang.labels))
                if not self.multilabel:
                    prob_label = torch.sum(curr_label_ls)/len(curr_label_ls)
                    label = (prob_label > 0.5).long().item()
                else:
                    
                    local_prob_label_ls = []
                    for cid in range(n_labels):
                        prob_label = torch.sum(curr_label_ls == cid).item()*1.0/len(curr_label_ls)
                        local_prob_label_ls.append(prob_label)
                    prob_label = np.array(local_prob_label_ls)
                    label = np.argmax(prob_label)
                
                sub_label_ls.append(label)
                sub_prob_label_ls.append(prob_label)
            if self.multilabel:
                if len(sub_label_ls) <= 0:
                     #(np.array([-1]*len(self.train_dataset.data[self.label_cln][0])))
                    prob_label = np.random.rand(n_labels)
                    prob_label_ls.append(prob_label)  #(np.array([-1]*len(self.train_dataset.data[self.label_cln][0])))
                else:
                    prob_label = np.mean(np.stack(sub_prob_label_ls), axis=0)
                    prob_label_ls.append(prob_label)
                label_ls.append(np.argmax(prob_label))
            else:
                if len(sub_label_ls) <= 0:    
                    label_ls.append(-1)
                    prob_label_ls.append(-1)
                else:
                    
                    prob_label = np.mean(np.array(sub_prob_label_ls))
                    prob_label_ls.append(prob_label)
                    
                    if prob_label == 0.5:
                        label_ls.append(-1)
                    elif prob_label > 0.5:
                        label_ls.append(1)
                    else:
                        label_ls.append(0)
                
            
        return label_ls, prob_label_ls

    def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
        if len(data) == 0:
            return 0
        same = data.loc[data[self.label_cln] == y]["PAT_ID"].nunique()
        total = data['PAT_ID'].nunique()
        return same / total

    def check_db_constrants_ls(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            sub_data_ls = data_ls[idx]
            sub_rwd_ls = []
            for data in sub_data_ls:
                # if y_ls[idx].numel() == 1:
                if not self.multilabel:
                    y = int(y_ls[idx].item())
                    # same = data.loc[data[self.label_cln] == y]["PAT_ID"].nunique()
                    # total = data['PAT_ID'].nunique()
                    if len(data) == 0:
                        sub_rwd_ls.append(0)
                    else:
                        sub_rwd_ls.append(np.mean(data[self.label_cln] == y))
                else:
                    y = y_ls[idx]
                    score_ls = []
                    total = data['PAT_ID'].nunique()
                    if total == 0:
                        sub_rwd_ls.append([0]*y_ls[idx].numel())
                    else:
                        for cid in range(y_ls[idx].numel()):
                            curr_same = data.loc[np.array(list(data[self.label_cln]))[:,cid] == y[cid].item()]["PAT_ID"].nunique()
                            curr_score = curr_same/total
                            score_ls.append(curr_score)
                        # score = score/y_ls[idx].numel()
                        sub_rwd_ls.append(score_ls)


                
            
            rwd_ls.append(sub_rwd_ls) 
            # if total == 0:
            #     rwd_ls.append(0)
            # else:
            #     rwd_ls.append(same / total) 
        return np.array(rwd_ls)
    
    def check_db_constrants_ls2(self, data_ls,  y_ls):
        # if len(data) == 0:
        #     return 0
        rwd_ls = []
        for idx in range(len(data_ls)):
            sub_data_ls = data_ls[idx]
            sub_rwd_ls = []
            for data in sub_data_ls:
                # if y_ls[idx].numel() == 1:
                if not self.multilabel:
                    y = int(y_ls[idx].item())
                    # same = data.loc[data[self.label_cln] == y]["PAT_ID"].nunique()
                    # total = data['PAT_ID'].nunique()
                    curr_label_ls = self.lang.labels[data]
                    if len(curr_label_ls) == 0:
                        sub_rwd_ls.append(0)
                    else:
                        sub_rwd_ls.append(torch.sum(curr_label_ls == y)/len(curr_label_ls))
                else:
                    y = y_ls[idx]
                    n_labels = len(torch.unique(self.lang.labels))
                    curr_label_ls = self.lang.labels[data]
                    score_ls = []
                    total = len(curr_label_ls)
                    if total == 0:
                        sub_rwd_ls.append([0]*n_labels)
                    else:
                        for cid in range(n_labels):
                            curr_same =  torch.sum(curr_label_ls == cid).item() #data.loc[np.array(list(data[self.label_cln]))[:,cid] == y[cid].item()]["PAT_ID"].nunique()
                            curr_score = curr_same/total
                            score_ls.append(curr_score)
                        # score = score/y_ls[idx].numel()
                        sub_rwd_ls.append(score_ls)


                
            
            rwd_ls.append(sub_rwd_ls) 
            # if total == 0:
            #     rwd_ls.append(0)
            # else:
            #     rwd_ls.append(same / total) 
        return np.array(rwd_ls)

    def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
        return lang.evaluate_atom_on_sample(atom, X)

    def check_program_constraint(self, prog: list) -> bool:
        return len(prog) < self.program_max_len
    
    def identify_op(self, X:pd, atom:dict):

        atom_ls = []
        

        atom1 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom1[k] = atom[k]
            else:
                atom1[k] = atom[k][0][0]
                atom1[k + "_prob"] = atom[k][1][0]

        atom1["num_op"] = operator.__ge__

        atom2 = dict()
        for k in atom:
            if k not in self.lang.syntax["num_feat"]:
                atom2[k] = atom[k]
            else:
                atom2[k] = atom[k][0][1]
                atom2[k + "_prob"] = atom[k][1][1]
        atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    
    def identify_op_ls(self, batch_size:int, atom:dict):

        atom_ls = []        

        # atom1 = [dict()]*batch_size
        atom1 = []

        atom2 = []
        for _ in range(batch_size):
            atom2.append(dict())
        for _ in range(batch_size):
            atom1.append(dict())
        for k in atom:
            # if k not in self.lang.syntax["num_feat"]:
            if type(k) is not tuple:
                if type(atom[k]) is not dict:
                    for atom_id in range(batch_size):
                        atom1[atom_id][k] = atom[k]
                        atom2[atom_id][k] = atom[k]
                else:
                    # atom1[k] = [None]*batch_size
                    for selected_item in atom[k]:
                        sample_ids = atom[k][selected_item]
                        for sample_id in sample_ids:
                            atom1[sample_id.item()][k] = selected_item
                            atom2[sample_id.item()][k] = selected_item
            else:
                
                # atom1[k] = [None]*batch_size
                # atom1[k + "_prob"] = [None]*batch_size
                if k[0].endswith("_lb"):
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom1[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                else:
                    for selected_item in atom[k][2]:
                        sample_ids = atom[k][2][selected_item]
                        for sample_id_id in range(len(sample_ids)):
                            atom2[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
                            # atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
                        # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


                # atom1[k] = atom[k][0][0]
                # atom1[k + "_prob"] = atom[k][1][0]
                # atom1[k + "_sample_ids"] = atom[k][2][0]
        for sample_id in range(len(atom1)):
            atom1[sample_id]["num_op"] = operator.__ge__   

        for sample_id in range(len(atom2)):
            atom2[sample_id]["num_op"] = operator.__le__   

        
        # for k in atom:
        #     # if k not in self.lang.syntax["num_feat"]:
        #     if type(k) is not tuple:
        #         if type(atom[k]) is not dict:
        #             for atom_id in range(batch_size):
        #                 atom2[atom_id][k] = atom[k]
        #         else:
        #             for selected_item in atom[k]:
        #                 sample_ids = atom[k][selected_item]
        #                 for sample_id in sample_ids:
        #                     atom2[sample_id.item()][k] = selected_item
        #     else:
                
        #         for selected_item in atom[k][2]:
        #             sample_ids = atom[k][2][selected_item]
        #             for sample_id_id in range(len(sample_ids)):
        #                 atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
        #                 atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
        #                 # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


        #         # atom1[k] = atom[k][0][0]
        #         # atom1[k + "_prob"] = atom[k][1][0]
        #         # atom1[k + "_sample_ids"] = atom[k][2][0]
        # for sample_id in range(len(atom2)):
        #     atom2[sample_id]["num_op"] = operator.__le__  


        # atom2 = dict()
        # for k in atom:
        #     if k not in self.lang.syntax["num_feat"]:
        #         atom2[k] = atom[k]
        #     else:
        #         atom2[k] = atom[k][0][1]
        #         atom2[k + "_prob"] = atom[k][1][1]
        # atom2["num_op"] = operator.__le__
        atom_ls.append(atom1)
        atom_ls.append(atom2)
            
        return atom_ls
    def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
        satisfy_bool=True
        for atom in atom_ls:
            curr_bool = lang.evaluate_atom_on_sample(atom, X)
            satisfy_bool = satisfy_bool & curr_bool
        return satisfy_bool
   

    def train_epoch_ppo(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        t = 0
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        
        for episode_i, val in iterator:
            (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
            all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)
            X_feat_embedding = None
            if self.train_feat_embeddings is not None:
                X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            # program_str = [[] for _ in range(len(X_pd_ls))]
            # program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            
            X_pd_full = pd.concat(X_pd_ls)
            prev_reward = np.zeros([len(X), self.topk_act])
            ep_rews = []
            for arr_idx in range(self.program_max_len):
                # (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                if X_feat_embedding is None:
                    atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, train=True)
                else:
                    atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, train=True)

                next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = self.process_curr_atoms(atom_ls, program, program_str, all_other_pats_ls, program_col_ls)

                db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                #derive reward
                reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                done = (arr_idx == self.program_max_len-1)
                if X_feat_embedding is None:
                    batch_obs.append((X, X_pd_full, program))
                else:
                    batch_obs.append(((X, X_feat_embedding), X_pd_full, program))
                ep_rews.append(reward - prev_reward)
                batch_acts.append(atom_ls)
                # atom_probs = atom_ls[pred_prob_key] #self.PPO.idx_to_logs(atom_pred, atom_idx)
                # atom_log_probs = atom_probs[torch.tensor(list(range(atom_probs.shape[0]))), atom_ls[pred_prob_id]]
                atom_log_probs = self.dqn.idx_to_logs(atom_ls, atom_ls)
                batch_log_probs.append(atom_log_probs)
                
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    success += np.sum(reward > 0.5)
                    failure += np.sum(reward <= 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
            
            batch_rews.append(ep_rews)
            batch_lens.append(len(program))

            t += 1

            if t == self.timesteps_per_batch:
                batch_rtgs = self.dqn.compute_rtgs(batch_rews=batch_rews)
                batch = (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)
                sum_loss += self.dqn.learn(batch=batch)
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = [], [], [], [], [], []
                t = 0

            # Print information
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{(episode_i + 1)*self.batch_size} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)


    def train_epoch(self, epoch, train_loader):
        success, failure, sum_loss = 0, 0, 0.
        
        iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
        for episode_i, val in iterator:
            (all_other_pats_ls, all_other_pats_ls2, X_pd_ls2, X, X_sample_ids, X_pd_ls, (X_num, X_cat), (abnormal_feature_indicator, activated_indicator)), y = val
            # (all_other_pats_ls, X_pd_ls, X, X_sample_ids, (X_num, X_cat), _), y = val
            # all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)
            all_other_pats_ls = self.copy_data_in_database2(all_other_pats_ls2)
            X_feat_embedding = None
            if self.train_feat_embeddings is not None:
                X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
            # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
            program = []
            outbound_mask_ls = []
            # program_str = [[] for _ in range(len(X_pd_ls))]
            # program_atom_ls = [[] for _ in range(len(X_pd_ls))]
            # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            program_str = []
            program_col_ls = []
            
            X_pd_full = pd.concat(X_pd_ls)
            X_pd_full2 = torch.stack(X_pd_ls2)
            
            
            # col_comp_ls = zip()
            # while True: # episode
            # for 
            # for col_id in col_ids:
            prev_reward = np.zeros([len(X), self.topk_act])
            # random.shuffle(col_op_ls)
            # last_col , last_op  = col_op_ls[-1]


            # for arr_idx in range(len(col_op_ls)):
            for arr_idx in range(self.program_max_len):
                # (col, op) = col_op_ls[arr_idx]
                # col_name = col_list[col_id]
                if X_feat_embedding is None:
                    # atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=self.epsilon)
                    atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=self.epsilon, X_pd_ls2=X_pd_full2, abnormal_info=(abnormal_feature_indicator, activated_indicator))
                else:
                    atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=self.epsilon, X_pd_ls2=X_pd_full2, abnormal_info=(abnormal_feature_indicator, activated_indicator))
                
                # curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings)
                
                # next_program = program.copy()
                
                # next_program_str = program_str.copy()
                
                # curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

                # if len(program) > 0:                        
                #     next_program, program_col_ls, next_program_str= self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls)
                # else:
                #     next_program.append(curr_vec_ls)
                #     for vec_idx in range(len(curr_vec_ls)):
                #         # vec = curr_vec_ls[vec_idx]
                #         atom_str = curr_atom_str_ls[vec_idx]
                #         for k in range(len(atom_ls[col_key][vec_idx])):
                #             program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                #             next_program_str[vec_idx][k].append(atom_str[k])
                
                # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                # next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = process_curr_atoms0(self, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls)
                next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = process_curr_atoms0_2(self, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls2, outbound_mask_ls, other_keys=[col_id_key])
                # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                db_cons = self.check_db_constrants_ls2(next_all_other_pats_ls, y) #entropy
                #derive reward
                if not self.multilabel:
                    reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
                else:
                    reward = np.mean(db_cons,axis=-1)
                done = (arr_idx == self.program_max_len-1)
                #record transition in buffer
                if done:
                    next_state = None
                    next_program = None
                else:
                    next_state = (next_program, next_outbound_mask_ls)
                    
                if atom_ls["prev_prog"] is not None:
                    prev_prog_ids = atom_ls["prev_prog"]
                    new_prev_reward = np.zeros([len(X), self.topk_act])
                    for cl_idx in range(prev_reward.shape[-1]):
                        new_prev_reward[np.arange(len(X)), cl_idx] = prev_reward[np.arange(len(X)), prev_prog_ids[:,cl_idx].cpu().numpy()]
                    prev_reward = new_prev_reward
                
                if X_feat_embedding is None:
                    transition = Transition((X, X_num, X_cat), X_pd_full2,(program, outbound_mask_ls), atom_ls, next_state, reward - prev_reward)
                    # transition = Transition(X, X_pd_full2,(program, outbound_mask_ls), atom_ls, next_state, reward - prev_reward)
                else:
                    transition = Transition((X, X_feat_embedding), X_pd_full2,(program, outbound_mask_ls), atom_ls, next_state, reward - prev_reward)
                self.dqn.observe_transition(transition)
                #update model
                loss = self.dqn.optimize_model_ls0()
                # print(loss)
                sum_loss += loss
                #update next step
                if done: #stopping condition
                    # if reward > 0.5: success += 1
                    # else: failure += 1
                    if not self.multilabel:
                        success += np.sum(np.max(reward, axis = -1) > 0.5)
                    else:
                        success += np.sum(db_cons > 0.5)
                    break
                else:
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    prev_reward = reward
                    outbound_mask_ls = next_outbound_mask_ls
            # Update the target net
            if episode_i % self.target_update == 0:
                self.dqn.update_target()
            # Print information
            total_count = ((episode_i + 1)*self.batch_size)
            success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
            avg_loss = sum_loss/(episode_i+1)
            desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
            iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        self.epsilon *= self.epsilon_falloff

    
    def test_epoch(self, epoch):
        success, failure, sum_loss = 0, 0, 0.
        iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        self.dqn.policy_net.eval()
        self.dqn.target_net.eval()
        with torch.no_grad():
            for episode_i, val in iterator:
                # if episode_i == 28:
                #     print()
                (all_other_pats, X_pd, X), y = val
                program = []
                program_str = []
                program_atom_ls = []
                while True: # episode
                    atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
                    atom_ls = self.identify_op(X_pd, atom)
                    next_program = program.copy()
                    next_program_str = program_str.copy()
                    for new_atom in atom_ls:
                        next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
                        # atom["num_op"] = atom_op
                        
                        
                        next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
                        program_atom_ls.append(new_atom)
                    #apply new atom
                    next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
                    db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
                    #derive reward
                    done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    if done:
                        next_program = None
                    #update next step
                    if done: #stopping condition
                        if self.is_log:
                            msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
                            self.logger.log(level=logging.DEBUG, msg=msg)
                        if y == y_pred: success += 1
                        else: failure += 1
                        y_true_ls.append(y.item())
                        y_pred_ls.append(y_pred)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        all_other_pats = next_all_other_pats

                y_true_array = np.array(y_true_ls, dtype=float)
                y_pred_array = np.array(y_pred_ls, dtype=float)
                y_pred_array[y_pred_array < 0] = 0.5
                if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                #     recall = 0
                #     f1 = 0
                    auc_score= 0
                else:
                    auc_score = roc_auc_score(y_true_array, y_pred_array)

                # if episode_i == self.batch_size:
                #     print(y_true_array.reshape(-1))
                #     print(y_pred_array.reshape(-1))

                # Print information
                success_rate = (success / (episode_i + 1)) * 100.00
                avg_loss = sum_loss/(episode_i+1)
                desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
                iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
            
        self.dqn.policy_net.train()
        self.dqn.target_net.train()
        return y_pred_array
    
    

    def copy_data_in_database(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(all_other_pats_ls[idx].copy())
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    def copy_data_in_database2(self, all_other_pats_ls):
        all_other_pats_ls_ls = []
        for idx in range(len(all_other_pats_ls)):
            curr_other_pats_ls = []
            for k in range(self.topk_act):
                curr_other_pats_ls.append(torch.clone(all_other_pats_ls[idx]))
                # if self.lang.db_backend.startswith("tql"):
                #     Query("sub_data_" + str(idx) + "_" + str(k), base="feature_label").run(self.lang.db, disable=True)
                # if self.lang.db_backend == "Scallop":
                #     attr_str = ",".join(["x_" + str(idx) for idx in range(self.lang.features.shape[1])])
                #     rule_head = "res_" + str(idx) + "_" + str(k) + "("+ attr_str + ")"
                #     rule_body = "features(" + attr_str + ")"
                #     self.lang.ctx.add_rule(rule_head + " = " + rule_body)
                #     self.lang.ctx.run()
            
            all_other_pats_ls_ls.append(curr_other_pats_ls)
            
        return all_other_pats_ls_ls
    
    def concatenate_program_across_samples(self, generated_program_ls, generated_program_str_ls, program_label_ls):
        full_program_ls = []
        full_program_str_ls = []
        full_label_ls = []
        # for k in range(len(generated_program_ls[0])):
        #     curr_full_program_ls = []
        #     curr_full_program_str_ls = []
        #     curr_full_label_ls = []
        #     for idx in range(len(generated_program_ls)):
        #         curr_full_program_ls.append(generated_program_ls[idx][k].view(-1,generated_program_ls[idx][k].shape[-1]))
        #         curr_full_label_ls.append(program_label_ls[idx][:,k].unsqueeze(1).repeat(1, generated_program_ls[idx][k].shape[1]).view(-1, 1))
        #         curr_generated_program_str_ls = []
        #         for i in range(len(generated_program_str_ls[idx])):
        #             sub_curr_generated_program_str_ls = []
        #             for j in range(len(generated_program_str_ls[idx][0])):
        #                 sub_curr_generated_program_str_ls.append(generated_program_str_ls[idx][i][j][k])
                    
        #             curr_generated_program_str_ls.append(sub_curr_generated_program_str_ls)
                    
        #         curr_full_program_str_ls.extend(curr_generated_program_str_ls)
            
        #     full_program_str_ls.extend(curr_full_program_str_ls)
        #     full_program_ls.append(torch.cat(curr_full_program_ls))
        #     full_label_ls.append(torch.cat(curr_full_label_ls))
        
        for idx in range(len(generated_program_ls)):
            full_program_ls.append(torch.stack(generated_program_ls[idx]))
            full_label_ls.append(torch.cat(program_label_ls[idx]))
        for p_str_ls in generated_program_str_ls:
            full_program_str_ls.extend(p_str_ls)
        
        
        
        return torch.cat(full_program_ls), full_program_str_ls, torch.cat(full_label_ls)
    
    def decode_program_to_str(self, single_program):
        program_str = self.dqn.vector_ls_to_str_ls0(single_program)
        return program_str

    def redundancy_metrics(self, existing_data, target_data):
        if len(existing_data) == len(target_data):
            return True

        return False

    def concat_all_elements(self, reduced_program_ls, reduced_program_str_ls, labels):
        flatten_reduced_program_ls = []
        flatten_reduced_program_str_ls = []
        flatten_labels = []
        for i in range(len(reduced_program_ls)):
            for j in range(len(reduced_program_ls[i])):
                for k in range(len(reduced_program_ls[i][j])):
                    flatten_reduced_program_ls.append(reduced_program_ls[i][j][k])
                    flatten_reduced_program_str_ls.append(reduced_program_str_ls[i][j][k])
                    flatten_labels.append(labels[i])
                    
        return flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_labels
                

    def remove_redundant_predicates(self, all_other_pats_ls, all_transformed_expr_ls, next_all_other_pats_ls, next_program, next_program_str):
        transposed_expr_ls = []
        transposed_next_program = []
        for j in range(len(all_transformed_expr_ls[0])):
            curr_transposed_expr_ls = []
            curr_program_ls = []
            for k in range(len(all_transformed_expr_ls[0][0])):
                sub_curr_transposed_expr_ls = []
                sub_curr_program_ls = []
                for i in range(len(all_transformed_expr_ls)):
                    
                    sub_curr_transposed_expr_ls.append(all_transformed_expr_ls[i][j][k])
                    sub_curr_program_ls.append(next_program[i][j][k])
                curr_transposed_expr_ls.append(sub_curr_transposed_expr_ls)
                curr_program_ls.append(sub_curr_program_ls)
            
            transposed_expr_ls.append(curr_transposed_expr_ls)
            transposed_next_program.append(curr_program_ls)


        all_other_pats_ls = self.copy_data_in_database2(all_other_pats_ls)

        reduced_program_ls = []
        reduced_program_str_ls = []

        for i in range(len(transposed_expr_ls)):
            curr_reduced_program_ls = []
            curr_reduced_program_str_ls = []
            for j in range(len(transposed_expr_ls[i])):
                redundant_clause_id_ls = []
                sub_curr_reduced_program_ls = []
                sub_curr_reduced_program_str_ls = []

                for k in range(len(transposed_expr_ls[i][j])):
                    temp_expr_ls = transposed_expr_ls[i][j].copy()
                    del temp_expr_ls[k]
                    existing_data = all_other_pats_ls[i][j].copy()
                    for expr_c in temp_expr_ls:
                        curr_op = expr_c[1]
                        curr_const = expr_c[2]
                        expr = curr_op(existing_data[expr_c[0]], curr_const)
                        existing_data = self.lang.evaluate_expression_on_data(existing_data, expr)
                    if self.redundancy_metrics(existing_data, next_all_other_pats_ls[i][j]):
                        redundant_clause_id_ls.append(k)
                    else:
                        sub_curr_reduced_program_ls.append(transposed_next_program[i][j][k])
                        sub_curr_reduced_program_str_ls.append(next_program_str[i][j][k])
                curr_reduced_program_ls.append(sub_curr_reduced_program_ls)
                curr_reduced_program_str_ls.append(sub_curr_reduced_program_str_ls)
            reduced_program_ls.append(curr_reduced_program_ls)
            reduced_program_str_ls.append(curr_reduced_program_str_ls)
                
        return reduced_program_ls, reduced_program_str_ls

    def cluster_programs(self, full_program_ls, full_program_str_ls, full_label_ls):
        # full_label_ls_tensor = torch.cat(full_label_ls)
        # full_program_ls_tensor = torch.cat(full_program_ls)
        
        full_label_ls_tensor = full_label_ls
        full_program_ls_tensor = full_program_ls

        unique_labels = full_label_ls_tensor.unique().tolist()

        for label in unique_labels:

            print("print info for label ", str(label))

            curr_full_program_ls_tensor = full_program_ls_tensor[full_label_ls_tensor.view(-1) == label][0:-1]
            curr_idx_ls = torch.nonzero(full_label_ls_tensor.view(-1) == label).view(-1).tolist()
            cluster_assignment_ids, cluster_centroids = KMeans(curr_full_program_ls_tensor, K=5)
            approx_cluster_centroids, min_program_ids = get_closet_samples_per_clusters(cluster_centroids, curr_full_program_ls_tensor)
            for idx in range(len(min_program_ids.tolist())):
                selected_program_str = full_program_str_ls[curr_idx_ls[min_program_ids[idx]]]
                print("cluster idx %d:%s"%(idx, selected_program_str))
                print("cluster count for cluster idx %d:%d"%(idx, torch.sum(cluster_assignment_ids==idx).item()))

            program_str_ls = []
            for idx in range(len(cluster_centroids)):
                if len(torch.nonzero(cluster_centroids[idx])) <= 0:
                    continue

                program_str = self.decode_program_to_str(cluster_centroids[idx])
                print("cluster idx %d:%s"%(idx, program_str))
            # program_str_ls.append(program_str) 
        # return program_str_ls
        print()
        
    def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None, feat_embedding = None):
        pd.options.mode.chained_assignment = None

        success, failure, sum_loss = 0, 0, 0.

        iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
        # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
        y_true_ls=[]
        y_pred_ls=[]
        y_pred_prob_ls=[]
        if self.rl_algorithm == "dqn":
            self.dqn.policy_net.eval()
            self.dqn.target_net.eval()
        else:
            self.dqn.actor.eval()
            self.dqn.critic.eval()
            
        generated_program_ls = []
        generated_program_str_ls = []
        program_label_ls = []
        
        
        with torch.no_grad():
            # col_list = list(self.train_dataset.data.columns)
        
            # op_ls = list([operator.__le__, operator.__ge__])
            
            # col_op_ls = []
            
            # last_col = None

            # last_op = None
            
            # for col in col_list:
            #     if col == "PAT_ID" or col == "label":
            #         continue
            #     last_col = col
            #     for op in op_ls:
            #         col_op_ls.append((col, op))
            #         last_op = op
            for episode_i, val in iterator:
                # if episode_i == 13:
                #     print()
                # (origin_all_other_pats_ls, origin_all_other_pats_ls2, X_pd_ls2, X, X_sample_ids, X_pd_ls), y = val
                (origin_all_other_pats_ls, origin_all_other_pats_ls2, X_pd_ls2, X, X_sample_ids, X_pd_ls, (X_num, X_cat), (abnormal_feature_indicator, activated_indicator)), y = val
                all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls2)
                # all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
                # for x_pd_idx in range(len(X_pd_ls)):
                #     if np.sum(X_pd_ls[x_pd_idx]["PAT_ID"] == 277) >= 1:
                #         print(x_pd_idx)
                #         break                
                X_feat_embeddings = None
                if feat_embedding is not None:
                    X_feat_embeddings = feat_embedding[X_sample_ids]
                
                # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
                program = []
                outbound_mask_ls = []
                # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                program_str = []
                program_col_ls = []
                # for p_k in range(len(program_str)):
                #     program_str[p_k].append([[] for _ in range(self.topk_act)])
                #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
                X_pd_full = pd.concat(X_pd_ls)
                X_pd_full2 = torch.stack(X_pd_ls2)
                
                y_pred = torch.zeros([len(X)])
                if self.is_log:
                    all_transformed_expr_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
                # for arr_idx in range(len(col_op_ls)):
                for arr_idx in range(self.program_max_len):
                    # (col, op) = col_op_ls[arr_idx]
                    # col_name = col_list[col_id]
                    if X_feat_embeddings is None:
                        if self.rl_algorithm == "dqn":
                            atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=0, X_pd_ls2=X_pd_full2, abnormal_info = (abnormal_feature_indicator, activated_indicator))
                        else:
                            atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, train=False, X_pd_ls2=X_pd_full2, abnormal_info = (abnormal_feature_indicator, activated_indicator))
                    else:
                        if self.rl_algorithm == "dqn":
                            atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=0, X_pd_ls2=X_pd_full2, abnormal_info = (abnormal_feature_indicator, activated_indicator))
                        else:
                            atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, train=False, X_pd_ls2=X_pd_full2, abnormal_info = (abnormal_feature_indicator, activated_indicator))
                    
                    # if not self.do_medical:
                    #     curr_atom_str_ls = self.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings, self.test_dataset.cat_id_unique_vals_mappings)
                    # else:
                    #     curr_atom_str_ls = self.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, self.test_dataset.feat_range_mappings)
                        # curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings)

                    # next_program = program.copy()
                    # next_outbound_mask_ls = outbound_mask_ls.copy()
                    # next_program_str = program_str.copy()
                    # curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)
                    
                    # if len(program) > 0:                        
                        

                    #     next_program, program_col_ls, next_program_str, next_outbound_mask_ls = integrate_curr_program_with_prev_programs(self, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)



                    # else:

                    #     next_program.append(curr_vec_ls)
                    #     next_outbound_mask_ls.append(atom_ls[outbound_key])

                    #     for vec_idx in range(len(curr_vec_ls)):
                    #         # vec = curr_vec_ls[vec_idx]
                    #         atom_str = curr_atom_str_ls[vec_idx]
                    #         for k in range(len(atom_ls[col_key][vec_idx])):
                    #             program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                    #             next_program_str[vec_idx][k].append(atom_str[k])
                    # trainer, atom_ls, program, program_str, all_other_pats_ls, all_other_pats_ids_ls_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=None
                    if not self.is_log:
                        next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = process_curr_atoms0_2(self, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls2, outbound_mask_ls, other_keys=[col_id_key])
                    else:
                        next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls, transformed_expr_ls = process_curr_atoms0_2(self, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls2, outbound_mask_ls, other_keys=[col_id_key], return_expr_ls=True)
                            # next_program_str[vec_idx].append(atom_str)

                    if self.is_log:
                        for pat_idx in range(len(X_pd_ls)):
                            for prog_idx in range(self.topk_act):
                                all_transformed_expr_ls[pat_idx][prog_idx].append(transformed_expr_ls[pat_idx][prog_idx])
                    program = next_program
                    program_str = next_program_str
                    all_other_pats_ls = next_all_other_pats_ls
                    outbound_mask_ls = next_outbound_mask_ls
                    # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                    # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    
                    
                    # for new_atom_ls in atom_ls_ls:

                    #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

                    #     next_program.append(torch.stack(curr_vec_ls))

                    #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    
                # while True: # episode
                #     atom,_ = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
                #     atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
                #     reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                #     next_program = program.copy()
                #     next_program_str = program_str.copy()
                #     for new_atom_ls in atom_ls_ls:

                #         curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

                #         next_program.append(torch.stack(curr_vec_ls))

                #         curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                #         for vec_idx in range(len(curr_vec_ls)):
                #             vec = curr_vec_ls[vec_idx]
                #             atom_str = curr_atom_str_ls[vec_idx]
                            
                #             next_program_str[vec_idx].append(atom_str)
                #             program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
                #             reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
                        # atom["num_op"] = atom_op
                        
                        
                        # next_program_str = next_program_str + []
                        
                        # program_atom_ls.append(new_atom_ls)
                    #apply new atom
                    # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                    # if not self.do_medical:
                    #     # next_all_other_pats_ls, transformed_expr_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
                    #     next_all_other_pats_ls, transformed_expr_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi2(atom_ls, all_other_pats_ls, col_id_key, op_key, pred_v_key)
                    # else:
                    #     next_all_other_pats_ls =  self.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_key, range_key)
                    # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
                    # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
                    #check constraints
                    # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
                    # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
                    # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
                    # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
                    # y_pred, y_pred_prob = self.get_test_decision_from_db_ls_multi(next_all_other_pats_ls)
                    curr_y_pred, y_pred_prob = self.get_test_decision_from_db_ls_multi2(next_all_other_pats_ls)
                    curr_y_pred = torch.tensor(curr_y_pred, dtype=torch.float32)
                    y_pred[curr_y_pred >= 0] = curr_y_pred[curr_y_pred >= 0]
                    # final_y_pred,_ = stats.mode(np.array(y_pred), axis = -1)
                    # final_y_pred_prob = np.mean(np.array(y_pred_prob), axis = -1)
                    
                    # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
                    # done = (col == last_col) and (op == last_op)
                    done = (arr_idx == self.program_max_len - 1)

                    # all_transformed_expr_ls.append(transformed_expr_ls)
                    # if done:
                    #     next_program = None
                    #update next step
                    if done: #stopping condition
                        
                        
                        
                        if self.is_log:
                            all_other_pats_ls = self.copy_data_in_database2(origin_all_other_pats_ls2)
                            sorted_conjunction_ids_ls = []
                            for pat_idx in range(len(X_pd_ls)):
                                # if X_pd_ls[0]["PAT_ID"].values[0] == 'FFB92A393FF8B':
                                other_data_booleans_ls = self.lang.evaluate_atom_ls_ls_on_one_patient_with_full_programs_leave_one_out(pat_idx, all_transformed_expr_ls[pat_idx], all_other_pats_ls[pat_idx])
                                curr_test_y_pred_ls, curr_test_y_pred_prob_ls = self.get_test_decision_from_db_ls_multi2(other_data_booleans_ls)
                                origin_y_pred = y_pred_prob[pat_idx]
                                if not self.multilabel:
                                    curr_test_y_pred_prob_tensor = torch.tensor(curr_test_y_pred_prob_ls, dtype=torch.float32)
                                    sorted_conjunction_ids = torch.argsort(origin_y_pred - curr_test_y_pred_prob_tensor, descending=True)
                                else:
                                    curr_test_y_pred_prob_tensor = torch.tensor(curr_test_y_pred_prob_ls, dtype=torch.float32)
                                    sorted_conjunction_ids = torch.argsort(torch.norm(torch.from_numpy(origin_y_pred) - curr_test_y_pred_prob_tensor, dim=-1), descending=True)
                                sorted_conjunction_ids_ls.append(sorted_conjunction_ids)
                                    # curr_y_pred_prob_ls = []
                                    # curr_y_pred_ls = []
                                    # for test_idx in range(len(other_data_booleans_ls)):
                                    #     curr_test_y_pred, curr_test_y_pred_prob = self.get_test_decision_from_db_ls_multi2([other_data_booleans_ls[test_idx]])
                                    #     curr_y_pred_ls.append(curr_test_y_pred)
                                    #     curr_y_pred_prob_ls.append(curr_test_y_pred_prob)
                                    # other_data_booleans = self.lang.evaluate_atom_ls_ls_on_one_patient_with_full_programs(pat_idx, all_transformed_expr_ls[pat_idx], all_other_pats_ls[pat_idx])
                                    # curr_y_pred, curr_y_pred_prob = self.get_test_decision_from_db_ls_multi2([other_data_booleans])

                                    

                            # reduced_program_ls, reduced_program_str_ls = self.remove_redundant_predicates(origin_all_other_pats_ls, all_transformed_expr_ls, next_all_other_pats_ls, next_program, next_program_str)
                            # flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_label_ls = self.concat_all_elements(reduced_program_ls, reduced_program_str_ls, y)
                            flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_label_ls = next_program, next_program_str, y
                            generated_program_ls.append(flatten_reduced_program_ls)
                            generated_program_str_ls.append(flatten_reduced_program_str_ls)
                            program_label_ls.append(flatten_label_ls)
                            save_data_path = os.path.join(self.work_dir, "save_data_dir/")
                            os.makedirs(save_data_path, exist_ok=True)


                            for pat_idx in range(len(y_pred)):
                                curr_pat_program_cols_ls = program_col_ls[pat_idx]
                                # col_ls = list(set(program_col_ls[pat_idx]))
                                for program_idx in range(len(curr_pat_program_cols_ls)):
                                    col_ls = curr_pat_program_cols_ls[program_idx]
                                    col_ls.append(self.id_cln)
                                    col_ls = list(set(col_ls))
                                    x_pat_sub = X_pd_ls[pat_idx][col_ls]
                                    x_pat_sub.reset_index(inplace=True)
                                    
                                    for col in col_ls:
                                        if not col == self.id_cln:
                                            if not col in self.test_dataset.cat_cols:
                                                x_pat_sub[col] = x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]
                                            else:
                                                x_pat_sub[col] = self.test_dataset.cat_id_unique_vals_mappings[col][x_pat_sub[col].values[0]]#x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]

                                    pat_count = torch.sum(next_all_other_pats_ls[pat_idx][program_idx]).item()

                                    # x_pat_sub.to_csv(os.path.join(save_data_path, "patient_" + str(list(X_pd_ls[pat_idx]["PAT_ID"])[0]) + ".csv"))
                                    if not self.multilabel:
                                        msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Matched Patient Count: {},  Patient Info: {}, Explanation of number {}: {}, importance score {}".format(epoch, int(y[pat_idx]), y_pred[pat_idx], y_pred_prob[pat_idx], pat_count, str(x_pat_sub.to_dict()), int(program_idx), str(next_program_str[pat_idx][program_idx]), sorted_conjunction_ids_ls[pat_idx].tolist())
                                    else:
                                        msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Matched Patient Count: {},  Patient Info: {}, Explanation of number {}: {}, importance score {}".format(epoch, int(y[pat_idx]), y_pred[pat_idx], y_pred_prob[pat_idx][int(y_pred[pat_idx])], pat_count, str(x_pat_sub.to_dict()), int(program_idx), str(next_program_str[pat_idx][program_idx]), sorted_conjunction_ids_ls[pat_idx].tolist())
                                    self.logger.log(level=logging.DEBUG, msg=msg)
                        # if y == y_pred: success += 1
                        # else: failure += 1
                        if not self.multilabel:
                            success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
                            failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
                            y_true_ls.extend(y.view(-1).tolist())
                            y_pred_ls.extend(y_pred)
                            y_pred_prob_ls.extend(y_pred_prob)
                        else:
                            y_true_ls.append(y.numpy())
                            y_pred_ls.extend(y_pred)
                            y_pred_prob_ls.extend(y_pred_prob)
                        break
                    else:
                        program = next_program
                        program_str = next_program_str
                        outbound_mask_ls = next_outbound_mask_ls
                        all_other_pats_ls = next_all_other_pats_ls
                if not self.multilabel:
                    y_true_array = np.array(y_true_ls, dtype=float)
                    y_pred_array = np.array(y_pred_ls, dtype=float)
                    y_pred_prob_array = np.array(y_pred_prob_ls, dtype=float)
                    # y_pred_prob_array = np.concatenate(y_pred_prob_ls, axis = 0)
                    y_pred_array[y_pred_array < 0] = 0.5
                    y_pred_prob_array[y_pred_prob_array < 0] = 0.5
                    if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                        auc_score_2 = 0
                    else:
                        auc_score_2 = roc_auc_score(y_true_array.reshape(-1), y_pred_prob_array.reshape(-1))
                    success_rate = (success / len(y_pred_array)) * 100.00
                    avg_loss = sum_loss/len(y_pred_array)
                    desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score:{auc_score_2}"
                    iterator.set_description(desc)
                else:
                    y_true_array = np.concatenate(y_true_ls)
                    y_pred_array = np.stack(y_pred_ls)
                    y_pred_prob_array = np.stack(y_pred_prob_ls)
                    # y_pred_prob_array = np.concatenate(y_pred_prob_ls, axis = 0)
                    # y_pred_array[y_pred_array < 0] = 0.5
                    # y_pred_prob_array[y_pred_prob_array < 0] = 0.5
                    # if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
                    #     auc_score_2 = 0
                    # else:
                    # selected_label_ids = (np.mean(y_true_array, axis=0) > 0)
                    try:
                        auc_score_2 = roc_auc_score(y_true_array, y_pred_prob_array, average=None, multi_class="ovr")
                    except ValueError:
                        auc_score_2 = np.zeros(y_pred_prob_array.shape[-1])
                    # success_rate = (success / len(y_pred_array)) * 100.00
                    success_rate = np.mean(y_true_array.reshape(-1) == y_pred_array.reshape(-1))*100
                    avg_loss = sum_loss/len(y_pred_array)
                    # desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: ({success_rate:.2f}%), auc score list:{auc_score_2.tolist()}, auc score mean:{np.mean(auc_score_2)}"
                    desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: ({success_rate:.2f}%), auc score mean:{np.mean(auc_score_2)}"
                    iterator.set_description(desc)
        if self.is_log:
            self.logger.log(level=logging.DEBUG, msg = desc)
        
        if not self.multilabel:
            additional_score_str = ""
            full_y_pred_prob_array = np.stack([1 - y_pred_prob_array.reshape(-1), y_pred_prob_array.reshape(-1)], axis=1)
            for metric_name in metrics_maps:
                curr_score = metrics_maps[metric_name](y_true_array.reshape(-1),full_y_pred_prob_array)
                additional_score_str += metric_name + ": " + str(curr_score) + " "
            print(additional_score_str)
            if self.is_log:
                self.logger.log(level=logging.DEBUG, msg = additional_score_str)
        # Print information
        
        # if exp_y_pred_arr is not None:
        #     nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
        #     print(nonzero_ids[0])
        
        if self.rl_algorithm == "dqn":
            self.dqn.policy_net.train()
            self.dqn.target_net.train()
        else:
            self.dqn.actor.train()
            self.dqn.critic.train()

        if self.is_log:
            exit(1)
            # full_generated_program_ls, full_program_str_ls, full_label_ls = self.concatenate_program_across_samples(generated_program_ls, generated_program_str_ls, program_label_ls)
            # self.cluster_programs(full_generated_program_ls, full_program_str_ls,full_label_ls)

    def run(self):
        # exp_pred_array = self.test_epoch(0)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
        if self.valid_dataset is not None:
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        # if self.valid_dataset is not None:
        # if self.rl_algorithm == "dqn":
        self.test_epoch_ls(test_loader, 0, feat_embedding=self.test_feat_embeddings)
        if self.is_log:
            self.test_epoch_ls(test_loader, 0, feat_embedding=self.test_feat_embeddings)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
        # with torch.autograd.set_detect_anomaly(True):
        for i in range(1, self.epochs + 1):
            # if self.rl_algorithm == "dqn":
            self.train_epoch(i, train_loader)
            # else:
            #     self.train_epoch_ppo(i, train_loader)
            # if self.rl_algorithm == "dqn":
            torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
            torch.save(self.dqn.target_net.state_dict(), os.path.join(self.work_dir, "target_net_" + str(i)))
            torch.save(self.dqn.memory, os.path.join(self.work_dir, "memory"))
            # else:
            #     torch.save(self.dqn.actor.state_dict(), os.path.join(self.work_dir, "actor_" + str(i)))
            #     torch.save(self.dqn.critic.state_dict(), os.path.join(self.work_dir, "critic_" + str(i)))
            # self.test_epoch(i)
            if self.valid_dataset is not None:
                self.test_epoch_ls(valid_loader, i, feat_embedding=self.valid_feat_embeddings)    
            self.test_epoch_ls(test_loader, i, feat_embedding=self.test_feat_embeddings)
            torch.cuda.empty_cache() 

            # self.test_epoch_ls(test_loader, i)


# class Trainer_all2:
#     # lang=lang, train_dataset=train_dataset, valid_dataset=valid_dataset,test_dataset = test_dataset, train_feat_embeddings=train_feat_embeddings, valid_feat_embeddings=valid_feat_embeddings, test_feat_embeddings=test_feat_embeddings, program_max_len=program_max_len, topk_act=args.num_ors,   learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs,       is_log = args.is_log, dropout_p=args.dropout_p, feat_range_mappings=feat_range_mappings, seed=args.seed, work_dir=work_dir, numeric_count=numeric_count, category_count=category_count , model=args.model, rl_algorithm=args.rl_algorithm,model_config = model_config,rl_config = rl_config
#     def __init__(self, lang:Language, train_dataset, valid_dataset, test_dataset, train_feat_embeddings, valid_feat_embeddings, test_feat_embeddings, program_max_len, topk_act, learning_rate, batch_size, epochs,is_log, dropout_p, feat_range_mappings, seed, work_dir, numeric_count=None, category_count=None, category_sum_count=None, model="mlp", rl_algorithm= "dqn", model_config=None, rl_config=None, feat_group_names=None, removed_feat_ls=None, prefer_smaller_range = False, prefer_smaller_range_coeff = 0.5, method_two=False, args = None):
#         self.topk_act =topk_act
#         if rl_algorithm == "dqn":
#             self.dqn = DQN_all2(lang=lang, replay_memory_capacity=rl_config["replay_memory_capacity"], learning_rate=learning_rate, batch_size=batch_size, gamma=rl_config["gamma"], program_max_len=program_max_len, dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, mem_sample_size=rl_config["mem_sample_size"], seed=seed, numeric_count=numeric_count, category_count=category_count,category_sum_count = category_sum_count, has_embeddings=(train_feat_embeddings is not None), model=model, topk_act=topk_act, model_config = model_config, feat_group_names = feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, method_two=method_two, args = args)
#             self.epsilon = rl_config["epsilon"]
#             self.epsilon_falloff = rl_config["epsilon_falloff"]
#             self.target_update = rl_config["target_update"]
#         else:
#             # (self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, patient_max_appts,latent_size, tf_latent_size, dropout_p, n_updates_per_iteration, clip,feat_range_mappings, seed=0, numeric_count=None, category_count=None, has_embeddings=False, model="mlp", pretrained_model_path=None, topk_act=1)
#             self.dqn = PPO_all(lang=lang, learning_rate=learning_rate, batch_size=batch_size, gamma=rl_config["gamma"], program_max_len=program_max_len, dropout_p=dropout_p, n_updates_per_iteration = rl_config["n_updates_per_iteration"], clip=rl_config["clip"], feat_range_mappings=feat_range_mappings, seed=seed, numeric_count=numeric_count, category_count=category_count, category_sum_count = category_sum_count, has_embeddings=(train_feat_embeddings is not None), model=model, topk_act=topk_act, continue_act = rl_config["continue_act"], model_config = model_config, feat_group_names = feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff= prefer_smaller_range_coeff, args = args)
#             self.timesteps_per_batch = rl_config["timesteps_per_batch"]

#         self.rl_algorithm = rl_algorithm

        
#         self.work_dir = work_dir
        
#         self.epochs = epochs
#         self.train_dataset = train_dataset
#         self.valid_dataset = valid_dataset
#         self.test_dataset = test_dataset
#         self.lang = lang
        
#         self.program_max_len = program_max_len
#         self.is_log = is_log
#         self.batch_size = batch_size
#         self.train_feat_embeddings=train_feat_embeddings
#         self.valid_feat_embeddings=valid_feat_embeddings
#         self.test_feat_embeddings=test_feat_embeddings
#         # self.do_medical = args.do_medical
#         self.multilabel = (args.dataset_name == four)
#         if self.is_log:
#             self.logger = logging.getLogger()


#     def get_test_decision_from_db(self, data: pd.DataFrame):
#         if data.shape[0] == 0:
#             return -1
#         return data[self.label_cln].value_counts().idxmax()
    
#     def get_test_decision_from_db_ls(self, data_ls: pd.DataFrame):
#         if len(data_ls) == 0:
#             return -1
        
#         label_ls = []
#         prob_label_ls = []
#         for data in data_ls:
#             if len(data) == 0:
#                 label_ls.append(-1)
#                 prob_label_ls.append(-1)
#                 continue
#             label = data[self.label_cln].value_counts().idxmax()
#             prob_label = np.mean(data[self.label_cln])
#             label_ls.append(label)
#             prob_label_ls.append(prob_label)
#         return label_ls, prob_label_ls
    
#     def get_test_decision_from_db_ls_multi(self, data_ls):
#         if len(data_ls) == 0:
#             return -1
        
#         label_ls = []
#         prob_label_ls = []
#         for data in data_ls:
#             # sub_label_ls = []
#             # sub_prob_label_ls = []
#             # for data in sub_data_ls:
#             if len(data) == 0:
#                 # sub_label_ls.append(-1)
#                 # sub_prob_label_ls.append(-1)
#                 prob_label = -1
#                 continue
#             label = data[self.label_cln].value_counts().idxmax()
#             if self.multilabel:
#                 prob_label = np.mean(np.array(list(data[self.label_cln])), axis=0)
#             else:
#                 prob_label = np.mean(data[self.label_cln])
#             label_ls.append(label)
#             prob_label_ls.append(prob_label)
                        
#         return label_ls, prob_label_ls

#     def check_db_constrants(self, data: pd.DataFrame,  y: int) -> float:
#         if len(data) == 0:
#             return 0
#         same = data.loc[data[self.label_cln] == y]["PAT_ID"].nunique()
#         total = data['PAT_ID'].nunique()
#         return same / total

#     def check_db_constrants_ls(self, data_ls,  y_ls):
#         # if len(data) == 0:
#         #     return 0
#         rwd_ls = []
#         for idx in range(len(data_ls)):
#             data = data_ls[idx]
#             # sub_rwd_ls = []
#             # for data in sub_data_ls:
#                 # if y_ls[idx].numel() == 1:
#             if not self.multilabel:
#                 y = int(y_ls[idx].item())
#                 # same = data.loc[data[self.label_cln] == y]["PAT_ID"].nunique()
#                 # total = data['PAT_ID'].nunique()
#                 if len(data) == 0:
#                     rwd_ls.append(0)
#                 else:
#                     rwd_ls.append(np.mean(data[self.label_cln] == y))
#             else:
#                 y = y_ls[idx]
#                 score_ls = []
#                 total = data['PAT_ID'].nunique()
#                 if total == 0:
#                     rwd_ls.append([0]*y_ls[idx].numel())
#                 else:
#                     for cid in range(y_ls[idx].numel()):
#                         curr_same = data.loc[np.array(list(data[self.label_cln]))[:,cid] == y[cid].item()]["PAT_ID"].nunique()
#                         curr_score = curr_same/total
#                         score_ls.append(curr_score)
#                     # score = score/y_ls[idx].numel()
#                     rwd_ls.append(score_ls)


                
            
#             # rwd_ls.append(sub_rwd_ls) 
#             # if total == 0:
#             #     rwd_ls.append(0)
#             # else:
#             #     rwd_ls.append(same / total) 
#         return np.array(rwd_ls)

#     def check_x_constraint(self, X: pd.DataFrame, atom: dict, lang) -> bool:
#         return lang.evaluate_atom_on_sample(atom, X)

#     def check_program_constraint(self, prog: list) -> bool:
#         return len(prog) < self.program_max_len
    
#     def identify_op(self, X:pd, atom:dict):

#         atom_ls = []
        

#         atom1 = dict()
#         for k in atom:
#             if k not in self.lang.syntax["num_feat"]:
#                 atom1[k] = atom[k]
#             else:
#                 atom1[k] = atom[k][0][0]
#                 atom1[k + "_prob"] = atom[k][1][0]

#         atom1["num_op"] = operator.__ge__

#         atom2 = dict()
#         for k in atom:
#             if k not in self.lang.syntax["num_feat"]:
#                 atom2[k] = atom[k]
#             else:
#                 atom2[k] = atom[k][0][1]
#                 atom2[k + "_prob"] = atom[k][1][1]
#         atom2["num_op"] = operator.__le__
#         atom_ls.append(atom1)
#         atom_ls.append(atom2)
            
#         return atom_ls
    
#     def identify_op_ls(self, batch_size:int, atom:dict):

#         atom_ls = []        

#         # atom1 = [dict()]*batch_size
#         atom1 = []

#         atom2 = []
#         for _ in range(batch_size):
#             atom2.append(dict())
#         for _ in range(batch_size):
#             atom1.append(dict())
#         for k in atom:
#             # if k not in self.lang.syntax["num_feat"]:
#             if type(k) is not tuple:
#                 if type(atom[k]) is not dict:
#                     for atom_id in range(batch_size):
#                         atom1[atom_id][k] = atom[k]
#                         atom2[atom_id][k] = atom[k]
#                 else:
#                     # atom1[k] = [None]*batch_size
#                     for selected_item in atom[k]:
#                         sample_ids = atom[k][selected_item]
#                         for sample_id in sample_ids:
#                             atom1[sample_id.item()][k] = selected_item
#                             atom2[sample_id.item()][k] = selected_item
#             else:
                
#                 # atom1[k] = [None]*batch_size
#                 # atom1[k + "_prob"] = [None]*batch_size
#                 if k[0].endswith("_lb"):
#                     for selected_item in atom[k][2]:
#                         sample_ids = atom[k][2][selected_item]
#                         for sample_id_id in range(len(sample_ids)):
#                             atom1[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
#                             # atom1[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
#                 else:
#                     for selected_item in atom[k][2]:
#                         sample_ids = atom[k][2][selected_item]
#                         for sample_id_id in range(len(sample_ids)):
#                             atom2[sample_ids[sample_id_id].item()][self.dqn.policy_net.get_prefix(selected_item)] = atom[k][0][selected_item][sample_id_id]
#                             # atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][0][sample_id_id]
#                         # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


#                 # atom1[k] = atom[k][0][0]
#                 # atom1[k + "_prob"] = atom[k][1][0]
#                 # atom1[k + "_sample_ids"] = atom[k][2][0]
#         for sample_id in range(len(atom1)):
#             atom1[sample_id]["num_op"] = operator.__ge__   

#         for sample_id in range(len(atom2)):
#             atom2[sample_id]["num_op"] = operator.__le__   

        
#         # for k in atom:
#         #     # if k not in self.lang.syntax["num_feat"]:
#         #     if type(k) is not tuple:
#         #         if type(atom[k]) is not dict:
#         #             for atom_id in range(batch_size):
#         #                 atom2[atom_id][k] = atom[k]
#         #         else:
#         #             for selected_item in atom[k]:
#         #                 sample_ids = atom[k][selected_item]
#         #                 for sample_id in sample_ids:
#         #                     atom2[sample_id.item()][k] = selected_item
#         #     else:
                
#         #         for selected_item in atom[k][2]:
#         #             sample_ids = atom[k][2][selected_item]
#         #             for sample_id_id in range(len(sample_ids)):
#         #                 atom2[sample_ids[sample_id_id].item()][selected_item] = atom[k][0][1][selected_item][sample_id_id]
#         #                 atom2[sample_ids[sample_id_id].item()][selected_item + "_prob"] = atom[k][1][selected_item][1][sample_id_id]
#         #                 # atom1[sample_ids[sample_id_id].item()][k + "_"] = atom[k][1][selected_item][0][sample_id_id.item()]


#         #         # atom1[k] = atom[k][0][0]
#         #         # atom1[k + "_prob"] = atom[k][1][0]
#         #         # atom1[k + "_sample_ids"] = atom[k][2][0]
#         # for sample_id in range(len(atom2)):
#         #     atom2[sample_id]["num_op"] = operator.__le__  


#         # atom2 = dict()
#         # for k in atom:
#         #     if k not in self.lang.syntax["num_feat"]:
#         #         atom2[k] = atom[k]
#         #     else:
#         #         atom2[k] = atom[k][0][1]
#         #         atom2[k + "_prob"] = atom[k][1][1]
#         # atom2["num_op"] = operator.__le__
#         atom_ls.append(atom1)
#         atom_ls.append(atom2)
            
#         return atom_ls
#     def check_x_constraint_with_atom_ls(self, X: pd.DataFrame, atom_ls:list, lang) -> bool:
#         satisfy_bool=True
#         for atom in atom_ls:
#             curr_bool = lang.evaluate_atom_on_sample(atom, X)
#             satisfy_bool = satisfy_bool & curr_bool
#         return satisfy_bool

#     def process_curr_atoms(self, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls):
#         # if not self.do_medical:    
#         curr_atom_str_ls = self.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings, self.train_dataset.cat_id_unique_vals_mappings)
#         # else:
#         #     curr_atom_str_ls = self.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, self.train_dataset.feat_range_mappings)
        
#         # outbound_mask_ls = atom_ls[outbound_key]
        
#         next_program = program.copy()
        
#         next_outbound_mask_ls=outbound_mask_ls.copy()
        
#         next_program_str = program_str.copy()
        
#         curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

#         # if len(program) > 0:                        
#         #     next_program, program_col_ls, next_program_str, next_outbound_mask_ls = self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)
#         # else:
#         next_program.append(curr_vec_ls)
#         next_outbound_mask_ls.append(atom_ls[outbound_key])
#         for vec_idx in range(len(curr_vec_ls)):
#             # vec = curr_vec_ls[vec_idx]
#             atom_str = curr_atom_str_ls[vec_idx]
#             program_col_ls[vec_idx].append(atom_ls[col_key][vec_idx])
#             next_program_str[vec_idx].append(atom_str)
#             # for k in range(len(atom_ls[col_key][vec_idx])):
#             #     program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
#             #     next_program_str[vec_idx][k].append(atom_str[k])
#         # if not self.do_medical:
#         next_all_other_pats_ls,_ = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi_2(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
#         # else:
#         #     next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_key, range_key)
#         return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls

#     def train_epoch_ppo(self, epoch, train_loader):
#         success, failure, sum_loss = 0, 0, 0.
#         t = 0
#         iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
#         batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        
#         for episode_i, val in iterator:
#             (all_other_pats_ls, X_pd_ls, X, X_sample_ids), y = val
#             all_other_pats_ls = self.copy_data_in_database2(all_other_pats_ls)
#             X_feat_embedding = None
#             if self.train_feat_embeddings is not None:
#                 X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
#             # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
#             program = []
#             # program_str = [[] for _ in range(len(X_pd_ls))]
#             # program_atom_ls = [[] for _ in range(len(X_pd_ls))]
#             program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
#             program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            
#             X_pd_full = pd.concat(X_pd_ls)
#             prev_reward = np.zeros([len(X), self.topk_act])
#             ep_rews = []
#             for arr_idx in range(self.program_max_len):
#                 # (col, op) = col_op_ls[arr_idx]
#                 # col_name = col_list[col_id]
#                 if X_feat_embedding is None:
#                     atom_ls = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_full, program=program, train=True)
#                 else:
#                     atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, train=True)

#                 next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = self.process_curr_atoms(atom_ls, program, program_str, all_other_pats_ls, program_col_ls)

#                 db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
#                 #derive reward
#                 reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
#                 done = (arr_idx == self.program_max_len-1)
#                 if X_feat_embedding is None:
#                     batch_obs.append((X, X_pd_full, program))
#                 else:
#                     batch_obs.append(((X, X_feat_embedding), X_pd_full, program))
#                 ep_rews.append(reward - prev_reward)
#                 batch_acts.append(atom_ls)
#                 # atom_probs = atom_ls[pred_prob_key] #self.PPO.idx_to_logs(atom_pred, atom_idx)
#                 # atom_log_probs = atom_probs[torch.tensor(list(range(atom_probs.shape[0]))), atom_ls[pred_prob_id]]
#                 atom_log_probs = self.dqn.idx_to_logs(atom_ls, atom_ls)
#                 batch_log_probs.append(atom_log_probs)
                
#                 if done: #stopping condition
#                     # if reward > 0.5: success += 1
#                     # else: failure += 1
#                     success += np.sum(reward > 0.5)
#                     failure += np.sum(reward <= 0.5)
#                     break
#                 else:
#                     program = next_program
#                     program_str = next_program_str
#                     all_other_pats_ls = next_all_other_pats_ls
#                     prev_reward = reward
            
#             batch_rews.append(ep_rews)
#             batch_lens.append(len(program))

#             t += 1

#             if t == self.timesteps_per_batch:
#                 batch_rtgs = self.dqn.compute_rtgs(batch_rews=batch_rews)
#                 batch = (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)
#                 sum_loss += self.dqn.learn(batch=batch)
#                 batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = [], [], [], [], [], []
#                 t = 0

#             # Print information
#             success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
#             avg_loss = sum_loss/(episode_i+1)
#             desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{(episode_i + 1)*self.batch_size} ({success_rate:.2f}%)"
#             iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)


#     def train_epoch(self, epoch, train_loader):
#         success, failure, sum_loss = 0, 0, 0.
        
#         iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
        
#         for episode_i, val in iterator:
#             (all_other_pats_ls, X_pd_ls, X, X_sample_ids, (X_num, X_cat), _), y = val
#             # all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)
#             X_feat_embedding = None
#             if self.train_feat_embeddings is not None:
#                 X_feat_embedding = self.train_feat_embeddings[X_sample_ids]
            
#             # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
#             program = []
#             outbound_mask_ls = []
#             program_str = [[] for _ in range(len(X_pd_ls))]
#             program_col_ls = [[] for _ in range(len(X_pd_ls))]
#             # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
#             # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
            
#             X_pd_full = pd.concat(X_pd_ls)
            
            
            
#             # col_comp_ls = zip()
#             # while True: # episode
#             # for 
#             # for col_id in col_ids:
#             prev_reward = np.zeros([len(X)])
#             # random.shuffle(col_op_ls)
#             # last_col , last_op  = col_op_ls[-1]


#             # for arr_idx in range(len(col_op_ls)):
#             for arr_idx in range(self.program_max_len):
#                 # (col, op) = col_op_ls[arr_idx]
#                 # col_name = col_list[col_id]
#                 if X_feat_embedding is None:
#                     atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=self.epsilon)
#                 else:
#                     atom_ls = self.dqn.predict_atom_ls(features=(X, X_feat_embedding), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=self.epsilon)
                
#                 # curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.train_dataset.feat_range_mappings)
                
#                 # next_program = program.copy()
                
#                 # next_program_str = program_str.copy()
                
#                 # curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)

#                 # if len(program) > 0:                        
#                 #     next_program, program_col_ls, next_program_str= self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls)
#                 # else:
#                 #     next_program.append(curr_vec_ls)
#                 #     for vec_idx in range(len(curr_vec_ls)):
#                 #         # vec = curr_vec_ls[vec_idx]
#                 #         atom_str = curr_atom_str_ls[vec_idx]
#                 #         for k in range(len(atom_ls[col_key][vec_idx])):
#                 #             program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
#                 #             next_program_str[vec_idx][k].append(atom_str[k])
                
#                 # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
#                 next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls = self.process_curr_atoms(atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls)
#                 db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
#                 #derive reward
#                 if not self.multilabel:
#                     reward = db_cons# if x_cons else 0 # NOTE: these become part of reward
#                 else:
#                     reward = np.mean(db_cons,axis=-1)
#                 done = (arr_idx == self.program_max_len-1)
#                 #record transition in buffer
#                 if done:
#                     next_state = None
#                     next_program = None
#                 else:
#                     next_state = (next_program, next_outbound_mask_ls)
#                 if X_feat_embedding is None:
#                     transition = Transition((X, X_num, X_cat), X_pd_full,(program, outbound_mask_ls), atom_ls, next_state, reward - prev_reward)
#                 else:
#                     transition = Transition((X, X_feat_embedding), X_pd_full,(program, outbound_mask_ls), atom_ls, next_state, reward - prev_reward)
#                 self.dqn.observe_transition(transition)
#                 #update model
#                 loss = self.dqn.optimize_model_ls0()
#                 # print(loss)
#                 sum_loss += loss
#                 #update next step
#                 if done: #stopping condition
#                     # if reward > 0.5: success += 1
#                     # else: failure += 1
#                     if not self.multilabel:
#                         # success += np.sum(np.max(reward, axis = -1) > 0.5)
#                         success += np.sum(db_cons > 0.5)
#                     else:
#                         success += np.sum(db_cons > 0.5)
#                     break
#                 else:
#                     program = next_program
#                     program_str = next_program_str
#                     all_other_pats_ls = next_all_other_pats_ls
#                     prev_reward = reward
#                     outbound_mask_ls = next_outbound_mask_ls
#             # Update the target net
#             if episode_i % self.target_update == 0:
#                 self.dqn.update_target()
#             # Print information
#             total_count = ((episode_i + 1)*self.batch_size)
#             success_rate = (success / ((episode_i + 1)*self.batch_size)) * 100.0
#             avg_loss = sum_loss/(episode_i+1)
#             desc = f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{total_count} ({success_rate:.2f}%)"
#             iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
#         self.epsilon *= self.epsilon_falloff

    
#     def test_epoch(self, epoch):
#         success, failure, sum_loss = 0, 0, 0.
#         iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
#         y_true_ls=[]
#         y_pred_ls=[]
#         self.dqn.policy_net.eval()
#         self.dqn.target_net.eval()
#         with torch.no_grad():
#             for episode_i, val in iterator:
#                 # if episode_i == 28:
#                 #     print()
#                 (all_other_pats, X_pd, X), y = val
#                 program = []
#                 program_str = []
#                 program_atom_ls = []
#                 while True: # episode
#                     atom = self.dqn.predict_atom(features=X, X_pd=X_pd, program=program, epsilon=0)
#                     atom_ls = self.identify_op(X_pd, atom)
#                     next_program = program.copy()
#                     next_program_str = program_str.copy()
#                     for new_atom in atom_ls:
#                         next_program = next_program + [self.dqn.atom_to_vector(new_atom)]
#                         # atom["num_op"] = atom_op
                        
                        
#                         next_program_str = next_program_str + [self.lang.atom_to_str(new_atom)]
                        
#                         program_atom_ls.append(new_atom)
#                     #apply new atom
#                     next_all_other_pats = self.lang.evaluate_atom_ls_on_dataset(program_atom_ls, all_other_pats)
#                     # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                     # next_program_str = program_str.copy()+[self.lang.atom_to_str(atom)]
#                     #check constraints
#                     # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
#                     prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                     y_pred = self.get_test_decision_from_db(next_all_other_pats)# if x_cons else -1
#                     db_cons = self.check_db_constrants(next_all_other_pats, y=y_pred)  # entropy
#                     #derive reward
#                     done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
#                     if done:
#                         next_program = None
#                     #update next step
#                     if done: #stopping condition
#                         if self.is_log:
#                             msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Patient Info: {}, Explanation: {}".format(epoch, int(y[0]), y_pred, db_cons, str(X_pd.to_dict()),str(next_program_str))
#                             self.logger.log(level=logging.DEBUG, msg=msg)
#                         if y == y_pred: success += 1
#                         else: failure += 1
#                         y_true_ls.append(y.item())
#                         y_pred_ls.append(y_pred)
#                         break
#                     else:
#                         program = next_program
#                         program_str = next_program_str
#                         all_other_pats = next_all_other_pats

#                 y_true_array = np.array(y_true_ls, dtype=float)
#                 y_pred_array = np.array(y_pred_ls, dtype=float)
#                 y_pred_array[y_pred_array < 0] = 0.5
#                 if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#                 #     recall = 0
#                 #     f1 = 0
#                     auc_score= 0
#                 else:
#                     auc_score = roc_auc_score(y_true_array, y_pred_array)

#                 # if episode_i == self.batch_size:
#                 #     print(y_true_array.reshape(-1))
#                 #     print(y_pred_array.reshape(-1))

#                 # Print information
#                 success_rate = (success / (episode_i + 1)) * 100.00
#                 avg_loss = sum_loss/(episode_i+1)
#                 desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%), auc score: {auc_score}"
#                 iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
            
#         self.dqn.policy_net.train()
#         self.dqn.target_net.train()
#         return y_pred_array
    
#     def integrate_curr_program_with_prev_programs(self, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls):
#         prev_prog_ids = atom_ls[prev_prog_key].cpu()
#         curr_col_ids = atom_ls[col_key]
#         outbound_mask = atom_ls[outbound_key]
#         program = []
#         outbound_mask_ls = []
#         sample_ids = torch.arange(len(next_program[0]))
#         # program length
#         for pid in range(len(next_program)):
#             program.append(torch.stack([next_program[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])],dim=1))
#             outbound_mask_ls.append(torch.stack([next_outbound_mask_ls[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])], dim=-1))
#         program.append(curr_vec_ls)
#         outbound_mask_ls.append(outbound_mask)
#         new_program_col_ls = []
#         new_program_str = []
#         for idx in range(len(program_col_ls)):
#             curr_sample_new_program_col_ls = []
#             curr_sample_new_program_str = []
#             for k in range(self.topk_act):
#                 curr_new_program_col_ls = []
#                 curr_new_program_str = []
#                 # for pid in range(len(program_col_ls[idx])):
                
#                 #     curr_new_program_col_ls.append(program_col_ls[idx][prev_prog_ids[idx,k].item()][pid])
#                 #     # [k].append()
#                 #     curr_new_program_str.append(next_program_str[idx][prev_prog_ids[idx,k].item()][pid])
#                 curr_new_program_col_ls.extend(program_col_ls[idx][prev_prog_ids[idx,k].item()])
#                 curr_new_program_str.extend(next_program_str[idx][prev_prog_ids[idx,k].item()])
                
                
#                 curr_new_program_col_ls.append(curr_col_ids[idx][k])
#                 curr_new_program_str.append(curr_atom_str_ls[idx][k])
#                 curr_sample_new_program_col_ls.append(curr_new_program_col_ls)
#                 curr_sample_new_program_str.append(curr_new_program_str)
#             new_program_col_ls.append(curr_sample_new_program_col_ls)
#             new_program_str.append(curr_sample_new_program_str)
#         return program, new_program_col_ls, new_program_str, outbound_mask_ls

#     def copy_data_in_database(self, all_other_pats_ls):
#         all_other_pats_ls_ls = []
#         for idx in range(len(all_other_pats_ls)):
#             curr_other_pats_ls = []
#             for k in range(self.topk_act):
#                 curr_other_pats_ls.append(all_other_pats_ls[idx].copy())
            
#             all_other_pats_ls_ls.append(curr_other_pats_ls)
            
#         return all_other_pats_ls_ls
    
#     def concatenate_program_across_samples(self, generated_program_ls, generated_program_str_ls, program_label_ls):
#         full_program_ls = []
#         full_program_str_ls = []
#         full_label_ls = []
#         # for k in range(len(generated_program_ls[0])):
#         #     curr_full_program_ls = []
#         #     curr_full_program_str_ls = []
#         #     curr_full_label_ls = []
#         #     for idx in range(len(generated_program_ls)):
#         #         curr_full_program_ls.append(generated_program_ls[idx][k].view(-1,generated_program_ls[idx][k].shape[-1]))
#         #         curr_full_label_ls.append(program_label_ls[idx][:,k].unsqueeze(1).repeat(1, generated_program_ls[idx][k].shape[1]).view(-1, 1))
#         #         curr_generated_program_str_ls = []
#         #         for i in range(len(generated_program_str_ls[idx])):
#         #             sub_curr_generated_program_str_ls = []
#         #             for j in range(len(generated_program_str_ls[idx][0])):
#         #                 sub_curr_generated_program_str_ls.append(generated_program_str_ls[idx][i][j][k])
                    
#         #             curr_generated_program_str_ls.append(sub_curr_generated_program_str_ls)
                    
#         #         curr_full_program_str_ls.extend(curr_generated_program_str_ls)
            
#         #     full_program_str_ls.extend(curr_full_program_str_ls)
#         #     full_program_ls.append(torch.cat(curr_full_program_ls))
#         #     full_label_ls.append(torch.cat(curr_full_label_ls))
        
#         for idx in range(len(generated_program_ls)):
#             full_program_ls.append(torch.stack(generated_program_ls[idx]))
#             full_label_ls.append(torch.cat(program_label_ls[idx]))
#         for p_str_ls in generated_program_str_ls:
#             full_program_str_ls.extend(p_str_ls)
        
        
        
#         return torch.cat(full_program_ls), full_program_str_ls, torch.cat(full_label_ls)
    
#     def decode_program_to_str(self, single_program):
#         program_str = self.dqn.vector_ls_to_str_ls0(single_program)
#         return program_str

#     def redundancy_metrics(self, existing_data, target_data):
#         if len(existing_data) == len(target_data):
#             return True

#         return False

#     def concat_all_elements(self, reduced_program_ls, reduced_program_str_ls, labels):
#         flatten_reduced_program_ls = []
#         flatten_reduced_program_str_ls = []
#         flatten_labels = []
#         for i in range(len(reduced_program_ls)):
#             for j in range(len(reduced_program_ls[i])):
#                 for k in range(len(reduced_program_ls[i][j])):
#                     flatten_reduced_program_ls.append(reduced_program_ls[i][j][k])
#                     flatten_reduced_program_str_ls.append(reduced_program_str_ls[i][j][k])
#                     flatten_labels.append(labels[i])
                    
#         return flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_labels
                

#     def remove_redundant_predicates(self, all_other_pats_ls, all_transformed_expr_ls, next_all_other_pats_ls, next_program, next_program_str):
#         transposed_expr_ls = []
#         transposed_next_program = []
#         for j in range(len(all_transformed_expr_ls[0])):
#             curr_transposed_expr_ls = []
#             curr_program_ls = []
#             for k in range(len(all_transformed_expr_ls[0][0])):
#                 sub_curr_transposed_expr_ls = []
#                 sub_curr_program_ls = []
#                 for i in range(len(all_transformed_expr_ls)):
                    
#                     sub_curr_transposed_expr_ls.append(all_transformed_expr_ls[i][j][k])
#                     sub_curr_program_ls.append(next_program[i][j][k])
#                 curr_transposed_expr_ls.append(sub_curr_transposed_expr_ls)
#                 curr_program_ls.append(sub_curr_program_ls)
            
#             transposed_expr_ls.append(curr_transposed_expr_ls)
#             transposed_next_program.append(curr_program_ls)


#         all_other_pats_ls = self.copy_data_in_database(all_other_pats_ls)

#         reduced_program_ls = []
#         reduced_program_str_ls = []

#         for i in range(len(transposed_expr_ls)):
#             curr_reduced_program_ls = []
#             curr_reduced_program_str_ls = []
#             for j in range(len(transposed_expr_ls[i])):
#                 redundant_clause_id_ls = []
#                 sub_curr_reduced_program_ls = []
#                 sub_curr_reduced_program_str_ls = []

#                 for k in range(len(transposed_expr_ls[i][j])):
#                     temp_expr_ls = transposed_expr_ls[i][j].copy()
#                     del temp_expr_ls[k]
#                     existing_data = all_other_pats_ls[i][j].copy()
#                     for expr_c in temp_expr_ls:
#                         curr_op = expr_c[1]
#                         curr_const = expr_c[2]
#                         expr = curr_op(existing_data[expr_c[0]], curr_const)
#                         existing_data = self.lang.evaluate_expression_on_data(existing_data, expr)
#                     if self.redundancy_metrics(existing_data, next_all_other_pats_ls[i][j]):
#                         redundant_clause_id_ls.append(k)
#                     else:
#                         sub_curr_reduced_program_ls.append(transposed_next_program[i][j][k])
#                         sub_curr_reduced_program_str_ls.append(next_program_str[i][j][k])
#                 curr_reduced_program_ls.append(sub_curr_reduced_program_ls)
#                 curr_reduced_program_str_ls.append(sub_curr_reduced_program_str_ls)
#             reduced_program_ls.append(curr_reduced_program_ls)
#             reduced_program_str_ls.append(curr_reduced_program_str_ls)
                
#         return reduced_program_ls, reduced_program_str_ls

#     def cluster_programs(self, full_program_ls, full_program_str_ls, full_label_ls):
#         # full_label_ls_tensor = torch.cat(full_label_ls)
#         # full_program_ls_tensor = torch.cat(full_program_ls)
        
#         full_label_ls_tensor = full_label_ls
#         full_program_ls_tensor = full_program_ls

#         unique_labels = full_label_ls_tensor.unique().tolist()

#         for label in unique_labels:

#             print("print info for label ", str(label))

#             curr_full_program_ls_tensor = full_program_ls_tensor[full_label_ls_tensor.view(-1) == label][0:-1]
#             curr_idx_ls = torch.nonzero(full_label_ls_tensor.view(-1) == label).view(-1).tolist()
#             cluster_assignment_ids, cluster_centroids = KMeans(curr_full_program_ls_tensor, K=5)
#             approx_cluster_centroids, min_program_ids = get_closet_samples_per_clusters(cluster_centroids, curr_full_program_ls_tensor)
#             for idx in range(len(min_program_ids.tolist())):
#                 selected_program_str = full_program_str_ls[curr_idx_ls[min_program_ids[idx]]]
#                 print("cluster idx %d:%s"%(idx, selected_program_str))
#                 print("cluster count for cluster idx %d:%d"%(idx, torch.sum(cluster_assignment_ids==idx).item()))

#             program_str_ls = []
#             for idx in range(len(cluster_centroids)):
#                 if len(torch.nonzero(cluster_centroids[idx])) <= 0:
#                     continue

#                 program_str = self.decode_program_to_str(cluster_centroids[idx])
#                 print("cluster idx %d:%s"%(idx, program_str))
#             # program_str_ls.append(program_str) 
#         # return program_str_ls
#         print()
        
#     def test_epoch_ls(self, test_loader, epoch, exp_y_pred_arr = None, feat_embedding = None):
#         pd.options.mode.chained_assignment = None

#         success, failure, sum_loss = 0, 0, 0.

#         iterator = tqdm(enumerate(test_loader), desc="Training Synthesizer", total=len(test_loader))
#         # iterator = tqdm(enumerate(self.test_dataset), desc="Testing Synthesizer", total=len(self.test_dataset))
#         y_true_ls=[]
#         y_pred_ls=[]
#         y_pred_prob_ls=[]
#         if self.rl_algorithm == "dqn":
#             self.dqn.policy_net.eval()
#             self.dqn.target_net.eval()
#         else:
#             self.dqn.actor.eval()
#             self.dqn.critic.eval()
            
#         generated_program_ls = []
#         generated_program_str_ls = []
#         program_label_ls = []
        
        
#         with torch.no_grad():
#             # col_list = list(self.train_dataset.data.columns)
        
#             # op_ls = list([operator.__le__, operator.__ge__])
            
#             # col_op_ls = []
            
#             # last_col = None

#             # last_op = None
            
#             # for col in col_list:
#             #     if col == "PAT_ID" or col == "label":
#             #         continue
#             #     last_col = col
#             #     for op in op_ls:
#             #         col_op_ls.append((col, op))
#             #         last_op = op
#             for episode_i, val in iterator:
#                 # if episode_i == 13:
#                 #     print()
#                 (all_other_pats_ls, X_pd_ls, X, X_sample_ids, (X_num, X_cat), _), y = val
#                 # all_other_pats_ls = self.copy_data_in_database(origin_all_other_pats_ls)
                
#                 # for x_pd_idx in range(len(X_pd_ls)):
#                 #     if np.sum(X_pd_ls[x_pd_idx]["PAT_ID"] == 277) >= 1:
#                 #         print(x_pd_idx)
#                 #         break                
#                 X_feat_embeddings = None
#                 if feat_embedding is not None:
#                     X_feat_embeddings = feat_embedding[X_sample_ids]
                
#                 # (all_other_pats, X_pd, X), y = self.train_dataset[all_rand_ids[sample_idx].item()]
#                 program = []
#                 outbound_mask_ls = []
#                 # program_str = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
#                 # program_col_ls = [[[] for _ in range(self.topk_act)] for _ in range(len(X_pd_ls))]
#                 program_str = [[] for _ in range(len(X_pd_ls))]
#                 program_col_ls = [[] for _ in range(len(X_pd_ls))]
#                 # for p_k in range(len(program_str)):
#                 #     program_str[p_k].append([[] for _ in range(self.topk_act)])
#                 #     program_col_ls[p_k].append([[] for _ in range(self.topk_act)])
                
                
#                 X_pd_full = pd.concat(X_pd_ls)
#                 all_transformed_expr_ls = []
#                 # for arr_idx in range(len(col_op_ls)):
#                 for arr_idx in range(self.program_max_len):
#                     # (col, op) = col_op_ls[arr_idx]
#                     # col_name = col_list[col_id]
#                     if X_feat_embeddings is None:
#                         if self.rl_algorithm == "dqn":
#                             atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=0)
#                         else:
#                             atom_ls = self.dqn.predict_atom_ls(features=(X, X_num, X_cat), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, train=False)
#                     else:
#                         if self.rl_algorithm == "dqn":
#                             atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, epsilon=0)
#                         else:
#                             atom_ls = self.dqn.predict_atom_ls(features=(X,X_feat_embeddings), X_pd_ls=X_pd_full, program=program, outbound_mask_ls=outbound_mask_ls, train=False)
                    
#                     # if not self.do_medical:
#                     curr_atom_str_ls = self.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings, self.test_dataset.cat_id_unique_vals_mappings)
#                     # else:
#                     #     curr_atom_str_ls = self.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, self.test_dataset.feat_range_mappings)
#                         # curr_atom_str_ls = self.lang.atom_to_str_ls_full(atom_ls, col_key, op_key, pred_v_key, self.test_dataset.feat_range_mappings)

#                     next_program = program.copy()
#                     next_outbound_mask_ls = outbound_mask_ls.copy()
#                     next_program_str = program_str.copy()
#                     curr_vec_ls = self.dqn.atom_to_vector_ls0(atom_ls)
                    
#                     # if len(program) > 0:                        
                        

#                     #     next_program, program_col_ls, next_program_str, next_outbound_mask_ls = self.integrate_curr_program_with_prev_programs(next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)



#                     # else:

#                     next_program.append(curr_vec_ls)
#                     next_outbound_mask_ls.append(atom_ls[outbound_key])

#                     for vec_idx in range(len(curr_vec_ls)):
#                         # vec = curr_vec_ls[vec_idx]
#                         atom_str = curr_atom_str_ls[vec_idx]
#                         program_col_ls[vec_idx].append(atom_ls[col_key][vec_idx])
#                         next_program_str[vec_idx].append(atom_str)
#                         # for k in range(len(atom_ls[col_key][vec_idx])):
#                         #     program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
#                         #     next_program_str[vec_idx][k].append(atom_str[k])
#                             # next_program_str[vec_idx].append(atom_str)

                    
#                     # atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
#                     # reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

                    
                    
#                     # for new_atom_ls in atom_ls_ls:

#                     #     curr_vec_ls = self.dqn.atom_to_vector_ls0(new_atom_ls)

#                     #     next_program.append(torch.stack(curr_vec_ls))

#                     #     curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

                    
#                 # while True: # episode
#                 #     atom,_ = self.dqn.predict_atom_ls(features=X, X_pd_ls=X_pd_ls, program=program, epsilon=0)
#                 #     atom_ls_ls = self.identify_op_ls(X.shape[0], atom)
#                 #     reorg_atom_ls_ls= [[] for _ in range(len(X_pd_ls))]

#                 #     next_program = program.copy()
#                 #     next_program_str = program_str.copy()
#                 #     for new_atom_ls in atom_ls_ls:

#                 #         curr_vec_ls = self.dqn.atom_to_vector_ls(new_atom_ls)

#                 #         next_program.append(torch.stack(curr_vec_ls))

#                 #         curr_atom_str_ls = self.lang.atom_to_str_ls(new_atom_ls)

#                 #         for vec_idx in range(len(curr_vec_ls)):
#                 #             vec = curr_vec_ls[vec_idx]
#                 #             atom_str = curr_atom_str_ls[vec_idx]
                            
#                 #             next_program_str[vec_idx].append(atom_str)
#                 #             program_atom_ls[vec_idx].append(new_atom_ls[vec_idx])
#                 #             reorg_atom_ls_ls[vec_idx].append(new_atom_ls[vec_idx])
#                         # atom["num_op"] = atom_op
                        
                        
#                         # next_program_str = next_program_str + []
                        
#                         # program_atom_ls.append(new_atom_ls)
#                     #apply new atom
#                     # next_all_other_pats_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
#                     # if not self.do_medical:
#                     next_all_other_pats_ls, transformed_expr_ls = self.lang.evaluate_atom_ls_ls_on_dataset_full_multi_2(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key)
#                     # else:
#                     #     next_all_other_pats_ls =  self.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_key, range_key)
#                     # next_program = program.copy() + [self.dqn.atom_to_vector(atom)]
#                     # next_program_str = program_str.copy() + [self.lang.atom_to_str(atom)]
#                     #check constraints
#                     # x_cons = self.check_x_constraint_with_atom_ls(X_pd, program_atom_ls, lang) #is e(r)?
#                     # prog_cons = self.check_program_constraint(next_program) #is len(e) < 10
#                     # db_cons = self.check_db_constrants_ls(next_all_other_pats_ls, y) #entropy
#                     # y_pred = self.get_test_decision_from_db(next_all_other_pats_ls) if x_cons else -1
#                     y_pred, y_pred_prob = self.get_test_decision_from_db_ls_multi(next_all_other_pats_ls)
#                     # final_y_pred,_ = stats.mode(np.array(y_pred), axis = -1)
#                     # final_y_pred_prob = np.mean(np.array(y_pred_prob), axis = -1)
                    
#                     # done = atom["formula"] == "end" or not prog_cons# or not x_cons # NOTE: Remove reward check
#                     # done = (col == last_col) and (op == last_op)
#                     done = (arr_idx == self.program_max_len - 1)

#                     all_transformed_expr_ls.append(transformed_expr_ls)
#                     # if done:
#                     #     next_program = None
#                     #update next step
#                     if done: #stopping condition
                        
                        
                        
#                         if self.is_log:
#                             # reduced_program_ls, reduced_program_str_ls = self.remove_redundant_predicates(origin_all_other_pats_ls, all_transformed_expr_ls, next_all_other_pats_ls, next_program, next_program_str)
#                             # flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_label_ls = self.concat_all_elements(reduced_program_ls, reduced_program_str_ls, y)
#                             flatten_reduced_program_ls, flatten_reduced_program_str_ls, flatten_label_ls = next_program, next_program_str, y
                            
#                             generated_program_ls.append(flatten_reduced_program_ls)
#                             generated_program_str_ls.append(flatten_reduced_program_str_ls)
#                             program_label_ls.append(flatten_label_ls)
#                             save_data_path = os.path.join(self.work_dir, "save_data_dir/")
#                             os.makedirs(save_data_path, exist_ok=True)


#                             for pat_idx in range(len(y_pred)):
#                                 curr_pat_program_cols_ls = program_col_ls[pat_idx]
#                                 # col_ls = list(set(program_col_ls[pat_idx]))
#                                 for program_idx in range(len(curr_pat_program_cols_ls)):
#                                     col_ls = curr_pat_program_cols_ls[program_idx]
#                                     col_ls.append("PAT_ID")
#                                     col_ls = list(set(col_ls))
#                                     x_pat_sub = X_pd_ls[pat_idx][col_ls]
#                                     x_pat_sub.reset_index(inplace=True)
                                    
#                                     for col in col_ls:
#                                         if not col == "PAT_ID":
#                                             if not col in self.test_dataset.cat_cols:
#                                                 x_pat_sub[col] = x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]
#                                             else:
#                                                 x_pat_sub[col] = self.test_dataset.cat_id_unique_vals_mappings[col][x_pat_sub[col].values[0]]#x_pat_sub[col]*(self.test_dataset.feat_range_mappings[col][1] - self.test_dataset.feat_range_mappings[col][0]) + self.test_dataset.feat_range_mappings[col][0]

#                                     pat_count = len(next_all_other_pats_ls[pat_idx][program_idx])

#                                     x_pat_sub.to_csv(os.path.join(save_data_path, "patient_" + str(list(X_pd_ls[pat_idx]["PAT_ID"])[0]) + ".csv"))
                                    
#                                     msg = "Test Epoch {},  Label: {}, Prediction: {}, Match Score:{:7.4f}, Matched Patient Count: {},  Patient Info: {}, Explanation of number {}: {}".format(epoch, int(y[pat_idx]), y_pred[pat_idx], y_pred_prob[pat_idx], pat_count, str(x_pat_sub.to_dict()), int(program_idx), str(reduced_program_str_ls[pat_idx][program_idx]))
#                                     self.logger.log(level=logging.DEBUG, msg=msg)
#                         # if y == y_pred: success += 1
#                         # else: failure += 1
#                         if not self.multilabel:
#                             success += np.sum(y.view(-1).numpy() == np.array(y_pred).reshape(-1))
#                             failure += np.sum(y.view(-1).numpy() != np.array(y_pred).reshape(-1))
#                             y_true_ls.extend(y.view(-1).tolist())
#                             y_pred_ls.extend(y_pred)
#                             y_pred_prob_ls.extend(y_pred_prob)
#                         else:
#                             y_true_ls.append(y.numpy())
#                             y_pred_ls.extend(y_pred)
#                             y_pred_prob_ls.extend(y_pred_prob)
#                         break
#                     else:
#                         program = next_program
#                         program_str = next_program_str
#                         outbound_mask_ls = next_outbound_mask_ls
#                         all_other_pats_ls = next_all_other_pats_ls
#                 if not self.multilabel:
#                     y_true_array = np.array(y_true_ls, dtype=float)
#                     y_pred_array = np.array(y_pred_ls, dtype=float)
#                     y_pred_prob_array = np.array(y_pred_prob_ls, dtype=float)
#                     # y_pred_prob_array = np.concatenate(y_pred_prob_ls, axis = 0)
#                     y_pred_array[y_pred_array < 0] = 0.5
#                     y_pred_prob_array[y_pred_prob_array < 0] = 0.5
#                     if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#                         auc_score_2 = 0
#                     else:
#                         auc_score_2 = roc_auc_score(y_true_array.reshape(-1), y_pred_prob_array.reshape(-1))
#                     success_rate = (success / len(y_pred_array)) * 100.00
#                     avg_loss = sum_loss/len(y_pred_array)
#                     desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{len(y_pred_array)} ({success_rate:.2f}%), auc score:{auc_score_2}"
#                     iterator.set_description(desc)
#                 else:
#                     y_true_array = np.concatenate(y_true_ls)
#                     y_pred_array = np.stack(y_pred_ls)
#                     y_pred_prob_array = np.stack(y_pred_prob_ls)
#                     # y_pred_prob_array = np.concatenate(y_pred_prob_ls, axis = 0)
#                     # y_pred_array[y_pred_array < 0] = 0.5
#                     y_pred_prob_array[y_pred_prob_array < 0] = 0.5
#                     # if np.sum(y_pred_array == 1) <= 0 or np.sum(y_true_array == 1) <= 0:
#                     #     auc_score_2 = 0
#                     # else:
#                     selected_label_ids = (np.mean(y_true_array, axis=0) > 0)
#                     try:
#                         auc_score_2 = roc_auc_score(y_true_array[:,selected_label_ids], y_pred_prob_array[:,selected_label_ids], average=None)
#                     except ValueError:
#                         auc_score_2 = np.zeros(y_true_array[selected_label_ids].shape[-1])
#                     # success_rate = (success / len(y_pred_array)) * 100.00
#                     success_rate = np.mean(y_true_array == y_pred_array)*100
#                     avg_loss = sum_loss/len(y_pred_array)
#                     # desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: ({success_rate:.2f}%), auc score list:{auc_score_2.tolist()}, auc score mean:{np.mean(auc_score_2)}"
#                     desc = f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Success: ({success_rate:.2f}%), auc score mean:{np.mean(auc_score_2)}"
#                     iterator.set_description(desc)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = desc)
        

#         additional_score_str = ""
#         full_y_pred_prob_array = np.stack([1 - y_pred_prob_array.reshape(-1), y_pred_prob_array.reshape(-1)], axis=1)
#         for metric_name in metrics_maps:
#             curr_score = metrics_maps[metric_name](y_true_array.reshape(-1),full_y_pred_prob_array)
#             additional_score_str += metric_name + ": " + str(curr_score) + " "
#         print(additional_score_str)
#         if self.is_log:
#             self.logger.log(level=logging.DEBUG, msg = additional_score_str)
#         # Print information
        
#         # if exp_y_pred_arr is not None:
#         #     nonzero_ids = np.nonzero(exp_y_pred_arr != y_pred_array)
#         #     print(nonzero_ids[0])
        
#         if self.rl_algorithm == "dqn":
#             self.dqn.policy_net.train()
#             self.dqn.target_net.train()
#         else:
#             self.dqn.actor.train()
#             self.dqn.critic.train()

#         if self.is_log:
#             full_generated_program_ls, full_program_str_ls, full_label_ls = self.concatenate_program_across_samples(generated_program_ls, generated_program_str_ls, program_label_ls)
#             self.cluster_programs(full_generated_program_ls, full_program_str_ls,full_label_ls)

#     def run(self):
#         # exp_pred_array = self.test_epoch(0)
#         train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
#         if self.valid_dataset is not None:
#             valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
#         test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
#         # if self.valid_dataset is not None:
#         # if self.rl_algorithm == "dqn":
#         # self.test_epoch_ls(test_loader, 0, feat_embedding=self.test_feat_embeddings)
#         # train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
#         # with torch.autograd.set_detect_anomaly(True):
#         for i in range(1, self.epochs + 1):
#             if self.rl_algorithm == "dqn":
#                 self.train_epoch(i, train_loader)
#             else:
#                 self.train_epoch_ppo(i, train_loader)
#             if self.rl_algorithm == "dqn":
#                 torch.save(self.dqn.policy_net.state_dict(), os.path.join(self.work_dir, "policy_net_" + str(i)))
#                 torch.save(self.dqn.target_net.state_dict(), os.path.join(self.work_dir, "target_net_" + str(i)))
#                 torch.save(self.dqn.memory, os.path.join(self.work_dir, "memory"))
#             else:
#                 torch.save(self.dqn.actor.state_dict(), os.path.join(self.work_dir, "actor_" + str(i)))
#                 torch.save(self.dqn.critic.state_dict(), os.path.join(self.work_dir, "critic_" + str(i)))
#             # self.test_epoch(i)
#             if self.valid_dataset is not None:
#                 self.test_epoch_ls(valid_loader, i, feat_embedding=self.valid_feat_embeddings)    
#             self.test_epoch_ls(test_loader, i, feat_embedding=self.test_feat_embeddings)
#             torch.cuda.empty_cache() 

#             # self.test_epoch_ls(test_loader, i)

