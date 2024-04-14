from collections import namedtuple, deque
import random
import torch
import os, sys
from torch import nn, optim

from functools import reduce


sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from enc_dec import *
from enc_dec_medical import RLSynthesizerNetwork_mlp0, RLSynthesizerNetwork_transformer0

class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)



class DQN_all:
    def __init__(self, lang, args, rl_config, feat_range_mappings, numeric_count=None, category_count=None, category_sum_count = None, has_embeddings=False, model_config=None, feat_group_names = None, removed_feat_ls=None):
        self.mem_sample_size = rl_config["mem_sample_size"]
        self.batch_size = args.batch_size
        self.gamma = rl_config["gamma"]
        self.lang = lang
        torch.manual_seed(args.seed)
        self.topk_act = args.topk_act
        torch.manual_seed(args.seed)
        # self.do_medical = args.do_medical
        # if not args.do_medical:
        if args.model == "mlp":
            self.policy_net = RLSynthesizerNetwork_mlp(args=args, lang=lang, model_config=model_config, rl_config=rl_config, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls)
        else:
            self.policy_net = RLSynthesizerNetwork_transformer(args=args, lang=lang, model_config=model_config, rl_config=rl_config, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls)
        
        if args.model == "mlp":
            self.target_net = RLSynthesizerNetwork_mlp(args=args, lang=lang, model_config=model_config, rl_config=rl_config, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls)
            # self.target_net = RLSynthesizerNetwork_mlp(lang=lang, program_max_len=args.program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=args.topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= args.prefer_smaller_range, prefer_smaller_range_coeff = args.prefer_smaller_range_coeff, args = args, discretize_feat_value_count=rl_config["discretize_feat_value_count"])
        else:
            self.target_net = RLSynthesizerNetwork_transformer(args=args, lang=lang, model_config=model_config, rl_config=rl_config, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls)
            # self.target_net = RLSynthesizerNetwork_transformer(lang=lang, program_max_len=args.program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=args.topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range=args.prefer_smaller_range, prefer_smaller_range_coeff=args.prefer_smaller_range_coeff, method_two=args.method_two, args = args, discretize_feat_value_count=rl_config["discretize_feat_value_count"])
        # else:
        #     if model == "mlp":
        #         self.policy_net = RLSynthesizerNetwork_mlp0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)
        #     else:
        #         self.policy_net = RLSynthesizerNetwork_transformer0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, category_sum_count=category_sum_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, method_two=method_two, args = args, discretize_feat_value_count=discretize_feat_value_count)
            
        #     if model == "mlp":
        #         self.target_net = RLSynthesizerNetwork_mlp0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args, discretize_feat_value_count=discretize_feat_value_count)
        #     else:
        #         self.target_net = RLSynthesizerNetwork_transformer0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, category_sum_count=category_sum_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range=prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, method_two=method_two, args = args, discretize_feat_value_count=discretize_feat_value_count)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.memory = ReplayMemory(rl_config["replay_memory_capacity"])

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), args.learning_rate)

        self.first_prog_embed = torch.tensor([0]*self.policy_net.ATOM_VEC_LENGTH, device=DEVICE, dtype=torch.float)#torch.randn(self.policy_net.ATOM_VEC_LENGTH, requires_grad=True)

    #turns atom into one-hot encoding
    def atom_to_vector(self, atom:dict):
        return self.policy_net.atom_to_vector(atom)

    def atom_to_vector_ls(self, atom:dict):
        return self.policy_net.atom_to_vector_ls(atom)

    def atom_to_vector_ls0(self, atom):
        return self.policy_net.atom_to_vector_ls0(atom)

    def vector_ls_to_str_ls0(self, atom):
        return self.policy_net.vector_ls_to_str0(atom)

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
    
    def predict_atom_ls(self, features, X_pd_ls, program, outbound_mask_ls, epsilon, X_pd_ls2, abnormal_info):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)
            pred = self.policy_net.forward_ls0(features, X_pd_ls, [init_program], outbound_mask_ls, ["formula"], epsilon, init=True, X_pd_full2=X_pd_ls2, abnormal_info=abnormal_info)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features, X_pd_ls, program, outbound_mask_ls, ["formula"], epsilon, X_pd_full2=X_pd_ls2, abnormal_info=abnormal_info)
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
        program, outbound_mask_ls = state
        
        
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            pred = self.target_net.forward_ls0(features, data, [init_program], outbound_mask_ls, ["formula"], 0, eval=False, init=True, X_pd_full2=data)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls0(features, data, program, outbound_mask_ls, ["formula"], 0, eval=False, X_pd_full2=data)
            
        max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)
        selected_num_feat_tensor_bool = pred[select_num_feat_key]
        if op_Q_key in pred:

            max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors + max_op_tensors

            max_tensors = max_tensors/3
        
        else:
            # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors

            max_tensors = max_tensors/2
            
        max_tensors = max_tensors*selected_num_feat_tensor_bool + max_col_tensors*(1-selected_num_feat_tensor_bool)
        # max_tensors = torch.mean(max_tensors, dim=-1)
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
    
    def predict_next_state_with_tensor_info_ls0_medical(self, features, data, state):
        program = state
        
        if len(state) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            pred = self.target_net.forward_ls0(features, data, [init_program], ["formula"], 0, eval=False, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls0(features, data, program, ["formula"], 0, eval=False)
            
        # max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)

        # if op_Q_key in pred:

        #     max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

        #     # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

        #     max_tensors += max_col_tensors + max_op_tensors

        #     max_tensors = max_tensors/3
        
        # else:
        #     # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

        #     # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

        #     max_tensors += max_col_tensors

        #     max_tensors = max_tensors/2

        return max_col_tensors.to(DEVICE)

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

    # def get_state_action_prediction_tensors_ls(self, features, X_pd, program, atom_pair):
    #     atom = atom_pair[0]
    #     origin_atom = atom_pair[1]
    #     queue = list(atom.keys())
    #     if len(program) == 0:
    #         init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
    #         pred = self.policy_net.forward_ls(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
    #         del init_program
    #     else:
    #         #program.sort()
    #         pred = self.policy_net.forward_ls(features, X_pd, program, queue, 0, eval=True, replay=True, existing_atom=origin_atom)

    #     tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
    #     atom_prediction_tensors = {}
    #     for token, token_val in atom.items():
    #         # if token == "num_op" or token.endswith("_prob"):
    #         #     continue

    #         # if token not in self.lang.syntax["num_feat"]:
    #         if not type(token) is tuple:
    #             # if not token.endswith("_prob"):
    #                 # tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
    #                 if not type(token_val) is dict:
    #                     tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val]
    #                     val = pred[token][:,tensor_idx].reshape((len(X_pd),1))
    #                     atom_prediction_tensors[token] = val
    #                     del val
    #                 else:
    #                     atom_prediction_tensors[token] = torch.zeros(len(X_pd), device = DEVICE)
    #                     for token_val_key in token_val:
    #                         token_val_sample_ids = token_val[token_val_key]
    #                         tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val_key]
    #                         val = pred[token][token_val_sample_ids,tensor_idx]
    #                         atom_prediction_tensors[token][token_val_sample_ids] = val
    #                         del val
                        
    #         else:
    #             if not "pred_score" in atom_prediction_tensors:
    #                 atom_prediction_tensors["pred_score"] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
    #             # atom_prediction_tensors[token] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
    #             pred_val = pred[token]
    #             for token_key in token:
                    
    #                 # token_key = token_key[0]
    #                 # lb_probs = pred_val[1][token_key][0]
    #                 probs = pred_val[1][token_key]
    #                 sample_ids = token_val[2][token_key].view(-1)
    #                 sample_cln_id_ls = token_val[3][token_key]
    #                 val = probs[sample_ids.view(-1), sample_cln_id_ls.view(-1)]
    #                 if token_key.endswith("_lb"):
    #                     atom_prediction_tensors["pred_score"][0][sample_ids] = val
    #                 elif token_key.endswith("_ub"):
    #                     atom_prediction_tensors["pred_score"][1][sample_ids] = val
    #                 del val
    #                 # val = ub_probs[sample_ids.view(-1), sample_cln_id_ls[1].view(-1)]
    #                 # atom_prediction_tensors[token][1][sample_ids] = val
    #                 # del val


    #             # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
    #         # else:
    #         #     tensor_indeces[token] = 0
        
    #     # for token, tensor_idx in tensor_indeces.items():
    #     #     if token not in self.lang.syntax["num_feat"]:
    #     #         atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
    #     #     else:
    #     #         atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
    #     # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
    #     del pred
    #     return atom_prediction_tensors
    
    def get_state_action_prediction_tensors_ls0(self, features, X_pd, state, atom):
        # atom = atom_pair[0]
        # origin_atom = atom_pair[1]
        queue = list(atom.keys())
        
        program, outbound_mask_ls = state
        
        # if atom[col_id_key].max() == 116:
        #     print()
        
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
            # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            pred = self.policy_net.forward_ls0(features,X_pd, [init_program], outbound_mask_ls, atom, 0, eval=True, init=True, X_pd_full2=X_pd)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features,X_pd, program, outbound_mask_ls, atom, 0, eval=True, X_pd_full2=X_pd)
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
        
        selected_num_feat_tensor_bool = atom[select_num_feat_key].to(atom_prediction_tensors.device)
        
        if op_Q_key in atom:
            op_tensor_indices = atom[op_Q_key].argmax(-1)

            op_prediction_Q_tensor_ls = []
            for k in range(op_tensor_indices.shape[-1]):
                op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
            op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
            op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

            assert torch.sum(atom_prediction_tensors**selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) + torch.sum(op_prediction_Q_tensor == min_Q_val) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3
        else:
            assert torch.sum(atom_prediction_tensors*selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3
            
        atom_prediction_tensors = atom_prediction_tensors*selected_num_feat_tensor_bool + col_prediction_Q_tensor*(1-selected_num_feat_tensor_bool)

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
    
    def get_state_action_prediction_tensors_ls0_medical(self, features, X_pd, state, atom):
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
        # tensor_indeces = atom[pred_Q_key].argmax(-1)
        
        x_idx = torch.tensor(list(range(len(X_pd))))
        
        # atom_prediction_tensors_ls = []
        # for k in range(tensor_indeces.shape[-1]):
        #     atom_prediction_tensors_ls.append(pred[pred_Q_key][x_idx, k, tensor_indeces[:,k]])
        # atom_prediction_tensors = torch.stack(atom_prediction_tensors_ls, dim=1) #atom_prediction_tensors/tensor_indeces.shape[-1]

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
        
        # if op_Q_key in atom:
        #     op_tensor_indices = atom[op_Q_key].argmax(-1)

        #     op_prediction_Q_tensor_ls = []
        #     for k in range(op_tensor_indices.shape[-1]):
        #         op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
        #     op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
        #     op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

        #     assert torch.sum(atom_prediction_tensors == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) + torch.sum(op_prediction_Q_tensor == min_Q_val) == 0

        #     atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3
        # else:
        #     assert torch.sum(atom_prediction_tensors == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

        #     atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3


        return col_prediction_Q_tensor
    
    
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
    
    # def optimize_model_ls(self):
    #     if len(self.memory) < self.mem_sample_size: return 0.0

    #     # Pull out a batch and its relevant features
    #     batch = self.memory.sample(self.mem_sample_size)
    #     non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
    #     non_final_samples = [transition for transition in batch if transition.next_program is not None]
    #     state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
    #     reward_batch = torch.stack([torch.from_numpy(transition.reward).view(-1) for transition in batch]).to(DEVICE)

    #     #get Q value for (s,a)
    #     state_action_pred = [(a,self.get_state_action_prediction_tensors_ls(f,d, p,a)) for f,d, p,a in state_action_batch]
    #     state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
    #     state_action_values = state_action_values.to(DEVICE)
        
    #     #get Q value for (s', max_a')
    #     next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
    #     next_state_values = torch.zeros([self.mem_sample_size, self.batch_size], dtype=torch.float, device=DEVICE)
    #     if len(next_state_pred_non_final) > 0:
    #         next_state_values_non_final = torch.stack([self.get_atom_Q_value_ls(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
    #         next_state_values[non_final_mask] = next_state_values_non_final
    #         del next_state_values_non_final
    #     next_state_values = next_state_values.to(DEVICE)
    #     # Prepare the loss function
    #     expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    #     # Compute the loss
    #     loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))
    #     self.optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
    #     self.optimizer.step()
        
    #     # for item in non_final_samples:
    #     #     del item
    #     # for  item in state_action_batch:
    #     #     del item
    #     # for item in state_action_pred:
    #     #     del item
    #     # for item in next_state_pred_non_final:
    #     #     del item
    #     # non_final_samples.clear()
    #     # state_action_pred.clear()
    #     # state_action_batch.clear()
    #     # non_final_samples.clear()
    #     # del state_action_values, expected_state_action_values, next_state_values, reward_batch, state_action_pred, next_state_pred_non_final, non_final_mask
    #     # del non_final_samples, batch, state_action_batch
    #     # for i in range(len(batch)):
    #     #     print(batch[i].data)
    #     #     print("program::")
    #     #     for pid in range(len(batch[i].program)):
    #     #         print(batch[i].program[pid])
    #     # print(batch[0].data)
    #     # print(batch[1].data)
    #     # print("loss::", loss)
    #     # print("expected_state_action_values::", expected_state_action_values)
    #     # print("next_state_values::", next_state_values)
    #     # print("reward_batch::", reward_batch)
    #     # print("state_action_values::", state_action_values)
    #     # Return loss
    #     return_loss = loss.detach()
    #     del loss
    #     return return_loss
    
    def optimize_model_ls0(self):
        if len(self.memory) < self.mem_sample_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.mem_sample_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.stack([torch.from_numpy(transition.reward) for transition in batch]).to(DEVICE)

        #get Q value for (s,a)
        # if not self.do_medical:
        state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0(f,d, p,a)) for f,d, p,a in state_action_batch]
        # else:
        #     state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0_medical(f,d, p,a)) for f,d, p,a in state_action_batch]
        # state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
        state_action_values = torch.stack([t for a,t in state_action_pred])
        state_action_values = state_action_values.to(DEVICE)
        
        #get Q value for (s', max_a')
        # if not self.do_medical:
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        # else:
            # next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0_medical(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
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


class DQN_all2:
    def __init__(self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, program_max_len, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0, numeric_count=None, category_count=None, category_sum_count = None, has_embeddings=False, model="mlp", topk_act=1, model_config=None, feat_group_names = None, removed_feat_ls=None, prefer_smaller_range = False, prefer_smaller_range_coeff=0.5, args = None):
        self.mem_sample_size = mem_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lang = lang
        torch.manual_seed(seed)
        self.topk_act = topk_act
        torch.manual_seed(seed)
        # self.do_medical = args.do_medical
        # if not args.do_medical:
        if model == "mlp":
            self.policy_net = RLSynthesizerNetwork_mlp2(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args)
        else:
            self.policy_net = RLSynthesizerNetwork_transformer(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args)
        
        if model == "mlp":
            self.target_net = RLSynthesizerNetwork_mlp2(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, num_feat_count=numeric_count, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args)
        else:
            self.target_net = RLSynthesizerNetwork_transformer(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range=prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args)
        # else:
        #     if model == "mlp":
        #         self.policy_net = RLSynthesizerNetwork_mlp0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, args = args)
        #     else:
        #         self.policy_net = RLSynthesizerNetwork_transformer0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, category_sum_count=category_sum_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range = prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, method_two=method_two, args = args)
            
        #     if model == "mlp":
        #         self.target_net = RLSynthesizerNetwork_mlp0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p = 0, category_sum_count=category_sum_count, feat_range_mappings=feat_range_mappings, topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range= prefer_smaller_range, prefer_smaller_range_coeff = prefer_smaller_range_coeff, args = args)
        #     else:
        #         self.target_net = RLSynthesizerNetwork_transformer0(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, category_sum_count=category_sum_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, feat_group_names=feat_group_names, removed_feat_ls=removed_feat_ls, prefer_smaller_range=prefer_smaller_range, prefer_smaller_range_coeff=prefer_smaller_range_coeff, method_two=method_two, args = args)


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

    def vector_ls_to_str_ls0(self, atom):
        return self.policy_net.vector_ls_to_str0(atom)

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
    
    def predict_atom_ls(self, features, X_pd_ls, program, outbound_mask_ls, epsilon):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)
            pred = self.policy_net.forward_ls0(features, X_pd_ls, [init_program], outbound_mask_ls, ["formula"], epsilon, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features, X_pd_ls, program, outbound_mask_ls, ["formula"], epsilon)
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
        program, outbound_mask_ls = state
        
        
        if len(program) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            pred = self.target_net.forward_ls0(features, data, [init_program], outbound_mask_ls, ["formula"], 0, eval=False, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls0(features, data, program, outbound_mask_ls, ["formula"], 0, eval=False)
            
        max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)
        selected_num_feat_tensor_bool = pred[select_num_feat_key]
        if op_Q_key in pred:

            max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors + max_op_tensors

            max_tensors = max_tensors/3
        
        else:
            # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

            # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

            max_tensors += max_col_tensors

            max_tensors = max_tensors/2
            
        max_tensors = max_tensors*selected_num_feat_tensor_bool + max_col_tensors*(1-selected_num_feat_tensor_bool)
        max_tensors = torch.mean(max_tensors, dim=-1)
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
    
    def predict_next_state_with_tensor_info_ls0_medical(self, features, data, state):
        program = state
        
        if len(state) == 0:
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(data), 1)
            pred = self.target_net.forward_ls0(features, data, [init_program], ["formula"], 0, eval=False, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.target_net.forward_ls0(features, data, program, ["formula"], 0, eval=False)
            
        # max_tensors,_ = pred[pred_Q_key].max(dim=-1)

        # max_tensors = torch.mean(max_tensors, dim=-1)

        max_col_tensors,_ = torch.topk(pred[col_Q_key].view(len(pred[col_Q_key]), -1), k = self.topk_act, dim=-1)#.max(dim=-1)

        # max_col_tensors  =torch.mean(max_col_tensors, dim=-1)

        # if op_Q_key in pred:

        #     max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

        #     # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

        #     max_tensors += max_col_tensors + max_op_tensors

        #     max_tensors = max_tensors/3
        
        # else:
        #     # max_op_tensors,_ = pred[op_Q_key].max(dim=-1)

        #     # max_op_tensors = torch.mean(max_op_tensors, dim=-1)

        #     max_tensors += max_col_tensors

        #     max_tensors = max_tensors/2

        return max_col_tensors.to(DEVICE)

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

    # def get_state_action_prediction_tensors_ls(self, features, X_pd, program, atom_pair):
    #     atom = atom_pair[0]
    #     origin_atom = atom_pair[1]
    #     queue = list(atom.keys())
    #     if len(program) == 0:
    #         init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
    #         pred = self.policy_net.forward_ls(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
    #         del init_program
    #     else:
    #         #program.sort()
    #         pred = self.policy_net.forward_ls(features, X_pd, program, queue, 0, eval=True, replay=True, existing_atom=origin_atom)

    #     tensor_indeces = {}#{token:self.policy_net.grammar_token_val_to_num[token][token_val] for token, token_val in atom.items()}
    #     atom_prediction_tensors = {}
    #     for token, token_val in atom.items():
    #         # if token == "num_op" or token.endswith("_prob"):
    #         #     continue

    #         # if token not in self.lang.syntax["num_feat"]:
    #         if not type(token) is tuple:
    #             # if not token.endswith("_prob"):
    #                 # tensor_indeces[token] = self.policy_net.grammar_token_val_to_num[token][token_val]
    #                 if not type(token_val) is dict:
    #                     tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val]
    #                     val = pred[token][:,tensor_idx].reshape((len(X_pd),1))
    #                     atom_prediction_tensors[token] = val
    #                     del val
    #                 else:
    #                     atom_prediction_tensors[token] = torch.zeros(len(X_pd), device = DEVICE)
    #                     for token_val_key in token_val:
    #                         token_val_sample_ids = token_val[token_val_key]
    #                         tensor_idx = self.policy_net.grammar_token_val_to_num[token][token_val_key]
    #                         val = pred[token][token_val_sample_ids,tensor_idx]
    #                         atom_prediction_tensors[token][token_val_sample_ids] = val
    #                         del val
                        
    #         else:
    #             if not "pred_score" in atom_prediction_tensors:
    #                 atom_prediction_tensors["pred_score"] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
    #             # atom_prediction_tensors[token] = [torch.zeros(len(X_pd), device = DEVICE), torch.zeros(len(X_pd), device = DEVICE)]
    #             pred_val = pred[token]
    #             for token_key in token:
                    
    #                 # token_key = token_key[0]
    #                 # lb_probs = pred_val[1][token_key][0]
    #                 probs = pred_val[1][token_key]
    #                 sample_ids = token_val[2][token_key].view(-1)
    #                 sample_cln_id_ls = token_val[3][token_key]
    #                 val = probs[sample_ids.view(-1), sample_cln_id_ls.view(-1)]
    #                 if token_key.endswith("_lb"):
    #                     atom_prediction_tensors["pred_score"][0][sample_ids] = val
    #                 elif token_key.endswith("_ub"):
    #                     atom_prediction_tensors["pred_score"][1][sample_ids] = val
    #                 del val
    #                 # val = ub_probs[sample_ids.view(-1), sample_cln_id_ls[1].view(-1)]
    #                 # atom_prediction_tensors[token][1][sample_ids] = val
    #                 # del val


    #             # tensor_indeces[token] = [torch.argmax(atom[token][1][0]).item(),torch.argmax(atom[token][1][1]).item()]
    #         # else:
    #         #     tensor_indeces[token] = 0
        
    #     # for token, tensor_idx in tensor_indeces.items():
    #     #     if token not in self.lang.syntax["num_feat"]:
    #     #         atom_prediction_tensors[token] = pred[token].view(-1)[tensor_idx].reshape((1,1))
    #     #     else:
    #     #         atom_prediction_tensors[token] = [pred[token][1][0][tensor_idx[0]].view(-1).reshape((1,1)),pred[token][1][1][tensor_idx[1]].view(-1).reshape((1,1))]#.view(-1).reshape((1,1))
            
    #     # {token:pred[token].view(-1)[tensor_idx].reshape((1,1)) for token, tensor_idx in tensor_indeces.items()}
    #     del pred
    #     return atom_prediction_tensors
    
    def get_state_action_prediction_tensors_ls0(self, features, X_pd, state, atom):
        # atom = atom_pair[0]
        # origin_atom = atom_pair[1]
        queue = list(atom.keys())
        
        program, outbound_mask_ls = state
        
        # if atom[col_id_key].max() == 116:
        #     print()
        
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd),1)
            init_program =self.first_prog_embed.unsqueeze(0).repeat(len(X_pd), 1)
            # pred = self.policy_net.forward_ls0(features, X_pd, [init_program], queue, 0, eval=True, replay=True, existing_atom=origin_atom)
            pred = self.policy_net.forward_ls0(features,X_pd, [init_program], outbound_mask_ls, atom, 0, eval=True, init=True)
            del init_program
        else:
            #program.sort()
            pred = self.policy_net.forward_ls0(features,X_pd, program, outbound_mask_ls, atom, 0, eval=True)
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
        
        selected_num_feat_tensor_bool = atom[select_num_feat_key]
        
        if op_Q_key in atom:
            op_tensor_indices = atom[op_Q_key].argmax(-1)

            op_prediction_Q_tensor_ls = []
            for k in range(op_tensor_indices.shape[-1]):
                op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
            op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
            op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

            assert torch.sum(atom_prediction_tensors**selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) + torch.sum(op_prediction_Q_tensor == min_Q_val) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3
        else:
            assert torch.sum(atom_prediction_tensors*selected_num_feat_tensor_bool == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

            atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3
            
        atom_prediction_tensors = atom_prediction_tensors*selected_num_feat_tensor_bool + col_prediction_Q_tensor*(1-selected_num_feat_tensor_bool)

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
        return torch.mean(atom_prediction_tensors, dim=-1)
    
    def get_state_action_prediction_tensors_ls0_medical(self, features, X_pd, state, atom):
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
        # tensor_indeces = atom[pred_Q_key].argmax(-1)
        
        x_idx = torch.tensor(list(range(len(X_pd))))
        
        # atom_prediction_tensors_ls = []
        # for k in range(tensor_indeces.shape[-1]):
        #     atom_prediction_tensors_ls.append(pred[pred_Q_key][x_idx, k, tensor_indeces[:,k]])
        # atom_prediction_tensors = torch.stack(atom_prediction_tensors_ls, dim=1) #atom_prediction_tensors/tensor_indeces.shape[-1]

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
        
        # if op_Q_key in atom:
        #     op_tensor_indices = atom[op_Q_key].argmax(-1)

        #     op_prediction_Q_tensor_ls = []
        #     for k in range(op_tensor_indices.shape[-1]):
        #         op_prediction_Q_tensor_ls.append(pred[op_Q_key][x_idx, k, op_tensor_indices[:,k]])
        #     op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
        #     op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

        #     assert torch.sum(atom_prediction_tensors == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) + torch.sum(op_prediction_Q_tensor == min_Q_val) == 0

        #     atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3
        # else:
        #     assert torch.sum(atom_prediction_tensors == min_Q_val) + torch.sum(col_prediction_Q_tensor == min_Q_val) == 0# + torch.sum(op_prediction_Q_tensor < -1) == 0

        #     atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor)/2# + op_prediction_Q_tensor)/3


        return col_prediction_Q_tensor
    
    
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
    
    # def optimize_model_ls(self):
    #     if len(self.memory) < self.mem_sample_size: return 0.0

    #     # Pull out a batch and its relevant features
    #     batch = self.memory.sample(self.mem_sample_size)
    #     non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
    #     non_final_samples = [transition for transition in batch if transition.next_program is not None]
    #     state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
    #     reward_batch = torch.stack([torch.from_numpy(transition.reward).view(-1) for transition in batch]).to(DEVICE)

    #     #get Q value for (s,a)
    #     state_action_pred = [(a,self.get_state_action_prediction_tensors_ls(f,d, p,a)) for f,d, p,a in state_action_batch]
    #     state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
    #     state_action_values = state_action_values.to(DEVICE)
        
    #     #get Q value for (s', max_a')
    #     next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
    #     next_state_values = torch.zeros([self.mem_sample_size, self.batch_size], dtype=torch.float, device=DEVICE)
    #     if len(next_state_pred_non_final) > 0:
    #         next_state_values_non_final = torch.stack([self.get_atom_Q_value_ls(atom, max_tensors) for atom, max_tensors in next_state_pred_non_final])
    #         next_state_values[non_final_mask] = next_state_values_non_final
    #         del next_state_values_non_final
    #     next_state_values = next_state_values.to(DEVICE)
    #     # Prepare the loss function
    #     expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    #     # Compute the loss
    #     loss = self.criterion(state_action_values.view(-1), expected_state_action_values.view(-1))
    #     self.optimizer.zero_grad()
    #     loss.backward(retain_graph=True)
    #     # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
    #     self.optimizer.step()
        
    #     # for item in non_final_samples:
    #     #     del item
    #     # for  item in state_action_batch:
    #     #     del item
    #     # for item in state_action_pred:
    #     #     del item
    #     # for item in next_state_pred_non_final:
    #     #     del item
    #     # non_final_samples.clear()
    #     # state_action_pred.clear()
    #     # state_action_batch.clear()
    #     # non_final_samples.clear()
    #     # del state_action_values, expected_state_action_values, next_state_values, reward_batch, state_action_pred, next_state_pred_non_final, non_final_mask
    #     # del non_final_samples, batch, state_action_batch
    #     # for i in range(len(batch)):
    #     #     print(batch[i].data)
    #     #     print("program::")
    #     #     for pid in range(len(batch[i].program)):
    #     #         print(batch[i].program[pid])
    #     # print(batch[0].data)
    #     # print(batch[1].data)
    #     # print("loss::", loss)
    #     # print("expected_state_action_values::", expected_state_action_values)
    #     # print("next_state_values::", next_state_values)
    #     # print("reward_batch::", reward_batch)
    #     # print("state_action_values::", state_action_values)
    #     # Return loss
    #     return_loss = loss.detach()
    #     del loss
    #     return return_loss
    
    def optimize_model_ls0(self):
        if len(self.memory) < self.mem_sample_size: return 0.0

        # Pull out a batch and its relevant features
        batch = self.memory.sample(self.mem_sample_size)
        non_final_mask = torch.tensor([transition.next_program is not None for transition in batch], dtype=torch.bool, device=DEVICE)
        non_final_samples = [transition for transition in batch if transition.next_program is not None]
        state_action_batch = [(transition.features, transition.data, transition.program, transition.action) for transition in batch]
        reward_batch = torch.stack([torch.from_numpy(transition.reward) for transition in batch]).to(DEVICE)

        #get Q value for (s,a)
        # if not self.do_medical:
        state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0(f,d, p,a)) for f,d, p,a in state_action_batch]
        # else:
        #     state_action_pred = [(a,self.get_state_action_prediction_tensors_ls0_medical(f,d, p,a)) for f,d, p,a in state_action_batch]
        # state_action_values = torch.stack([self.get_atom_Q_value_ls(a,t) for a,t in state_action_pred])
        state_action_values = torch.stack([t for a,t in state_action_pred])
        state_action_values = state_action_values.to(DEVICE)
        
        #get Q value for (s', max_a')
        # if not self.do_medical:
        next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        # else:
        #     next_state_pred_non_final = [self.predict_next_state_with_tensor_info_ls0_medical(sample.features, sample.data, sample.next_program) for sample in non_final_samples]
        next_state_values = torch.zeros([self.mem_sample_size, self.batch_size], dtype=torch.float, device=DEVICE)
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



class PPO_all:
    # self, lang, replay_memory_capacity, learning_rate, batch_size, gamma, provenance, program_max_len, latent_size, tf_latent_size, dropout_p, feat_range_mappings, mem_sample_size=1, seed=0, numeric_count=None, category_count=None, has_embeddings=False, model="mlp", pretrained_model_path=None, topk_act=1
    def __init__(self, lang, learning_rate, batch_size, gamma, program_max_len, dropout_p, n_updates_per_iteration, clip,feat_range_mappings, seed=0, numeric_count=None, category_count=None, has_embeddings=False, model="mlp", topk_act=1, continue_act = False, model_config=None, removed_feat_ls=None):
        self.batch_size = batch_size
        self.clip = clip
        self.gamma = gamma
        self.lang = lang
        self.n_updates_per_iteration = n_updates_per_iteration
        self.continue_act = continue_act
        torch.manual_seed(seed)
        self.removed_feat_ls = removed_feat_ls
        if model == "mlp":
            self.actor = RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], dropout_p=dropout_p, feat_range_mappings=feat_range_mappings, topk_act=topk_act, continue_act=continue_act)
            # self.actor = RLSynthesizerNetwork_mlp(lang=lang, program_max_len=program_max_len,latent_size=latent_size, feat_range_mappings=feat_range_mappings)
            self.critic = Critic_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], topk_act=topk_act)
        else:
            self.actor = RLSynthesizerNetwork_transformer(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], tf_latent_size=model_config["tf_latent_size"], dropout_p = 0, feat_range_mappings=feat_range_mappings, numeric_count=numeric_count, category_count=category_count, has_embeddings=has_embeddings, pretrained_model_path=model_config["pretrained_model_path"], topk_act=topk_act, continue_act = continue_act)
            # self.actor = Actor2(lang=lang, program_max_len=program_max_len,latent_size=latent_size, feat_range_mappings=feat_range_mappings)
            # lang,  program_max_len, latent_size, topk_act, has_embeddings, category_count, numeric_count, tf_latent_size, pretrained_model_path=None
            
            
            self.critic = Critic_transformer(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], topk_act=topk_act,has_embeddings = has_embeddings, category_count=category_count, numeric_count=numeric_count, tf_latent_size=model_config["tf_latent_size"],pretrained_model_path=model_config["pretrained_model_path"])
            # self.critic = Critic_mlp(lang=lang, program_max_len=program_max_len,latent_size=model_config["latent_size"], topk_act=topk_act)
    
        self.actor_optimizer = optim.Adam(self.actor.parameters(), learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), learning_rate)
        self.topk_act = topk_act
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
        batch_rtgs = torch.from_numpy(np.stack(batch_rtgs, axis=0)).type(torch.float).to(DEVICE)
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
        A_k = (A_k - A_k.mean(dim=(0,1))) / (A_k.std(dim=(0,1)) + 1e-10)
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

    def atom_to_vector_ls0(self, atom):
        return self.actor.atom_to_vector_ls0(atom)
    
    def vector_to_atom_vector_ls0(self, atom):
        return self.actor.vector_ls_to_str0(atom)

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

    def idx_to_logs(self, pred:dict, atom:dict):
        x_idx = torch.tensor(list(range(len(atom[pred_probs_key]))))

        if not self.continue_act:
            tensor_indeces = atom[pred_probs_key].argmax(-1)
            atom_prediction_tensors_ls = []
            for k in range(tensor_indeces.shape[-1]):
                atom_prediction_tensors_ls.append(pred[pred_probs_key][x_idx, k, tensor_indeces[:,k]])
            atom_prediction_tensors = torch.stack(atom_prediction_tensors_ls, dim=1) #atom_prediction_tensors/tensor_indeces.shape[-1]

        else:
            pred_probs = torch.clamp(pred[pred_probs_key], min=1e-6, max=1)
            pred_dist = torch.distributions.normal.Normal(pred_probs[:,:, 0], 1e-3)

            atom_prediction_tensors = torch.exp(pred_dist.log_prob(atom[pred_probs_key][:,:,0]) + np.log(1e-3*np.sqrt(2*torch.pi))) + 1e-4
            
            # atom_prediction_tensors = torch.exp(atom_prediction_tensors - 0.5)
        
        
        # col_tensor_indices = atom[col_Q_key].argmax(-1)
        # _,col_tensor_indices = torch.topk(atom[col_Q_key], k = self.topk_act, dim=-1)
        
        _,col_tensor_indices = torch.topk(atom[col_probs_key].view(len(atom[col_probs_key]),-1), k=self.topk_act, dim=-1)


        col_prediction_Q_tensor_ls = []
        
        for k in range(self.topk_act):
            col_prediction_Q_tensor_ls.append(pred[col_probs_key].view(len(pred[col_probs_key]), -1)[x_idx, col_tensor_indices[:,k]])
        
        col_prediction_Q_tensor = torch.stack(col_prediction_Q_tensor_ls, dim=1)
        # col_prediction_Q_tensor_ls = []
        # for k in range(col_tensor_indices.shape[-1]):
        #     col_prediction_Q_tensor_ls += pred[col_Q_key][x_idx, col_tensor_indices[:,k]]
        # col_prediction_Q_tensor = pred[col_Q_key][x_idx, col_tensor_indices]
        # col_prediction_Q_tensor = col_prediction_Q_tensor/col_tensor_indices.shape[-1]
        
        op_tensor_indices = atom[op_probs_key].argmax(-1)

        op_prediction_Q_tensor_ls = []
        for k in range(op_tensor_indices.shape[-1]):
            op_prediction_Q_tensor_ls.append(pred[op_probs_key][x_idx, k, op_tensor_indices[:,k]])
        op_prediction_Q_tensor = torch.stack(op_prediction_Q_tensor_ls, dim=1)
        # op_prediction_Q_tensor = op_prediction_Q_tensor/op_tensor_indices.shape[-1]

        assert torch.sum(atom_prediction_tensors < -1) + torch.sum(col_prediction_Q_tensor < -1) + torch.sum(op_prediction_Q_tensor < -1) == 0

        atom_prediction_tensors = atom_prediction_tensors*col_prediction_Q_tensor*op_prediction_Q_tensor
        
        # atom_prediction_tensors = (atom_prediction_tensors + col_prediction_Q_tensor + op_prediction_Q_tensor)/3


        # atom_log_probs = pred[torch.tensor(list(range(pred.shape[0]))), idx]
        
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
        return torch.log(atom_prediction_tensors)


    # def predict_atom(self, features, X_pd_ls, program, train,col, op):
    #     if len(program) == 0:
    #         program = [self.first_prog_embed]
    #     # features,X_pd_full, program, train, col, op, eval=False
    #     atom_preds = self.actor.forward(features,X_pd_ls, program, train = train, col = col, op = op)
    #     return atom_preds

    def predict_atom_ls(self, features, X_pd_ls, program, outbound_mask_ls, train):
        if len(program) == 0:
            # init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), self.topk_act, 1)
            init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_ls), 1)
            pred = self.actor.forward_ls0(features, X_pd_ls, [init_program],outbound_mask_ls, ["formula"], init=True, train=train, is_ppo=True)
            del init_program
        else:
            #program.sort()
            pred = self.actor.forward_ls0(features, X_pd_ls, program, outbound_mask_ls, ["formula"], train=train, is_ppo=True)
        # return self.policy_net.prediction_to_atom_ls(pred), pred
        return pred

    def evaluate(self, batch_obs, batch_acts):
        # V = torch.tensor([self.critic(f, p) for _, f,p in batch_obs], dtype = torch.float, requires_grad=True).squeeze()
        V = torch.stack([self.critic(f, p, init=(len(p) == 0)) for f,_,p in batch_obs])#.squeeze(-1)
        batch_eval_probs = []
        for obs, act in zip(batch_obs, batch_acts):
            X, X_pd_full, program = obs
            if len(program) == 0:
                init_program = self.first_prog_embed.unsqueeze(0).repeat(len(X_pd_full), 1)
                preds = self.actor.forward_ls0(X, X_pd_full, [init_program], act, train=False, eval=True, is_ppo=True, init=True)
            else:
                preds = self.actor.forward_ls0(X, X_pd_full, program, act, train=False, eval=True, is_ppo=True)
            atom_probs = self.idx_to_logs(preds, act)
            batch_eval_probs.append(atom_probs)
        return V, batch_eval_probs
    

    

