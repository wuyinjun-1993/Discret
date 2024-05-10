import pandas as pd
import os, sys
from parse_args import parse_args, load_configs


import random
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import pickle

import torch
import yaml

from create_language import *
from trainer import *
from miracle_local.MIRACLE import MIRACLE
from miracle_local.impute import impute_train_eval_all
# from rl_enc_dec.train_rl_synthesizer_2 import Trainer6_2,Trainer7_2,obtain_feat_range_mappings,obtain_numeric_categorical_value_count
from baselines.baseline_main import construct_feat_label_mat, remove_empty_classes
from baselines.dt import *
from baselines.gb import *
from baselines.rf import *
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import logging

# from rl_enc_dec.ehr_lang import *


from datetime import datetime
from full_experiments.pretrain_main import obtain_embeddings_over_data, construct_model

from classification_task.datasets.EHR_datasets import EHRDataset, read_data, create_train_val_test_datasets, obtain_numeric_categorical_value_count #, synthetic_lang_mappings
import classification_task.full_experiments.rule_lang as rule_lang


from utils_mortality.metrics import metrics_maps
from trainer import Trainer_all

from preprocess_abnormal import generate_abnormal_value_indicator_mat
from utils_mortality.evaluation_utils import evaluate_performance

def pre_compute_thresholds(cols, drop_cols, bin_size=20):
    col_thres_mappings = dict()
    for col in cols:
        if col in drop_cols:
            continue
        col_thres_mappings[col] = []
        for idx in range(bin_size):
            col_thres_mappings[col].append(idx/bin_size*1.0)

        col_thres_mappings[col].append(1)

    for col in col_thres_mappings:
        col_thres_mappings[col] = np.array(col_thres_mappings[col])

    return col_thres_mappings



def set_lang_data(lang, train_dataset):
    lang.features = train_dataset.features
    lang.transformed_features = train_dataset.transformed_features
    lang.labels = train_dataset.labels
    lang.data = train_dataset.data
    lang.dataset = train_dataset
    return lang
# TBD: 3. different samples can have varied explanation lengths


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    rl_config, model_config = load_configs(args)
    train_data, valid_data, test_data, label_cln, id_cln = read_data(args.data_folder, dataset_name=args.dataset_name)
    args.label_cln = label_cln
    args.id_cln = id_cln
    
    program_max_len = args.num_ands
    print("program max len::", program_max_len)
    
    dataset_folder = os.path.join(args.data_folder, args.dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    
    train_data_file_path  = os.path.join(dataset_folder, "train_transformed_feat")
    valid_data_file_path  = os.path.join(dataset_folder, "valid_transformed_feat")
    test_data_file_path  = os.path.join(dataset_folder, "test_transformed_feat")
    
    train_feat_file_path  = os.path.join(dataset_folder, "train_feat")
    valid_feat_file_path  = os.path.join(dataset_folder, "valid_feat")
    test_feat_file_path  = os.path.join(dataset_folder, "test_feat")
    # train_valid_data_file_path  = os.path.join(args.data_folder, "train_valid_dataset")
    train_dataset, train_valid_dataset, valid_dataset, test_dataset, feat_range_mappings = create_train_val_test_datasets(train_data, valid_data, test_data, rule_lang, args)
    
    # if not os.path.exists(train_data_file_path) or not os.path.exists(valid_data_file_path) or not os.path.exists(train_valid_data_file_path) or not os.path.exists(test_data_file_path):  
    if True: #not os.path.exists(train_data_file_path) or not os.path.exists(valid_data_file_path) or not os.path.exists(test_data_file_path) or not os.path.exists(train_feat_file_path) or not os.path.exists(valid_feat_file_path) or not os.path.exists(test_feat_file_path):
    
    
        
        
            
        if torch.any(torch.isnan(train_dataset.transformed_features)):
            train_dataset.transformed_features, valid_dataset.transformed_features, test_dataset.transformed_features = impute_train_eval_all(train_dataset.transformed_features, valid_dataset.transformed_features, test_dataset.transformed_features)
            # train_valid_dataset.transformed_features, miracle = impute_train_eval(train_valid_dataset.transformed_features.numpy(), miracle=miracle)
        torch.save(train_dataset.transformed_features, train_data_file_path)
        torch.save(train_dataset.features, train_feat_file_path)
        
        torch.save(valid_dataset.transformed_features, valid_data_file_path)
        torch.save(valid_dataset.features, valid_feat_file_path)
        
        torch.save(test_dataset.transformed_features, test_data_file_path)
        torch.save(test_dataset.features, test_feat_file_path)
        
    else:
        train_dataset.transformed_features = torch.load(train_data_file_path)
        train_dataset.features = torch.load(train_feat_file_path)
        
        valid_dataset.transformed_features = torch.load(valid_data_file_path)
        valid_dataset.features = torch.load(valid_feat_file_path)
        
        test_dataset.transformed_features = torch.load(test_data_file_path)
        test_dataset.features = torch.load(test_feat_file_path)

    train_valid_dataset.transformed_features = torch.cat([train_dataset.transformed_features, valid_dataset.transformed_features], dim=0)
    train_valid_dataset.features = torch.cat([train_dataset.features, valid_dataset.features], dim=0)
    train_valid_dataset.labels = torch.cat([train_dataset.labels, valid_dataset.labels], dim=0)
    train_dataset.create_imputed_data()
    valid_dataset.create_imputed_data()
    test_dataset.create_imputed_data()
    train_valid_dataset.create_imputed_data()

    if args.use_precomputed_thres:
        col_thres_mappings = pre_compute_thresholds(train_dataset.data.columns,train_dataset.drop_cols, bin_size=10)
    else:
        col_thres_mappings = None

    train_feat_embeddings = None
    
    valid_feat_embeddings = None
    
    test_feat_embeddings = None
    
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    pretrained_net = None

    work_dir = os.path.join(args.log_folder, args.method)
    os.makedirs(work_dir, exist_ok=True)

    # if args.pretrained_model_path is not None:
    lang = Language(data=train_data, id_attr=args.id_cln, outcome_attr=args.label_cln, treatment_attr=None, text_attr=None, precomputed=col_thres_mappings, lang=rule_lang, num_feats=train_dataset.num_cols)

    feat_bound_mappings = None

    # if args.do_medical:
    #     curr_dir = os.path.dirname(os.path.realpath(__file__))

    #     with open(os.path.join(curr_dir, "feat_range_mappings"), "rb") as f:
    #         feat_bound_mappings = pickle.load(f)
    args.feat_bound_mappings = feat_bound_mappings

    feat_bound_point_ls = None
    # if args.feat_bound_point_file_name is not None:
    # if not args.not_use_feat_bound_point_ls:
    #     curr_dir = os.path.dirname(os.path.realpath(__file__))

    #     with open(os.path.join(curr_dir, "feat_bound_point_mappings_" + args.dataset_name), "rb") as f:
    #         feat_bound_point_ls = pickle.load(f)
    args.feat_bound_point_ls = feat_bound_point_ls
    
    # if args.use_kg and feat_bound_point_ls is not None:
    #     normal_feat_range_file_name = os.path.join(args.data_folder, feat_range_file_name_mappings[args.dataset_name])
    #     normal_feat_range_df = pd.read_csv(normal_feat_range_file_name)
    #     generate_abnormal_value_indicator_mat(normal_feat_range_df, train_dataset, feat_range_mappings, feat_bound_point_ls)
    #     generate_abnormal_value_indicator_mat(normal_feat_range_df, valid_dataset, feat_range_mappings, feat_bound_point_ls)
    #     generate_abnormal_value_indicator_mat(normal_feat_range_df, test_dataset, feat_range_mappings, feat_bound_point_ls)
    # else:
    if feat_bound_point_ls is not None:
        unique_thres_count = len(feat_bound_point_ls[list(feat_bound_point_ls.keys())[0]])
    else:
        unique_thres_count = rl_config["discretize_feat_value_count"]
    
    train_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)
    valid_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)
    test_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)

        
    lang = set_lang_data(lang, train_dataset)
    if args.method == "ours":
        if "pretrained_model_path" in model_config and model_config["pretrained_model_path"] is not None:
            pretrained_net = construct_model(model_config, lang, train_dataset.cat_unique_val_count_ls)
            
            pretrained_net.load_state_dict(torch.load(model_config["pretrained_model_path"]))
            
            pretrained_net = pretrained_net.to(args.device)

            if "fix_pretrained_model" in model_config and model_config["fix_pretrained_model"]:
                
                # if os.path.exists(os.path.join(work_dir, "train_feat_embeddings")):
                #     train_feat_embeddings = torch.load(os.path.join(work_dir, "train_feat_embeddings"))    
                # else:        
                    train_feat_embeddings = obtain_embeddings_over_data(args, pretrained_net, train_dataset)
                    torch.save(train_feat_embeddings, os.path.join(work_dir, "train_feat_embeddings"))

                # if os.path.exists(os.path.join(work_dir, "valid_feat_embeddings")):            
                #     valid_feat_embeddings = torch.load(os.path.join(work_dir, "valid_feat_embeddings")) 
                # else:
                    valid_feat_embeddings = obtain_embeddings_over_data(args, pretrained_net, valid_dataset)
                    torch.save(valid_feat_embeddings, os.path.join(work_dir, "valid_feat_embeddings"))

                # if os.path.exists(os.path.join(work_dir, "test_feat_embeddings")):
                #     test_feat_embeddings = torch.load(os.path.join(work_dir, "test_feat_embeddings")) 
                # else:
                    test_feat_embeddings = obtain_embeddings_over_data(args, pretrained_net, test_dataset)
                    torch.save(test_feat_embeddings, os.path.join(work_dir, "test_feat_embeddings"))

        else:
            model_config["pretrained_model_path"] = None
            print("no pretrained model")

    

    if args.is_log:
        # log_path = os.path.join(work_dir, datetime.now().strftime("%d-%m-%YT%H:%M::%s") + '.txt')
        log_path = os.path.join(work_dir, args.log_file_name + '.txt')
        logging.basicConfig(filename=log_path,
                    filemode='w',
                    format='%(message)s',
                    level=logging.DEBUG)
        # logging.info("EHR Explanation Synthesis\n Seed: {}, train_path: {}, test_path: {}, precomputed_path: {}, mem_cap: {}, learning_rate: {}, batch_\
        # size: {}, gamma: {}, epsilon: {}, epsilon_falloff: {}, target_update: {}, epochs: {}, prog_max_len: {}, pat_max_appt: {}, latent_size: {}, dropout: {}".format(
        #     args.seed, work_dir,work_dir,work_dir,args.replay_memory_capacity,args.learning_rate,args.batch_size,args.gamma,args.epsilon,args.epsilon_falloff,args.target_update,args.epochs,
        #     program_max_len,patient_max_appts,args.latent_size, args.dropout_p))

    # if args.selected_col_ratio > 0:
    #     work_dir = os.path.join(work_dir, "col_ratio_" + str(args.selected_col_ratio))
    #     os.makedirs(work_dir, exist_ok=True)

    
    if args.method == "ours":
        numeric_count  = len(lang.syntax["num_feat"]) if "num_feat" in lang.syntax else 0
        # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
        # category_count = list(train_dataset.cat_unique_count_mappings.values()) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
        category_count = [train_dataset.cat_unique_count_mappings[key] for key in train_dataset.cat_cols]
        category_sum_count = train_dataset.cat_sum_count
        # numeric_count, category_count = obtain_numeric_categorical_value_count(train_dataset)
        trainer = Trainer_all(lang=lang, train_dataset=train_dataset, valid_dataset=valid_dataset,
                            test_dataset = test_dataset, train_feat_embeddings=train_feat_embeddings, valid_feat_embeddings=valid_feat_embeddings, test_feat_embeddings=test_feat_embeddings,
                            feat_range_mappings=feat_range_mappings, args=args, work_dir=work_dir, numeric_count=numeric_count, category_count=category_count, category_sum_count=category_sum_count,
                            model_config = model_config,
                            rl_config = rl_config, id_cln=args.id_cln, label_cln=args.label_cln
                            )
        # if args.backbone_model == ""
        # trainer.dqn.policy_net.input_embedding
        
        if args.is_log:
            if args.model_folder is not None:
                if args.rl_algorithm == "dqn":
                    trainer.dqn.policy_net.load_state_dict(torch.load(os.path.join(args.model_folder, "policy_net_" + str(args.model_suffix)), map_location="cpu"))
                    trainer.dqn.target_net.load_state_dict(torch.load(os.path.join(args.model_folder, "target_net_" + str(args.model_suffix)), map_location="cpu"))
                    
                    # trainer.dqn.memory = torch.load(os.path.join(args.model_folder, "memory_" + str(args.model_suffix)))
                else:
                    trainer.dqn.actor.load_state_dict(torch.load(os.path.join(args.model_folder, "actor_" + str(args.model_suffix)), map_location="cpu"))
                    trainer.dqn.critic.load_state_dict(torch.load(os.path.join(args.model_folder, "critic_" + str(args.model_suffix)), map_location="cpu"))
        else:
            if args.model.lower() == "transformer" and args.pretrained_model_path is not None:
                trainer.dqn.policy_net.input_embedding.load_state_dict(torch.load(args.pretrained_model_path,map_location="cpu"), strict=False)
        # (torch.tensor(feat_range_mappings["ALBUMIN..last"])[1] - torch.tensor(feat_range_mappings["ALBUMIN..last"])[0])*torch.tensor(feat_bound_point_ls["ALBUMIN..last"]) + torch.tensor(feat_range_mappings["ALBUMIN..last"])[0]
        trainer.run()
    elif args.method == "dt":
        multi_label=(len(torch.unique(train_dataset.labels)) > 2)
        train_feat, train_labels = construct_feat_label_mat(train_valid_dataset)
        valid_feat, valid_labels = construct_feat_label_mat(valid_dataset)
        test_feat, test_labels = construct_feat_label_mat(test_dataset)

        # gb_model = dt_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=1000, multi_label=multi_label)
        gb_model = dt_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=args.num_ands, multi_label=multi_label)

        gb_pred_labels, pred_prob_labels = dt_model_pred(test_feat, gb_model, multi_label=multi_label)

        evaluate_performance(pred_prob_labels, test_labels, multi_label=multi_label)
    
    elif args.method == "gb":
        multi_label=(len(torch.unique(train_dataset.labels)) > 2)
        train_feat, train_labels = construct_feat_label_mat(train_valid_dataset)
        valid_feat, valid_labels = construct_feat_label_mat(valid_dataset)
        test_feat, test_labels = construct_feat_label_mat(test_dataset)
        # if args.dataset_name == four:
        #     train_labels, valid_labels, test_labels = remove_empty_classes(train_labels, valid_labels, test_labels)
        
        gb_model = gb_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=1000, multi_label=multi_label)
        # gb_model = gb_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=args.num_ands, multi_label=multi_label)

        gb_pred_labels, pred_prob_labels = gb_model_pred(test_feat, gb_model, multi_label=multi_label)

        evaluate_performance(pred_prob_labels, test_labels, multi_label=multi_label)

    elif args.method == "rf":
        multi_label=(len(torch.unique(train_dataset.labels)) > 2)
        train_feat, train_labels = construct_feat_label_mat(train_valid_dataset)
        valid_feat, valid_labels = construct_feat_label_mat(valid_dataset)
        test_feat, test_labels = construct_feat_label_mat(test_dataset)
        # if args.dataset_name == four:
        #     train_labels, valid_labels, test_labels = remove_empty_classes(train_labels, valid_labels, test_labels)

        gb_model = rf_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=1000, multi_label=multi_label)
        # gb_model = rf_model_train(feat_mats=train_feat, label_mats=train_labels, feat_count=args.num_ands, multi_label=multi_label)

        gb_pred_labels, pred_prob_labels = rf_model_pred(test_feat, gb_model, multi_label=multi_label)

        evaluate_performance(pred_prob_labels, test_labels, multi_label=multi_label)
         
        # if args.dataset_name == one:
        #     with open(os.path.join(args.data_folder, "rf_model"), "wb") as f:
        #         pickle.dump(gb_model, f)
        #     var_clns = []
        #     var_clns.extend(list(train_valid_dataset.cat_cols))
        #     var_clns.extend(list(train_valid_dataset.num_cols))
        #     with open(os.path.join(args.data_folder, "columns_for_prediction"), "wb") as f:
        #         pickle.dump(var_clns, f)