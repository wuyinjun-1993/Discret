import torch
import numpy as np
import random
import os,sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from parse_args import *
from classification_task.datasets.EHR_datasets import read_data, EHRDataset, create_train_val_test_datasets
from rl_enc_dec.ehr_lang import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from miracle_local.impute import impute_train_eval_all

from feat_encoder.ft_transformer0 import FTTransformer, Transformer_wrapper
# from classification_task.datasets.EHR_datasets import obtain_feat_range_mappings, obtain_numeric_categorical_value_count , synthetic_lang_mappings, pred_mortality_feat
from tqdm import tqdm

from sklearn.metrics import recall_score, f1_score, roc_auc_score
from utils_mortality.metrics import print_additional_metrics
from rl_enc_dec.ehr_lang import *
import rl_enc_dec.synthetic_lang as synthetic_lang
from create_language import *
import shap
import json
import classification_task.full_experiments.rule_lang as rule_lang



def obtain_embeddings_over_data(args, net, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False)
    # evaluate_net(args, net, train_loader)
    feature_embedding_ls = []
    net.eval()
    with torch.no_grad():
        for val in tqdm(train_loader):
            (all_other_pats_ls, X_pd_ls, X, _, (X_num, X_cat),_), y = val
            X = X.to(args.device)
            y = y.to(args.device)
            X_num = X_num.to(args.device)
            X_cat = X_cat.to(args.device)
            feature_embedding = net(X_cat, X_num, return_embedding=True)
            
            feature_embedding_ls.append(feature_embedding)

    feature_embedding_mat = torch.cat(feature_embedding_ls)

    # torch.save(feature_embedding_mat, os.path.join(args.work_dir, "feat_embedding"))
    return feature_embedding_mat



def construct_model(model_config, lang, catergory_count_map={}):
    numeric_count  = len(lang.syntax["num_feat"]) if "num_feat" in lang.syntax else 0
    # category_count = len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
    # numeric_count, category_count = obtain_numeric_categorical_value_count(lang)
    transformer_net = FTTransformer(
                categories = catergory_count_map,      # tuple containing the number of unique values within each category
                num_continuous = numeric_count,                # number of continuous values
                dim = model_config["tf_latent_size"],                           # dimension, paper set at 32
                dim_out = 1,                        # binary prediction, but could be anything
                depth = model_config["depth"],                          # depth, paper recommended 6
                heads = model_config["heads"],                          # heads, paper recommends 8
                attn_dropout = model_config["attn_dropout"],                 # post-attention dropout
                ff_dropout = model_config["ff_dropout"]                  # feed forward dropout
            )
    
    return transformer_net

def get_dataloader(train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=True, drop_last=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn = EHRDataset.collate_fn, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader


def evaluate_net(args, net, test_loader):
    net.eval()

    pred_correct = 0
    total = 0

    with torch.no_grad():
        y_pred_ls = []
        y_ls = []
        y_logit_ls = []
        for val in tqdm(test_loader):
            # (all_other_pats_ls, X_pd_ls, X, _, (X_num, X_cat), _), y = val
            (all_other_pats_ls, _, X_pd_ls, X, _, _, (X_num, X_cat),_), y = val
            X = X.to(args.device)
            y = y.to(args.device)
            X_num = X_num.to(args.device)
            X_cat = X_cat.to(args.device)
            feature_embedding = net(X_cat, X_num, return_embedding=True)
            y_logits = torch.sigmoid(net.to_logits(feature_embedding))
            y_pred = (y_logits > 0.5).type(torch.long).view(-1)
            pred_correct += torch.sum(y.view(-1) == y_pred)
            total += y.shape[0]
            y_ls.append(y.view(-1))
            y_pred_ls.append(y_pred.view(-1))
            y_logit_ls.append(y_logits.view(-1))

        y_pred_array = torch.cat(y_pred_ls)
        y_array = torch.cat(y_ls)
        y_logit_ls_array = torch.cat(y_logit_ls)
           
        auc_score = roc_auc_score(y_array.cpu().numpy(), y_logit_ls_array.cpu().numpy())

        accuracy = pred_correct/total*1.0

        print("Test accuracy::", accuracy)

        print("auc score::", auc_score)

        print_additional_metrics(y_logit_ls_array.cpu().numpy(), y_array.cpu().numpy())


    net.train()
    
    return accuracy.cpu().item(), auc_score

def evalute_shap(args, net, test_loader, tree_depth=2):
    wrapped_net = Transformer_wrapper(net, len(test_loader.dataset.num_cols))
    e = shap.DeepExplainer(wrapped_net, shap.utils.sample(test_loader.dataset.features, 100).to(args.device)) 
    
    all_shap_values = {}
    for sample_id in tqdm(range(len(test_loader.dataset))):
        #     shap_values = torch.from_numpy(e.shap_values(feature_tensor[sample_id].view(1,-1)))
        # for sample_id in tqdm(range(feature_tensor.shape[0])):
            (all_other_pats, appts, X, idx, (X_num, X_cat), _, num_cols, cat_cols, feat_onehot_mat_mappings), y = test_loader.dataset[sample_id]
            all_cols = num_cols + cat_cols
            X_num_cat = torch.cat([X_num, X_cat], dim=1)
            shap_val = e.shap_values(X_num_cat.view(1,-1))
            if type(shap_val) is list:
            
                shap_values = torch.from_numpy(shap_val[0])
            else:
                shap_values = torch.from_numpy(shap_val)
            topk_s_val, topk_ids= torch.topk(shap_values.view(-1), k=tree_depth)
                        
            # selected_col_ids = {int(topk_ids[idx]):topk_s_val[idx] for idx in range(len(topk_ids))}
            selected_col_ids = [int(topk_ids[idx]) for idx in range(len(topk_ids))]
            selected_col_names = [all_cols[idx] for idx in selected_col_ids]
            selected_col_vals = []
            for col_id in range(len(selected_col_ids)):
                col_name = selected_col_names[col_id]
                if col_id < len(num_cols):
                    min, max = test_loader.dataset.feat_range_mappings[col_name]
                    col_val = X_num[0][col_id]*(max-min)+min
                    selected_col_vals.append(col_val.item())
                else:
                    selected_col_vals.append(list(appts[col_name])[0])
            
            if y == 1:
                all_shap_values[sample_id] = {"features": selected_col_names, "label": y.item(), "feature_vals": selected_col_vals}
    with open(os.path.join(args.log_folder, "transformer_explanation.json"), "w") as f:
        json.dump(all_shap_values, f, indent=4)
                        
        exit(1)
            
    


def pretrain_supervised(args, net, train_loader, valid_loader, test_loader, optimizer, criterion):
    
    # iterator = tqdm(enumerate(train_loader), desc="Training Synthesizer", total=len(train_loader))
    # val_accuracy, val_auc_score = evaluate_net(args, net, valid_loader)
    # test_accuracy, test_auc_score = evaluate_net(args, net, test_loader)
    # best_val_auc_score = val_auc_score
    # best_val_accuracy = val_accuracy
    # best_test_auc_score = test_auc_score
    # best_test_accuracy = test_accuracy
    
    best_val_auc_score = -1
    best_val_accuracy = -1
    best_test_auc_score = -1
    best_test_accuracy = -1
    val_accuracy, val_auc_score = evaluate_net(args, net, valid_loader)
    for e in range(args.epochs):
        print("start training at epoch " + str(e))
        avg_loss = 0
        count = 0
        for val in tqdm(train_loader):
            # (all_other_pats2, appts2, X, idx, appts, all_other_pats, (X_num, X_cat), (abnormal_feature_indicator, activated_indicator)), y
            (all_other_pats_ls, _, X_pd_ls, X, _, _, (X_num, X_cat), _), y = val
            # (origin_all_other_pats_ls, origin_all_other_pats_ls2, X_pd_ls2, X, X_sample_ids, X_pd_ls, (X_num, X_cat)), y = val
            X = X.to(args.device)
            y = y.to(args.device)
            X_num = X_num.to(args.device)
            X_cat = X_cat.to(args.device)
            feature_embedding = net(X_cat, X_num, return_embedding=True)
            y_logits = torch.sigmoid(net.to_logits(feature_embedding))

            loss = criterion(y_logits, y)
            count += len(y)
            avg_loss += loss.item()*len(y)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        torch.save(net.state_dict(), os.path.join(args.work_dir, "pretrain_net_" + str(e)))
        avg_loss /= count
        print("epoch %d, avg loss %f"%(e, avg_loss))

        val_accuracy, val_auc_score = evaluate_net(args, net, valid_loader)
        test_accuracy, test_auc_score = evaluate_net(args, net, test_loader)
        if val_auc_score > best_val_auc_score and val_accuracy > 0.8:
            best_val_accuracy = val_accuracy
            best_test_accuracy = test_accuracy
            best_val_auc_score = val_auc_score
            best_test_auc_score = test_auc_score
        print("val accuracy:%f, val auc score:%f, test accuracy:%f, test auc score:%f"%(val_accuracy, val_auc_score, test_accuracy, test_auc_score))
        print("best val accuracy:%f, best val auc score:%f, best test accuracy:%f, best test auc score:%f"%(best_val_accuracy, best_val_auc_score, best_test_accuracy, best_test_auc_score))


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    work_dir = os.path.join(args.log_folder, "pretrain/")
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir

    rl_config, model_config = load_configs(args)

    train_data, valid_data, test_data, label_column, id_column = read_data(args.data_folder, dataset_name=args.dataset_name)
    args.label_cln = label_column
    args.id_cln = id_column

    # col_str = None
    # for col in train_data.columns:
    #     if col_str is None:
    #         col_str = col
    #     else:
    #         col_str +=",  " + col
    # print(col_str)
    program_max_len = args.program_max_len
    print("program max len::", program_max_len)
    # train_dataset = EHRDataset(data= train_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    # train_valid_dataset = EHRDataset(data= pd.concat([train_data, valid_data]), drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=True)
    # valid_dataset = EHRDataset(data = valid_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)
    # test_dataset = EHRDataset(data = test_data, drop_cols=DROP_FEATS, patient_max_appts = patient_max_appts, balance=False)

    # feat_range_mappings = obtain_feat_range_mappings(train_dataset)   

    # train_dataset.rescale_data(feat_range_mappings) 
    # valid_dataset.rescale_data(feat_range_mappings) 
    # test_dataset.rescale_data(feat_range_mappings) 
    lang = Language(data=train_data, id_attr=args.id_cln, outcome_attr=args.label_cln, treatment_attr=None, text_attr=None, precomputed=None, lang=rule_lang)
    train_dataset, train_valid_dataset, valid_dataset, test_dataset, feat_range_mappings = create_train_val_test_datasets(train_data, valid_data, test_data, rule_lang, args)
    dataset_folder = os.path.join(args.data_folder, args.dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    train_data_file_path  = os.path.join(dataset_folder, "train_transformed_feat")
    valid_data_file_path  = os.path.join(dataset_folder, "valid_transformed_feat")
    test_data_file_path  = os.path.join(dataset_folder, "test_transformed_feat")
    
    train_feat_file_path  = os.path.join(dataset_folder, "train_feat")
    valid_feat_file_path  = os.path.join(dataset_folder, "valid_feat")
    test_feat_file_path  = os.path.join(dataset_folder, "test_feat")
    if not os.path.exists(train_data_file_path) or not os.path.exists(valid_data_file_path) or not os.path.exists(test_data_file_path) or not os.path.exists(train_feat_file_path) or not os.path.exists(valid_feat_file_path) or not os.path.exists(test_feat_file_path):
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
    
    unique_thres_count = rl_config["discretize_feat_value_count"]
    
    train_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)
    valid_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)
    test_dataset.set_abnormal_feature_vals(None, None, unique_thres_count)



    transformer_net = construct_model(model_config, lang, train_dataset.cat_unique_val_count_ls)
    transformer_net = transformer_net.to(args.device)

    train_loader, valid_loader, test_loader = get_dataloader(train_dataset, valid_dataset, test_dataset)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(transformer_net.parameters(), lr=args.learning_rate)
    if args.cached_model_name is not None:
        transformer_net.load_state_dict(torch.load(args.cached_model_name))
        # evalute_shap(args, transformer_net, test_loader)
    # transformer_net.load_state_dict(torch.load("/data6/wuyinjun/cancer_data_four/logs2/pretrain/pretrain_net_299"))
    pretrain_supervised(args, transformer_net, train_loader, valid_loader, test_loader, optimizer, criterion)