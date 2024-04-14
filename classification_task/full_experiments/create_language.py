import pandas as pd
import operator as op
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch

class Language:
    def __init__(self, data, id_attr, outcome_attr, treatment_attr, text_attr, precomputed, lang, num_feats=None):
        self.lang = lang
        self.syntax = lang.LANG_SYNTAX
        self.dataset = data
        self.precomputed = precomputed
        if num_feats is None:
            num_feats = [col for col in self.dataset.columns if col not in lang.CAT_FEATS and col not in lang.DROP_FEATS and not col == id_attr and not col == outcome_attr and not col == treatment_attr and not col == text_attr] 
        if "num_feat" in self.syntax:
            num_feats.sort()
            for col in num_feats:
                self.syntax["num_feat"][col] = [col]
                # if precomputed is not None:
                self.syntax[col] = []#{i:[] for i in precomputed[col]}
        if "cat_feat" in self.syntax:
            lang.CAT_FEATS.sort()
            for col in lang.CAT_FEATS:
                self.syntax["cat_feat"][col] = [col]
                # if precomputed is not None:
                self.syntax[col] = []#{i:[] for i in self.dataset[col].unique()}
    
    #returns filtered dataset from given expression
    def evaluate_atom_on_dataset(self, expr: dict, data):
        assert "formula" in expr
        if expr["formula"] == "end":
            return data
        # no num_op
        assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
        op = expr["num_op" if "num_op" in expr else "cat_op"]
        feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
        #  this should a tuple
        target_const = expr[feat]
        patient_matches = data.loc[op(data[feat], target_const),"PAT_ID"].unique()
        return data[data['PAT_ID'].isin(patient_matches)]

    def evaluate_atom_ls_on_dataset(self, expr_ls, data):
        existing_data = data.copy()
        for expr in expr_ls:
            assert "formula" in expr
            if expr["formula"] == "end":
                return data
            assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            op = expr["num_op" if "num_op" in expr else "cat_op"]
            feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            target_const = expr[feat]
            patient_matches = existing_data.loc[op(existing_data[feat], target_const),"PAT_ID"].unique()
            existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
        return existing_data
    
    def evaluate_atom_ls_on_dataset2(self, expr_ls, data):
        existing_data = data.copy()
        res_data_ls = []
        for expr in expr_ls:
            assert "formula" in expr
            if expr["formula"] == "end":
                return data
            assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            op = expr["num_op" if "num_op" in expr else "cat_op"]
            feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            target_const = expr[feat]
            patient_matches = existing_data.loc[op(existing_data[feat], target_const),"PAT_ID"].unique()
            existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
            res_data_ls.append(existing_data.copy())
        return res_data_ls
    
    def evaluate_atom_ls_ls_on_dataset(self, expr_ls_ls, data_ls):

        existing_data_ls = []
        for idx in range(len(expr_ls_ls)):
            expr_ls = expr_ls_ls[idx]
            data = data_ls[idx]

            existing_data = data.copy()
            for expr in expr_ls:
                assert "formula" in expr
                if expr["formula"] == "end":
                    return data
                assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                op = expr["num_op" if "num_op" in expr else "cat_op"]
                feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
                target_const = expr[feat]
                patient_matches = existing_data.loc[op(existing_data[feat], target_const),"PAT_ID"].unique()
                existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
            
            existing_data_ls.append(existing_data)
        return existing_data_ls
    
    def evaluate_atom_ls_ls_on_dataset0(self, expr_ls, data_ls, col, op, pred_v_col):

        existing_data_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            data = data_ls[idx]

            existing_data = data.copy()
            # for expr in expr_ls:
                # assert "formula" in expr
                # if expr["formula"] == "end":
                #     return data
                # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                # op = expr["num_op" if "num_op" in expr else "cat_op"]
                # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
                # target_const = expr[feat]
            patient_matches = existing_data.loc[op(existing_data[col], target_const),"PAT_ID"].unique()
            existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
            
            existing_data_ls.append(existing_data)
        return existing_data_ls
    
    def evaluate_atom_ls_ls_on_dataset_full(self, expr_ls, data_ls, col_name_col, op_name_col, pred_v_col):

        existing_data_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        op_ls = expr_ls[op_name_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            data = data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            sub_op_ls = op_ls[idx]

            existing_data = data.copy()
            # for expr in expr_ls:
                # assert "formula" in expr
                # if expr["formula"] == "end":
                #     return data
                # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                # op = expr["num_op" if "num_op" in expr else "cat_op"]
                # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

                # target_const = expr[feat]
            full_expr = None
            for sub_idx in range(len(sub_op_ls)):
                col = sub_col_name_ls[sub_idx]
                op = sub_op_ls[sub_idx]
                const = target_const[sub_idx]
                expr = op(existing_data[col], const)
                if full_expr is None:
                    full_expr = (expr)
                else:
                    full_expr = (expr) | full_expr

            patient_matches = existing_data.loc[full_expr,"PAT_ID"].unique()
            existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
            
            existing_data_ls.append(existing_data)
        return existing_data_ls
    
    def evaluate_atom_ls_ls_on_dataset_full_multi(self, expr_ls, full_data_ls, col_name_col, op_name_col, pred_v_col, other_keys=None):

        existing_data_ls = []
        transformed_expr_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        op_ls = expr_ls[op_name_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            data_ls = full_data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            sub_op_ls = op_ls[idx]
            curr_existing_data_ls = []
            curr_expr_ls = []
            # for data in data_ls:
            '''previous version'''
            for sub_idx in range(len(sub_op_ls)):
                existing_data = data_ls[sub_idx]
                # for expr in expr_ls:
                    # assert "formula" in expr
                    # if expr["formula"] == "end":
                    #     return data
                    # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                    # op = expr["num_op" if "num_op" in expr else "cat_op"]
                    # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

                    # target_const = expr[feat]
                # full_expr = None
              
                col = sub_col_name_ls[sub_idx]
                # if col in self.lang.CAT_FEATS:
                #     print()  
                curr_op = sub_op_ls[sub_idx]
                const = target_const[sub_idx]
                expr = curr_op(existing_data[col], const)
                # if full_expr is None:
                #     full_expr = (expr)
                # else:
                #     full_expr = (expr) | full_expr
                curr_expr_ls.append((col, curr_op, const))
                existing_data = self.evaluate_expression_on_data(existing_data, expr)
                curr_existing_data_ls.append(existing_data)
            existing_data_ls.append(curr_existing_data_ls)
            transformed_expr_ls.append(curr_expr_ls)

            # '''new version'''
            # full_expr = None
            # existing_data = data_ls
            # for sub_idx in range(len(sub_op_ls)):
            #     #.copy()
            #     # for expr in expr_ls:
            #         # assert "formula" in expr
            #         # if expr["formula"] == "end":
            #         #     return data
            #         # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            #         # op = expr["num_op" if "num_op" in expr else "cat_op"]
            #         # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

            #         # target_const = expr[feat]
                
              
            #     col = sub_col_name_ls[sub_idx]
            #     # if col in self.lang.CAT_FEATS:
            #     #     print()  
            #     curr_op = sub_op_ls[sub_idx]
            #     const = target_const[sub_idx]
            #     expr = curr_op(existing_data[col], const)
            #     if full_expr is None:
            #         full_expr = (expr)
            #     else:
            #         full_expr = (expr) | full_expr
            #     curr_expr_ls.append((col, curr_op, const))
                
            # existing_data = self.evaluate_expression_on_data(existing_data, full_expr)
            # # curr_existing_data_ls.append(existing_data)
            

            # #     existing_data = self.evaluate_expression_on_data(existing_data, expr)
            # #     curr_existing_data_ls.append(existing_data)
            # #     curr_expr_ls.append((col, curr_op, const))
            # existing_data_ls.append(existing_data)
            # transformed_expr_ls.append(curr_expr_ls)
        return existing_data_ls, transformed_expr_ls
    
    def evaluate_atom_ls_ls_on_dataset_full_multi2(self, expr_ls, full_data_ls, col_name_col, op_name_col, pred_v_col, other_keys=None):

        existing_data_ls = []
        transformed_expr_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        op_ls = expr_ls[op_name_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            data_ls = full_data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            sub_op_ls = op_ls[idx]
            curr_existing_data_ls = []
            curr_expr_ls = []
            # for data in data_ls:
            '''previous version'''
            for sub_idx in range(len(sub_op_ls)):
                existing_data = data_ls[sub_idx]
                # for expr in expr_ls:
                    # assert "formula" in expr
                    # if expr["formula"] == "end":
                    #     return data
                    # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                    # op = expr["num_op" if "num_op" in expr else "cat_op"]
                    # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

                    # target_const = expr[feat]
                # full_expr = None
              
                col = sub_col_name_ls[sub_idx]
                # if col in self.lang.CAT_FEATS:
                #     print()  
                curr_op = sub_op_ls[sub_idx]
                const = target_const[sub_idx]
                
                expr = torch.logical_and(curr_op(self.features[:,col], const), ~torch.isnan(self.features[:,col]))
                existing_data = torch.logical_and(existing_data, expr)
                curr_existing_data_ls.append(existing_data)
                # if full_expr is None:
                #     full_expr = (expr)
                # else:
                #     full_expr = (expr) | full_expr
                curr_expr_ls.append((col, curr_op, const))
                # existing_data = self.evaluate_expression_on_data(existing_data, expr)
                
            existing_data_ls.append(curr_existing_data_ls)
            transformed_expr_ls.append(curr_expr_ls)

            # '''new version'''
            # full_expr = None
            # existing_data = data_ls
            # for sub_idx in range(len(sub_op_ls)):
            #     #.copy()
            #     # for expr in expr_ls:
            #         # assert "formula" in expr
            #         # if expr["formula"] == "end":
            #         #     return data
            #         # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            #         # op = expr["num_op" if "num_op" in expr else "cat_op"]
            #         # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

            #         # target_const = expr[feat]
                
              
            #     col = sub_col_name_ls[sub_idx]
            #     # if col in self.lang.CAT_FEATS:
            #     #     print()  
            #     curr_op = sub_op_ls[sub_idx]
            #     const = target_const[sub_idx]
            #     expr = curr_op(existing_data[col], const)
            #     if full_expr is None:
            #         full_expr = (expr)
            #     else:
            #         full_expr = (expr) | full_expr
            #     curr_expr_ls.append((col, curr_op, const))
                
            # existing_data = self.evaluate_expression_on_data(existing_data, full_expr)
            # # curr_existing_data_ls.append(existing_data)
            

            # #     existing_data = self.evaluate_expression_on_data(existing_data, expr)
            # #     curr_existing_data_ls.append(existing_data)
            # #     curr_expr_ls.append((col, curr_op, const))
            # existing_data_ls.append(existing_data)
            # transformed_expr_ls.append(curr_expr_ls)
        return existing_data_ls, transformed_expr_ls

    def evaluate_atom_ls_ls_on_one_patient_with_full_programs(self, pid, transformed_expr_ls, full_data_ls):

        existing_data_ls = []
        # [[A < a, B > b], [C < c, D > d],....]
        transformed_exprs = transformed_expr_ls
        for idx in range(len(transformed_exprs)):
            expr_ls = transformed_exprs[idx]
            existing_data = full_data_ls[idx].clone()
            for sub_idx in range(len(expr_ls)):
                col, curr_op, const = expr_ls[sub_idx]
                expr = torch.logical_and(curr_op(self.features[:,col], const), ~torch.isnan(self.features[:,col]))
                existing_data = torch.logical_and(existing_data, expr)

            existing_data_ls.append(existing_data)
        return existing_data_ls
    def evaluate_atom_ls_ls_on_one_patient_with_full_programs_leave_one_out(self, pid, transformed_expr_ls, full_data_ls):

        existing_data_ls = []
        # [[A < a, B > b], [C < c, D > d],....]
        transformed_exprs = transformed_expr_ls
        for sub_idx in range(len(transformed_exprs[0])):
            curr_existing_data_ls = []
            for idx in range(len(transformed_exprs)):
                expr_ls = transformed_exprs[idx]
                # existing_data = full_data_ls[idx].clone()
                
                
                curr_expr_ls= expr_ls[:sub_idx] + expr_ls[sub_idx+1:]
                curr_existing_data = self.evaluate_atom_ls_ls_on_one_patient_with_full_programs(pid, [curr_expr_ls], full_data_ls)
                curr_existing_data_ls.append(curr_existing_data[0])

            existing_data_ls.append(curr_existing_data_ls)
        return existing_data_ls
    
    def evaluate_atom_ls_ls_on_dataset_full_multi_2(self, expr_ls, full_data_ls, col_name_col, op_name_col, pred_v_col):

        existing_data_ls = []
        transformed_expr_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        op_ls = expr_ls[op_name_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            existing_data = full_data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            sub_op_ls = op_ls[idx]
            # curr_existing_data_ls = []
            curr_expr_ls = []
            # existing_data = data_ls[0]
            # for data in data_ls:
            '''previous version'''
            full_expr = None
            for sub_idx in range(len(sub_op_ls)):
                # existing_data = data_ls[sub_idx].copy()
                # for expr in expr_ls:
                    # assert "formula" in expr
                    # if expr["formula"] == "end":
                    #     return data
                    # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                    # op = expr["num_op" if "num_op" in expr else "cat_op"]
                    # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

                    # target_const = expr[feat]
                # full_expr = None
              
                col = sub_col_name_ls[sub_idx]
                # if col in self.lang.CAT_FEATS:
                #     print()  
                curr_op = sub_op_ls[sub_idx]
                const = target_const[sub_idx]
                expr = curr_op(existing_data[col], const)
                if full_expr is None:
                    full_expr = (expr)
                else:
                    full_expr = (expr) | full_expr
                curr_expr_ls.append((col, curr_op, const))
            existing_data = self.evaluate_expression_on_data(existing_data, full_expr)
            # curr_existing_data_ls.append(existing_data)
            existing_data_ls.append(existing_data)
            transformed_expr_ls.append(curr_expr_ls)
        return existing_data_ls, transformed_expr_ls
    
    def evaluate_expression_on_data(self, existing_data, expr):
        # import time
        # existing_data_test = existing_data.copy()
        # expr_test = expr.copy()
        # t1 = time.time()
        # patient_matches = existing_data.loc[expr,"PAT_ID"].unique()
        # existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
        # t2 = time.time()

        # t3 = time.time()
        # patient_matches_test = existing_data_test.loc[expr_test].index.unique()
        # existing_data_test = existing_data_test[existing_data_test.index.isin(patient_matches_test)]
        # t4 = time.time()

        # print(t4 - t3, t2 - t1)
        existing_data = existing_data.loc[expr]
        return existing_data

    def evaluate_atom_ls_ls_on_dataset_full_multi_medicine(self, expr_ls, full_data_ls, col_name_col, range_name_col):

        existing_data_ls = []
        # pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        # op_ls = expr_ls[op_name_col]
        ranges_ls = expr_ls[range_name_col]
        
        for idx in range(len(ranges_ls)):
            target_ranges = ranges_ls[idx]
            data_ls = full_data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            # sub_op_ls = op_ls[idx]
            curr_existing_data_ls = []
            # for data in data_ls:
            for sub_idx in range(len(target_ranges)):
                existing_data = data_ls[sub_idx].copy()
                range_min, range_max = target_ranges[sub_idx]
                
                # for expr in expr_ls:
                    # assert "formula" in expr
                    # if expr["formula"] == "end":
                    #     return data
                    # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                    # op = expr["num_op" if "num_op" in expr else "cat_op"]
                    # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]

                    # target_const = expr[feat]
                full_expr = None
                
                col = sub_col_name_ls[sub_idx]
                # op = sub_op_ls[sub_idx]
                # const = target_const[sub_idx]
                if not np.isinf(range_min):
                    expr = op.__ge__(existing_data[col], range_min)
                else:
                    expr = None

                if not np.isinf(range_max):
                    if expr is not None:
                        expr = (expr & op.__le__(existing_data[col], range_max))
                    else:
                        expr = op.__le__(existing_data[col], range_max)
                
                # if full_expr is None:
                #     full_expr = (expr)
                # else:
                #     full_expr = (expr) | full_expr

                patient_matches = existing_data.loc[expr,"PAT_ID"].unique()
                existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
                curr_existing_data_ls.append(existing_data)
            existing_data_ls.append(curr_existing_data_ls)
        return existing_data_ls
    
    def evaluate_union_atom_ls_on_dataset(self, union_expr_ls, data):
        existing_data = data.copy()

        where_cond = None

        for expr_ls in union_expr_ls:
            curr_where_cond = None
            for expr in expr_ls:
                assert "formula" in expr
                if expr["formula"] == "end":
                    return data
                assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
                op = expr["num_op" if "num_op" in expr else "cat_op"]
                feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
                target_const = expr[feat]
                if curr_where_cond is None:
                    curr_where_cond = op(existing_data[feat], target_const)
                else:
                    curr_where_cond = op(existing_data[feat], target_const) & curr_where_cond

            if where_cond is None:
                where_cond = (curr_where_cond)
            else:
                where_cond = where_cond | (curr_where_cond)
            

        patient_matches = existing_data.loc[where_cond,"PAT_ID"].unique()
        return patient_matches
        # existing_data = existing_data[existing_data['PAT_ID'].isin(patient_matches)]
        # return existing_data

    def evaluate_atom_ls_on_dataset_for_remaining_data(self, expr_ls, data):
        existing_data = data.copy()
        all_patient_matches = None
        for expr in expr_ls:
            assert "formula" in expr
            if expr["formula"] == "end":
                return data
            assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            op = expr["num_op" if "num_op" in expr else "cat_op"]
            feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            target_const = expr[feat]
            patient_matches = op(existing_data[feat], target_const)
            if all_patient_matches is None:
                all_patient_matches = patient_matches
            else:
                all_patient_matches = patient_matches & all_patient_matches
        
        all_patient_matches_ids = existing_data.loc[all_patient_matches,"PAT_ID"].unique()
        
        existing_data = existing_data[~existing_data['PAT_ID'].isin(all_patient_matches_ids)]
        return existing_data
    
    def evaluate_atom_on_sample(self, expr:dict, X) -> bool:
        assert "formula" in expr
        if expr["formula"] == "end":
            return True
        assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
        op = expr["num_op" if "num_op" in expr else "cat_op"]
        feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
        target_const = expr[feat]
        filtered = X.loc[op(X[feat], target_const)]
        return len(filtered) != 0

    def get_dataset(self):
        return self.dataset

    def atom_to_str(self, expr:dict):
        assert "formula" in expr
        if expr["formula"] == "end":
            return True
        assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
        op = self.lang.NON_STR_REP[expr["num_op" if "num_op" in expr else "cat_op"]]
        feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
        target_const = str(expr[feat])
        return feat+op+target_const

    def atom_to_str_ls(self, expr_ls:dict):
        atom_str_ls = []
        for expr in expr_ls:
            assert "formula" in expr
            if expr["formula"] == "end":
                return True
            assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            op = self.lang.NON_STR_REP[expr["num_op" if "num_op" in expr else "cat_op"]]
            feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            target_const = str(expr[feat])
            atom_str = feat+op+target_const
            atom_str_ls.append(atom_str)
        return atom_str_ls
    
    def atom_to_str_ls0(self, expr_ls, col, op, col_v_key):
        atom_str_ls = []
        const_arr = expr_ls[col_v_key]
        # for expr in expr_ls:
        for const in const_arr:
            # assert "formula" in expr
            # if expr["formula"] == "end":
            #     return True
            # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            # op = self.lang.NON_STR_REP[expr["num_op" if "num_op" in expr else "cat_op"]]
            # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            # target_const = str(expr[col_v_key])
            atom_str = col+self.lang.NON_STR_REP[op]+str(const)
            atom_str_ls.append(atom_str)
        return atom_str_ls

    def atom_to_str_ls_full(self, X_pd_ls, expr_ls, col_name_key, op_name_key, col_v_key, feat_range_mappings, cat_id_unique_vals_mappings, other_keys=None):
        atom_str_ls = []
        const_arr = expr_ls[col_v_key]
        col_name_arr = expr_ls[col_name_key]
        op_arr = expr_ls[op_name_key]
        col_id_key = other_keys[0]
        col_id_arr = expr_ls[col_id_key]
        # for expr in expr_ls:
        # for const in const_arr:
        for idx in range(len(const_arr)):
            # if idx == 23:
            #     print()
            # assert "formula" in expr
            # if expr["formula"] == "end":
            #     return True
            # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            # op = self.lang.NON_STR_REP[expr["num_op" if "num_op" in expr else "cat_op"]]
            # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            # target_const = str(expr[col_v_key])
            curr_atom_str_ls = []
            # feat_range_ls = 
            for sub_idx in range(len(col_name_arr[idx])):
                # if col_name_arr[idx][sub_idx] in self.test_dataset.num_cols:
                if not col_name_arr[idx][sub_idx] in self.lang.CAT_FEATS:
                    feat_range = feat_range_mappings[col_name_arr[idx][sub_idx]]
                    real_val = const_arr[idx][sub_idx]*(feat_range[1] - feat_range[0]) + feat_range[0]
                    atom_str = col_name_arr[idx][sub_idx]+self.lang.NON_STR_REP[op_arr[idx][sub_idx]]+str(real_val)
                else:
                    
                    curr_feat_val = X_pd_ls[idx][col_id_arr[idx][sub_idx]].item()
                    real_val = cat_id_unique_vals_mappings[col_name_arr[idx][sub_idx]][curr_feat_val]
                    atom_str = col_name_arr[idx][sub_idx]+"="+str(real_val)
                curr_atom_str_ls.append(atom_str)
            atom_str_ls.append(curr_atom_str_ls)
        return atom_str_ls

    def atom_to_str_ls_full_medical(self, expr_ls, col_name_key, range_name_key, feat_range_mappings):
        atom_str_ls = []
        # const_arr = expr_ls[col_v_key]
        col_name_arr = expr_ls[col_name_key]
        selected_range = expr_ls[range_name_key]
        # for expr in expr_ls:
        # for const in const_arr:
        for idx in range(len(col_name_arr)):
            # assert "formula" in expr
            # if expr["formula"] == "end":
            #     return True
            # assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
            # op = self.lang.NON_STR_REP[expr["num_op" if "num_op" in expr else "cat_op"]]
            # feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
            # target_const = str(expr[col_v_key])
            curr_atom_str_ls = []
            # feat_range_ls = 
            for sub_idx in range(len(col_name_arr[idx])):
                feat_range = feat_range_mappings[col_name_arr[idx][sub_idx]]
                # real_val = const_arr[idx][sub_idx]*(feat_range[1] - feat_range[0]) + feat_range[0]
                curr_min, curr_max = selected_range[idx][sub_idx]
                
                atom_str = ""
                if not np.isinf(curr_min):
                    atom_str += col_name_arr[idx][sub_idx]+">"+str(curr_min)        
                if not np.isinf(curr_max):
                    atom_str += col_name_arr[idx][sub_idx]+"<="+str(curr_max)        
                
                # atom_str = col_name_arr[idx][sub_idx]+self.lang.NON_STR_REP[op_arr[idx][sub_idx]]+str(real_val)
                curr_atom_str_ls.append(atom_str)
            atom_str_ls.append(curr_atom_str_ls)
        return atom_str_ls

if __name__ == "__main__":
    lang = Language(dataset=pd.read_csv("dataset.pd"))
    test_expr = {
        "formula":"num_comp",
        "num_op":op.__lt__,
        "num_feat":"RED.BLOOD.CELLS..first",
        "RED.BLOOD.CELLS..first":4.23
    }
    d = lang.evaluate_atom(test_expr, lang.dataset)
    a=0