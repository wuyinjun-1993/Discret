import pandas as pd
import operator as op
import numpy as np
class Language:
    def __init__(self, data, precomputed, lang):
        self.lang = lang
        self.syntax = lang.LANG_SYNTAX
        self.dataset = data
        self.precomputed = precomputed
        num_feats = [col for col in self.dataset.columns if col not in lang.CAT_FEATS and col not in lang.DROP_FEATS]
        if "num_feat" in self.syntax:
            for col in num_feats:
                self.syntax["num_feat"][col] = [col]
                # if precomputed is not None:
                self.syntax[col] = []#{i:[] for i in precomputed[col]}
        if "cat_feat" in self.syntax:
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
    
    def evaluate_atom_ls_ls_on_dataset_full_multi(self, expr_ls, full_data_ls, col_name_col, op_name_col, pred_v_col):

        existing_data_ls = []
        pred_v_ls = expr_ls[pred_v_col]
        col_name_ls = expr_ls[col_name_col]
        op_ls = expr_ls[op_name_col]
        
        for idx in range(len(pred_v_ls)):
            target_const = pred_v_ls[idx]
            data_ls = full_data_ls[idx]
            sub_col_name_ls = col_name_ls[idx]
            sub_op_ls = op_ls[idx]
            curr_existing_data_ls = []
            # for data in data_ls:
            for sub_idx in range(len(sub_op_ls)):
                existing_data = data_ls[sub_idx].copy()
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
                op = sub_op_ls[sub_idx]
                const = target_const[sub_idx]
                expr = op(existing_data[col], const)
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

    def atom_to_str_ls_full(self, expr_ls, col_name_key, op_name_key, col_v_key, feat_range_mappings):
        atom_str_ls = []
        const_arr = expr_ls[col_v_key]
        col_name_arr = expr_ls[col_name_key]
        op_arr = expr_ls[op_name_key]
        # for expr in expr_ls:
        # for const in const_arr:
        for idx in range(len(const_arr)):
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
                real_val = const_arr[idx][sub_idx]*(feat_range[1] - feat_range[0]) + feat_range[0]
                atom_str = col_name_arr[idx][sub_idx]+self.lang.NON_STR_REP[op_arr[idx][sub_idx]]+str(real_val)
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