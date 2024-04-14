import pandas as pd
import operator as op

class Language:
    def __init__(self, data, precomputed, lang):
        self.lang = lang
        self.syntax = lang.LANG_SYNTAX
        self.dataset = data
        num_feats = [col for col in self.dataset.columns if col not in lang.CAT_FEATS and col not in lang.DROP_FEATS]
        if "num_feat" in self.syntax:
            for col in num_feats:
                self.syntax["num_feat"][col] = [col]
                self.syntax[col] = {i:[] for i in precomputed[col]}
        if "cat_feat" in self.syntax:
            for col in lang.CAT_FEATS:
                self.syntax["cat_feat"][col] = [col]
                self.syntax[col] = {i:[] for i in self.dataset[col].unique()}
    
    #returns filtered dataset from given expression
    def evaluate_atom_on_dataset(self, expr: dict, data):
        assert "formula" in expr
        if expr["formula"] == "end":
            return data
        assert ("num_op" in expr) ^ ("cat_op" in expr) and ("num_feat" in expr) ^ ("cat_feat" in expr)
        op = expr["num_op" if "num_op" in expr else "cat_op"]
        feat = expr["num_feat" if "num_feat" in expr else "cat_feat"]
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