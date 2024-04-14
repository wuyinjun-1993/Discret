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