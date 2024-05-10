import operator as op

LANG_SYNTAX = {
    "formula":{"num_comp":["num_feat", "num_op"]},
    "num_feat":{}, #dynamically populated
    "cat_feat":{},
    # comment this line, pick an interval
    "cat_op":{op.__eq__:[]},
    "num_op":{op.__le__:[], op.__ge__:[]},
    #feature aggregated value syntax dynamically added
}

# CAT_FEATS = ["Cholesterol", "Glucose", "Smoking", "Alcohol intake", "Physical activity"]
CAT_FEATS = []
DROP_FEATS = []

# DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID"])

NON_STR_REP= {
    op.__le__:"<=",
    op.__ge__:">=",
    op.__eq__:"=="
}