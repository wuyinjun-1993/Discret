import operator as op

LANG_SYNTAX = {
    "formula":{"num_comp":["num_feat", "num_op"]},
    "num_feat":{}, #dynamically populated
    "num_op":{op.__le__:[], op.__ge__:[]},
    #feature aggregated value syntax dynamically added
}

CAT_FEATS = []
DROP_FEATS = sorted(["PAT_ID", 'label'])