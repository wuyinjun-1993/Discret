import operator as op


LANG_SYNTAX = {
    "formula":{"end":[], "num_comp":["num_feat", "num_op"], "cat_comp":["cat_feat", "cat_op"]},
    "num_feat":{}, #dynamically populated
    "cat_feat":{}, #dynamically populated
    # no need on operators
    "num_op":{op.__lt__:[], op.__le__:[], op.__gt__:[], op.__ge__:[], op.__eq__:[], op.__ne__:[]}, 
    "cat_op":{op.__eq__:[], op.__ne__:[]},
    #feature aggregated value syntax dynamically added 
}

CAT_FEATS = sorted(['ETHNIC_GROUP', 'fin_class_name', 'QUEST_ANSWER.UPHS ONC PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS', 'RACE', 'STAGE_GROUP', 'MARITAL_STATUS'])
DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID", "EMPI", "SDE_VALUE"])
# DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID"])