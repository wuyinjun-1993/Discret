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

CAT_FEATS = ["DECREASED APPETITE SEVERITY", "MARITAL_STATUS", "PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS", "ETHNIC_GROUP", "SADNESS INTERFERENCE", "NAUSEA FREQUENCY","PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS", "STAGE_GROUP", "NUMBNESS & TINGLING SEVERITY", "SEX_C.y", "SEX_C.x", "FATIGUE INTERFERENCE", "CONSTIPATION SEVERITY", "DIARRHEA FREQUENCY", "GLOBAL02", "fin_class_name", "RASH YES/NO", "RACE", "ACTIVITIES & FUNCTION SCORING QUESTION", "SHORTNESS OF BREATH INTERFERENCE", "GLOBAL02 SCORING QUESTION", "ANXIETY INTERFERENCE", "ACTIVITIES & FUNCTION", "FEVER 100.5 OR GREATER", "ECOG_strict", "subtype", "CONSTIPATION ", "DECREASED APPETITE ", "NUMBNESS & TINGLING ", "n_is_Lymp_recent", "n_is_Fluid_recent", "n_is_Myeloproliferative.neoplasms","n_is_LD", "n_is_Coag", "n_is_Paralysis_recent","n_is_LD_recent", "n_is_Lymp", "n_is_METS_recent", "n_is_METS", "n_is_Tumor",  "n_is_WL", "n_is_VD", "n_is_Weakness", "n_is_WL_recent", "n_is_Leukemia.and.Myeloma", "n_is_Tumor_recent", "n_is_VD_recent", "other"]
DROP_FEATS = sorted(["Unnamed: 0", "TreatedCycles", "line", "SubjectId", "EpicPatientId", "Empi", "HupMrn", "TreatmentPlanId", "ProtocolId", "ProtocolName", "Regimen", "FirstTreatmentDate", "LastTreatmentDate", 'label', "Pred", "EMPI", "SDE_VALUE", "SEX_C_x", "SEX_C_y", "batch_id", "count", "NPI", "X", "APPT_DATE", "CODE_TIME_DATE", "DATE_DIFF_DAYS", "Platinum chemotherapy_indicator", "Non-platinum chemotherapy_indicator", "Immunotherapy - PD1/PDL1 inhibitor_indicator", "VEGF inhibitor_indicator", "Targeted therapy_indicator", "Immunotherapy - CTLA-4 inhibitor_indicator"])

# DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID"])

NON_STR_REP= {
    op.__le__:"<=",
    op.__ge__:">=",
    op.__eq__:"=="
}