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

CAT_FEATS = ["DIARRHEA FREQUENCY", "SADNESS INTERFERENCE", "ACTIVITIES & FUNCTION", "SEX_C.x", "ETHNIC_GROUP", "CONSTIPATION SEVERITY", "FEVER 100.5 OR GREATER","NAUSEA FREQUENCY", "GLOBAL02", "STAGE_GROUP", "RACE", "RASH YES/NO", "ANXIETY INTERFERENCE", "DECREASED APPETITE SEVERITY", "FATIGUE INTERFERENCE", "MARITAL_STATUS", "PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS", "SHORTNESS OF BREATH INTERFERENCE", "NUMBNESS & TINGLING SEVERITY"]
OTHER_SCORE_FEATS = ["ACTIVITIES & FUNCTION SCORING QUESTION", "PAT_AGE", "ANXIETY INTERFERENCE SCORE", "DECREASED APPETITE SEVERITY SCORE", "SHORTNESS OF BREATH INTERFERENCE SCORE", "SADNESS INTERFERENCE SCORE", "GLOBAL02 SCORING QUESTION", "NAUSEA FREQUENCY SCORE", "DIARRHEA FREQUENCY SCORE", "FATIGUE INTERFERENCE SCORE", "CONSTIPATION SEVERITY SCORE"]
DROP_FEATS = sorted(["ZIP", "Unnamed: 0", "Unnamed: 0.1", "SubjectId", "EpicPatientId", "Empi", "HupMrn", "TreatmentPlanId", "ProtocolId", "ProtocolName", "Regimen", "other", "fin_class_name", "EMPI", "NPI", "DEPARTMENT_NAME", "CSN", "PROVIDER_PENN_ID", "Pred", "PROVIDER_NAME", "count", "batch_id", "X_id", "BIRTH_DATE", "SDE_VALUE", "SEX_C.y"])
UNKNOWN=["PROV_TYPE", "SPECIALTY", "TreatedCycles"]
DATE_FEATS=["FirstTreatmentDate", "LastTreatmentDate", "stage_date", "contact_date", "date", "pred_date", "APPT_TIME", "appt_date", "DEATH_DATE", "EDIT_DATETIME"]
OTHER_FEATS={"PAT_ID", "line"}
Prob_FEATS={"NUMBNESS & TINGLING SEVERITY SCORE"}

# DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID"])

NON_STR_REP= {
    op.__le__:"<=",
    op.__ge__:">=",
    op.__eq__:"=="
}