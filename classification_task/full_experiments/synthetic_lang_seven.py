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

# CAT_FEATS = ["CONSTIPATION SEVERITY SCORE", "DECREASED APPETITE SEVERITY", "MARITAL_STATUS", "PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS", "DIARRHEA FREQUENCY SCORE", "NUMBNESS & TINGLING SEVERITY SCORE", "ETHNIC_GROUP", "FATIGUE INTERFERENCE SCORE", "SADNESS INTERFERENCE", "NAUSEA FREQUENCY","PRO QUESTIONNAIRE SYMPTOMS TO ADDRESS", "STAGE_GROUP", "NAUSEA FREQUENCY SCORE", "NUMBNESS & TINGLING SEVERITY", "SEX_C.y", "SEX_C.x", "FATIGUE INTERFERENCE", "CONSTIPATION SEVERITY", "ANXIETY INTERFERENCE SCORE", "DIARRHEA FREQUENCY", "GLOBAL02", "fin_class_name", "RASH YES/NO", "RACE", "SHORTNESS OF BREATH INTERFERENCE SCORE", "ACTIVITIES & FUNCTION SCORING QUESTION", "DECREASED APPETITE SEVERITY SCORE", "SADNESS INTERFERENCE SCORE", "SHORTNESS OF BREATH INTERFERENCE", "GLOBAL02 SCORING QUESTION", "ANXIETY INTERFERENCE", "ACTIVITIES & FUNCTION", "FEVER 100.5 OR GREATER", "ECOG_strict", "subtype", "CONSTIPATION ", "CONSTIPATION  SCORE", "DECREASED APPETITE ", "DECREASED APPETITE  SCORE", "NUMBNESS & TINGLING ", "NUMBNESS & TINGLING  SCORE"]
# CAT_FEATS = ["Gender", "Race", "Ethnicity", "Histology", "SmokingStatus", "GroupStage", "EcogValue", "DrugCategory", "SESIndex2015_2019", "ALK", "BRAF", "EGFR", "KRAS", "PDL1", "ROS1"]
CAT_FEATS = ['gender', 'race', 'ethnicity', 'region', 'Histology', 'SmokingStatus', 'stage', 'steroid_diag', 'opioid_PO_diag', 'nonopioid_PO_diag', 'pain_IV_diag', 'ac_diag', 'antiinfective_IV_diag', 'antiinfective_diag', 'antihyperglycemic_diag', 'ppi_diag', 'antidepressant_diag', 'bta_diag', 'thyroid_diag', 'is_diag', 'ALK', 'BRAF', 'EGFR', 'KRAS', 'MET', 'NTRK - other', 'NTRK - unknown gene type', 'NTRK1', 'NTRK2', 'NTRK3', 'RET', 'ROS1', 'pdl1', 'pdl1_n', 'commercial', 'medicare', 'medicaid', 'other_insurance', 'ecog_diagnosis', 'bmi_diag_na', 'weight_pct_na', 'albumin_diag_na', 'alp_diag_na', 'alt_diag_na', 'ast_diag_na', 'bicarb_diag_na', 'bun_diag_na', 'calcium_diag_na', 'chloride_diag_na', 'creatinine_diag_na', 'hemoglobin_diag_na', 'neutrophil_count_diag_na', 'platelet_diag_na', 'potassium_diag_na', 'sodium_diag_na', 'total_bilirubin_diag_na', 'wbc_diag_na', 'albumin_slope_na', 'alp_slope_na', 'alt_slope_na', 'ast_slope_na', 'bicarb_slope_na', 'bun_slope_na', 'calcium_slope_na', 'chloride_slope_na', 'creatinine_slope_na', 'hemoglobin_slope_na', 'neutrophil_count_slope_na', 'platelet_slope_na', 'potassium_slope_na', 'sodium_slope_na', 'total_bilirubin_slope_na', 'wbc_slope_na', 'chf', 'cardiac_arrhythmias', 'valvular_disease', 'pulmonary_circulation', 'peripheral_vascular', 'htn_uncomplicated', 'htn_complicated', 'paralysis', 'other_neuro_disorders', 'chronic_pulmonary', 'diabetes_uncomplicated', 'diabetes_complicated', 'hypothyroidism', 'renal_failure', 'liver_disease', 'peptic_ulcer_disease', 'aids_hiv', 'lymphoma', 'metastatic_cancer', 'solid_tumor_wout_mets', 'rheumatoid_arthritis', 'coagulopathy', 'obesity', 'weight_loss', 'fluid_electrolyte', 'blood_loss_anemia', 'deficiency_anemia', 'alcohol_abuse', 'drug_abuse', 'psychoses', 'depression', 'elixhauser_other', 'other_cancer']
DROP_FEATS = sorted(["Unnamed: 0", "State"])

# DROP_FEATS = sorted(["PAT_ID", 'label', "Pred", "PROVIDER_PENN_ID"])

NON_STR_REP= {
    op.__le__:"<=",
    op.__ge__:">=",
    op.__eq__:"=="
}