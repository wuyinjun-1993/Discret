import pandas as pd
import numpy as np
import datetime as dt
from dateutil import parser

PRIORITY = ['Normal with no limitations','Not my normal self, but able to be up and about with fairly normal activities','Not feeling up to most things, but in bed or chair less than half the day', 'Able to do little activity and spend most of the day in bed or chair', 'Pretty much bedridden, rarely out of bed', 'Not at all', 'A little bit', 'Somewhat', 'Quite a bit', 'Very much', 'None', 'Mild', 'Moderate', 'Severe', 'Very severe', 'Very Severe', 'Never', 'Rarely', 'Occasionally', 'Frequently', 'Almost constantly', 'N', 'Y','0','1','2','3','4','5','6','Incomplete', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent', 'NA']

def sort_fn(x):
    for i,v in enumerate(PRIORITY):
        if x == v:
            return i
    if isinstance(x, int) or isinstance(x, float):
        return x
    return 0


def construct_label_column_helper(appt, death):
    if pd.isna(death) or pd.isna(appt):
        return 0
    appt, death = parser.parse(appt), parser.parse(death)
    return 1 if death - appt < dt.timedelta(days=183) else 0

def process_dataset():
    df = pd.read_csv("r35_pah.EHR.PRO.csv", index_col=0)
    df['label'] = df.apply(lambda row: construct_label_column_helper(row.appt_date, row.DEATH_DATE), axis=1)
    #df = df.select_dtypes(['number'])
    #df = (df-df.min())/(df.max()-df.min())
    df = df.drop(['CSN', 'PROVIDER_PENN_ID', "APPT_TIME", "appt_date", "BIRTH_DATE","DEPARTMENT_NAME","PROVIDER_PENN_ID","PROV_TYPE","SPECIALTY", "X_id", "date", "DEATH_DATE", "pred_date", "stage_date", "contact_date", 'EDIT_DATETIME', 'PROVIDER_NAME', 'ZIP','Pred', 'NPI', 'EMPI'], axis=1)
    df = df.fillna(value='NA')
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df2 = df.select_dtypes(['number', None])
    df = df.replace('Incomplete','NA')
    cat_cols = [col for col in df.columns if col not in df2.columns]
    df = df.replace('NA', np.NaN)
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    replace_dict = {i:dict(map(reversed,dict(enumerate(sorted(df[i].unique(), key=sort_fn))).items())) for i in cat_cols}
    df = df.replace(replace_dict)
    #df = df.dropna(axis=0)

    df.to_csv('dataset.pd', index=False)
    stds = [-1,0,1]
    precomputed = {}
    for col in df.columns:
        std = df[col].std()
        mean = df[col].mean()
        precomputed[col] = [mean+std*i for i in stds]
    np.save("precomputed", precomputed, allow_pickle=True)
if __name__ == "__main__":
    process_dataset()
