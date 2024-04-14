import pandas as pd
import numpy as np
import random

def create_synthetic_df(size, name):
    data = {"PAT_ID": [i for i in range(size)], "A": np.random.rand(size), "B": np.random.rand(size)}
    df = pd.DataFrame(data=data)
    df["label"] = df.apply(lambda row: 1 if (row.A <= 0.25 and row.B <= 0.25) or (row.A >= 0.75 and row.B >= 0.75) else 0, axis=1)
    df.to_csv(name, index=False)
    precomputed = {"A": [0.25,0.5,0.75], "B": [0.25,0.5,0.75]}
    np.save("synthetic_precomputed", precomputed, allow_pickle=True)


def create_ehr_df():
    # data = pd.read_csv("simple_df")
    # data.columns = data.columns.str.replace('[.]','')
    # data.to_csv("simple_df", index=False)
    precomputed = { "PAT_AGE": [25, 47, 65, 80],
                    "ALBUMINlast": [3,3.4,4, 4.3],
                    "ALKALINEPHOSPHATASEmean": [60,75,123,135],
                    "REDBLOODCELLSlast": [2.5,2.85, 3.9,4.2],
                    "XLYMPHOCYTESmin": [4.5, 6.5, 15, 29],
                    "HEMOGLOBINlast": [7, 8.4, 11.4],
                    }
    np.save("ehr_precomputed", precomputed, allow_pickle=True)


if __name__ == "__main__":
    train_size, test_size = 1000, 300
    create_synthetic_df(size=train_size, name="synthetic_dataset.pd")
    create_synthetic_df(size=test_size, name="synthetic_test_dataset.pd")
    create_ehr_df()