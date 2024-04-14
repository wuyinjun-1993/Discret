import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from parse_args import *

if __name__ == "__main__":

    args = parse_args()

    lab_measurement_data = pd.read_csv(os.path.join(args.data_folder, "enc_data_static_12month"))

    final_treatment_data = pd.read_csv(os.path.join(args.data_folder, "Treatment_Data/final_thoracic_patients.csv"))

    lab_treatment_data = pd.read_csv(os.path.join(args.data_folder, "Treatment_Data/Labs_Treat_Thoracic.csv"))

    patient_treatment_data = pd.read_csv(os.path.join(args.data_folder, "Treatment_Data/thoracic_patients_treatments.csv"))

    print()