import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from full_experiments.parse_args import parse_args
from full_experiments.full_cancer_experiment import read_cancer_data

if __name__ == "__main__":
    args = parse_args()
    train_data, valid_data, test_data, program_max_len = read_cancer_data(args.data_folder)

    