import argparse

import os, yaml, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from classification_task.datasets.EHR_datasets import one, two, three, four, five, six, seven


def load_configs(args, root_dir=None):
    if args.model_config is None:
        args.model_config = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs/configs.yaml")

    # yamlfile_name = os.path.join(args.model_config, "model_config.yaml")
    # elif args.model_type == "csdi":
    #     yamlfile_name = os.path.join(model_config_file_path_base, "csdi_config.yaml")
    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(os.path.join(root_dir, args.model_config), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        rl_config = config["rl"][args.rl_algorithm]
        model_config = config["model"][args.model]
    return rl_config, model_config

def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data_folder', type=str, default="data/", help="std of the initial phi table")
    # parser.add_argument('--dataset_name', type=str, default=one, choices=[one,two,three, four, five, six, seven], help="std of the initial phi table")
    parser.add_argument('--dataset_name', type=str, default="cardio", help="std of the initial phi table")
    parser.add_argument('--method', type=str, choices=["ours", "dt", "rf", "gb"], help="std of the initial phi table")
    parser.add_argument('--seed', type=int, default=0, help="std of the initial phi table")
    parser.add_argument('--batch_size', type=int, default=128, help="std of the initial phi table")
    # parser.add_argument('--replay_memory_capacity', type=int, default=20000, help="std of the initial phi table")
    # parser.add_argument('--gamma', type=float, default=0.999, help="std of the initial phi table")
    # parser.add_argument('--epsilon', type=float, default=0.2, help="std of the initial phi table")
    # parser.add_argument('--selected_col_ratio', type=float, default=-1, help="std of the initial phi table")
    # parser.add_argument('--epsilon_falloff', type=float, default=0.9, help="std of the initial phi table")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="std of the initial phi table")
    parser.add_argument('--dropout_p', type=float, default=0, help="std of the initial phi table")
    # parser.add_argument('--target_update', type=int, default=20, help="std of the initial phi table")
    parser.add_argument('--epochs', type=int, default=100, help="std of the initial phi table")
    parser.add_argument('--program_max_len', type=int, default=6, help="std of the initial phi table")
    parser.add_argument('--log_folder', type=str, default="logs/", help="std of the initial phi table")
    # parser.add_argument('--latent_size', type=int, default=100, help="std of the initial phi table")
    # parser.add_argument('--tf_latent_size', type=int, default=30, help="std of the initial phi table")
    # parser.add_argument('--mem_sample_size', type=int, default=16, help="std of the initial phi table")
    parser.add_argument('--model', type=str, default="mlp", help="std of the initial phi table")
    parser.add_argument('--removed_feats_file_name', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--pretrained_model_path', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--rl_algorithm', type=str, default="dqn", choices=["dqn", "ppo"], help="std of the initial phi table")
    parser.add_argument('--topk_act', type=int, default=1, help="std of the initial phi table")

    parser.add_argument('--model_suffix', type=int, default=0, help="std of the initial phi table")
    parser.add_argument('--model_folder', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--cached_model_name', type=str, default=None, help="std of the initial phi table")
    # parser.add_argument('--timesteps_per_batch', type=int, default=5, help="std of the initial phi table")
    # parser.add_argument('--clip', type=float, default=0.2, help="std of the initial phi table")

    # model_config
    parser.add_argument('--model_config', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--not_use_feat_bound_point_ls', action='store_true', help="std of the initial phi table")


    parser.add_argument('--is_log', action='store_true', help='specifies what features to extract')
    parser.add_argument('--log_file_name', type=str, default="logs", help='specifies what features to extract')
    # parser.add_argument('--method_two', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--group_aware', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--do_medical', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--continue_act', action='store_true', help='specifies what features to extract')
    # parser.add_argument('--fix_pretrained_model', action='store_true', help='specifies what features to extract')
    parser.add_argument('--use_precomputed_thres', action='store_true', help='specifies what features to extract')


    # parser.add_argument('--prefer_smaller_range', action='store_true', help='specifies what features to extract')
    parser.add_argument('--use_kg', action='store_true', help='incorporate domain knowledge')
    # parser.add_argument('--prefer_smaller_range_coeff', type=float, default=0.5, help='specifies what features to extract')
    
    # parser.add_argument('--treatment_var_ids', nargs='+', type=int, help='List of integers', default=[0])
    # parser.add_argument('--treatment_var_ids', type=list, default=0.5, help='specifies what features to extract')
    # timesteps_per_batch
    # 
    # # discretize_feat_value_count
    # parser.add_argument('--discretize_feat_value_count', type=int, default=10, help="std of the initial phi table")

    args = parser.parse_args()
    print(args)

    return args