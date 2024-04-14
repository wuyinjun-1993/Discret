
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from ft_transformer import FTTransformer

import torch

import logging


def load_ft_transformer(model_args, num_numerical, unique_categories, num_outputs, device):
    model_path = model_args.model_path
    d_embedding = model_args.d_embedding
    epoch = 0
    optimizer = None
    net = FTTransformer(num_numerical, unique_categories, num_outputs, d_embedding, model_args)
    # net = get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_args)
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]

    return net, epoch, optimizer