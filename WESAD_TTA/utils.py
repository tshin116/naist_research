import random

import numpy as np
import torch

from data_processing import wesad
from models.cnn1d import StressCNN1D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_model(args):
    if args.model == "cnn1d":
        return StressCNN1D(num_channels=args.num_channels)
    raise ValueError(f"Unknown model: {args.model}")


def get_dataset(args, subjects_data=None):
    if args.dataset == "wesad":
        return wesad.get_loso_loaders(args, subjects_data=subjects_data)
    raise ValueError(f"Unknown dataset: {args.dataset}")
