"""学習・評価スクリプトから共通利用する補助関数。

`train.py` や `adapt.py` が具体的なモデルクラスやデータセット実装を直接
知りすぎないように、このファイルで生成処理を一段まとめている。
"""

import random

import numpy as np
import torch

from data_processing import wesad
from models.cnn1d import StressCNN1D


def set_seed(seed):
    """NumPy、Python、PyTorch の乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name="auto"):
    """実行に使う device を返す。

    `auto` の場合は CUDA が使えるときだけ GPU を選び、それ以外は CPU を使う。
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_model(args):
    """設定に従ってモデルを生成する。

    現在は WESAD 用の `cnn1d` のみを実装している。別モデルを追加する場合は、
    `models/` に定義を追加し、この関数に分岐を足す。
    """
    if args.model == "cnn1d":
        return StressCNN1D(
            num_channels=args.num_channels,
            num_classes=getattr(args, "num_classes", 2),
            label_mode=getattr(args, "label_mode", "binary"),
        )
    raise ValueError(f"Unknown model: {args.model}")


def get_dataset(args, subjects_data=None):
    """設定に従って LOSO 用 DataLoader を作成する。"""
    if args.dataset == "wesad":
        return wesad.get_loso_loaders(args, subjects_data=subjects_data)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def get_target_dataset(args):
    """評価専用に target 被験者だけの DataLoader を作成する。"""
    if args.dataset == "wesad":
        return wesad.get_target_loader(args)
    raise ValueError(f"Unknown dataset: {args.dataset}")
