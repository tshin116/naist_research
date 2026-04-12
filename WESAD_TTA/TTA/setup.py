"""Test-Time Adaptation 手法を選択してモデルを組み立てる。"""

import torch

from TTA.adapt_algorithm import tent


def setup_source(args, model):
    """適応なしの source モデルとして評価モードにする。"""
    model.eval()
    return model


def setup_tent(args, model):
    """Tent 用に BatchNorm1d だけを更新可能にしてラップする。"""
    model = tent.configure_model(model)
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return tent.Tent(model, optimizer, steps=args.tent_steps, episodic=args.episodic)


def get_adaptation(args, base_model):
    """`args.adaption` に応じて source または Tent のモデルを返す。"""
    if args.adaption == "source":
        return setup_source(args, base_model)
    if args.adaption == "tent":
        return setup_tent(args, base_model)
    raise ValueError(f"Unknown adaptation: {args.adaption}")
