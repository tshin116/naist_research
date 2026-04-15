"""Test-Time Adaptation 手法を選択してモデルを組み立てる。"""

import torch

from TTA.adapt_algorithm import ema_tent, oftta, t3a, tent


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


def setup_t3a(args, model):
    model = t3a.configure_model(model)
    return t3a.t3a(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def setup_ema_tent(args, model):
    """EMA-TENT 用に EmaBN1d を差し替え、γ/β を optimizer に登録する。"""
    momentum = getattr(args, "ema_momentum", 0.9)
    model = ema_tent.configure_model(model, momentum=momentum)
    params, _ = ema_tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return ema_tent.EmaTent(model, optimizer, steps=args.tent_steps, episodic=args.episodic)


def setup_oftta(args, model):
    model = oftta.configure_model(model)
    return oftta.OFTTA(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def get_adaptation(args, base_model):
    """`args.adaption` に応じて TTA モデルを返す。"""
    if args.adaption == "source":
        return setup_source(args, base_model)
    if args.adaption == "tent":
        return setup_tent(args, base_model)
    if args.adaption == "t3a":
        return setup_t3a(args, base_model)
    if args.adaption == "ema_tent":
        return setup_ema_tent(args, base_model)
    if args.adaption == "oftta":
        return setup_oftta(args, base_model)
    raise ValueError(f"Unknown adaptation: {args.adaption}. Supported: source, tent, ema_tent, t3a, oftta")
