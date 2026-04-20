"""Test-Time Adaptation 手法を選択してモデルを組み立てる。"""

import torch

from TTA.adapt_algorithm import mem_oftta, norm, oftta, pl, sar, shot, t3a, tast, tast_bn, tent


def setup_source(args, model):
    """適応なしの source モデルとして評価モードにする。"""
    model.eval()
    return model


def setup_tent(args, model):
    """Tent 用に BatchNorm1d だけを更新可能にしてラップする。"""
    model = tent.configure_model(model)
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return tent.Tent(
        model,
        optimizer,
        steps=args.tent_steps,
        episodic=args.episodic,
        label_mode=getattr(args, "label_mode", "binary"),
    )


def setup_norm(args, model):
    return norm.NORM(args, model)


def setup_pl(args, model):
    model = pl.configure_model(model)
    params, _ = pl.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tta_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return pl.PL(model=model, args=args, optimizer=optimizer, steps=args.tta_steps, episodic=args.episodic)


def setup_shot(args, model):
    model = shot.configure_model(model)
    params, _ = shot.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tta_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return shot.SHOT(model=model, args=args, optimizer=optimizer, steps=args.tta_steps, episodic=args.episodic)


def setup_sar(args, model):
    model = sar.configure_model(model)
    params, _ = sar.collect_params(model)
    optimizer = sar.SAM(params, torch.optim.SGD, lr=args.tta_lr, momentum=0.9)
    return sar.SAR(model=model, args=args, optimizer=optimizer, steps=args.tta_steps, episodic=args.episodic)


def setup_t3a(args, model):
    model = t3a.configure_model(model)
    return t3a.t3a(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def setup_oftta(args, model):
    model = oftta.configure_model(model)
    return oftta.OFTTA(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def setup_mem_oftta(args, model):
    model = mem_oftta.configure_model(model)
    return mem_oftta.MemOFTTA(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def setup_tast(args, model):
    model = tast.configure_model(model)
    return tast.TAST(args=args, model=model, optimizer=None, steps=args.tta_steps, episodic=args.episodic)


def setup_tast_bn(args, model):
    model = tast_bn.configure_model(model)
    params, _ = tast_bn.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tta_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return tast_bn.TAST_BN(args=args, model=model, optimizer=optimizer, steps=args.tta_steps, episodic=args.episodic)


def get_adaptation(args, base_model):
    """`args.adaption` に応じて TTA モデルを返す。"""
    if args.adaption == "source":
        return setup_source(args, base_model)
    if args.adaption == "tent":
        return setup_tent(args, base_model)
    if args.adaption == "norm":
        return setup_norm(args, base_model)
    if args.adaption == "pl":
        return setup_pl(args, base_model)
    if args.adaption == "shot":
        return setup_shot(args, base_model)
    if args.adaption == "sar":
        return setup_sar(args, base_model)
    if args.adaption == "t3a":
        return setup_t3a(args, base_model)
    if args.adaption == "oftta":
        return setup_oftta(args, base_model)
    if args.adaption == "mem_oftta":
        return setup_mem_oftta(args, base_model)
    if args.adaption == "tast":
        return setup_tast(args, base_model)
    if args.adaption == "tast_bn":
        return setup_tast_bn(args, base_model)
    raise ValueError(f"Unknown adaptation: {args.adaption}")
