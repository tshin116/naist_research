"""Test-Time Adaptation 手法を選択してモデルを組み立てる。"""

import torch

from TTA.adapt_algorithm import delta, dua, ema_tent, mem_oftta, mi_dynamic_ema_tent, norm, note, oftta, pl, realistic_tta, rotta, sar, shot, t3a, tast, tast_bn, tema, tent


def setup_source(args, model):
    """適応なしの source モデルとして評価モードにする。"""
    model.eval()
    return model


def setup_tent(args, model):
    """Tent 用に BatchNorm1d だけを更新可能にしてラップする。"""
    model = tent.configure_model(model, disable_dropout=getattr(args, "disable_dropout", False))
    params, _ = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return tent.Tent(
        model,
        optimizer,
        steps=args.tent_steps,
        episodic=args.episodic,
        label_mode=getattr(args, "label_mode", "binary"),
        disable_dropout=getattr(args, "disable_dropout", False),
    )


def setup_ema_tent(args, model):
    """EMA-Tent 用に BatchNorm1d 統計を source/EMA/current batch の混合にする。"""
    model = ema_tent.configure_model(model, args)
    params, _ = ema_tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return ema_tent.EMATent(
        model,
        optimizer,
        steps=args.tent_steps,
        episodic=args.episodic,
        label_mode=getattr(args, "label_mode", "binary"),
        args=args,
    )


def setup_mi_dynamic_ema_tent(args, model):
    """MI-gated Dynamic EMA-Tent 用に BN1d 統計混合を設定する。"""
    model = mi_dynamic_ema_tent.configure_model(model, args)
    params, _ = mi_dynamic_ema_tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.tent_lr, betas=(0.9, 0.999), weight_decay=0.0)
    return mi_dynamic_ema_tent.MIDynamicEMATent(
        model,
        optimizer,
        steps=args.tent_steps,
        episodic=args.episodic,
        label_mode=getattr(args, "label_mode", "binary"),
        args=args,
    )


def setup_norm(args, model):
    return norm.NORM(args, model)


def setup_realistic_tta(args, model):
    return realistic_tta.RealisticTTA(args, model)


def setup_tema(args, model):
    return tema.TEMA(args, model)


def setup_dua(args, model):
    return dua.DUA(args, model)


def setup_note(args, model):
    return note.NOTE(args, model)


def setup_rotta(args, model):
    return rotta.RoTTA(args, model)


def setup_delta(args, model):
    return delta.DELTA(args, model)


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
    if args.adaption == "ema_tent":
        return setup_ema_tent(args, base_model)
    if args.adaption == "mi_dynamic_ema_tent":
        return setup_mi_dynamic_ema_tent(args, base_model)
    if args.adaption == "norm":
        return setup_norm(args, base_model)
    if args.adaption == "realistic_tta":
        return setup_realistic_tta(args, base_model)
    if args.adaption == "tema":
        return setup_tema(args, base_model)
    if args.adaption == "dua":
        return setup_dua(args, base_model)
    if args.adaption == "note":
        return setup_note(args, base_model)
    if args.adaption == "rotta":
        return setup_rotta(args, base_model)
    if args.adaption == "delta":
        return setup_delta(args, base_model)
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
