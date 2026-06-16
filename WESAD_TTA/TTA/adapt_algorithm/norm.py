"""WESAD 1D-CNN 用の test-time normalization."""

from copy import deepcopy

import torch.nn as nn

from TTA.adapt_algorithm.common import set_dropout_eval


class NORM(nn.Module):
    """BatchNorm1d をテストバッチ統計で動かす適応なし更新系 TTA。"""

    def __init__(self, args, model, eps=1e-5, momentum=0.1, reset_stats=False, no_stats=True):
        super().__init__()
        self.args = args
        self.model = configure_model(
            model,
            eps,
            momentum,
            reset_stats,
            no_stats,
            disable_dropout=getattr(args, "disable_dropout", False),
        )
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def collect_stats(model):
    """BatchNorm1d の running stats を集める。"""
    stats = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            state = module.state_dict()
            if module.affine:
                state.pop("weight", None)
                state.pop("bias", None)
            for stat_name, stat in state.items():
                stats.append(stat)
                names.append(f"{module_name}.{stat_name}")
    return stats, names


def configure_model(model, eps, momentum, reset_stats, no_stats, disable_dropout=False):
    """BatchNorm1d をテストバッチ統計で forward するよう設定する。"""
    if disable_dropout:
        model.eval()
        set_dropout_eval(model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.eps = eps
            module.momentum = momentum
            if reset_stats:
                module.reset_running_stats()
            if no_stats:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    return model
