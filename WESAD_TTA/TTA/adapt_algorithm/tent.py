"""WESAD の 1D-CNN 向け Tent 実装。"""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import entropy_from_logits, set_dropout_eval


class Tent(nn.Module):
    """forward 時にテストバッチで自己適応するラッパーモジュール。"""

    def __init__(self, model, optimizer, steps=1, episodic=False, label_mode="binary", disable_dropout=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.label_mode = label_mode
        self.disable_dropout = disable_dropout
        if steps <= 0:
            raise ValueError("tent requires at least one adaptation step")

        # episodic=True のときに各バッチ前の状態へ戻せるよう、初期状態を保持する。
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        """入力バッチで `steps` 回 Tent 更新を行い、更新前 forward の logits を返す。"""
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(
                x,
                self.model,
                self.optimizer,
                self.label_mode,
                self.disable_dropout,
            )
        return outputs

    def reset(self):
        """保存しておいた初期状態へモデルと optimizer を戻す。"""
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, label_mode, disable_dropout=False):
    """元の Tent 実装に合わせ、forward 出力を返した後続バッチへ更新を反映する。"""
    model.train()
    if disable_dropout:
        set_dropout_eval(model)

    outputs = model(x)

    if isinstance(outputs, tuple):
        outputs, _ = outputs

    loss = entropy_from_logits(outputs, label_mode=label_mode).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


def collect_params(model):
    """Tent で更新する BatchNorm1d の affine パラメータだけを集める。"""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            for param_name, param in module.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def configure_model(model, disable_dropout=False):
    """Tent 用にモデルを設定する。

    モデル全体を train mode にしたうえで、勾配更新は BatchNorm1d の
    weight と bias だけに限定する。また running statistics は使わず、
    テストバッチ自身の統計を使うようにする。
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    if disable_dropout:
        set_dropout_eval(model)
    check_model(model)
    return model


def check_model(model):
    """Tent の前提条件を満たしているか確認する。"""
    if not model.training:
        raise AssertionError("tent needs train mode")

    param_grads = [param.requires_grad for param in model.parameters()]
    if not any(param_grads):
        raise AssertionError("tent needs parameters to update")
    if all(param_grads):
        raise AssertionError("tent should not update all parameters")
    if not any(isinstance(module, nn.BatchNorm1d) for module in model.modules()):
        raise AssertionError("tent needs BatchNorm1d layers")


def copy_model_and_optimizer(model, optimizer):
    """reset 用にモデルと optimizer の状態を deepcopy する。"""
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """保存済み状態をモデルと optimizer に読み戻す。"""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
