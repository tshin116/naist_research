"""WESAD の 1D-CNN TTA 実装で共有する関数。"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(logits):
    """多クラス logits からサンプルごとの entropy を返す。"""
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def split_model_output(output):
    """モデル出力から logits と feature を取り出す。"""
    if isinstance(output, tuple):
        logits, feature = output
        return logits, feature
    return output, None


def forward_with_features(model, x):
    """feature 取得に対応した forward を実行する。"""
    try:
        output = model(x, return_feature=True)
    except TypeError:
        output = model(x)
    return split_model_output(output)


def get_final_linear(model):
    """WESAD 1D-CNN の最後の線形分類層を返す。"""
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        for module in reversed(classifier):
            if isinstance(module, nn.Linear):
                return module
    if isinstance(classifier, nn.Linear):
        return classifier
    raise ValueError("A final nn.Linear classifier is required for this adaptation.")


def classifier_weights(model):
    """分類層の重みと bias を返す。"""
    linear = get_final_linear(model)
    return linear.weight.detach().clone(), linear.bias.detach().clone() if linear.bias is not None else None


def collect_bn1d_params(model):
    """BatchNorm1d の affine パラメータだけを集める。"""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            for param_name, param in module.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def configure_bn1d_for_adaptation(model, train=True, update_params=True, use_batch_stats=True):
    """BN1d を TTA 用に設定する。"""
    model.train(train)
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train(train)
            module.requires_grad_(update_params)
            if use_batch_stats:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    return model


def copy_model_and_optimizer(model, optimizer=None):
    """reset 用に model と optimizer の状態を保存する。"""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """保存済み状態を読み戻す。"""
    model.load_state_dict(model_state, strict=True)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)


def safe_mean_loss(loss_values, device):
    """空 tensor になった場合でも backward 可能な 0 loss を返す。"""
    if loss_values.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return loss_values.mean()


def pseudo_label_loss(logits, confidence_threshold=0.9):
    """高信頼サンプルだけを使う pseudo-label loss。"""
    probs = torch.softmax(logits, dim=1)
    confidence, pseudo = probs.max(dim=1)
    mask = confidence > confidence_threshold
    if mask.any():
        return F.cross_entropy(logits[mask], pseudo[mask])
    return torch.zeros((), device=logits.device, requires_grad=True)
