"""WESAD の 1D-CNN TTA 実装で共有する関数。

二値分類では既存 checkpoint と互換性を保つため、モデルは single-logit を返す。
TTA アルゴリズム内ではそれを 2 クラス logits に変換して扱う。3分類ではモデルが
最初から multi-class logits を返すため、そのまま softmax 系の処理に渡す。
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_logits_to_two_class(logits):
    """single-logit を 2 クラス logits に変換する。

    `BCEWithLogitsLoss` の logit は `log p(y=1)/p(y=0)` を表すため、
    `[-logit/2, logit/2]` とすれば softmax のクラス 1 確率が sigmoid(logit) と一致する。
    """
    if logits.ndim == 2 and logits.size(1) == 2:
        return logits
    logits = logits.view(-1)
    return torch.stack([-0.5 * logits, 0.5 * logits], dim=1)


def two_class_to_binary_logits(two_class_logits):
    """2 クラス logits を single-logit に戻す。"""
    if two_class_logits.ndim == 1:
        return two_class_logits
    return two_class_logits[:, 1] - two_class_logits[:, 0]


def two_class_probs_to_binary_logits(two_class_probs):
    """2 クラス確率を single-logit に変換する。"""
    probs = two_class_probs.clamp_min(1e-12)
    return torch.log(probs[:, 1]) - torch.log(probs[:, 0])


def label_mode_from_args(args):
    """設定から分類モードを取り出す。未指定なら従来の二値分類として扱う。"""
    return getattr(args, "label_mode", "binary")


def logits_to_class_logits(logits, label_mode="binary"):
    """モデル出力を softmax で扱える class logits に変換する。"""
    if label_mode == "binary":
        return binary_logits_to_two_class(logits)
    return logits


def class_logits_to_model_logits(class_logits, label_mode="binary"):
    """class logits をモデルの出力形式に戻す。"""
    if label_mode == "binary":
        return two_class_to_binary_logits(class_logits)
    return class_logits


def probs_to_model_logits(probs, label_mode="binary"):
    """クラス確率をモデルの出力形式に変換する。"""
    if label_mode == "binary":
        return two_class_probs_to_binary_logits(probs)
    return torch.log(probs.clamp_min(1e-12))


def entropy_from_logits(logits, label_mode="binary"):
    """モデル出力からサンプルごとの entropy を返す。"""
    probs = torch.softmax(logits_to_class_logits(logits, label_mode), dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def binary_entropy_from_logits(logits):
    """後方互換用。single-logit または 2 クラス logits から entropy を返す。"""
    return entropy_from_logits(logits, label_mode="binary")


def softmax_entropy(class_logits):
    """multi-class logits 用 entropy。"""
    probs = torch.softmax(class_logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def split_model_output(output):
    """モデル出力から single-logit と feature を取り出す。"""
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


def classifier_weights(model, label_mode="binary"):
    """最終線形層から class logits 用の重みと bias を返す。"""
    linear = get_final_linear(model)
    weight = linear.weight
    if label_mode == "binary" and weight.size(0) == 1:
        pos_w = 0.5 * weight[0]
        neg_w = -0.5 * weight[0]
        if linear.bias is None:
            pos_b = neg_b = torch.zeros((), device=weight.device, dtype=weight.dtype)
        else:
            pos_b = 0.5 * linear.bias[0]
            neg_b = -0.5 * linear.bias[0]
        return torch.stack([neg_w, pos_w], dim=0), torch.stack([neg_b, pos_b], dim=0)
    return weight, linear.bias


def binary_classifier_weights(model):
    """後方互換用。single-logit classifier から 2 クラス用の重みと bias を作る。"""
    return classifier_weights(model, label_mode="binary")


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


def pseudo_label_loss(logits, confidence_threshold=0.9, label_mode="binary"):
    """高信頼サンプルだけを使う pseudo-label loss。"""
    class_logits = logits_to_class_logits(logits, label_mode)
    probs = torch.softmax(class_logits, dim=1)
    confidence, pseudo = probs.max(dim=1)
    mask = confidence > confidence_threshold
    if mask.any():
        return F.cross_entropy(class_logits[mask], pseudo[mask])
    return torch.zeros((), device=class_logits.device, requires_grad=True)
