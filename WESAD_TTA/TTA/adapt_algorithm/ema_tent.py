"""EMA-TENT: BN 統計量に EMA を導入した TENT の改良版。

通常の TENT はテストバッチの統計量でBN を正規化するため、感情状態が
ブロック状に連続する時系列データでは、ブロック切り替え時に統計量が
急変して性能が低下する。EMA-TENT はソースモデルの running stats を
初期値とした EMA で統計量を平滑化し、この問題を緩和する。
"""

from copy import deepcopy

import torch
import torch.nn as nn


class EmaBN1d(nn.Module):
    """BN 統計量に EMA を適用する BatchNorm1d ラッパー。

    forward のたびにバッチ統計量で EMA を更新し、更新後の EMA 統計量で
    正規化する。元の BN 層の weight(γ) と bias(β) はそのまま使い、
    TENT のエントロピー最小化で勾配更新される。
    """

    def __init__(self, layer, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.layer = layer

        self.register_buffer(
            "ema_mean", layer.running_mean.detach().clone()
        )
        self.register_buffer(
            "ema_var", layer.running_var.detach().clone()
        )

        self.layer.track_running_stats = False
        self.layer.running_mean = None
        self.layer.running_var = None

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, unbiased=False)

        self.ema_mean = (
            self.momentum * self.ema_mean + (1.0 - self.momentum) * batch_mean.detach()
        )
        self.ema_var = (
            self.momentum * self.ema_var + (1.0 - self.momentum) * batch_var.detach()
        )

        view_shape = [1, -1] + [1] * (x.ndim - 2)
        x = (x - self.ema_mean.view(*view_shape)) / torch.sqrt(
            self.ema_var.view(*view_shape) + self.layer.eps
        )
        return x * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)


def softmax_entropy(logits):
    """多クラス logits からカテゴリカルエントロピーを計算する。"""
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


class EmaTent(nn.Module):
    """EMA-BN + エントロピー最小化による TTA。"""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        if steps <= 0:
            raise ValueError("ema_tent requires at least one adaptation step")
        self.model_state, self.optimizer_state = (
            deepcopy(model.state_dict()),
            deepcopy(optimizer.state_dict()),
        )

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    """エントロピー最小化で EmaBN1d の γ/β を更新する。"""
    model.train()
    outputs = model(x)
    if isinstance(outputs, tuple):
        outputs, _ = outputs
    loss = softmax_entropy(outputs).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
    return outputs


def _replace_bn_with_ema(model, momentum):
    """モデル内の全 BatchNorm1d を EmaBN1d に差し替える。"""
    replacements = []
    queue = [(model, "")]
    while queue:
        parent, prefix = queue.pop(0)
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm1d):
                replacements.append((parent, name, EmaBN1d(child, momentum)))
            else:
                module_name = f"{prefix}.{name}" if prefix else name
                queue.append((child, module_name))
    for parent, name, new_module in replacements:
        setattr(parent, name, new_module)


def configure_model(model, momentum=0.9):
    """EMA-TENT 用にモデルを設定する。

    全 BatchNorm1d を EmaBN1d に差し替え、γ/β だけを勾配更新対象にする。
    """
    _replace_bn_with_ema(model, momentum)
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, EmaBN1d):
            module.layer.weight.requires_grad_(True)
            module.layer.bias.requires_grad_(True)
    return model


def collect_params(model):
    """EmaBN1d 内の weight/bias を集める。"""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, EmaBN1d):
            for param_name, param in module.layer.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.layer.{param_name}")
    return params, names
