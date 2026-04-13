"""WESAD 1D-CNN 用 SAR-style test-time adaptation."""

from copy import deepcopy
import math

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import (
    binary_entropy_from_logits,
    collect_bn1d_params,
    configure_bn1d_for_adaptation,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    safe_mean_loss,
)


def update_ema(ema, new_data, momentum=0.9):
    if ema is None:
        return new_data
    return momentum * ema + (1.0 - momentum) * new_data


class SAR(nn.Module):
    """信頼できる低 entropy サンプルで SAM 更新する SAR。"""

    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.margin_e0 = getattr(args, "sar_margin", 0.4 * math.log(getattr(args, "num_classes", 2)))
        self.reset_constant_em = getattr(args, "sar_reset_constant", 0.2)
        self.ema = None
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs, self.ema, reset_flag = forward_and_adapt_sar(
                x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema
            )
            if reset_flag:
                self.reset()
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None


@torch.enable_grad()
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    optimizer.zero_grad()
    model.train()
    logits = model(x)
    entropies = binary_entropy_from_logits(logits)
    selected = entropies[entropies < margin]
    loss = safe_mean_loss(selected, x.device)
    loss.backward()

    optimizer.first_step(zero_grad=True)
    logits_second = model(x)
    entropies_second = binary_entropy_from_logits(logits_second)
    selected_second = entropies_second[entropies_second < margin]
    loss_second = safe_mean_loss(selected_second, x.device)
    if selected_second.numel() > 0:
        ema = update_ema(ema, float(loss_second.detach().cpu()))
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    reset_flag = ema is not None and ema < reset_constant
    with torch.no_grad():
        outputs = model(x)
    return outputs, ema, reset_flag


def collect_params(model):
    return collect_bn1d_params(model)


def configure_model(model):
    return configure_bn1d_for_adaptation(model, train=True, update_params=True, use_batch_stats=True)


class SAM(torch.optim.Optimizer):
    """SAR で使う Sharpness-Aware Minimization optimizer。"""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                e_w = (torch.pow(param, 2) if group["adaptive"] else 1.0) * param.grad * scale.to(param)
                param.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data = self.state[param]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        grads = [
            ((torch.abs(param) if group["adaptive"] else 1.0) * param.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not grads:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(grads), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
