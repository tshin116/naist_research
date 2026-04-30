"""EMA-Tent for WESAD 1D-CNN.

This Tent variant replaces BatchNorm1d with a source/test-EMA/current-batch
normalizer. It keeps Tent's entropy minimization on BN affine parameters, but
uses batch-diversity gating to reduce the influence of class-biased test batches.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import entropy_from_logits, logits_to_class_logits


class EMABatchNorm1d(nn.Module):
    """BatchNorm1d using source stats, past test EMA stats, and current batch stats."""

    def __init__(self, layer, source_weight, ema_weight, batch_weight, momentum, eps=None):
        super().__init__()
        total = source_weight + ema_weight + batch_weight
        if total <= 0:
            raise ValueError("EMA-Tent BN weights must have a positive sum")
        self.layer = layer
        self.layer.eval()
        self.source_weight = float(source_weight / total)
        self.ema_weight = float(ema_weight / total)
        self.batch_weight = float(batch_weight / total)
        self.momentum = float(momentum)
        self.eps = layer.eps if eps is None else eps
        self.gate = 1.0

        source_mean = layer.running_mean.detach().clone()
        source_std = torch.sqrt(layer.running_var.detach().clone() + self.eps)
        self.register_buffer("source_mean", source_mean)
        self.register_buffer("source_std", source_std)
        self.register_buffer("ema_mean", source_mean.clone())
        self.register_buffer("ema_std", source_std.clone())
        self.register_buffer("last_batch_mean", source_mean.clone())
        self.register_buffer("last_batch_std", source_std.clone())

    def set_gate(self, gate):
        self.gate = float(max(0.0, min(1.0, gate)))

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_std = torch.sqrt(x.var(dim=reduce_dims, unbiased=False) + self.eps)
        self.last_batch_mean = batch_mean.detach()
        self.last_batch_std = batch_std.detach()

        batch_weight = self.batch_weight * self.gate
        source_weight = self.source_weight + self.batch_weight * (1.0 - self.gate)
        ema_weight = self.ema_weight
        total = source_weight + ema_weight + batch_weight
        source_weight = source_weight / total
        ema_weight = ema_weight / total
        batch_weight = batch_weight / total

        source_mean = self.source_mean.to(x.device)
        source_std = self.source_std.to(x.device)
        ema_mean = self.ema_mean.to(x.device)
        ema_std = self.ema_std.to(x.device)

        mixed_mean = (
            source_weight * source_mean
            + ema_weight * ema_mean
            + batch_weight * batch_mean.detach()
        )
        mixed_std = (
            source_weight * source_std
            + ema_weight * ema_std
            + batch_weight * batch_std.detach()
        )
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        x = (x - mixed_mean.view(*view_shape)) / mixed_std.view(*view_shape)
        return x * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)

    @torch.no_grad()
    def update_ema(self):
        device = self.ema_mean.device
        batch_mean = self.last_batch_mean.to(device)
        batch_std = self.last_batch_std.to(device)
        self.ema_mean.mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)
        self.ema_std.mul_(self.momentum).add_(batch_std, alpha=1.0 - self.momentum)


class EMATent(nn.Module):
    """Tent wrapper with EMA-smoothed test-time BatchNorm statistics."""

    def __init__(self, model, optimizer, steps=1, episodic=False, label_mode="binary", args=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.label_mode = label_mode
        self.args = args
        if steps <= 0:
            raise ValueError("ema_tent requires at least one adaptation step")

        self.use_gate = getattr(args, "ema_tent_use_gate", True)
        self.gate_threshold = getattr(args, "ema_tent_gate_threshold", 0.2)
        self.gate_power = getattr(args, "ema_tent_gate_power", 1.0)
        self.min_confidence = getattr(args, "ema_tent_min_confidence", 0.0)
        self.loss_gate = getattr(args, "ema_tent_loss_gate", True)
        self.probe_gate = getattr(args, "ema_tent_probe_gate", 0.0)
        self.bn_layers = [module for module in model.modules() if isinstance(module, EMABatchNorm1d)]

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.bn_layers = [module for module in self.model.modules() if isinstance(module, EMABatchNorm1d)]

    def set_bn_gate(self, gate):
        for layer in self.bn_layers:
            layer.set_gate(gate)

    @torch.no_grad()
    def update_ema(self):
        for layer in self.bn_layers:
            layer.update_ema()


@torch.enable_grad()
def forward_and_adapt(x, wrapper):
    """Return current-batch logits, update BN affine, then update EMA stats."""
    model = wrapper.model
    optimizer = wrapper.optimizer
    model.train()

    gate = 1.0
    confidence = None
    if wrapper.use_gate:
        gate, confidence = estimate_batch_gate(x, wrapper)

    wrapper.set_bn_gate(gate)
    outputs = model(x)
    if isinstance(outputs, tuple):
        outputs, _ = outputs

    should_update = gate >= wrapper.gate_threshold
    if confidence is not None:
        should_update = should_update and confidence >= wrapper.min_confidence

    if should_update:
        loss = entropy_from_logits(outputs, label_mode=wrapper.label_mode).mean()
        if wrapper.loss_gate:
            loss = loss * gate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        optimizer.zero_grad()

    wrapper.update_ema()
    return outputs


@torch.no_grad()
def estimate_batch_gate(x, wrapper):
    wrapper.set_bn_gate(wrapper.probe_gate)
    outputs = wrapper.model(x)
    if isinstance(outputs, tuple):
        outputs, _ = outputs
    class_logits = logits_to_class_logits(outputs, wrapper.label_mode)
    probs = torch.softmax(class_logits, dim=1)
    pred_dist = probs.mean(dim=0)
    entropy = -(pred_dist * torch.log(pred_dist.clamp_min(1e-12))).sum()
    diversity = entropy / torch.log(torch.tensor(float(probs.size(1)), device=probs.device))
    confidence = probs.max(dim=1).values.mean()
    gate = diversity.clamp(0.0, 1.0).pow(wrapper.gate_power)
    return float(gate.item()), float(confidence.item())


def collect_params(model):
    """Collect affine parameters of wrapped EMA BatchNorm1d layers."""
    params = []
    names = []
    for module_name, module in model.named_modules():
        if isinstance(module, EMABatchNorm1d):
            for param_name, param in module.layer.named_parameters():
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.layer.{param_name}")
    return params, names


def configure_model(model, args):
    """Replace BatchNorm1d with EMA BatchNorm and enable only BN affine updates."""
    model.train()
    model.requires_grad_(False)
    source_weight = getattr(args, "ema_tent_source_weight", 0.3)
    ema_weight = getattr(args, "ema_tent_ema_weight", 0.5)
    batch_weight = getattr(args, "ema_tent_batch_weight", 0.2)
    momentum = getattr(args, "ema_tent_momentum", 0.9)

    for parent, name, child in find_bns(model):
        setattr(parent, name, EMABatchNorm1d(child, source_weight, ema_weight, batch_weight, momentum))

    for module in model.modules():
        if isinstance(module, EMABatchNorm1d):
            module.layer.requires_grad_(True)
    check_model(model)
    return model


def find_bns(model):
    replace_mods = []
    queue = [(model, "")]
    while queue:
        parent, prefix = queue.pop(0)
        for child_name, child in parent.named_children():
            module_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.BatchNorm1d):
                replace_mods.append((parent, child_name, child))
            else:
                queue.append((child, module_name))
    return replace_mods


def check_model(model):
    if not model.training:
        raise AssertionError("ema_tent needs train mode")
    param_grads = [param.requires_grad for param in model.parameters()]
    if not any(param_grads):
        raise AssertionError("ema_tent needs parameters to update")
    if all(param_grads):
        raise AssertionError("ema_tent should not update all parameters")
    if not any(isinstance(module, EMABatchNorm1d) for module in model.modules()):
        raise AssertionError("ema_tent needs EMABatchNorm1d layers")


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
