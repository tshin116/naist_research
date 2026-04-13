"""WESAD 1D-CNN 用 OFTTA."""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm import t3a


class WeightedBN1d(nn.Module):
    """source running stats と test batch stats を重み付きで混ぜる BatchNorm1d。"""

    def __init__(self, layer, prior):
        super().__init__()
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        self.layer = layer
        self.layer.eval()
        self.prior = prior
        self.running_mean = deepcopy(layer.running_mean.detach().clone())
        self.running_std = deepcopy(torch.sqrt(layer.running_var.detach().clone()) + layer.eps)

    def forward(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_std = torch.sqrt(x.var(dim=reduce_dims, unbiased=False) + self.layer.eps)
        running_mean = self.running_mean.to(x.device)
        running_std = self.running_std.to(x.device)
        weighted_mean = self.prior * running_mean + (1.0 - self.prior) * batch_mean.detach()
        weighted_std = self.prior * running_std + (1.0 - self.prior) * batch_std.detach()
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        x = (x - weighted_mean.view(*view_shape)) / weighted_std.view(*view_shape)
        return x * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)


def find_bns(model, priors):
    replace_mods = []
    bn_index = 0
    queue = [(model, "")]
    while queue:
        parent, prefix = queue.pop(0)
        for child_name, child in parent.named_children():
            module_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.BatchNorm1d):
                prior = priors[min(bn_index, len(priors) - 1)]
                replace_mods.append((parent, child_name, WeightedBN1d(child, prior)))
                bn_index += 1
            else:
                queue.append((child, module_name))
    return replace_mods


class OFTTA(t3a.t3a):
    """WeightedBN1d と T3A classifier adjustment を組み合わせた OFTTA。"""

    def __init__(self, args, model, optimizer=None, steps=1, episodic=False):
        priors = get_priors(args, model)
        for parent, name, child in find_bns(model, priors):
            setattr(parent, name, child)
        super().__init__(args, model, optimizer=optimizer, steps=steps, episodic=episodic)


def get_priors(args, model):
    prior_max = getattr(args, "oftta_prior_max", 0.99)
    prior_min = getattr(args, "oftta_prior_min", 0.1)
    num_bn = sum(1 for module in model.modules() if isinstance(module, nn.BatchNorm1d))
    if num_bn <= 1:
        return [prior_min]
    factor = (prior_max / prior_min) ** (1 / (num_bn - 1))
    priors = []
    value = prior_min
    for _ in range(num_bn):
        priors.append(value)
        value *= factor
    return priors


def collect_params(model):
    return [], []


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    return model
