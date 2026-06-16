"""RealisticTTA-style BatchNorm rectification for WESAD 1D-CNN.

This is a BatchNorm1d adaptation of the TTBN/BayesianBatchNorm implementation
in `/home/shinsaku-t/work/naist_reserch/RealisticTTA`. It performs no gradient
updates. Each BatchNorm layer keeps source statistics and target running
statistics, then mixes them with a layer-wise prior estimated from the
source-target statistic divergence.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from TTA.adapt_algorithm.common import set_dropout_eval


class RealisticBatchNorm1d(nn.Module):
    """Layer-wise source/target BN statistic rectification."""

    def __init__(self, layer, target_momentum=0.1, prior_scale=0.5, eps=None):
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.target_momentum = float(target_momentum)
        self.prior_scale = float(prior_scale)
        self.eps = layer.eps if eps is None else eps

        source_mean = layer.running_mean.detach().clone()
        source_var = layer.running_var.detach().clone()
        self.register_buffer("source_mean", source_mean)
        self.register_buffer("source_var", source_var)
        self.register_buffer("target_mean", source_mean.clone())
        self.register_buffer("target_var", source_var.clone())
        self.register_buffer("prior", torch.full((), self.prior_scale))
        self.register_buffer("divergence", torch.zeros(()))

    @torch.no_grad()
    def update_target_stats(self, x):
        reduce_dims = [0] + list(range(2, x.ndim))
        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, unbiased=False)
        self.target_mean.mul_(1.0 - self.target_momentum).add_(batch_mean.detach(), alpha=self.target_momentum)
        self.target_var.mul_(1.0 - self.target_momentum).add_(batch_var.detach(), alpha=self.target_momentum)
        self.divergence.copy_(symmetric_diag_gaussian_kl(self.source_mean, self.source_var, self.target_mean, self.target_var))

    def set_prior(self, prior):
        prior_tensor = torch.as_tensor(prior, device=self.prior.device, dtype=self.prior.dtype).clamp(0.0, 1.0)
        self.prior.copy_(prior_tensor)

    def forward(self, x):
        self.update_target_stats(x)
        prior = self.prior.to(x.device, dtype=x.dtype)
        source_mean = self.source_mean.to(x.device, dtype=x.dtype)
        source_var = self.source_var.to(x.device, dtype=x.dtype)
        target_mean = self.target_mean.to(x.device, dtype=x.dtype)
        target_var = self.target_var.to(x.device, dtype=x.dtype)

        mixed_mean = prior * source_mean + (1.0 - prior) * target_mean
        mixed_var = (
            prior * source_var
            + (1.0 - prior) * target_var
            + prior * (1.0 - prior) * (source_mean - target_mean).pow(2)
        )
        view_shape = [1, -1] + [1] * (x.ndim - 2)
        normalized = (x - mixed_mean.view(*view_shape)) / torch.sqrt(mixed_var.view(*view_shape) + self.eps)
        return normalized * self.layer.weight.view(*view_shape) + self.layer.bias.view(*view_shape)


class RealisticTTA(nn.Module):
    """RealisticTTA wrapper for evaluation-time BN rectification."""

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        self.bn_layers = [module for module in self.model.modules() if isinstance(module, RealisticBatchNorm1d)]
        self.prior_ema = torch.zeros(len(self.bn_layers), device=next(self.model.parameters()).device)
        self.prior_ema_initialized = False
        self.prior_ema_momentum = float(getattr(args, "realistic_tta_prior_ema_momentum", 0.9))
        self.prior_scale = float(getattr(args, "realistic_tta_prior_scale", 0.5))
        self.diagnostics = []
        self._batch_index = 0
        self.model_state = deepcopy(self.model.state_dict())

    @torch.no_grad()
    def forward(self, x):
        # Probe pass updates target statistics and measures per-layer divergence.
        _ = self.model(x)
        divergences = torch.stack([layer.divergence.detach() for layer in self.bn_layers]) if self.bn_layers else torch.empty(0, device=x.device)
        priors = normalized_layer_priors(divergences, self.prior_scale)
        for layer, prior in zip(self.bn_layers, priors):
            layer.set_prior(prior)

        outputs = self.model(x)

        if priors.numel() > 0:
            if not self.prior_ema_initialized:
                self.prior_ema = priors.detach().clone()
                self.prior_ema_initialized = True
            else:
                self.prior_ema.mul_(self.prior_ema_momentum).add_(priors.detach(), alpha=1.0 - self.prior_ema_momentum)
            for layer, prior in zip(self.bn_layers, self.prior_ema):
                layer.set_prior(prior)

        self.add_diagnostic(divergences, priors)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.bn_layers = [module for module in self.model.modules() if isinstance(module, RealisticBatchNorm1d)]
        self.prior_ema = torch.zeros(len(self.bn_layers), device=next(self.model.parameters()).device)
        self.prior_ema_initialized = False
        self.diagnostics = []
        self._batch_index = 0

    def add_diagnostic(self, divergences, priors):
        row = {
            "batch_index": self._batch_index,
            "realistic_divergence_mean": float(divergences.mean().item()) if divergences.numel() else 0.0,
            "realistic_divergence_min": float(divergences.min().item()) if divergences.numel() else 0.0,
            "realistic_divergence_max": float(divergences.max().item()) if divergences.numel() else 0.0,
            "realistic_prior_mean": float(priors.mean().item()) if priors.numel() else 0.0,
            "realistic_prior_min": float(priors.min().item()) if priors.numel() else 0.0,
            "realistic_prior_max": float(priors.max().item()) if priors.numel() else 0.0,
        }
        self._batch_index += 1
        self.diagnostics.append(row)

    def record_batch_labels(self, y_batch):
        if not self.diagnostics:
            return
        labels = y_batch.detach().cpu().reshape(-1).to(torch.long)
        if labels.numel() == 0:
            return
        num_classes = max(int(labels.max().item()) + 1, getattr(self.args, "num_classes", 1))
        counts = torch.bincount(labels, minlength=num_classes).tolist()
        total = float(sum(counts))
        self.diagnostics[-1]["batch_size"] = int(total)
        for class_index, count in enumerate(counts):
            self.diagnostics[-1][f"true_class_{class_index}_count"] = int(count)
            self.diagnostics[-1][f"true_class_{class_index}_ratio"] = float(count / total) if total else 0.0
        self.diagnostics[-1]["true_max_class_ratio"] = float(max(counts) / total) if total else 0.0
        self.diagnostics[-1]["true_num_present_classes"] = int(sum(1 for count in counts if count > 0))

    def get_diagnostics(self):
        return list(self.diagnostics)


def symmetric_diag_gaussian_kl(mean_a, var_a, mean_b, var_b):
    var_a = var_a.clamp_min(1e-6)
    var_b = var_b.clamp_min(1e-6)
    diff = (mean_a - mean_b).pow(2)
    kl_ab = 0.5 * ((var_a / var_b) + (diff / var_b) - 1.0 + torch.log(var_b / var_a)).sum()
    kl_ba = 0.5 * ((var_b / var_a) + (diff / var_a) - 1.0 + torch.log(var_a / var_b)).sum()
    return 0.5 * (kl_ab + kl_ba)


def normalized_layer_priors(divergences, prior_scale):
    if divergences.numel() == 0:
        return divergences
    if divergences.numel() == 1 or torch.std(divergences, unbiased=False) < 1e-12:
        return torch.full_like(divergences, 0.5 * prior_scale)
    z = (divergences - divergences.mean()) / divergences.std(unbiased=False).clamp_min(1e-12)
    z = z.clamp(-1.0, 1.0)
    return ((z + 1.0) * 0.5 * prior_scale).clamp(0.0, 1.0)


def configure_model(model, args):
    model.eval()
    model.requires_grad_(False)
    target_momentum = getattr(args, "realistic_tta_target_momentum", 0.1)
    prior_scale = getattr(args, "realistic_tta_prior_scale", 0.5)
    for parent, name, child in find_bns(model):
        setattr(parent, name, RealisticBatchNorm1d(child, target_momentum=target_momentum, prior_scale=prior_scale))
    if getattr(args, "disable_dropout", False):
        set_dropout_eval(model)
    return model


def find_bns(model):
    replace_mods = []
    queue = [(model, "")]
    while queue:
        parent, _ = queue.pop(0)
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm1d):
                replace_mods.append((parent, name, child))
            else:
                queue.append((child, name))
    return replace_mods
